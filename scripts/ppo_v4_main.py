"""Five independent training seeds; all selection uses development data only."""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import time
import numpy as np
import torch
from ppo_followup import setup,train_run,evaluate
from research_stage import unfiltered_tasks
from experimentlab.tasks import mechanism_hash
from experimentlab.private_cache import CachedScientistEnv
from experimentlab.inference import Belief
from experimentlab.baselines import DecisionPlanner
from experimentlab.learning import ActorCritic,collect_episode,episode_batch,imitation_update,save_checkpoint
from experimentlab.cli import write_json,code_hash

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reports/ppo-v4'

SELECTOR={'batch':64,'epochs':4,'entropy':.05,'learning_rate':.001,'selector_fraction':1.,
          'normalize_advantage':False,'evaluation_every':100000,'episode_namespace':11,
          'graduation_minimum':1000000,'graduation_return':-.51}
JOINT={'batch':128,'epochs':2,'clip':.1,'entropy':.03,'entropy_final':.001,'learning_rate':.0003,
       'normalize_advantage':False,'curriculum':True,'bias_end_fraction':.3,'initial_outcome_bias':3.,
       'minimum_assays':2,'maximum_assays':3,'value_coef':.1,'evaluation_every':100000,
       'episode_namespace':12,'selection_on_validation':True,'select_after_fraction':.5}
FINETUNE={'batch':128,'epochs':2,'clip':.1,'entropy':.01,'entropy_final':.001,'learning_rate':.0001,
          'normalize_advantage':False,'value_coef':.1,'evaluation_every':100000,'episode_namespace':13,
          'selection_on_validation':True}


def protocol():
    doc={'scope':'development_selected_followup','seeds':[20,21,22,23,24],
         'architecture':{'kind':'recurrent','hidden':64,'input_size':75,'actions':45},
         'selector':SELECTOR,'selector_cap':2000000,'joint':JOINT,'joint_budget':2000000,
         'imitation':{'demonstrations':2048,'optimizer_updates':12000,'episode_minibatch':256,
                      'learning_rate':.001,'validation_every':2000,'public_nuisance_augmentation':True},
         'finetune':FINETUNE,'finetune_budget':1000000,'critic_only_warmup_updates':500,
         'selection':'best mean deterministic development return; all seeds retained; PPO joint selection only after bias removed and >=1M joint transitions',
         'strong_baselines':'v3 exhaustive fixed/quality schedules unchanged, plus original imitation/PPO controls; BO,EIG,VOI,random and prior included',
         'fresh_test':{'reserved_pool_seed':27401,'mechanisms':1024,'not_opened':True},
         'source_hash':code_hash()}
    path=OUTPUT/'main-protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        # Source can acquire reporting tools, but experimental settings cannot drift.
        if {k:v for k,v in old.items() if k!='source_hash'}!={k:v for k,v in doc.items() if k!='source_hash'}:
            raise ValueError('main protocol already exists with different settings')
    else:write_json(path,doc)
    return doc


def create_demonstrations(context):
    config,pool,val,cache,bank,_=context
    path=OUTPUT/'teacher-2048.pt'
    if path.exists():return
    tasks=unfiltered_tasks(pool,bank,2048,75441)
    write_json(OUTPUT/'teacher-manifest.json',{'mechanisms':[mechanism_hash(t.mechanism) for t in tasks],
                'episode_generator_seed':75441,'prior':'unfiltered','teacher':'public-history one-step VOI,128 hypothetical mechanisms,12 draws',
                'true_mechanisms_inserted_in_inference_bank':False})
    torch.manual_seed(719);shell=ActorCritic(kind='recurrent',hidden=64);demos=[];tick=time.perf_counter()
    hypothetical=0
    for i,task in enumerate(tasks):
        belief=Belief(bank,task.target);planner=DecisionPlanner(bank,'voi',12);seen=[0];rng=np.random.default_rng(i+510)
        def choose(obs):
            while seen[0]<len(obs['history']):belief.update(obs['history'][seen[0]]);seen[0]+=1
            return planner.choose(obs,belief,rng)
        demos.append(collect_episode(shell,CachedScientistEnv(task,config,cache),'cpu',teacher=choose))
        hypothetical+=planner.hypothetical_outcomes
        if (i+1)%128==0:print(f'public teacher {i+1}/2048,elapsed {time.perf_counter()-tick:.1f}s',flush=True)
    torch.save(demos,path)
    write_json(OUTPUT/'teacher-compute.json',{'episodes':len(demos),'transitions':sum(len(d['actions']) for d in demos),
               'seconds':time.perf_counter()-tick,'hypothetical_outcomes':hypothetical,'source_hash':code_hash()})


def augmented_minibatch(batch,ids,config):
    out={k:v[ids] for k,v in batch.items()};tokens=out['tokens'].clone()
    tokens[:,0,5:13]=torch.randn_like(tokens[:,0,5:13])*config.readout_sd/(np.sqrt(config.cultures)*.1)
    normal=torch.randn((*tokens.shape[:2],8,config.cultures-1))
    tokens[:,:,13:21]=(config.readout_sd/.1)*torch.sqrt(normal.square().sum(-1)/config.cultures)*tokens[:,:,21:29]
    out['tokens']=tokens
    return out


def imitation(seed,context):
    config,pool,val,cache,bank,_=context
    demos=torch.load(OUTPUT/'teacher-2048.pt',weights_only=False)
    batch=episode_batch(demos,'cpu');torch.manual_seed(seed)
    model=ActorCritic(kind='recurrent',hidden=64);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    destination=OUTPUT/'main'/f'seed{seed}'/'imitation';destination.mkdir(parents=True,exist_ok=True)
    curve=[];best=-np.inf;tick=time.perf_counter()
    for step in range(12000):
        ids=torch.randint(len(demos),(256,));metrics=imitation_update(model,optimizer,augmented_minibatch(batch,ids,config))
        if (step+1)%2000==0:
            free=evaluate(model,val,config,cache);curve.append({'updates':step+1,'training':metrics,'validation':free})
            print(json.dumps({'seed':seed,'imitation_updates':step+1,'validation_return':free['mean_return']}),flush=True)
            if free['mean_return']>best:
                best=free['mean_return']
                save_checkpoint(destination/'selected.pt',model,optimizer,{'optimizer_updates':step+1,'demonstrations':len(demos)},
                                {'manifest_path':str(OUTPUT/'teacher-manifest.json'),'resolved_config':asdict(config),
                                 'seed':seed,'scope':'ppo_v4_imitation','prior':'unfiltered'})
    result={'seed':seed,'curve':curve,'selected_validation_return':best,'selected_checkpoint':str(destination/'selected.pt'),
            'seconds':time.perf_counter()-tick,'optimizer_updates':12000,'teacher_demonstrations':len(demos)}
    write_json(destination/'result.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('teacher','scratch','imitation','finetune'),required=True)
    p.add_argument('--seed',type=int,default=20);args=p.parse_args();torch.set_num_threads(1)
    doc=protocol();context=setup(OUTPUT)
    if args.arm=='teacher':create_demonstrations(context);return
    seed=args.seed
    if seed not in doc['seeds']:raise SystemExit('Seed is not in frozen development protocol')
    if args.arm=='scratch':
        prefix=train_run('main_selector',SELECTOR,seed,doc['selector_cap'],context,OUTPUT)
        if not prefix['graduated']:
            write_json(OUTPUT/'main'/f'seed{seed}'/'selector-gate-failure.json',prefix)
            print('Selector graduation failed; seed retained as a declared failure, not discarded.',flush=True)
            return
        joint=train_run('main_curriculum_ppo',JOINT,seed,doc['joint_budget'],context,OUTPUT,initial=prefix['checkpoint'])
        write_json(OUTPUT/'main'/f'seed{seed}'/'scratch-result.json',{'selector':prefix,'joint':joint,
                   'total_interactions':prefix['transitions']+joint['transitions']})
    elif args.arm=='imitation':imitation(seed,context)
    else:
        config,pool,val,cache,bank,_=context
        initial=OUTPUT/'main'/f'seed{seed}'/'imitation'/'selected.pt'
        if not initial.exists():raise SystemExit('Imitation arm must finish first')
        state=torch.load(initial,weights_only=False);model=ActorCritic(**state['architecture']);model.load_state_dict(state['model'])
        demos=torch.load(OUTPUT/'teacher-2048.pt',weights_only=False);batch=episode_batch(demos,'cpu')
        returns=torch.zeros_like(batch['rewards'])
        for i,e in enumerate(demos):returns[i,:len(e['rewards'])]=torch.tensor(np.cumsum(np.asarray(e['rewards'])[::-1])[::-1].copy(),dtype=torch.float32)
        for param in model.parameters():param.requires_grad_(False)
        for param in model.critic.parameters():param.requires_grad_(True)
        optimizer=torch.optim.Adam(model.critic.parameters(),lr=.001);tick=time.perf_counter()
        for _ in range(500):
            _,values=model(batch['tokens']);loss=(values-returns).square()[batch['valid']].mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        for param in model.parameters():param.requires_grad_(True)
        destination=OUTPUT/'main'/f'seed{seed}'/'critic-warm.pt'
        save_checkpoint(destination,model,optimizer,{'critic_updates':500},
                        {'manifest_path':str(OUTPUT/'teacher-manifest.json'),'resolved_config':asdict(config),
                         'seed':seed,'scope':'ppo_v4_critic_only','prior':'unfiltered'})
        warm_seconds=time.perf_counter()-tick
        result=train_run('main_imitation_ppo',FINETUNE,seed,doc['finetune_budget'],context,OUTPUT,initial=destination)
        write_json(OUTPUT/'main'/f'seed{seed}'/'finetune-result.json',{'critic_warmup_seconds':warm_seconds,'training':result})


if __name__=='__main__':main()
