"""Development-only PPO collapse experiments. Fresh test is a separate runner."""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import sys
import time
import numpy as np
import torch

from research_stage import UnfilteredBank,quadrature_family,unfiltered_tasks
from experimentlab.cli import write_json,code_hash
from experimentlab.envs import PilotConfig,ScientistEnv
from experimentlab.tasks import continuous_family,make_task,mechanism_hash,stream
from experimentlab.private_cache import PrivateDynamicsCache,CachedScientistEnv
from experimentlab.inference import Belief
from experimentlab.learning import ActorCritic,save_checkpoint
from experimentlab.ppo_v4 import collect_batch,batch_episodes,update_ppo,behavior_distribution
from experimentlab.compute import peak_memory

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reports/ppo-v4'


def setup(output=OUTPUT):
    config=PilotConfig()
    pool=continuous_family(7401,4096)
    validation_pool=continuous_family(8401,1024)
    start=time.perf_counter();cache=PrivateDynamicsCache(pool+validation_pool,config)
    bank=UnfilteredBank(quadrature_family(2),config)
    validation=unfiltered_tasks(validation_pool,bank,128,54042)
    manifest={'train':[mechanism_hash(m) for m in pool],
              'validation':[mechanism_hash(t.mechanism) for t in validation],
              'train_pool_seed':7401,'validation_pool_seed':8401,'validation_episode_seed':54042,
              'fresh_test_pool_seed_reserved':27401,'fresh_test_not_opened':True,
              'prior':'unfiltered','config':asdict(config)}
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    write_json(output/'development-manifest.json',manifest)
    return config,pool,validation,cache,bank,time.perf_counter()-start


def task_batch(pool,config,seed,start,count,namespace=0):
    tasks=[]
    for i in range(start,start+count):
        episode_seed=87000000000+namespace*1000000000000+seed*10000000+i
        target=.5+stream(episode_seed,'target').choice([-.08,.08],2)
        mechanism=pool[(i*2053+seed*173)%len(pool)]
        tasks.append(make_task(mechanism,episode_seed,config,target))
    return tasks


def schedules_for(episodes,seed):
    schedules=[]
    for episode in episodes:
        # Public randomized designs; independent of each hidden mechanism.
        rng=np.random.default_rng(np.random.SeedSequence([seed,episode,928]))
        controls=rng.permutation(4);length=int(rng.integers(1,4))
        doses=rng.integers(0,2,length)
        schedule=[int((1+2*controls[0]+doses[0])*4+3)]
        schedule.extend(int((1+2*controls[j]+doses[j])*4) for j in range(1,length))
        schedules.append(tuple(schedule))
    return schedules


def evaluate(model,tasks,config,cache,device='cpu',schedules=None,stochastic=False,terminal_candidates=None):
    envs=[CachedScientistEnv(t,config,cache) for t in tasks]
    _,metrics=collect_batch(model,envs,device,deterministic=not stochastic,public_schedules=schedules,terminal_candidates=terminal_candidates)
    return metrics


def train_run(name,settings,seed,budget,context,output=OUTPUT,device='cpu',initial=None):
    config,pool,validation,cache,bank,_=context
    output=Path(output).resolve();destination=output/'development'/name/f'seed{seed}'
    destination.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed)
    model=ActorCritic(kind=settings.get('kind','recurrent'),hidden=64).to(device)
    if initial is not None:model.load_state_dict(torch.load(Path(initial).resolve(),map_location=device,weights_only=False)['model'])
    optimizer=torch.optim.Adam(model.parameters(),lr=settings.get('learning_rate',.0003),eps=1e-5)
    count=episodes=policy_decisions=updates=0;curve=[];evaluations=[]
    best_score=None;selected_count=None;graduated=False
    batch_size=settings.get('batch',256);tick=time.perf_counter()
    val_schedules=schedules_for(range(len(validation)),777)
    while count<budget:
        group_size=settings.get('paired_rollouts',1)
        if group_size>1:
            if batch_size%group_size or group_size>8:raise ValueError('paired rollout group must divide batch and be at most8')
            parents=task_batch(pool,config,seed,episodes//group_size,batch_size//group_size,settings.get('episode_namespace',0))
            tasks=[make_task(t.mechanism,t.seed*8+j,config,t.target) for t in parents for j in range(group_size)]
        else:tasks=task_batch(pool,config,seed,episodes,batch_size,settings.get('episode_namespace',0))
        selector_until=settings.get('selector_fraction',0.)*budget
        selector=count<selector_until
        for group in optimizer.param_groups:
            group['lr']=settings.get('selector_learning_rate',settings.get('learning_rate',.0003)) if selector else settings.get('learning_rate',.0003)
        schedule=schedules_for(range(episodes,episodes+batch_size),seed+310) if selector else None
        terminal_candidates=None
        if selector and settings.get('selector_mode')=='paired_single':
            schedule=[((2+2*((episodes+i)%4))*4+3,) for i in range(batch_size)]
            terminal_candidates=[(36,36+s[0]//4) for s in schedule]
        strength=0.
        if not selector and settings.get('curriculum',False):
            decay_end=settings.get('bias_end_fraction',.4)*budget
            strength=6.*max(0.,(decay_end-count)/max(1.,decay_end-selector_until))
        outcome_bias=settings.get('initial_outcome_bias',0.)*max(0.,1-count/max(1.,settings.get('bias_end_fraction',.4)*budget))
        followup_bias=settings.get('followup_duration_bias',0.)*max(0.,1-count/max(1.,settings.get('bias_end_fraction',.2)*budget))
        progress=count/max(1,budget)
        start_entropy=settings.get('entropy',.01)
        end_entropy=settings.get('entropy_final',start_entropy)
        entropy=start_entropy+(end_entropy-start_entropy)*min(1.,progress/settings.get('entropy_anneal_fraction',1.))
        records,rollout=collect_batch(model,[CachedScientistEnv(t,config,cache) for t in tasks],device,
                                     public_schedules=schedule,behavior_strength=strength,
                                     minimum_assays=settings.get('minimum_assays',1),maximum_assays=settings.get('maximum_assays',2),
                                     terminal_candidates=terminal_candidates,initial_outcome_bias=outcome_bias,
                                     followup_duration_bias=followup_bias)
        count+=rollout['transitions'];episodes+=batch_size;policy_decisions+=rollout['policy_decisions'];updates+=1
        batch=batch_episodes(records,device)
        prepared={}
        if group_size>1:
            from experimentlab.paired_returns import paired_trajectory_advantages
            a,t=paired_trajectory_advantages(batch,group_size)
            prepared={'prepared_advantages':a,'prepared_targets':t}
        metrics=update_ppo(model,optimizer,batch,epochs=settings.get('epochs',2),
                           clip=settings.get('clip',.15),entropy_coef=entropy,
                           gae_lambda=settings.get('gae_lambda',1.),value_coef=settings.get('value_coef',.5),
                           target_kl=settings.get('target_kl',.02),
                           advantage_std_floor=settings.get('advantage_std_floor',1e-8),
                           normalize_advantage=settings.get('normalize_advantage',True),audit_gradients=count<=10000,
                           center_advantage=settings.get('center_advantage',True),**prepared)
        record={'transitions':count,'episodes':episodes,'policy_decisions':policy_decisions,
                'selector_control':selector,'behavior_bias_strength':strength,'followup_duration_bias':followup_bias,'entropy_coef':entropy,
                'rollout':rollout,'update':metrics,'seconds':time.perf_counter()-tick}
        if count<=10000 or not curve or count//10000>curve[-1]['transitions']//10000:
            curve.append(record)
        eval_every=settings.get('evaluation_every',50000)
        if not evaluations or count//eval_every>evaluations[-1]['transitions']//eval_every or count>=budget:
            free=evaluate(model,validation,config,cache,device)
            stochastic=evaluate(model,validation[:64],config,cache,device,stochastic=True)
            control=evaluate(model,validation,config,cache,device,schedules=val_schedules) if selector or settings.get('selector_fraction',0)>0 else None
            paired=None
            if settings.get('selector_mode')=='paired_single':
                pair_schedules=[((2+2*(i%4))*4+3,) for i in range(len(validation))]
                pair_options=[(36,36+s[0]//4) for s in pair_schedules]
                paired=evaluate(model,validation,config,cache,device,schedules=pair_schedules,terminal_candidates=pair_options)
            evaluations.append({'transitions':count,'free':free,'stochastic':stochastic,'fixed_assay_control':control,'matched_paired_control':paired})
            eligible=(not selector and strength==0. and outcome_bias==0. and followup_bias==0. and progress>=settings.get('select_after_fraction',0.))
            if settings.get('selection_on_validation',False) and eligible and (best_score is None or free['mean_return']>best_score):
                best_score=free['mean_return'];selected_count=count
                save_checkpoint(destination/'selected.pt',model,optimizer,
                                {'transitions':count,'episodes':episodes,'policy_decisions':policy_decisions,'updates':updates},
                                {'manifest_path':str(Path(output).resolve()/'development-manifest.json'),
                                 'resolved_config':asdict(config),'settings':settings,'seed':seed,'prior':'unfiltered',
                                 'transition_budget':budget,'scope':'ppo_v4_development',
                                 'all_training_behavior_biases_zero_at_completion':True})
            write_json(destination/'progress.json',{'settings':settings,'seed':seed,'curve':curve,'evaluations':evaluations})
            if name.startswith('main_'):
                save_checkpoint(destination/'latest.pt',model,optimizer,
                                {'transitions':count,'episodes':episodes,'policy_decisions':policy_decisions,
                                 'updates':updates,'curve':curve,'evaluations':evaluations},
                                {'manifest_path':str(Path(output).resolve()/'development-manifest.json'),
                                 'resolved_config':asdict(config),'settings':settings,'seed':seed,'prior':'unfiltered',
                                 'transition_budget':budget,'scope':'ppo_v4_development',
                                 'all_training_behavior_biases_zero_at_completion':strength==0 and not selector and outcome_bias==0 and followup_bias==0})
            print(json.dumps({'run':name,'seed':seed,'transitions':count,'free_return':free['mean_return'],
                              'free_assays':free['mean_assays'],'buy_probability':free['initial_experiment_probability'],
                              'selector_return':None if control is None else control['mean_return'],
                              'rollout_return':rollout['mean_return'],'entropy':metrics['entropy']}),flush=True)
            if selector and control is not None and settings.get('graduation_return') is not None:
                if count>=settings.get('graduation_minimum',0) and control['mean_return']>=settings['graduation_return']:
                    graduated=True;break
    result={'name':name,'settings':settings,'seed':seed,'transitions':count,'policy_decisions':policy_decisions,
            'episodes':episodes,'updates':updates,'curve':curve,'evaluations':evaluations,
            'training_and_validation_seconds':time.perf_counter()-tick,'peak_process_rss_bytes':peak_memory(),
            'scope':'development_only','device':device,'graduated':graduated}
    result['initial_checkpoint']=None if initial is None else str(Path(initial).resolve())
    result['initial_checkpoint_sha256']=None if initial is None else hashlib.sha256(Path(initial).resolve().read_bytes()).hexdigest()
    checkpoint=destination/'checkpoint.pt'
    save_checkpoint(checkpoint,model,optimizer,{'transitions':count,'episodes':episodes,'policy_decisions':policy_decisions,'updates':updates},
                    {'manifest_path':str(Path(output).resolve()/'development-manifest.json'),
                     'resolved_config':asdict(config),'settings':settings,'seed':seed,'prior':'unfiltered',
                     'transition_budget':budget,'scope':'ppo_v4_development',
                     'initial_checkpoint_sha256':result['initial_checkpoint_sha256'],
                     'all_training_behavior_biases_zero_at_completion':strength==0 and not selector and outcome_bias==0 and followup_bias==0})
    result['checkpoint']=str(checkpoint)
    result['selected_checkpoint']=str(destination/'selected.pt') if best_score is not None else str(checkpoint)
    result['selected_at_transitions']=selected_count if selected_count is not None else count
    result['selected_validation_return']=best_score
    write_json(destination/'result.json',result)
    return result


def diagnose(context,output):
    _,_,_,_,bank,_=context
    old=json.loads((ROOT/'reports/unfiltered-v3/learned-results.json').read_text())
    expected={}
    for target in ((.42,.42),(.42,.58),(.58,.42),(.58,.58)):
        belief=Belief(bank,target)
        expected[str(target)]=(-belief.weights@belief.losses).tolist()
    data={'saved_curves':{k:v['curve'] for k,v in old.items() if 'ppo' in k},
          'prior_expected_terminal_returns':expected,
          'causal_hypothesis':'Before evidence interpretation is learned, assays add certain cost without improving an evidence-independent terminal policy. The resulting stop optimum removes future informative data.',
          'not_established':'No mask/reset/GAE-sign bug explains the collapse; shared critic interference is an unconfirmed amplifier.',
          'references':['https://spinningup.openai.com/en/latest/algorithms/ppo.html',
                        'https://stable-baselines3.readthedocs.io/en/v2.7.1/_modules/stable_baselines3/ppo/ppo.html']}
    write_json(Path(output).resolve()/'diagnosis.json',data)


def screen(context,output):
    settings={
        'original_like':{'batch':32,'epochs':4,'entropy':.01,'clip':.2,'gae_lambda':.95,'target_kl':None,'evaluation_every':10000},
        'entropy_only':{'batch':32,'epochs':4,'entropy':.1,'clip':.2,'gae_lambda':.95,'target_kl':None,'evaluation_every':10000},
        'batch_only':{'batch':256,'epochs':4,'entropy':.01,'clip':.2,'gae_lambda':.95,'target_kl':None,'evaluation_every':10000},
        'selector_control':{'batch':256,'epochs':4,'entropy':.05,'learning_rate':.001,'selector_fraction':1.,'evaluation_every':20000},
        'exploration':{'batch':256,'epochs':2,'entropy':.1,'entropy_final':.02,'learning_rate':.0003,'evaluation_every':20000},
        'curriculum':{'batch':256,'epochs':2,'entropy':.05,'entropy_final':.005,'selector_fraction':.4,'curriculum':True,
                      'bias_end_fraction':.65,'learning_rate':.0003,'evaluation_every':20000},
    }
    write_json(Path(output).resolve()/'screen-protocol.json',{'settings':settings,'seed':10,
                'budgets':{'single_factor_controls':30000,'selector':200000,'joint_candidates':200000},
                'selection':'development free-policy return and stability only; selector control is not a free-policy success',
                'fresh_locked_outcomes_opened':False})
    results={}
    for name,config in settings.items():
        budget=30000 if name in ('original_like','entropy_only','batch_only') else 200000
        results[name]=train_run(name,config,10,budget,context,output)
        write_json(Path(output).resolve()/'screen-results.json',results)
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('diagnose','screen','single'),default='screen')
    p.add_argument('--output',default=str(OUTPUT));p.add_argument('--name',default='curriculum')
    p.add_argument('--transitions',type=int,default=200000);p.add_argument('--seed',type=int,default=10)
    p.add_argument('--settings',help='absolute JSON settings file for a bounded development run')
    p.add_argument('--initial',help='absolute checkpoint of explicitly accounted earlier training phase')
    args=p.parse_args();torch.set_num_threads(1);output=Path(args.output).resolve()
    context=setup(output);diagnose(context,output)
    write_json(output/'runtime.json',{'private_cache_setup_seconds':context[-1],'source_hash':code_hash(),
               'script_hash':hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()})
    if args.stage=='screen':screen(context,output)
    elif args.stage=='single':
        if not args.settings:raise SystemExit('--settings is required for single runs')
        config=json.loads(Path(args.settings).expanduser().resolve().read_text())
        train_run(args.name,config,args.seed,args.transitions,context,output,initial=args.initial)
    else:print('Diagnosis and development manifest recorded.')


if __name__=='__main__':main()
