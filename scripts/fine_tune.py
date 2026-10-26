"""Conservative PPO refinement of a converged public-history imitation policy."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from research_stage import UnfilteredBank,quadrature_family,unfiltered_tasks,learned_evaluation
from experimentlab.cli import write_json
from experimentlab.envs import PilotConfig,ScientistEnv
from experimentlab.tasks import continuous_family,make_task
from experimentlab.learning import ActorCritic,episode_batch,collect_episode,ppo_update,save_checkpoint

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--transitions',type=int,default=100000)
    p.add_argument('--output',default=str(ROOT/'reports/unfiltered-v3'));args=p.parse_args()
    output=Path(args.output).resolve(); torch.set_num_threads(1)
    assessment=json.loads((output/'assessment.json').read_text())
    strongest=output/'exhaustive-fixed.json'
    if strongest.exists() and not json.loads(strongest.read_text())['gate_C']:
        raise SystemExit('Stronger fixed-design gate failed; substantial PPO refinement remains gated.')
    if not assessment['gate_C']: raise SystemExit('Scientific gate not met')
    config=PilotConfig(**assessment['config']); bank=UnfilteredBank(quadrature_family(2),config)
    training=unfiltered_tasks(continuous_family(7401,4096),bank,4096,3041)
    validation=unfiltered_tasks(continuous_family(8401,1024),bank,128,3042)
    teacher=torch.load(output/'public-history-demonstrations.pt',weights_only=False)
    batch=episode_batch(teacher,'cpu')
    returns=torch.zeros_like(batch['rewards'])
    for i,e in enumerate(teacher):
        returns[i,:len(e['rewards'])]=torch.tensor(np.cumsum(np.asarray(e['rewards'])[::-1])[::-1].copy(),dtype=torch.float32)
    destination=output/'strong-finetune';destination.mkdir(parents=True,exist_ok=True)
    write_json(destination/'protocol.json',{'seeds':[0,1,2],'on_policy_transitions_per_seed':args.transitions,
               'imitation':'512 public-history demos,5000 augmented epochs,hidden64 GRU',
               'critic_warmup':500,'critic_policy_frozen':True,'ppo_learning_rate':.0001,'ppo_epochs':2,'clip':.1,
               'selection':'fixed final transition budget; no validation checkpoint selection',
               'training_seed_prefix':600000,'locked_test_opened':False})
    results={}
    for seed in (0,1,2):
        path=output/'converged-imitation'/f'recurrent-imitation-seed{seed}.pt'
        if not path.exists(): raise SystemExit(f'Converged imitation seed{seed} not yet complete')
        state=torch.load(path,weights_only=False);torch.manual_seed(seed)
        model=ActorCritic(**state['architecture']);model.load_state_dict(state['model'])
        before=learned_evaluation(model,bank,validation)
        if before['mean_return']<=-.64: raise SystemExit('Imitation policy is not useful; learnability gate failed')
        for parameter in model.parameters(): parameter.requires_grad_(False)
        for parameter in model.critic.parameters(): parameter.requires_grad_(True)
        critic_optimizer=torch.optim.Adam(model.critic.parameters(),lr=.001)
        for _ in range(500):
            _,values=model(batch['tokens']);loss=((values-returns)**2)[batch['valid']].mean()
            critic_optimizer.zero_grad(set_to_none=True);loss.backward();critic_optimizer.step()
        for parameter in model.parameters(): parameter.requires_grad_(True)
        optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
        count=episodes=0; tick=time.perf_counter(); curve=[]
        while count<args.transitions:
            rollouts=[]
            for j in range(32):
                task=training[(episodes+j)%len(training)]
                task=make_task(task.mechanism,seed*10000000+episodes+j+600000,config,task.target)
                rollouts.append(collect_episode(model,ScientistEnv(task,config),'cpu'))
            count+=sum(len(r['actions']) for r in rollouts);episodes+=len(rollouts)
            metrics=ppo_update(model,optimizer,episode_batch(rollouts,'cpu'),epochs=2,clip=.1)
            if not curve or count//10000>curve[-1]['transitions']//10000:
                curve.append({'transitions':count,'metrics':metrics,'seconds':time.perf_counter()-tick})
                print(f'strong fine-tune seed{seed}: {count}/{args.transitions}',flush=True)
        after=learned_evaluation(model,bank,validation)
        name=f'recurrent-imitation_ppo-strong-seed{seed}'
        results[name]={'before':before,'validation':after,'curve':curve,'on_policy_transitions':count,
                       'training_seconds':time.perf_counter()-tick,'seed':seed}
        checkpoint=destination/f'{name}.pt'
        save_checkpoint(checkpoint,model,optimizer,{'episodes':episodes,'transitions':count},
                        {'resolved_config':config.__dict__,'seed':seed,'prior':'unfiltered','scope':'strong_imitation_ppo_development'})
        state=torch.load(checkpoint,weights_only=False)
        state['provenance']['manifest_hash']=hashlib.sha256((output/'learning-manifest.json').read_bytes()).hexdigest()
        state['provenance']['research_learning_manifest']=str(output/'learning-manifest.json')
        torch.save(state,checkpoint)
        write_json(destination/'results.json',results)
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
