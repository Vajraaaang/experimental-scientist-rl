"""Matched development evaluation for the revised unfiltered study.

Run after or during training to record baselines on exactly its validation
mechanisms and goals. Learned checkpoints can be evaluated with --learned.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from research_stage import UnfilteredBank,quadrature_family,unfiltered_tasks
from experimentlab.cli import write_json
from experimentlab.envs import PilotConfig,ScientistEnv
from experimentlab.tasks import continuous_family,mechanism_hash
from experimentlab.inference import Belief
from experimentlab.evaluation import run_strategy,reference_result,summarize,paired_interval
from experimentlab.learning import ActorCritic,load_checkpoint,collect_episode
from experimentlab.bo import DiscreteBO

ROOT=Path(__file__).resolve().parents[1]


def bo_run(task,bank):
    env=ScientistEnv(task,bank.config); obs,_=env.reset()
    belief=Belief(bank,task.target); belief.update(obs['history'][0]); selector=DiscreteBO(bank,belief)
    tick=time.perf_counter(); total=0.; choices=[]
    while True:
        action=selector.choose(obs,belief); choices.append(action)
        obs,reward,done,_,_=env.step(action); total+=reward
        if done: break
        belief.update(obs['history'][-1])
    row=dict(env.private_result)
    row.update(return_=total,assays=env.count,online_seconds=time.perf_counter()-tick,choices=choices)
    return row


def learned_run(path,bank,tasks):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=ActorCritic(**state['architecture']); load_checkpoint(path,model); model.eval()
    rows=[]; shared=[]
    for task in tasks:
        env=ScientistEnv(task,bank.config); tick=time.perf_counter()
        episode=collect_episode(model,env,'cpu',deterministic=True)
        row=dict(env.private_result); row.update(return_=sum(episode['rewards']),assays=env.count,online_seconds=time.perf_counter()-tick)
        rows.append(row)
        replay=ScientistEnv(task,bank.config); obs,_=replay.reset(); belief=Belief(bank,task.target)
        belief.update(obs['history'][0]); tick=time.perf_counter(); total=0.
        for a in episode['actions'][:-1]:
            obs,r,_,_,_=replay.step(a); total+=r; belief.update(obs['history'][-1])
        _,r,_,_,_=replay.step(belief.commitment()); total+=r
        row=dict(replay.private_result); row.update(return_=total,assays=replay.count,online_seconds=time.perf_counter()-tick)
        shared.append(row)
    return {'summary':summarize(rows),'shared_estimator_ablation':summarize(shared),'rows':rows,'shared_rows':shared}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--learned',action='store_true')
    p.add_argument('--output',default=str(ROOT/'reports/unfiltered-v3'))
    args=p.parse_args(); output=Path(args.output).resolve(); torch.set_num_threads(1)
    assessment=json.loads((output/'assessment.json').read_text())
    config=PilotConfig(**assessment['config']); bank=UnfilteredBank(quadrature_family(2),config)
    tasks=unfiltered_tasks(continuous_family(8401,1024),bank,128,3042)
    hashes=[mechanism_hash(t.mechanism) for t in tasks]
    if not args.learned:
        schedules={'prior':(),'fixed':tuple(assessment['tuning']['schedule']),
                   'fixed_quality':tuple(assessment['quality_tuning']['schedule']),
                   'random':(),'eig':(),'voi':(),'bo':()}
        rows={m:[] for m in schedules}; rows['reference']=[]
        for i,task in enumerate(tasks):
            for method,schedule in schedules.items():
                if method=='bo': row=bo_run(task,bank)
                else: row,_=run_strategy(task,bank,'fixed' if method.startswith('fixed') else method,schedule,planner_draws=12,seed=i+440)
                rows[method].append(row)
            rows['reference'].append(reference_result(task,bank))
            if (i+1)%16==0: print(f'matched baselines {i+1}/{len(tasks)}',flush=True)
        result={'scope':'paired_development_validation_unfiltered','mechanism_hashes':hashes,
                'summary':{k:summarize(v) for k,v in rows.items()},'rows':rows}
        write_json(output/'matched-baselines.json',result)
        print(json.dumps(result['summary'],indent=2),flush=True)
    else:
        result={}
        for checkpoint in sorted(output.glob('*-seed*.pt')):
            print(f'evaluating {checkpoint.name}',flush=True)
            result[checkpoint.stem]=learned_run(checkpoint,bank,tasks)
            write_json(output/'matched-learned.json',{'scope':'paired_development_validation_unfiltered','mechanism_hashes':hashes,'models':result})
        if not result: raise SystemExit('No completed checkpoints present')
        comparisons={}
        for seed in (0,1,2):
            im=result.get(f'recurrent-imitation-seed{seed}')
            if im is None: continue
            for method in ('ppo','imitation_ppo'):
                other=result.get(f'recurrent-{method}-seed{seed}')
                if other:
                    comparisons[f'{method}-minus-imitation-seed{seed}']=paired_interval([r['return_'] for r in other['rows']],
                                          [r['return_'] for r in im['rows']],groups=hashes)
        write_json(output/'paired-learning-differences.json',comparisons)
        print(json.dumps(comparisons,indent=2),flush=True)


if __name__=='__main__': main()
