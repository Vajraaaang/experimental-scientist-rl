"""One fresh, frozen comparison. Never select or tune from its outcomes."""
from pathlib import Path
from itertools import combinations
from dataclasses import asdict
import hashlib
import json
import time
import numpy as np
import torch
from ppo_followup import setup
from research_stage import UnfilteredBank,quadrature_family,unfiltered_tasks
from evaluate_research import bo_run,learned_run
from experimentlab.envs import PilotConfig,ScientistEnv
from experimentlab.tasks import continuous_family,mechanism_hash
from experimentlab.learning import ActorCritic,collect_episode
from experimentlab.evaluation import run_strategy,reference_result,summarize,paired_interval
from experimentlab.baselines import fixed_identification
from experimentlab.cli import write_json,code_hash

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'reports/ppo-v4'


def hierarchical_difference(a,b,samples=4000):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:raise ValueError('matched seed/mechanism shape required')
    S,N=a.shape;rng=np.random.default_rng(75043);diff=a-b
    draws=[]
    for _ in range(samples):
        si=rng.integers(S,size=S);mi=rng.integers(N,size=N)
        draws.append(float(diff[si[:,None],mi[None,:]].mean()))
    return {'mean':float(diff.mean()),'ci95':[float(x) for x in np.quantile(draws,[.025,.975])],
            'training_seeds':S,'mechanisms':N,'unit':'paired_training_seed_and_mechanism_bootstrap'}


def registry():
    files={}
    for seed in (20,21,22,23,24):
        scratch=STUDY/'main'/f'seed{seed}'/'scratch-result.json'
        if not scratch.exists():raise RuntimeError(f'Incomplete PPO seed{seed}')
        data=json.loads(scratch.read_text());files[f'curriculum_ppo-seed{seed}']=Path(data['joint']['selected_checkpoint'])
        files[f'imitation-seed{seed}']=STUDY/'main'/f'seed{seed}'/'imitation/selected.pt'
        fine=STUDY/'main'/f'seed{seed}'/'finetune-result.json'
        if not fine.exists():raise RuntimeError(f'Incomplete fine-tuning seed{seed}')
        files[f'imitation_ppo-seed{seed}']=Path(json.loads(fine.read_text())['training']['selected_checkpoint'])
    old=ROOT/'reports/unfiltered-v3'
    for seed in (0,1,2):
        files[f'legacy_ppo-seed{seed}']=old/f'recurrent-ppo-seed{seed}.pt'
        files[f'legacy_imitation-seed{seed}']=old/'converged-imitation'/f'recurrent-imitation-seed{seed}.pt'
    for name,path in files.items():
        if not path.exists():raise RuntimeError(f'Missing frozen model {name}')
        if name.startswith('curriculum_ppo') or name.startswith('imitation_ppo'):
            meta=torch.load(path,map_location='cpu',weights_only=False)['provenance']
            if not meta.get('all_training_behavior_biases_zero_at_completion'):
                raise RuntimeError('PPO selected before original unbiased training phase')
    return files


def additive_design(bank):
    config=bank.config
    calibration=unfiltered_tasks(continuous_family(7401,1024),bank,512,6041)
    scores={}
    for pair in combinations(range(4),2):
        schedule=fixed_identification(pair)
        scores[pair]=np.mean([run_strategy(t,bank,'additive',schedule)[0]['return_'] for t in calibration])
    pair=max(scores,key=scores.get)
    return fixed_identification(pair),{str(k):float(v) for k,v in scores.items()}


def stochastic_evaluation(path,bank,tasks,seed):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=ActorCritic(**state['architecture']);model.load_state_dict(state['model']);model.eval()
    rows=[]
    for i,task in enumerate(tasks):
        torch.manual_seed(9700000+seed*10000+i)
        env=ScientistEnv(task,bank.config);tick=time.perf_counter()
        e=collect_episode(model,env,'cpu',deterministic=False);row=dict(env.private_result)
        row.update(return_=sum(e['rewards']),assays=env.count,online_seconds=time.perf_counter()-tick)
        rows.append(row)
    return {'summary':summarize(rows),'rows':rows}


def main():
    output=STUDY/'fresh-locked'
    if (output/'results.json').exists():raise SystemExit('Fresh locked results exist: refusing rerun or retuning')
    torch.set_num_threads(1)
    files=registry()
    previous=json.loads((STUDY/'preserved-v3.json').read_text())
    old=ROOT/'reports/unfiltered-v3'
    assert hashlib.sha256((old/'locked-test/freeze.json').read_bytes()).hexdigest()==previous['previous_freeze_sha256']
    assert hashlib.sha256((old/'locked-test/results.json').read_bytes()).hexdigest()==previous['previous_results_sha256']
    assert hashlib.sha256((old/'exhaustive-fixed.json').read_bytes()).hexdigest()==previous['baseline_design_file_sha256']
    assert all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in previous['checkpoints'].values())
    development_config=PilotConfig()
    dev_bank=UnfilteredBank(quadrature_family(2),development_config)
    additive,additive_scores=additive_design(dev_bank)
    designs=json.loads((old/'exhaustive-fixed.json').read_text())
    config=PilotConfig(final_cultures=128);bank=UnfilteredBank(quadrature_family(2),config)
    tasks=unfiltered_tasks(continuous_family(27401,4096),bank,1024,64042)
    hashes=[mechanism_hash(t.mechanism) for t in tasks]
    manifest=json.loads((STUDY/'development-manifest.json').read_text())
    old_test=json.loads((old/'locked-test/freeze.json').read_text())['mechanism_hashes']
    assert not(set(hashes)&(set(manifest['train'])|set(manifest['validation'])|set(old_test)))
    assert not(set(hashes)&{mechanism_hash(m) for m in bank.mechanisms})
    freeze={'scope':'fresh_parameter_holdout_same_unfiltered_task','mechanisms':1024,'pool_seed':27401,
            'episode_seed':64042,'config':asdict(config),'mechanism_hashes':hashes,
            'source_hash':code_hash(),'runner_hash':hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
            'protocol':json.loads((STUDY/'main-protocol.json').read_text()),
            'checkpoints':{name:{'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in files.items()},
            'fixed':designs['tuning']['schedule'],'fixed_quality':designs['quality_tuning']['schedule'],
            'fixed_source_sha256':hashlib.sha256((old/'exhaustive-fixed.json').read_bytes()).hexdigest(),
            'additive_schedule':list(additive),'additive_calibration_scores':additive_scores,
            'primary_evaluation':'greedy learned policy; same original legal masks, zero curriculum bias, uncached reference simulator',
            'secondary_evaluation':'one stochastic rollout per seed/mechanism, plus same-assays/common-final-estimator ablation',
            'primary_metric':'mean final target MSE; report positive paired MSE reduction when the first named method improves on its comparator',
            'secondary_metrics':['variable experimental cost','return including cost and effort','success within RMSE0.04'],
            'primary_comparisons':['curriculum PPO versus imitation','imitation+PPO versus imitation','curriculum PPO versus exhaustive fixed'],
            'limitations':['Familiar motif support,held-out continuous parameters','Primary culture offset SD0','Five training seeds only']}
    if (output/'freeze.json').exists() and json.loads((output/'freeze.json').read_text())!=freeze:
        raise SystemExit('Existing freeze differs; cannot replace an opened study')
    write_json(output/'freeze.json',freeze)
    print('Fresh1024-mechanism protocol/checkpoint hashes frozen before scoring.',flush=True)
    tick=time.perf_counter()
    rows={k:[] for k in ('prior','fixed','fixed_quality','additive','random','bo','eig','voi','reference')}
    for i,task in enumerate(tasks):
        for method in rows:
            if method=='reference':r=reference_result(task,bank)
            elif method=='bo':r=bo_run(task,bank)
            else:
                schedule=freeze['fixed'] if method=='fixed' else freeze['fixed_quality'] if method=='fixed_quality' else additive if method=='additive' else ()
                r,_=run_strategy(task,bank,'fixed' if method.startswith('fixed') else method,schedule,planner_draws=12,seed=i+440)
            rows[method].append(r)
        if (i+1)%64==0:print(f'fresh baselines {i+1}/1024',flush=True)
    learned={}
    for name,path in files.items():
        print(f'fresh frozen model {name}',flush=True)
        learned[name]=learned_run(path,bank,tasks)
        if not name.startswith('legacy_'):
            seed=int(name.split('seed')[-1]);learned[name]['stochastic']=stochastic_evaluation(path,bank,tasks,seed)
    differences={}
    metric=lambda method,key='return_':np.array([[r[key] for r in learned[f'{method}-seed{s}']['rows']] for s in (20,21,22,23,24)])
    for method in ('curriculum_ppo','imitation_ppo'):
        differences[f'{method}_return_minus_imitation']=hierarchical_difference(metric(method),metric('imitation'))
        differences[f'{method}_mse_reduction_vs_imitation']=hierarchical_difference(metric('imitation','final_mse'),metric(method,'final_mse'))
        differences[f'{method}_cost_reduction_vs_imitation']=hierarchical_difference(metric('imitation','variable_cost'),metric(method,'variable_cost'))
    fixed=np.array([r['return_'] for r in rows['fixed']])
    differences['curriculum_ppo_return_minus_fixed']=hierarchical_difference(metric('curriculum_ppo'),np.broadcast_to(fixed,metric('curriculum_ppo').shape))
    for key,title in (('final_mse','mse'),('variable_cost','cost')):
        reference=np.broadcast_to(np.array([r[key] for r in rows['fixed']]),metric('curriculum_ppo',key).shape)
        differences[f'curriculum_ppo_{title}_reduction_vs_fixed']=hierarchical_difference(reference,metric('curriculum_ppo',key))
    result={'freeze':freeze,'baseline_summary':{k:summarize(v) for k,v in rows.items()},'baseline_rows':rows,
            'learned':learned,'paired_differences':differences,'seconds':time.perf_counter()-tick,
            'checkpoints_unchanged':all(hashlib.sha256(path.read_bytes()).hexdigest()==freeze['checkpoints'][name]['sha256'] for name,path in files.items()),
            'previous_v3_results_unchanged':hashlib.sha256((old/'locked-test/results.json').read_bytes()).hexdigest()==previous['previous_results_sha256']}
    write_json(output/'results.json',result)
    print(json.dumps({'baseline_summary':result['baseline_summary'],'differences':differences,'seconds':result['seconds']},indent=2),flush=True)


if __name__=='__main__':main()
