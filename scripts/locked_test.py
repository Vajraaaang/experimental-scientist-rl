"""Frozen, once-only parameter-held-out comparison for the unfiltered v3 task.

This runner refuses to overwrite a completed locked result. No method selection
or tuning occurs here. New methods require a new declared study, not a rerun.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from research_stage import UnfilteredBank,quadrature_family,unfiltered_tasks
from evaluate_research import bo_run,learned_run
from experimentlab.cli import code_hash,write_json
from experimentlab.envs import PilotConfig
from experimentlab.tasks import continuous_family,mechanism_hash
from experimentlab.evaluation import run_strategy,reference_result,summarize,paired_interval

ROOT=Path(__file__).resolve().parents[1]


def hierarchical_interval(a,b,samples=4000):
    """Paired bootstrap over training seeds and independent test mechanisms."""
    diff=np.asarray(a)-np.asarray(b)
    S,N=diff.shape;rng=np.random.default_rng(9207)
    values=[]
    for _ in range(samples):
        seed_idx=rng.integers(S,size=S);mechanism_idx=rng.integers(N,size=N)
        values.append(float(diff[seed_idx[:,None],mechanism_idx[None]].mean()))
    return {'mean':float(diff.mean()),'ci95':[float(v) for v in np.quantile(values,[.025,.975])],
            'training_seeds':S,'mechanisms':N,'unit':'paired_training_seed_and_mechanism_bootstrap',
            'limitation':'Three seeds limit inference over the training-seed distribution.'}


def main():
    study=ROOT/'reports/unfiltered-v3';output=study/'locked-test'
    if (output/'results.json').exists(): raise SystemExit('Locked results already exist; refusal to rerun or retune.')
    strongest=json.loads((study/'exhaustive-fixed.json').read_text())
    if not strongest['gate_C']: raise SystemExit('Scientific development gate not met')
    registry={}
    for seed in (0,1,2):
        for label,path in [('history_ppo',study/f'history-ppo-seed{seed}.pt'),
                           ('recurrent_ppo',study/f'recurrent-ppo-seed{seed}.pt'),
                           ('history_imitation',study/'converged-imitation'/f'history-imitation-seed{seed}.pt'),
                           ('recurrent_imitation',study/'converged-imitation'/f'recurrent-imitation-seed{seed}.pt'),
                           ('recurrent_imitation_ppo',study/'strong-finetune'/f'recurrent-imitation_ppo-strong-seed{seed}.pt')]:
            if not path.exists(): raise SystemExit(f'Missing frozen model: {path.name}')
            registry[f'{label}-seed{seed}']=path
    torch.set_num_threads(1)
    config=PilotConfig(readout_sd=.02,budget=8,final_cultures=128)
    bank=UnfilteredBank(quadrature_family(2),config)
    pool=continuous_family(9401,2048)
    tasks=unfiltered_tasks(pool,bank,512,4042)
    hashes=[mechanism_hash(t.mechanism) for t in tasks]
    manifest=json.loads((study/'learning-manifest.json').read_text())
    assert not(set(hashes)&set(manifest['train']) or set(hashes)&set(manifest['validation']))
    assert not(set(hashes)&{mechanism_hash(m) for m in bank.mechanisms})
    protocol={'scope':'locked_parameter_holdout_familiar_topology_unfiltered_v3','test_mechanisms':512,
              'pool_seed':9401,'episode_seed':4042,'config':config.__dict__,'mechanism_hashes':hashes,
              'all_generated_tasks_retained':True,'test_size_basis':'128-mechanism development intervals showed positive adaptive difference; quadruple to512 for precision before opening test.',
              'source_hash':code_hash(),'dependency_hash':hashlib.sha256((ROOT/'requirements.lock').read_bytes()).hexdigest(),
              'checkpoints':{k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in registry.items()},
              'fixed':strongest['tuning']['schedule'],'fixed_quality':strongest['quality_tuning']['schedule'],
              'primary_comparisons':['VOI versus tuned exhaustive fixed','recurrent imitation versus recurrent PPO','recurrent imitation followed by PPO versus recurrent imitation'],
              'selection':'All three seeds and fixed final training budgets; no best-seed selection',
              'limitations':['No structural or parameter-range shift claim.','Primary culture offset SD0; biological and culture-variation transfer unqualified.']}
    write_json(output/'freeze.json',protocol)
    print('Locked protocol and checkpoint hashes frozen before any test scoring.',flush=True)
    start=time.perf_counter()
    rows={m:[] for m in ('prior','fixed','fixed_quality','random','eig','voi','bo','reference')}
    for i,task in enumerate(tasks):
        for method in rows:
            if method=='reference':r=reference_result(task,bank)
            elif method=='bo':r=bo_run(task,bank)
            else:
                schedule=protocol['fixed'] if method=='fixed' else protocol['fixed_quality'] if method=='fixed_quality' else ()
                r,_=run_strategy(task,bank,'fixed' if method.startswith('fixed') else method,schedule,planner_draws=12,seed=i+440)
            rows[method].append(r)
        if (i+1)%32==0:print(f'locked baselines {i+1}/512',flush=True)
    learned={}
    for label,path in registry.items():
        print(f'locked checkpoint {label}',flush=True)
        learned[label]=learned_run(path,bank,tasks)
    summary={k:summarize(v) for k,v in rows.items()}
    differences={'voi_minus_fixed_return':paired_interval([r['return_'] for r in rows['voi']],[r['return_'] for r in rows['fixed']],groups=hashes),
                 'voi_error_reduction':paired_interval([r['final_mse'] for r in rows['fixed']],[r['final_mse'] for r in rows['voi']],groups=hashes)}
    for name in ('recurrent_ppo','recurrent_imitation_ppo'):
        a=np.array([[r['return_'] for r in learned[f'{name}-seed{s}']['rows']] for s in (0,1,2)])
        b=np.array([[r['return_'] for r in learned[f'recurrent_imitation-seed{s}']['rows']] for s in (0,1,2)])
        differences[f'{name}_minus_imitation_return']=hierarchical_interval(a,b)
    result={'protocol':protocol,'baseline_summary':summary,'baseline_rows':rows,'learned':learned,
            'paired_differences':differences,'seconds':time.perf_counter()-start,
            'frozen_checkpoint_hashes_unchanged':all(hashlib.sha256(path.read_bytes()).hexdigest()==protocol['checkpoints'][k]['sha256'] for k,path in registry.items())}
    write_json(output/'results.json',result)
    print(json.dumps({'baseline_summary':summary,'paired_differences':differences,'seconds':result['seconds']},indent=2),flush=True)


if __name__=='__main__':main()
