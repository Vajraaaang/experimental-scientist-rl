"""Freeze once, evaluate disjoint shards, then reduce without method selection."""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import gzip
import io
import json
import time
import numpy as np
import torch
from ppo_efficiency_v6 import ROOT,OUT,digest
from ppo_cost_locked import evaluate_record
from ppo_v4_locked import hierarchical_difference
from research_stage import UnfilteredBank,quadrature_family,unfiltered_tasks
from evaluate_research import bo_run
from experimentlab.envs import PilotConfig
from experimentlab.tasks import continuous_family,mechanism_hash
from experimentlab.evaluation import run_strategy,reference_result,summarize
from experimentlab.cli import write_json,code_hash,clean

LOCK=OUT/'fresh-locked'
PARTS=8


def write_packet(path,data):
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    try:
        with temporary.open('wb') as raw:
            with gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=6,mtime=0) as compressed:
                with io.TextIOWrapper(compressed,encoding='utf-8') as text:
                    json.dump(clean(data),text,indent=2,allow_nan=False)
                    text.write('\n')
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_packet(path):
    with gzip.open(Path(path).resolve(),'rt') as handle:return json.load(handle)


def read_freeze():
    freeze=json.loads((LOCK/'freeze.json').read_text())
    assert freeze['source_hash']==code_hash(),'scientific library changed after freeze'
    assert digest(OUT/'selection.json')==freeze['selection_sha256'],'development selection changed after freeze'
    for path,sha in freeze['evaluation_sources'].items():assert digest(ROOT/path)==sha,path
    assert all(digest(r['path'])==r['sha256'] for r in freeze['models'].values()),'model changed after freeze'
    return freeze


def tasks_for(freeze):
    config=PilotConfig(**freeze['config']);bank=UnfilteredBank(quadrature_family(2),config)
    tasks=unfiltered_tasks(continuous_family(freeze['pool_seed'],freeze['pool_size']),bank,freeze['mechanisms'],freeze['episode_seed'])
    assert [mechanism_hash(t.mechanism) for t in tasks]==freeze['mechanism_hashes']
    return config,bank,tasks


def freeze_once():
    if (LOCK/'freeze.json').exists():raise SystemExit('v6 freeze already exists; refusing replacement')
    selection=json.loads((OUT/'selection.json').read_text())
    assert selection['development_selection_complete']
    preserved=json.loads((OUT/'preserved-studies.json').read_text())
    assert all(digest(ROOT/path)==sha for path,sha in preserved.items()),'prior studies changed'
    models=selection['registry']
    assert all(digest(r['path'])==r['sha256'] for r in models.values())
    config=PilotConfig(final_cultures=128);bank=UnfilteredBank(quadrature_family(2),config)
    tasks=unfiltered_tasks(continuous_family(97401,8192),bank,4096,105442)
    hashes=[mechanism_hash(t.mechanism) for t in tasks]
    forbidden=set()
    for name in ('ppo-v4/development-manifest.json','ppo-cost-v5/development-manifest.json','ppo-efficiency-v6/development-manifest.json'):
        data=json.loads((ROOT/'reports'/name).read_text())
        for key in ('train','validation','tuning','confirmation'):forbidden.update(data.get(key,[]))
    for name in ('unfiltered-v3/locked-test/freeze.json','ppo-v4/fresh-locked/freeze.json','ppo-cost-v5/fresh-locked/freeze.json'):
        forbidden.update(json.loads((ROOT/'reports'/name).read_text())['mechanism_hashes'])
    forbidden.update(mechanism_hash(m) for m in bank.mechanisms)
    assert len(set(hashes))==4096 and not(set(hashes)&forbidden)
    designs=json.loads((ROOT/'reports/unfiltered-v3/exhaustive-fixed.json').read_text())
    evaluation_sources={str(path.relative_to(ROOT)):digest(path) for path in sorted((ROOT/'src').rglob('*.py'))}
    for path in sorted((ROOT/'scripts').glob('*.py')):
        evaluation_sources[str(path.relative_to(ROOT))]=digest(path)
    freeze={'scope':'another_fresh_parameter_holdout_v6','mechanisms':4096,'pool_seed':97401,'pool_size':8192,'episode_seed':105442,
            'mechanism_hashes':hashes,'config':asdict(config),'models':models,'source_hash':code_hash(),
            'evaluation_sources':evaluation_sources,'selection_sha256':digest(OUT/'selection.json'),
            'fixed':designs['tuning']['schedule'],'fixed_quality':designs['quality_tuning']['schedule'],
            'fixed_file_sha256':digest(ROOT/'reports/unfiltered-v3/exhaustive-fixed.json'),
            'additive_schedule':json.loads((ROOT/'reports/ppo-v4/fresh-locked/freeze.json').read_text())['additive_schedule'],
            'primary_comparators':['imitation','calibrated_imitation'],
            'closure_margins':{'cost_increase_upper':.10,'mse_increase_upper':.0001,'return_improvement_lower':-.005},
            'closure_rule':'all hierarchical two-sided95% interval bounds meet margins vs both primary imitation comparators',
            'evaluation':'greedy primary,original uncached reference dynamics and legal masks,independent final cultures',
            'secondary':'one stochastic rollout per seed/mechanism for selected PPO and imitation; shared-estimator ablations',
            'all_five_seeds_retained':True,'parts':PARTS,
            'limitations':['familiar32motif support','continuous parameters within original ranges','culture-offset SD0','five optimization seeds']}
    write_json(LOCK/'freeze.json',freeze)
    print('Frozen4096 new mechanisms,all models and evaluation sources before scoring.',flush=True)


def baseline_part(part):
    freeze=read_freeze();path=LOCK/f'baseline-part{part:02d}.json.gz'
    if path.exists():raise SystemExit('completed baseline shard exists; refusing overwrite')
    config,bank,tasks=tasks_for(freeze);indices=list(range(part,len(tasks),PARTS));tick=time.perf_counter()
    rows={name:[] for name in ('prior','fixed','fixed_quality','additive','random','bo','eig','voi','reference')}
    for j,i in enumerate(indices):
        task=tasks[i]
        for method in rows:
            if method=='reference':r=reference_result(task,bank)
            elif method=='bo':r=bo_run(task,bank)
            else:
                schedule=freeze['fixed'] if method=='fixed' else freeze['fixed_quality'] if method=='fixed_quality' else freeze['additive_schedule'] if method=='additive' else ()
                r,_=run_strategy(task,bank,'fixed' if method.startswith('fixed') else method,schedule,planner_draws=12,seed=i+440)
            rows[method].append(r)
        if (j+1)%64==0:print(f'baseline shard{part}:{j+1}/{len(indices)}',flush=True)
    read_freeze();write_packet(path,{'freeze_sha256':digest(LOCK/'freeze.json'),'indices':indices,'rows':rows,'seconds':time.perf_counter()-tick})


def neural_part(part):
    freeze=read_freeze();path=LOCK/f'neural-part{part:02d}.json.gz'
    if path.exists():raise SystemExit('completed neural shard exists; refusing overwrite')
    config,bank,tasks=tasks_for(freeze);tick=time.perf_counter();rows={}
    for name in sorted(freeze['models'])[part::PARTS]:
        print('frozen policy',name,flush=True)
        rows[name]=evaluate_record(freeze['models'][name],tasks,config,bank)
        if name.split('-seed')[0] in ('selected_ppo','imitation','calibrated_imitation'):
            rows[name]['stochastic']=evaluate_record(freeze['models'][name],tasks,config,bank,stochastic=True)
    read_freeze();write_packet(path,{'freeze_sha256':digest(LOCK/'freeze.json'),'rows':rows,'seconds':time.perf_counter()-tick})


def closure_components(difference,margins):
    return {'cost':difference['cost_increase']['ci95'][1]<=margins['cost_increase_upper'],
            'quality':difference['mse_increase']['ci95'][1]<=margins['mse_increase_upper'],
            'return':difference['return_improvement']['ci95'][0]>=margins['return_improvement_lower']}


def reduce_once():
    if (LOCK/'results.json').exists() or (LOCK/'results.json.gz').exists():raise SystemExit('v6 results already exist; refusing rerun')
    freeze=read_freeze();rows={name:[None]*freeze['mechanisms'] for name in ('prior','fixed','fixed_quality','additive','random','bo','eig','voi','reference')}
    neural={};seconds=0.;seen=[]
    for part in range(PARTS):
        base=read_packet(LOCK/f'baseline-part{part:02d}.json.gz');learn=read_packet(LOCK/f'neural-part{part:02d}.json.gz')
        assert base['freeze_sha256']==learn['freeze_sha256']==digest(LOCK/'freeze.json')
        seen.extend(base['indices']);seconds+=base['seconds']+learn['seconds']
        for method in rows:
            assert len(base['indices'])==len(base['rows'][method])
            for i,r in zip(base['indices'],base['rows'][method]):rows[method][i]=r
        assert not(set(neural)&set(learn['rows']));neural.update(learn['rows'])
    assert sorted(seen)==list(range(freeze['mechanisms']))
    assert set(neural)==set(freeze['models'])
    matrix=lambda method,key:np.array([[r[key] for r in neural[f'{method}-seed{s}']['rows']] for s in range(20,25)])
    differences={}
    for comparator in ('imitation','calibrated_imitation','v5_calibrated_imitation','imitation_ppo','v5_efficient_ppo','curriculum_ppo'):
        differences[f'selected_vs_{comparator}']={
            'cost_increase':hierarchical_difference(matrix('selected_ppo','variable_cost'),matrix(comparator,'variable_cost')),
            'mse_increase':hierarchical_difference(matrix('selected_ppo','final_mse'),matrix(comparator,'final_mse')),
            'return_improvement':hierarchical_difference(matrix('selected_ppo','return_'),matrix(comparator,'return_'))}
    fixed=np.broadcast_to(np.array([r['return_'] for r in rows['fixed']]),matrix('selected_ppo','return_').shape)
    differences['selected_vs_fixed_return']=hierarchical_difference(matrix('selected_ppo','return_'),fixed)
    components={c:closure_components(differences[f'selected_vs_{c}'],freeze['closure_margins']) for c in freeze['primary_comparators']}
    result={'freeze':freeze,'baseline_summary':{k:summarize(v) for k,v in rows.items()},'baseline_rows':rows,
            'neural':neural,'differences':differences,'gap_closure_components':components,
            'efficiency_gap_closed':all(all(v.values()) for v in components.values()),
            'evaluation_seconds_sum':seconds,'model_hashes_unchanged':True,
            'prior_studies_unchanged':all(digest(ROOT/path)==sha for path,sha in json.loads((OUT/'preserved-studies.json').read_text()).items())}
    assert result['prior_studies_unchanged'];write_packet(LOCK/'results.json.gz',result)
    print(json.dumps({'differences':differences,'closure':components,'gap_closed':result['efficiency_gap_closed']},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('freeze','baseline','neural','reduce'),required=True)
    parser.add_argument('--part',type=int,default=0);args=parser.parse_args();torch.set_num_threads(1)
    if not 0<=args.part<PARTS:raise SystemExit('invalid shard')
    if args.stage=='freeze':freeze_once()
    elif args.stage=='baseline':baseline_part(args.part)
    elif args.stage=='neural':neural_part(args.part)
    else:reduce_once()


if __name__=='__main__':main()
