"""Another fresh locked comparison for the development-selected efficiency method."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from ppo_cost_study import ROOT,OUT,model_path,load_model
from research_stage import UnfilteredBank,quadrature_family,unfiltered_tasks
from evaluate_research import bo_run
from ppo_v4_locked import hierarchical_difference
from experimentlab.cost_policy import PublicCostPolicy
from experimentlab.envs import PilotConfig,ScientistEnv
from experimentlab.tasks import continuous_family,mechanism_hash,stream
from experimentlab.ppo_v4 import collect_batch
from experimentlab.inference import Belief
from experimentlab.worlds import final_outcomes
from experimentlab.evaluation import run_strategy,reference_result,summarize
from experimentlab.cli import write_json,code_hash


def evaluate_record(record,tasks,config,bank,stochastic=False):
    model=load_model(record['path'])
    if any(record['calibration'].values()):model=PublicCostPolicy(model,**record['calibration'])
    rows=[];shared=[];hist={}
    for start in range(0,len(tasks),128):
        chunk=tasks[start:start+128];envs=[ScientistEnv(t,config) for t in chunk]
        if stochastic:torch.manual_seed(8600000+record['seed']*10000+start)
        tick=time.perf_counter();episodes,_=collect_batch(model,envs,'cpu',deterministic=not stochastic)
        seconds=(time.perf_counter()-tick)/len(chunk)
        for task,env,e in zip(chunk,envs,episodes):
            row=dict(env.private_result);row.update(return_=sum(e['rewards']),assays=env.count,online_seconds=seconds,actions=e['actions'])
            rows.append(row)
            for action in e['actions'][:-1]:
                a=env.actions[action];key=f't{a.duration:g}|panel{a.panel}|dose{max(a.intervention):g}'
                hist[key]=hist.get(key,0)+1
            if not stochastic:
                belief=Belief(bank,task.target)
                for packet in env.public_trace:belief.update(packet)
                chosen=bank.actions[belief.commitment()]
                y=final_outcomes(task.mechanism,np.array(chosen.intervention),config.final_cultures,
                                 config.offset_sd,stream(task.seed,'final'),dt=config.dt)[:,6:8]
                mse=float(np.mean((y-task.target)**2));effort=float(sum(chosen.intervention));copy=dict(row)
                copy.update(final_mse=mse,effort=effort,return_=-100*mse-config.effort_weight*effort-config.cost_weight*row['variable_cost'])
                shared.append(copy)
    result={'summary':summarize(rows),'rows':rows,'public_action_histogram':hist}
    if shared:result.update(shared_summary=summarize(shared),shared_rows=shared)
    return result


def main():
    torch.set_num_threads(1);output=OUT/'fresh-locked'
    if (output/'results.json').exists() or (output/'results.json.gz').exists():raise SystemExit('Fresh cost-test results already exist; refusing rerun')
    selection=json.loads((OUT/'selection.json').read_text())
    preserved=json.loads((OUT/'preserved-studies.json').read_text())
    for name,root in [('v3',ROOT/'reports/unfiltered-v3/locked-test'),('v4',ROOT/'reports/ppo-v4/fresh-locked')]:
        for file,sha in preserved[name].items():assert hashlib.sha256((root/file).read_bytes()).hexdigest()==sha
    assert all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in preserved['checkpoints'].values())
    fixed_file=ROOT/'reports/unfiltered-v3/exhaustive-fixed.json'
    assert hashlib.sha256(fixed_file.read_bytes()).hexdigest()==preserved['fixed_design_sha256']
    designs=json.loads(fixed_file.read_text());records={};zero={'duration_penalty':0.,'dose_penalty':0.,'stop_bonus':0.}
    for seed in range(20,25):
        choices={'original_ppo':('ppo',zero),'efficient_ppo':(selection['selected_model'],selection['selected_calibration']),
                 'paired_ppo':('trained',zero),'imitation':('imitation',zero),
                 'calibrated_imitation':('imitation',selection['imitation_calibration']),'imitation_ppo':('imitation_ppo',zero)}
        for name,(method,parameters) in choices.items():
            path=model_path(method,seed)
            records[f'{name}-seed{seed}']={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                         'seed':seed,'calibration':parameters}
    config=PilotConfig(final_cultures=128);bank=UnfilteredBank(quadrature_family(2),config)
    tasks=unfiltered_tasks(continuous_family(57401,4096),bank,2048,75442)
    hashes=[mechanism_hash(t.mechanism) for t in tasks];dev=json.loads((OUT/'development-manifest.json').read_text())
    forbidden=set(dev['train'])|set(dev['tuning'])|set(dev['confirmation'])
    for old in (ROOT/'reports/unfiltered-v3/locked-test/freeze.json',ROOT/'reports/ppo-v4/fresh-locked/freeze.json'):
        d=json.loads(old.read_text());forbidden.update(d['mechanism_hashes'])
    assert not(set(hashes)&forbidden) and len(set(hashes))==2048
    freeze={'scope':'fresh_cost_efficiency_parameter_holdout','mechanisms':2048,'pool_seed':57401,'episode_seed':75442,
            'mechanism_hashes':hashes,'config':config.__dict__,'source_hash':code_hash(),
            'runner_hash':hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
            'selection':{k:selection[k] for k in ('selected_name','selected_model','selected_calibration','imitation_calibration')},
            'models':records,'fixed':designs['tuning']['schedule'],'fixed_quality':designs['quality_tuning']['schedule'],
            'fixed_sha256':preserved['fixed_design_sha256'],
            'additive_schedule':json.loads((ROOT/'reports/ppo-v4/fresh-locked/freeze.json').read_text())['additive_schedule'],
            'primary_metrics':['experimental cost reduction vs original PPO','raw target MSE increase vs original PPO','original return improvement'],
            'mse_noninferiority_margin':.0001,'quality_rule':'upper two-sided95% CI for MSE increase <=0.0001; engineering screen,not universal criterion',
            'evaluation':'batched greedy neural inference; original legal masks,reference simulator and independent final scoring',
            'stochastic_secondary':['original_ppo','efficient_ppo','imitation','calibrated_imitation'],
            'all_five_seeds_retained':True,'prior_task_and_reward_unchanged':True,
            'limitations':['Familiar motif support and continuous parameter holdout','Primary culture-offset SD0']}
    if (output/'freeze.json').exists() and json.loads((output/'freeze.json').read_text())!=freeze:
        raise SystemExit('Existing freeze differs; cannot overwrite')
    write_json(output/'freeze.json',freeze);print('Frozen2048 new mechanisms,wrappers and checkpoint hashes before scoring.',flush=True)
    tick=time.perf_counter();base={name:[] for name in ('prior','fixed','fixed_quality','additive','random','bo','eig','voi','reference')}
    for i,task in enumerate(tasks):
        for method in base:
            if method=='reference':r=reference_result(task,bank)
            elif method=='bo':r=bo_run(task,bank)
            else:
                schedule=freeze['fixed'] if method=='fixed' else freeze['fixed_quality'] if method=='fixed_quality' else freeze['additive_schedule'] if method=='additive' else ()
                r,_=run_strategy(task,bank,'fixed' if method.startswith('fixed') else method,schedule,planner_draws=12,seed=i+440)
            base[method].append(r)
        if (i+1)%128==0:print(f'cost locked baselines {i+1}/2048',flush=True)
    neural={}
    for name,record in records.items():
        print(f'cost locked policy {name}',flush=True);neural[name]=evaluate_record(record,tasks,config,bank)
        if name.split('-seed')[0] in freeze['stochastic_secondary']:
            neural[name]['stochastic']=evaluate_record(record,tasks,config,bank,stochastic=True)
    matrix=lambda method,key:np.array([[r[key] for r in neural[f'{method}-seed{s}']['rows']] for s in range(20,25)])
    differences={}
    for comparator in ('original_ppo','imitation','calibrated_imitation'):
        differences[f'efficient_vs_{comparator}']={
            'cost_reduction':hierarchical_difference(matrix(comparator,'variable_cost'),matrix('efficient_ppo','variable_cost')),
            'mse_increase':hierarchical_difference(matrix('efficient_ppo','final_mse'),matrix(comparator,'final_mse')),
            'return_improvement':hierarchical_difference(matrix('efficient_ppo','return_'),matrix(comparator,'return_'))}
    primary=differences['efficient_vs_original_ppo']
    result={'freeze':freeze,'baseline_summary':{k:summarize(v) for k,v in base.items()},'baseline_rows':base,
            'neural':neural,'differences':differences,'seconds':time.perf_counter()-tick,
            'efficiency_screen':bool(primary['cost_reduction']['ci95'][0]>0 and
                                     primary['mse_increase']['ci95'][1]<=freeze['mse_noninferiority_margin'] and
                                     primary['return_improvement']['ci95'][0]>0),
            'model_hashes_unchanged':all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in records.values())}
    write_json(output/'results.json',result)
    print(json.dumps({'differences':differences,'efficiency_screen':result['efficiency_screen'],'seconds':result['seconds']},indent=2),flush=True)


if __name__=='__main__':main()
