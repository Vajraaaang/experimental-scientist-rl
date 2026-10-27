"""Development-only action-cost diagnosis. Immutable previous studies are read-only."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from research_stage import UnfilteredBank,quadrature_family,unfiltered_tasks
from experimentlab.envs import PilotConfig,ScientistEnv
from experimentlab.private_cache import PrivateDynamicsCache,CachedScientistEnv
from experimentlab.tasks import continuous_family,mechanism_hash
from experimentlab.learning import ActorCritic,collect_episode
from experimentlab.inference import Belief
from experimentlab.cost_policy import PublicCostPolicy
from experimentlab.cli import write_json,code_hash

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/ppo-cost-v5'


def setup():
    config=PilotConfig();train=continuous_family(7401,4096)
    # New development mechanisms from continuation of the validation generator,
    # disjoint from every v4 development mechanism (which used its first1024).
    dev=continuous_family(8401,2048)[1024:]
    cache=PrivateDynamicsCache(train+dev,config);bank=UnfilteredBank(quadrature_family(2),config)
    cases=unfiltered_tasks(dev,bank,512,85441);tuning,confirmation=cases[:256],cases[256:]
    OUT.mkdir(parents=True,exist_ok=True)
    manifest={'train':[mechanism_hash(m) for m in train],
              'tuning':[mechanism_hash(t.mechanism) for t in tuning],
              'confirmation':[mechanism_hash(t.mechanism) for t in confirmation],
              'development_generator_seed':8401,'development_parent_indices':[1024,2048],
              'episode_seed':85441,'reserved_fresh_test_seed':57401,'fresh_test_size':2048,
              'primary_reward_unchanged':True,'config':config.__dict__}
    assert not(set(manifest['tuning'])&set(manifest['confirmation']))
    write_json(OUT/'development-manifest.json',manifest)
    return config,train,tuning,confirmation,cache,bank


def model_path(method,seed):
    base=ROOT/'reports/ppo-v4'
    if method=='ppo':
        return Path(json.loads((base/'main'/f'seed{seed}'/'scratch-result.json').read_text())['joint']['selected_checkpoint'])
    if method=='imitation':return base/'main'/f'seed{seed}'/'imitation/selected.pt'
    if method=='imitation_ppo':
        return Path(json.loads((base/'main'/f'seed{seed}'/'finetune-result.json').read_text())['training']['selected_checkpoint'])
    if method=='trained':
        return Path(json.loads((OUT/'development/main_paired_cost'/f'seed{seed}/result.json').read_text())['selected_checkpoint'])
    raise ValueError(method)


def load_model(path):
    state=torch.load(Path(path).resolve(),map_location='cpu',weights_only=False)
    model=ActorCritic(**state['architecture']);model.load_state_dict(state['model']);model.eval()
    return model


def trace_policy(model,tasks,config,cache,bank=None,stochastic=False,policy_seed=930000):
    rows=[];hist=Counter();by_step=defaultdict(list);long_count=proxy_count=outcome_count=repeats=0
    prefixes=[]
    for index,task in enumerate(tasks):
        env=ScientistEnv(task,config) if cache is None else CachedScientistEnv(task,config,cache)
        if stochastic:torch.manual_seed(policy_seed+index)
        tick=time.perf_counter();records=collect_episode(model,env,'cpu',deterministic=not stochastic)
        row=dict(env.private_result);row['return_']=float(sum(records['rewards']));row['assays']=env.count
        row['online_seconds']=time.perf_counter()-tick
        row['actions']=records['actions'];rows.append(row)
        seen=set()
        belief=None if bank is None else Belief(bank,task.target)
        if belief is not None:belief.update(env.public_trace[0])
        packet_cost=0.
        for step,packet in enumerate(env.public_trace[1:],1):
            a=env.actions[packet['action_id']];u=tuple(a.intervention)
            control=int(np.argmax(u)) if sum(u)>0 else -1;dose=max(u)
            hist[f't{a.duration:g}|panel{a.panel}|dose{dose:g}|control{control}']+=1
            by_step[step].append(a.cost_ticks/20)
            long_count+=a.duration==6;proxy_count+=a.panel==(4,5);outcome_count+=a.panel==(6,7)
            repeats+=packet['action_id'] in seen;seen.add(packet['action_id'])
            packet_cost+=a.cost_ticks/20
            if belief is not None:
                belief.update(packet)
                prefixes.append({'step':step,'spent':packet_cost,'public_stop_risk':float(belief.risk()),
                                 'posterior_commitment':belief.commitment(),'selected_assay':packet['action_id']})
    total=sum(r['assays'] for r in rows)
    result={'rows':rows,'mean_variable_cost':float(np.mean([r['variable_cost'] for r in rows])),
            'aggregate_rmse':float(np.sqrt(np.mean([r['final_mse'] for r in rows]))),
            'mean_return':float(np.mean([r['return_'] for r in rows])),
            'mean_assays':float(total/len(rows)),'long_assays_per_episode':long_count/len(rows),
            'proxy_assays_per_episode':proxy_count/len(rows),'outcome_assays_per_episode':outcome_count/len(rows),
            'exact_repeated_assays_per_episode':repeats/len(rows),'cost_per_assay':sum(r['variable_cost'] for r in rows)/max(1,total),
            'mean_cost_by_step':{k:float(np.mean(v)) for k,v in by_step.items()},
            'stop_counts':dict(Counter(r['assays'] for r in rows)),'action_histogram':dict(hist),'prefix_diagnostics':prefixes}
    return result


def protect_previous():
    previous={}
    for name,root in [('v3',ROOT/'reports/unfiltered-v3/locked-test'),('v4',ROOT/'reports/ppo-v4/fresh-locked')]:
        previous[name]={k:hashlib.sha256((root/k).read_bytes()).hexdigest() for k in ('freeze.json','results.json')}
    old=json.loads((ROOT/'reports/ppo-v4/fresh-locked/freeze.json').read_text())
    previous['checkpoints']=old['checkpoints']
    previous['fixed_design_sha256']=hashlib.sha256((ROOT/'reports/unfiltered-v3/exhaustive-fixed.json').read_bytes()).hexdigest()
    write_json(OUT/'preserved-studies.json',previous)


def diagnose(context):
    config,train,tuning,confirmation,cache,bank=context
    output={};tick=time.perf_counter()
    for method in ('ppo','imitation','imitation_ppo'):
        output[method]={}
        for seed in range(20,25):
            path=model_path(method,seed);model=load_model(path)
            result=trace_policy(model,tuning,config,cache,bank)
            result['checkpoint']=str(path);result['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            output[method][seed]=result
            print(json.dumps({'method':method,'seed':seed,**{k:result[k] for k in ('mean_return','mean_variable_cost','mean_assays','long_assays_per_episode','proxy_assays_per_episode')}}),flush=True)
    write_json(OUT/'action-cost-diagnosis.json',{'scope':'development tuning only','models':output,'seconds':time.perf_counter()-tick,
               'source_hash':code_hash(),'fresh_locked_opened':False})


def evaluate_models(models,tasks,config,cache,parameters=None):
    from ppo_followup import evaluate
    summaries=[]
    for model in models:
        policy=PublicCostPolicy(model,**(parameters or {})) if parameters else model
        summaries.append(evaluate(policy,tasks,config,cache))
    return {'mean_return':float(np.mean([s['mean_return'] for s in summaries])),
            'mean_variable_cost':float(np.mean([s['mean_variable_cost'] for s in summaries])),
            'mean_assays':float(np.mean([s['mean_assays'] for s in summaries])),
            'aggregate_rmse':float(np.sqrt(np.mean([s['aggregate_rmse']**2 for s in summaries]))),'seeds':summaries}


def calibrate(context,methods=('ppo','imitation'),filename='calibration-results.json'):
    config,train,tuning,confirmation,cache,bank=context
    grids={'duration_penalty':[0.,.25,.5,1.,2.,4.],
           'dose_penalty':[0.,1.,2.,4.,8.,16.],
           'stop_bonus':[0.,.1,.25,.5,1.,2.]}
    write_json(OUT/filename.replace('results','protocol'),{'grids':grids,'passes':2,'seeds':[20,21,22,23,24],
               'selection':'coordinate search on original mean tuning return; confirmation separate',
               'policy':'public duration,dose prices and public experiment count only; no masks or reward changes',
               'imitation_receives_same_search':True,'fresh_test_opened':False})
    output={}
    for method in methods:
        models=[load_model(model_path(method,s)) for s in range(20,25)]
        parameters={k:0. for k in grids};history=[];cache_scores={}
        def score(p):
            key=tuple(p[k] for k in grids)
            if key not in cache_scores:cache_scores[key]=evaluate_models(models,tuning,config,cache,p)
            return cache_scores[key]
        original=score(parameters)
        for _ in range(2):
            for key,values in grids.items():
                trials=[]
                for value in values:
                    candidate={**parameters,key:value};metrics=score(candidate)
                    trials.append((metrics['mean_return'],candidate,metrics))
                    history.append({'parameters':candidate,'metrics':metrics})
                _,parameters,_=max(trials,key=lambda x:x[0])
        result={'parameters':parameters,'tuning_original':original,'tuning_selected':score(parameters),
                'confirmation_original':evaluate_models(models,confirmation,config,cache),
                'confirmation_selected':evaluate_models(models,confirmation,config,cache,parameters),'search':history}
        output[method]=result
        write_json(OUT/filename,output)
        print(json.dumps({'method':method,'parameters':parameters,'tuning':result['tuning_selected'],
                          'confirmation':result['confirmation_selected']}),flush=True)
    return output


def train_candidate(context,name,seed,budget,settings):
    from ppo_followup import train_run
    config,train,tuning,confirmation,cache,bank=context
    mapped=(config,train,tuning,cache,bank,0.)
    return train_run(name,settings,seed,budget,mapped,OUT,initial=model_path('ppo',seed))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('diagnose','calibrate','calibrate-trained','train'),default='diagnose')
    p.add_argument('--name',default='low_entropy');p.add_argument('--seed',type=int,default=20)
    p.add_argument('--transitions',type=int,default=300000);p.add_argument('--settings')
    args=p.parse_args();torch.set_num_threads(1);context=setup()
    if not (OUT/'preserved-studies.json').exists():protect_previous()
    if args.stage=='diagnose':diagnose(context)
    elif args.stage=='calibrate':calibrate(context)
    elif args.stage=='calibrate-trained':calibrate(context,('trained',),'calibration-trained-results.json')
    else:
        if not args.settings:raise SystemExit('--settings absolute JSON path required')
        train_candidate(context,args.name,args.seed,args.transitions,json.loads(Path(args.settings).expanduser().resolve().read_text()))


if __name__=='__main__':main()
