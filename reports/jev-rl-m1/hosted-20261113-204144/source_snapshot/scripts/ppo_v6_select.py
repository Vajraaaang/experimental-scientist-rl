"""Bounded selection on tuning/confirmation only, before any new test generation."""
from pathlib import Path
import json
import numpy as np
import torch
from ppo_efficiency_v6 import ROOT,OUT,CALIBRATION,SEEDS,setup,digest
from ppo_cost_study import load_model,model_path
from experimentlab.cost_policy import PublicCostPolicy
from experimentlab.private_cache import CachedScientistEnv
from experimentlab.ppo_v4 import collect_batch
from experimentlab.evaluation import summarize
from experimentlab.cli import write_json

ZERO={'duration_penalty':0.,'dose_penalty':0.,'stop_bonus':0.}


def evaluate_records(records,calibration,tasks,context):
    config,_,_,_,cache,_,_=context;runs=[]
    for record in records:
        model=PublicCostPolicy(load_model(record['path']),**calibration);envs=[CachedScientistEnv(t,config,cache) for t in tasks]
        episodes,_=collect_batch(model,envs,'cpu',deterministic=True)
        rows=[]
        for env,episode in zip(envs,episodes):
            row=dict(env.private_result);row.update(return_=sum(episode['rewards']),assays=env.count,online_seconds=0.,actions=episode['actions']);rows.append(row)
        runs.append({'summary':summarize(rows),'rows':rows})
    return {'mean_return':float(np.mean([r['summary']['mean_return'] for r in runs])),
            'mean_cost':float(np.mean([r['summary']['mean_variable_cost'] for r in runs])),
            'mean_mse':float(np.mean([r['summary']['aggregate_rmse']**2 for r in runs])),
            'runs':runs}


def main():
    torch.set_num_threads(1)
    if (OUT/'selection.json').exists():raise SystemExit('selection already finalized; refusing overwrite')
    context=setup();tune,confirm=context[2:4]
    main_records=json.loads((OUT/'main-registry.json').read_text())
    current=[{'seed':s,'path':str(model_path('trained',s))} for s in SEEDS]
    imitation=[{'seed':s,'path':str(model_path('imitation',s))} for s in SEEDS]
    calibrations={name:json.loads((OUT/f'calibration-{name}.json').read_text())['parameters'] for name in ('main_ppo','imitation')}
    candidates={'previous_efficient':(current,CALIBRATION),'main_training':(main_records,CALIBRATION),
                'main_calibrated':(main_records,calibrations['main_ppo'])}
    for name in ('self_curriculum','self_reset'):
        path=OUT/f'calibration-{name}.json'
        if path.exists():
            data=json.loads(path.read_text());candidates[name]=(data['records'],data['parameters'])
    protocol={'scope':'development_only','candidates':{k:{'records':v[0],'calibration':v[1]} for k,v in candidates.items()},
              'rule':'rank original mean tuning return; promote first candidate passing confirmation point gates; otherwise retain previous efficient PPO',
              'confirmation_gates':{'minimum_return_improvement_vs_previous':0.,'minimum_cost_reduction_vs_previous':.05,'maximum_mse_increase_vs_imitation':.0001},
              'all_five_seeds_retained':True,'extra_optimization_compute_and_per_seed_selection_disclosed':True,'fresh_test_generated':False}
    write_json(OUT/'final-selection-protocol.json',protocol)
    tuning={name:evaluate_records(*candidate,tune,context) for name,candidate in candidates.items()}
    baseline=evaluate_records(current,CALIBRATION,confirm,context)
    im_confirm=evaluate_records(imitation,ZERO,confirm,context)
    confirmation={};chosen='previous_efficient'
    for name in sorted(tuning,key=lambda n:tuning[n]['mean_return'],reverse=True):
        result=evaluate_records(*candidates[name],confirm,context);confirmation[name]=result
        if (result['mean_return']>=baseline['mean_return'] and
            baseline['mean_cost']-result['mean_cost']>=.05 and
            result['mean_mse']-im_confirm['mean_mse']<=.0001):
            chosen=name;break
    records,calibration=candidates[chosen]
    registry={}
    methods={'selected_ppo':(records,calibration),'v6_trained_ppo':(main_records,CALIBRATION),
             'v5_efficient_ppo':(current,CALIBRATION),'v5_paired_ppo':(current,ZERO),
             'curriculum_ppo':([{'seed':s,'path':str(model_path('ppo',s))} for s in SEEDS],ZERO),
             'imitation':(imitation,ZERO),'v5_calibrated_imitation':(imitation,{'duration_penalty':0.,'dose_penalty':16.,'stop_bonus':.1}),
             'calibrated_imitation':(imitation,calibrations['imitation']),
             'imitation_ppo':([{'seed':s,'path':str(model_path('imitation_ppo',s))} for s in SEEDS],ZERO)}
    for method,(files,parameters) in methods.items():
        assert [r['seed'] for r in files]==SEEDS
        for record in files:
            registry[f'{method}-seed{record["seed"]}']={**record,'path':str(Path(record['path']).resolve()),'sha256':digest(record['path']),'calibration':parameters}
    description=(f"Development selected `{chosen}` with deployment calibration {calibration}. All five seeds remain retained. "
                 "The main continuation compares two three-million-interaction settings per seed; each seed's checkpoint/setting is selected on original tuning return. "
                 "Additional bounded pilots and any self-curriculum restarts are explicitly counted. This is development-selected continuation with more compute, not an equal-compute algorithm comparison. "
                 "The selected paths and calibration are fully recorded; public calibration is distinguished from gradient learning.")
    result={'development_selection_complete':True,'selected_name':chosen,'selected_calibration':calibration,
            'selected_method_description':description,'registry':registry,'tuning':tuning,'confirmation':confirmation,
            'confirmation_previous':baseline,'confirmation_imitation':im_confirm,'protocol':protocol}
    write_json(OUT/'selection.json',result)
    print(json.dumps({'selected':chosen,'calibration':calibration,'tuning':{k:{a:v[a] for a in ('mean_return','mean_cost','mean_mse')} for k,v in tuning.items()},'confirmation':{k:{a:v[a] for a in ('mean_return','mean_cost','mean_mse')} for k,v in confirmation.items()}},indent=2),flush=True)


if __name__=='__main__':main()
