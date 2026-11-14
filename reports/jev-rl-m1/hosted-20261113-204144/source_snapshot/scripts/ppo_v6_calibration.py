"""Symmetric public-price calibration on the v6 tuning slice only."""
from pathlib import Path
import argparse
import json
import sys
import time
import torch
from ppo_efficiency_v6 import setup,OUT,CALIBRATION,SEEDS
from ppo_cost_study import load_model,model_path,evaluate_models
from experimentlab.cli import write_json

GRIDS={'duration_penalty':[0.,2.,4.,6.,8.,12.,16.,24.,32.],
       'dose_penalty':[0.,2.,4.,8.,16.,32.],
       'stop_bonus':[0.,.25,.5,1.,2.,4.]}


def calibrate(name,records,context,output=OUT):
    config,_,tuning,confirmation,cache,bank,_=context
    models=[load_model(r['path']) for r in records];scores={};history=[];tick=time.perf_counter()
    parameters=CALIBRATION.copy() if name!='imitation' else {k:0. for k in GRIDS}
    def score(p):
        key=tuple(p[k] for k in GRIDS)
        if key not in scores:
            metrics=evaluate_models(models,tuning,config,cache,p);scores[key]=metrics
            history.append({'parameters':dict(p),'metrics':metrics})
        return scores[key]
    original=score(parameters)
    for _ in range(2):
        for key,values in GRIDS.items():
            trials=[({**parameters,key:value},score({**parameters,key:value})) for value in values]
            eligible=[trial for trial in trials if trial[1]['aggregate_rmse']**2-original['aggregate_rmse']**2<=.0001]
            parameters=max(eligible,key=lambda trial:trial[1]['mean_return'])[0]
    zero={k:0. for k in GRIDS}
    # Include both uncalibrated and initial calibrated policies; no degradation is forced.
    candidates=[parameters,zero,CALIBRATION]
    eligible=[p for p in candidates if score(p)['aggregate_rmse']**2-original['aggregate_rmse']**2<=.0001]
    parameters=max(eligible,key=lambda p:score(p)['mean_return'])
    result={'name':name,'records':records,'parameters':parameters,'tuning_initial':original,
            'tuning_selected':score(parameters),'tuning_uncalibrated':score(zero),'search':history,
            'seconds':time.perf_counter()-tick,'confirmation_opened_for_this_calibration':False}
    write_json(output/f'calibration-{name}.json',result)
    print(json.dumps({'method':name,'parameters':parameters,'tuning_selected':result['tuning_selected'],'seconds':result['seconds']}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',default='v5_ppo');parser.add_argument('--registry')
    args=parser.parse_args();torch.set_num_threads(1);context=setup()
    protocol={'grids':GRIDS,'passes':2,'criterion':'original mean tuning return; raw-MSE increase<=0.0001 vs initial policy',
              'symmetric_search_offered_to_imitation':True,'justification':'v5 duration coefficient hit upper grid boundary4',
              'fresh_locked_generated':False}
    if not (OUT/'extended-calibration-protocol.json').exists():write_json(OUT/'extended-calibration-protocol.json',protocol)
    if args.registry:records=json.loads(Path(args.registry).resolve().read_text())
    else:records=[{'seed':s,'path':str(model_path('imitation' if args.method=='imitation' else 'trained',s))} for s in SEEDS]
    calibrate(args.method,records,context)


if __name__=='__main__':main()
