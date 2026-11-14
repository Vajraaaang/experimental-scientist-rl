"""Select efficiency methods entirely on development tuning/confirmation data."""
from pathlib import Path
import json
import numpy as np
import torch
from ppo_cost_study import ROOT,OUT,setup,model_path,load_model,trace_policy
from experimentlab.cost_policy import PublicCostPolicy
from experimentlab.cli import write_json
from ppo_v4_locked import hierarchical_difference


def evaluate_candidate(context,method,parameters,tasks):
    config,train,tuning,confirmation,cache,bank=context
    runs=[]
    for seed in range(20,25):
        model=load_model(model_path(method,seed))
        if any(parameters.values()):model=PublicCostPolicy(model,**parameters)
        runs.append(trace_policy(model,tasks,config,cache))
    return {'mean_return':float(np.mean([r['mean_return'] for r in runs])),
            'mean_variable_cost':float(np.mean([r['mean_variable_cost'] for r in runs])),
            'mean_mse':float(np.mean([r['aggregate_rmse']**2 for r in runs])),
            'rows':[r['rows'] for r in runs]}


def main():
    torch.set_num_threads(1);context=setup()
    config,train,tune,confirm,cache,bank=context
    calibration=json.loads((OUT/'calibration-results.json').read_text())
    trained=json.loads((OUT/'calibration-trained-results.json').read_text())
    zero={'duration_penalty':0.,'dose_penalty':0.,'stop_bonus':0.}
    candidates={'original':('ppo',zero),'calibrated_original':('ppo',calibration['ppo']['parameters']),
                'paired_training':('trained',zero),'paired_training_calibrated':('trained',trained['trained']['parameters'])}
    protocol={'candidates':{k:{'model':v[0],'calibration':v[1]} for k,v in candidates.items()},
              'selection':'rank original mean tuning return; confirmation screen before locked test',
              'minimum_development_cost_reduction':.15,'maximum_mean_mse_increase':.0001,
              'confirmation_is_development_and_has_been_inspected':True,
              'fresh_test_not_opened':True,'seeds':[20,21,22,23,24]}
    write_json(OUT/'selection-protocol.json',protocol)
    tuning={name:evaluate_candidate(context,*value,tune) for name,value in candidates.items()}
    baseline=tuning['original']
    ranked=sorted(candidates,key=lambda n:tuning[n]['mean_return'],reverse=True)
    confirmation={};chosen='original'
    original_confirm=evaluate_candidate(context,'ppo',zero,confirm)
    for name in ranked:
        r=tuning[name]
        if baseline['mean_variable_cost']-r['mean_variable_cost']<.15:continue
        if r['mean_mse']-baseline['mean_mse']>.0001:continue
        result=evaluate_candidate(context,*candidates[name],confirm);confirmation[name]=result
        if (original_confirm['mean_variable_cost']-result['mean_variable_cost']>=.15 and
            result['mean_mse']-original_confirm['mean_mse']<=.0001 and
            result['mean_return']>original_confirm['mean_return']):
            chosen=name;break
    result={'protocol':protocol,'selected_name':chosen,'selected_model':candidates[chosen][0],
            'selected_calibration':candidates[chosen][1],'tuning':tuning,'confirmation':confirmation,
            'confirmation_original':original_confirm,'imitation_calibration':calibration['imitation']['parameters']}
    if chosen!='original':
        rows=confirmation[chosen]['rows'];base=original_confirm['rows']
        matrix=lambda data,key:np.array([[r[key] for r in s] for s in data])
        result['confirmation_paired']={
            'cost_reduction':hierarchical_difference(matrix(base,'variable_cost'),matrix(rows,'variable_cost')),
            'mse_increase':hierarchical_difference(matrix(rows,'final_mse'),matrix(base,'final_mse')),
            'return_improvement':hierarchical_difference(matrix(rows,'return_'),matrix(base,'return_'))}
    write_json(OUT/'selection.json',result)
    print(json.dumps({'selected_name':chosen,'calibration':result['selected_calibration'],
                      'tuning':{k:{x:v[x] for x in ('mean_return','mean_variable_cost','mean_mse')} for k,v in tuning.items()},
                      'confirmation_paired':result.get('confirmation_paired')},indent=2))


if __name__=='__main__':main()
