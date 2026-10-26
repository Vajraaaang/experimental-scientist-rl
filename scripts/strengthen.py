"""Extra-compute checks: exhaustive fixed designs and converged imitation.

Development only. Original negative results and initial training are retained.
"""
from pathlib import Path
from dataclasses import replace
from itertools import combinations_with_replacement
import argparse
import json
import time
import numpy as np
import torch
from research_stage import UnfilteredBank,quadrature_family,unfiltered_tasks,learned_evaluation
from experimentlab.cli import write_json
from experimentlab.envs import PilotConfig
from experimentlab.tasks import continuous_family,mechanism_hash
from experimentlab.inference import Belief
from experimentlab.evaluation import run_strategy,summarize,paired_interval
import experimentlab.evaluation as evaluation
from experimentlab.learning import ActorCritic,episode_batch,imitation_update,save_checkpoint

ROOT=Path(__file__).resolve().parents[1]


def exhaustive(output):
    output=Path(output).resolve(); data=json.loads((output/'assessment.json').read_text())
    config=PilotConfig(**data['config']); bank=UnfilteredBank(quadrature_family(2),config)
    # Public prior predictions establish no-intervention assays uninformative at
    # matched baseline and zero culture variation. This is search pruning only.
    ids=[a for a,s in enumerate(bank.actions[:36]) if np.max(np.ptp(bank.predictions[:,a,list(s.panel)],axis=0))>1e-10]
    def designs(budget,max_experiments):
        candidates=[()]
        for n in range(1,max_experiments+1):
            candidates.extend(s for s in combinations_with_replacement(ids,n) if sum(bank.actions[a].cost_ticks for a in s)<=budget*20)
        return candidates
    candidates=designs(config.budget,config.max_experiments)
    write_json(output/'exhaustive-protocol.json',{'calibration_mechanisms':512,'canonical_multiset_designs':len(candidates),
               'informative_assays':ids,'search':'all affordable multisets through four assays, plus existing ordered-pair candidates',
               'pruning':'zero-information no-intervention assays under matched-baseline zero-offset prior',
               'validation_reused_for_development':True,'locked_test_opened':False})
    calibration=unfiltered_tasks(continuous_family(7401,1024),bank,512,6041)
    tasks=unfiltered_tasks(continuous_family(8401,1024),bank,128,2042)
    original=evaluation.fixed_candidates; evaluation.fixed_candidates=designs
    try:
        schedule,tuning=evaluation.tune_fixed(bank,calibration,beam_width=0)
        quality_bank=UnfilteredBank(quadrature_family(2),replace(config,cost_weight=0.))
        quality,quality_tuning=evaluation.tune_fixed(quality_bank,calibration,beam_width=0)
    finally: evaluation.fixed_candidates=original
    rows={name:[run_strategy(t,bank,'fixed',s)[0] for t in tasks] for name,s in [('fixed_exhaustive',schedule),('fixed_quality_exhaustive',quality)]}
    comparisons={}
    gap=np.mean([r['final_mse'] for r in data['rows']['prior']])-np.mean([r['final_mse'] for r in data['rows']['reference']])
    for name,r in rows.items():
        utility=paired_interval([v['return_'] for v in data['rows']['voi']],[v['return_'] for v in r],groups=data['validation_hashes'])
        error=paired_interval([v['final_mse'] for v in r],[v['final_mse'] for v in data['rows']['voi']],groups=data['validation_hashes'])
        comparisons[name]={'utility':utility,'error':error,'raw_gap_recovery':error['mean']/gap,
                           'gate_C':bool(utility['ci95'][0]>0 and error['ci95'][0]>0 and error['mean']/gap>=.1)}
    result={'tuning':tuning,'quality_tuning':quality_tuning,'summary':{k:summarize(v) for k,v in rows.items()},
            'rows':rows,'comparison':comparisons,'gate_C':all(v['gate_C'] for v in comparisons.values())}
    write_json(output/'exhaustive-fixed.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


def converge(output):
    output=Path(output).resolve(); data=json.loads((output/'assessment.json').read_text())
    config=PilotConfig(**data['config']); bank=UnfilteredBank(quadrature_family(2),config)
    teacher=torch.load(output/'public-history-demonstrations.pt',weights_only=False)
    batch=episode_batch(teacher,'cpu')
    validation=unfiltered_tasks(continuous_family(8401,1024),bank,128,3042)
    destination=output/'converged-imitation'; destination.mkdir(parents=True,exist_ok=True)
    write_json(destination/'protocol.json',{'epochs':5000,'seeds':[0,1,2],'learning_rate':.001,
                'demonstrations':len(teacher),'architecture_hidden':64,'policy_kinds':['history','recurrent'],
                'augmentation':'resample independent matched-baseline noise and ancillary within-cohort SD features; purchased assay means and teacher labels unchanged',
                'rationale':'At zero offset and known Gaussian variance, baseline observations and within-cohort SD are independent of mechanism; sample means are sufficient.',
                'validation_mechanism_hashes':[mechanism_hash(t.mechanism) for t in validation],
                'locked_test_opened':False})
    results={}
    for kind in ('history','recurrent'):
        for seed in (0,1,2):
            torch.manual_seed(seed); model=ActorCritic(kind=kind,hidden=64)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            tick=time.perf_counter(); curve=[]
            for epoch in range(5000):
                augmented=dict(batch); tokens=batch['tokens'].clone()
                tokens[:,0,5:13]=torch.randn_like(tokens[:,0,5:13])*config.readout_sd/(np.sqrt(config.cultures)*.1)
                # ddof=0 cohort SD follows sigma*sqrt(chi2(K-1)/K).
                normal=torch.randn((*tokens.shape[:2],8,config.cultures-1))
                sd=config.readout_sd*torch.sqrt((normal**2).sum(-1)/config.cultures)/.1
                tokens[:,:,13:21]=sd*tokens[:,:,21:29]
                augmented['tokens']=tokens
                metrics=imitation_update(model,optimizer,augmented)
                if (epoch+1)%1000==0:
                    curve.append({'epoch':epoch+1,**metrics})
                    print(f'converged {kind} seed{seed} epoch{epoch+1} {metrics}',flush=True)
            elapsed=time.perf_counter()-tick
            name=f'{kind}-imitation-seed{seed}'
            results[name]={'validation':learned_evaluation(model,bank,validation),'training_seconds':elapsed,
                           'curve':curve,'scope':'converged_imitation_development'}
            path=destination/f'{name}.pt'
            save_checkpoint(path,model,optimizer,{'epochs':5000,'demonstrations':len(teacher)},
                            {'resolved_config':config.__dict__,'seed':seed,'prior':'unfiltered','scope':'converged_imitation_development'})
            state=torch.load(path,weights_only=False)
            state['provenance']['research_learning_manifest']=str(output/'learning-manifest.json')
            import hashlib
            state['provenance']['manifest_hash']=hashlib.sha256((output/'learning-manifest.json').read_bytes()).hexdigest()
            torch.save(state,path)
            write_json(destination/'results.json',results)
    print(json.dumps(results,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('fixed','imitation'))
    p.add_argument('--output',default=str(ROOT/'reports/unfiltered-v3'))
    args=p.parse_args(); torch.set_num_threads(1)
    if args.mode=='fixed': exhaustive(args.output)
    else: converge(args.output)


if __name__=='__main__': main()
