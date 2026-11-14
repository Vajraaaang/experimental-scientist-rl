"""Render the frozen efficiency comparison; never select or train a method."""
from pathlib import Path
import json
import os
import numpy as np
from ppo_efficiency_v6 import ROOT,OUT
from ppo_v6_locked import read_packet
from experimentlab.cli import write_json
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'reports/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def averaged(values):
    result={k:float(np.mean([v[k] for v in values])) for k in values[0]}
    result['aggregate_rmse']=float(np.sqrt(np.mean([v['aggregate_rmse']**2 for v in values])))
    return result


def interval(value):return f"{value['mean']:.6g} (95% CI [{value['ci95'][0]:.6g}, {value['ci95'][1]:.6g}])"


def main():
    data=read_packet(OUT/'fresh-locked/results.json.gz');selection=json.loads((OUT/'selection.json').read_text())
    methods=dict(data['baseline_summary']);shared={};stochastic={}
    names=sorted({name.split('-seed')[0] for name in data['neural']})
    for name in names:
        runs=[data['neural'][f'{name}-seed{s}'] for s in range(20,25)]
        methods[name]=averaged([r['summary'] for r in runs]);shared[name]=averaged([r['shared_summary'] for r in runs])
        if 'stochastic' in runs[0]:stochastic[name]=averaged([r['stochastic']['summary'] for r in runs])
    accounting={'physical_interactions':0,'sampled_policy_decisions':0,'training_validation_seconds_sum':0.,'runs':{}}
    for path in sorted((OUT/'development').glob('*/*/result.json')):
        r=json.loads(path.read_text());key=str(path.parent.relative_to(OUT))
        accounting['physical_interactions']+=r['transitions'];accounting['sampled_policy_decisions']+=r['policy_decisions']
        accounting['training_validation_seconds_sum']+=r['training_and_validation_seconds']
        accounting['runs'][key]={k:r[k] for k in ('transitions','policy_decisions','training_and_validation_seconds','selected_at_transitions','maximum_initial_logp_error')}
    interrupted=json.loads((OUT/'interrupted-controls.json').read_text())
    accounting['interrupted_runs']=interrupted
    accounting['interrupted_interactions_lower']=sum(r['last_persisted_interactions'] for r in interrupted['controls'].values())
    accounting['interrupted_interactions_upper']=sum(r['interaction_upper_bound'] for r in interrupted['controls'].values())
    summary={'methods':methods,'shared_final_estimator':shared,'stochastic':stochastic,
             'differences':data['differences'],'gap_closure_components':data['gap_closure_components'],
             'efficiency_gap_closed':data['efficiency_gap_closed'],'compute':accounting,
             'evaluation_seconds_sum':data['evaluation_seconds_sum'],'selection':selection['selected_method_description']}
    write_json(OUT/'summary.json',summary)
    labels=['v5_efficient_ppo','selected_ppo','imitation','calibrated_imitation','imitation_ppo']
    titles=['Previous\nefficient PPO','Selected\nPPO','Imitation','Calibrated\nimitation','Imitation\nthen PPO']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));x=np.arange(len(labels));colors=['#94a3b8','#2563eb','#14b8a6','#0d9488','#a78bfa']
    for ax,key,title in zip(axes,['mean_variable_cost','aggregate_rmse'],['Experimental spending','Final target RMSE']):
        ax.bar(x,[methods[k][key] for k in labels],color=colors)
        for i,name in enumerate(labels):
            values=[data['neural'][f'{name}-seed{s}']['summary'][key] for s in range(20,25)]
            ax.scatter(np.linspace(i-.13,i+.13,5),values,color='black',s=15,zorder=3)
        ax.set_xticks(x,titles);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();figdir=OUT/'figures';figdir.mkdir(exist_ok=True);fig.savefig(figdir/'fresh-efficiency.png',dpi=180);plt.close(fig)
    status='met' if data['efficiency_gap_closed'] else 'did not meet'
    lines=['# PPO efficiency v6: another fresh locked comparison','',
           f"The selected policy {status} the predeclared efficiency-gap closure screen against both strong imitation comparators. All previous locked studies and fixed designs remain unchanged. The result below was opened only after development selection and freezing.",'',
           '## Independent final comparison','',
           'Five retained training seeds, 4,096 new mechanisms from generator seed 97401 and episode seed 105442. The unfiltered task, GRU64 (32,814 parameters), actual fees, reward, noise and budget 8 remain unchanged. Final scoring uses the original uncached float64 reference dynamics and independent final cohorts. Greedy deployment is primary.','',
           '| Method | Final RMSE | Variable cost | Mean return |','|---|---:|---:|---:|']
    order=['prior','fixed','fixed_quality','additive','random','bo','eig','voi','curriculum_ppo','v5_efficient_ppo','v6_trained_ppo','selected_ppo','imitation','v5_calibrated_imitation','calibrated_imitation','imitation_ppo','reference']
    for name in order:
        if name not in methods:continue
        r=methods[name];label=name.replace('_',' ')
        if name=='reference':label='private nominal reference (privileged)'
        lines.append(f"| {label} | {r['aggregate_rmse']:.5f} | {r['mean_variable_cost']:.3f} | {r['mean_return']:.5f} |")
    lines+=['','Compared with each named comparator, positive cost/MSE differences mean the selected PPO is worse; positive return difference means PPO is better. Intervals resample both training seeds and paired mechanisms.','']
    for name in ['imitation','calibrated_imitation','v5_efficient_ppo']:
        d=data['differences'][f'selected_vs_{name}']
        lines.append(f"- Versus {name.replace('_',' ')}: cost increase {interval(d['cost_increase'])}; raw MSE increase {interval(d['mse_increase'])}; return improvement {interval(d['return_improvement'])}.")
    lines+=['',
            'Closure requires upper two-sided 95% bounds of at most 0.10 additional cost and 0.0001 raw MSE, plus a lower return-difference bound of at least -0.005, against both unchanged and newly calibrated imitation. These are predeclared engineering noninferiority margins, not proof of identical performance. Zero-margin differences remain visible above.','',
            f"Screen components: `{data['gap_closure_components']}`.",'',
            '![Fresh cost and quality](../reports/ppo-efficiency-v6/figures/fresh-efficiency.png)','',
            '## Development findings and selected method','',
            'Previous paired continuation updates had near-zero policy KL and zero clipping. Direct stronger-update controls mostly changed probabilities without changing greedy assays. A new public-history belief audit separated excess spending from an interpretation deficit in one seed. Longer temporary-temperature exploration repaired some interpretation; all bounded controls and unsuccessful candidates are retained.','',
            selection['selected_method_description'],'',
            'Public-price calibration uses declared duration/dose prices and public experiment count only. If present, it is explicit post-processing and is not credited as a PPO gradient discovery. The same extended calibration search is offered to imitation, with unchanged imitation, previous calibration and imitation-plus-PPO preserved.','',
            'Temporary training temperature is used identically in old and new probabilities. Deployment returns to temperature 1. Any cheap-assay selector curriculum uses predetermined random public low-dose designs, full scientific fees and actual terminal rewards; forced assay positions are excluded from actor loss. No planner labels, true mechanisms, unpurchased observations or evaluation randomness enter the policy.','',
            'Training uses the original independent 4,096-mechanism pool. New tuning and confirmation each contain 512 mechanisms from development generator indices [2048,4096); confirmation remains development evidence. Checkpoint/candidate/calibration selection uses these development slices only. All five seeds remain retained.','',
            f"A common final estimator on selected PPO's purchased assays has RMSE {shared['selected_ppo']['aggregate_rmse']:.5f}, compared with its learned commitments at {methods['selected_ppo']['aggregate_rmse']:.5f}. The estimator ablation separates evidence selection from interpretation; replay timing is not end-to-end policy latency.",'',
            '## Compute, preservation and limits','',
            f"Two final restart pilots were interrupted by disk exhaustion and excluded from promotion. Their uncompleted additional work is bounded between {accounting['interrupted_interactions_lower']:,} and {accounting['interrupted_interactions_upper']:,} interactions, separately from the exact completed-run totals. Disposable project install/cache directories were removed; completed studies and baselines were preserved. Final evaluation packets use atomic lossless gzip writes.",'',
            f"This continuation consumed {accounting['physical_interactions']:,} new physical interactions and {accounting['sampled_policy_decisions']:,} sampled policy decisions across completed controls and main runs. Summed training/validation time is {accounting['training_validation_seconds_sum']:.1f} seconds across parallel processes, not elapsed wall time. Evaluation shard wall times sum to {data['evaluation_seconds_sum']:.1f} seconds. The original pretraining/teacher costs are additional historical compute, not hidden inside these counts.",'',
            f"The current native complete256-episode workload measured CPU {json.loads((OUT/'device-benchmark.json').read_text())['devices']['cpu']['median_transitions_per_second']:.0f} versus MPS {json.loads((OUT/'device-benchmark.json').read_text())['devices']['mps']['median_transitions_per_second']:.0f} transitions/s, with CPU training running concurrently. Fallback was unset.",'',
            'The exact evaluator-owned simulator cache accelerates training only. Native CPU execution remains complete; no larger network or GPU-only dependency was introduced. Full rollout/update measurements already favored CPU at this model size. The fresh test uses uncached scientific dynamics.','',
            'The exhaustive 10,373-design fixed frontier, quality schedule, additive single-control identification, prior, random, direct-assay BO, EIG and VOI remain in the comparison. Fixed/random/BO/EIG/VOI share the same final estimator. Setup cost 3 lies outside variable assay cost; the informed reference has privileged mechanism access and zero assay spending.','',
            'All earlier study artifacts are checked against pre-continuation hashes. The freeze includes model hashes and all existing evaluation/library helper sources. Source and model integrity is verified in each shard and at reduction. Completed results refuse overwrite. No fresh-test outcome was used for further tuning.','',
            'The task remains a continuous-parameter holdout within familiar motifs and original parameter ranges, with culture-offset SD zero. Structural extrapolation, culture variation and real biological usefulness remain unqualified. Five training seeds limit optimization uncertainty; more mechanisms do not replace more independent optimization seeds.','']
    (ROOT/'docs/ppo_efficiency_v6.md').write_text('\n'.join(lines))
    print('Rendered v6 report,summary and figure.',flush=True)


if __name__=='__main__':main()
