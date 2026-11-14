"""Render the measured report and scientific figures from frozen results."""
from pathlib import Path
import json
import os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'reports/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from locked_test import hierarchical_interval


def main():
    study=ROOT/'reports/unfiltered-v3'
    locked=json.loads((study/'locked-test/results.json').read_text())
    initial=json.loads((study/'learned-results.json').read_text())
    refined=json.loads((study/'strong-finetune/results.json').read_text())
    cloned=json.loads((study/'converged-imitation/results.json').read_text())
    online=json.loads((ROOT/'reports/online-compute.json').read_text())
    device=json.loads((ROOT/'reports/compute-final/benchmark.json').read_text())
    methods={k:dict(v) for k,v in locked['baseline_summary'].items()}
    names=('history_ppo','recurrent_ppo','history_imitation','recurrent_imitation','recurrent_imitation_ppo')
    for name in names:
        seeds=[locked['learned'][f'{name}-seed{s}']['summary'] for s in (0,1,2)]
        mean={k:float(np.mean([r[k] for r in seeds])) for k in seeds[0]}
        mean['aggregate_rmse']=float(np.sqrt(np.mean([r['aggregate_rmse']**2 for r in seeds])))
        methods[name]=mean
    interpretation={}
    for name in ('history_imitation','recurrent_imitation','recurrent_imitation_ppo'):
        rows=[locked['learned'][f'{name}-seed{s}']['shared_estimator_ablation'] for s in (0,1,2)]
        interpretation[name]={'same_assays_common_selector_rmse':float(np.sqrt(np.mean([r['aggregate_rmse']**2 for r in rows]))),
                              'same_assays_common_selector_return':float(np.mean([r['mean_return'] for r in rows]))}
    secondary={}
    for name in ('history_imitation','recurrent_imitation','recurrent_imitation_ppo'):
        a=np.array([[r['return_'] for r in locked['learned'][f'{name}-seed{s}']['rows']] for s in (0,1,2)])
        b=np.broadcast_to(np.array([r['return_'] for r in locked['baseline_rows']['fixed']]),a.shape)
        secondary[f'{name}_minus_fixed']=hierarchical_interval(a,b)
    transitions=sum(r['on_policy_transitions'] for r in initial.values())+sum(r['on_policy_transitions'] for r in refined.values())
    teacher_seconds=max(r['teacher_seconds_shared'] for r in initial.values())
    teacher_steps=max(r['teacher_transitions'] for r in initial.values())
    gap=.08**2-methods['reference']['aggregate_rmse']**2
    recovery=locked['paired_differences']['voi_error_reduction']['mean']/gap
    report_data={'scope':'frozen_parameter_holdout_unfiltered_v3','test_mechanisms':512,'training_seeds':3,
                 'methods':methods,'shared_final_estimator_ablation':interpretation,'primary_differences':locked['paired_differences'],'secondary_differences':secondary,
                 'on_policy_transitions_total':transitions,'teacher_seconds_shared':teacher_seconds,
                 'teacher_transitions_shared':teacher_steps,'raw_gap_recovery':recovery,
                 'device':device,'online':online,'locked_checkpoint_hashes_unchanged':locked['frozen_checkpoint_hashes_unchanged']}
    (ROOT/'reports/final-summary.json').write_text(json.dumps(report_data,indent=2)+'\n')
    figures=ROOT/'reports/figures';figures.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    for label in ('prior','fixed','bo','eig','voi','history_imitation','recurrent_imitation','recurrent_imitation_ppo'):
        row=methods[label]
        ax.scatter(row['mean_variable_cost'],row['aggregate_rmse'],s=55)
        name={'voi':'VOI','eig':'EIG','bo':'BO','history_imitation':'Full-history imitation',
              'recurrent_imitation':'Recurrent imitation','recurrent_imitation_ppo':'Imitation + PPO'}.get(label,label)
        offsets={'voi':(-30,4),'history_imitation':(7,22),'recurrent_imitation':(7,6),'recurrent_imitation_ppo':(7,-8)}
        ax.annotate(name,(row['mean_variable_cost'],row['aggregate_rmse']),xytext=offsets.get(label,(7,5)),textcoords='offset points',fontsize=8)
    ax.axhline(methods['reference']['aggregate_rmse'],color='gray',linestyle=':',label='private nominal reference')
    ax.set(xlabel='Mean purchased-assay cost (setup cost 3 reported separately)',ylabel='Final target RMSE',
           title='Locked test: adaptive planning and learned approximation')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'locked-cost-quality.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    labels=('history_ppo','recurrent_ppo','history_imitation','recurrent_imitation','recurrent_imitation_ppo')
    ax.bar(np.arange(len(labels)),[methods[k]['aggregate_rmse'] for k in labels],color=['#9ca3af','#9ca3af','#60a5fa','#2563eb','#14b8a6'])
    for i,label in enumerate(labels):
        values=[locked['learned'][f'{label}-seed{s}']['summary']['aggregate_rmse'] for s in (0,1,2)]
        ax.scatter([i-.08,i,i+.08],values,color='black',s=18,zorder=3)
    ax.axhline(methods['reference']['aggregate_rmse'],color='gray',linestyle=':',label='private nominal reference')
    ax.set_xticks(np.arange(len(labels)),['Full-history\nPPO','GRU\nPPO','Full-history\nimitation','GRU\nimitation','GRU imitation\nthen PPO'])
    ax.set(ylabel='Final target RMSE',ylim=(0,.09),title='Three training seeds; 512 untouched mechanisms')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(figures/'locked-learning-comparison.png',dpi=180);plt.close(fig)
    diff=locked['paired_differences'];im=methods['recurrent_imitation'];fine=methods['recurrent_imitation_ppo']
    old=ROOT/'docs/feasibility.md';archive=ROOT/'docs/feasibility-v1.md'
    if not archive.exists():archive.write_text(old.read_text())
    lines=['# Measured feasibility report','',
           'A useful adaptive experimental-design result was established after a documented development revision. A small learned policy approximates the strategy. PPO-specific added value was not established.','',
           '## Answers to the four questions','',
           '1. **Do experiments help after cost? Yes, within this synthetic family.** Prior-informed commitment returns -0.6400; adaptive VOI returns -0.3093 on 512 untouched mechanisms, with lower final error despite paying for experiments.',
           f"2. **Does adaptation improve on strong fixed designs? Yes, after removing attainability conditioning.** VOI improves return by {diff['voi_minus_fixed_return']['mean']:.4f}, 95% paired CI [{diff['voi_minus_fixed_return']['ci95'][0]:.4f}, {diff['voi_minus_fixed_return']['ci95'][1]:.4f}], and recovers {100*recovery:.1f}% more of the prior-to-reference raw error gap. It spends less and achieves lower error.",
           f"3. **Can a small learned policy reproduce the strategy? Approximately.** A 32,814-parameter GRU trained by imitation reaches aggregate RMSE {im['aggregate_rmse']:.5f} versus VOI {methods['voi']['aggregate_rmse']:.5f}, using three seeds. The full-history imitation model is similarly effective. The learned policy retains a small quality gap; this is an amortized-planning result.",
           f"4. **Does PPO add value over imitation? Not clearly in this protocol.** From-scratch PPO converges to buying no experiments. Conservative PPO after converged imitation changes return by {diff['recurrent_imitation_ppo_minus_imitation_return']['mean']:.5f}, with hierarchical 95% CI [{diff['recurrent_imitation_ppo_minus_imitation_return']['ci95'][0]:.5f}, {diff['recurrent_imitation_ppo_minus_imitation_return']['ci95'][1]:.5f}]. This does not establish an RL-specific improvement.",'',
           '## Frozen test results','',
           'All methods use the same 512 mechanisms and goals. Learned rows aggregate all three seeds; none was selected using test outcomes. Fixed/random/EIG/BO/VOI share one posterior final selector. Learned commitments are evaluated separately. The common baseline assay costs 3 units outside the variable budget.','',
           '| Method | Final RMSE | Variable cost | Mean return | Success within 0.04 |','|---|---:|---:|---:|---:|']
    for label in ('prior','fixed','fixed_quality','random','bo','eig','voi',*names,'reference'):
        r=methods[label];title=label.replace('_',' ')
        if label=='reference':title='private nominal reference (privileged)'
        lines.append(f"| {title} | {r['aggregate_rmse']:.5f} | {r['mean_variable_cost']:.3f} | {r['mean_return']:.5f} | {100*r['success_rate']:.2f}% |")
    lines.extend(['', 'The reference has private mechanism access and privileged zero variable assay cost. Unreachable targets remain in every reported average; its success rate is therefore below 100%. Lower absolute RMSE in the original admitted study cannot be compared directly with this unfiltered mixture.','',
                  '![Locked cost and quality](../reports/figures/locked-cost-quality.png)','',
                  '![Learned method comparison](../reports/figures/locked-learning-comparison.png)','',
                  '## What changed, and what was preserved','',
                  'The original feasibility-conditioned low-noise pilot was solved by a fixed design: fixed and VOI both reached RMSE 0.02891. A stronger 128-particle continuous check also failed the adaptivity screen. Those negative results remain in reports/feasibility-v1 and reports/research-v2.', '',
                  'The diagnosis was a simplifying prior constraint: knowing that the target is attainable can make an unmeasured sibling control inferable after its partner fails. The revision removes that conditioning from both generation and inference and retains unreachable targets. It preserves the simulator, primary readout noise, targets, actions, costs, final estimator and strong baselines. It was developed before opening the locked test. See revisions.md.', '',
                  'The strongest fixed search covers all affordable canonical assay multisets through four experiments, plus ordered-pair candidates. It examines 10,373 schedules on 512 calibration mechanisms. Zero-information no-intervention assays are pruned using the declared prior; public action masks do not use hidden information. This is exhaustive search over a declared finite design space with Monte Carlo calibration, not proof of a globally optimal expected fixed strategy.', '',
                  '## Scientific and learning protocol','',
                  'Eight states form two regulatory pathways with four controls, two intermediates and two outcomes. The prior contains 32 sign/route motifs, independent control-gain scale Uniform(0.9, 1.05) and shared-rate scale Uniform(0.9, 1.1). All baseline activities are 0.5. Targets independently use 0.5 +/- 0.08. Controls suppress production; they never clamp outcomes.', '',
                  'The pilot has 36 experimental actions and 9 commitments: no intervention or eight single-control doses, durations 1/6, proxy/outcome panels. The variable budget is 8, with at most four purchased assays and eight fresh cultures per assay. Readout SD is 0.02. The primary revised study has culture-offset SD zero; robustness to culture variation remains a separate qualification. Locked scoring uses 128 fresh cultures, independent of every purchased assay. With zero culture variation those latent outcomes are deterministic.', '',
                  'Reference simulation uses NumPy/SciPy float64 exponential Euler at dt 0.05, checked against dt/2 and DOP853. The admitted prior is used only in the preserved admitted study. The revised belief uses the unfiltered prior, with 128 independently specified quadrature hypotheses; true continuous parameters are never inserted. Policy features exclude mechanisms, hidden labels, unpurchased observations and final-evaluation random streams.', '',
                  'Training, validation and test use separate continuous mechanisms from seeds 7401, 8401 and 9401. Hashes are disjoint. This tests parameter generalization within familiar motif support. It does not establish structural, range-shift or simulator-misspecification generalization.', '',
                  'Full-history PPO precedes recurrent learning. The same GRU architecture is used for imitation, PPO and imitation followed by PPO. Updates reconstruct padded whole episodes, use valid dummy padding actions, exclude padding from every loss, respect public action masks and reset recurrent memory. External truncations are rejected by the complete-episode collector and tested separately for correct bootstrapping.', '',
                  'Initial 500-epoch imitation underfit its teacher and is preserved as an exploratory failure. Converged imitation uses 512 public-history demonstrations, 5,000 epochs and three seeds. Training resamples only mathematically ancillary baseline noise and within-cohort SD features; purchased means and teacher labels remain unchanged. Conservative refinement freezes the policy during 500 critic warmup updates, then uses 100,000 PPO transitions per seed at learning rate 0.0001 and clip 0.1. No checkpoint or seed is selected on test outcomes.', '',
                  'The direct-assay BO baseline pays for final-duration outcome measurements. Its noisy objective subtracts known readout variance and retains negative corrected estimates. EIG is a generic entropy heuristic; it is not a reproduction of DAD/Step-DAD. Two-step scientific lookahead was not expanded because one-step VOI already reached the private reference on development tasks.', '',
                  '## Actual compute','',
                  f"The machine is an Apple M3 Max with 36 GiB RAM, 14 CPU cores, macOS 26.3.1(a), native arm64 Python 3.12.1 and PyTorch {device['hardware']['dependencies']['torch']}. Both CPU and native MPS work. Scientific float64 stays on CPU; policies use float32. MPS CPU fallback is unset.", '',
                  f"Complete 24-episode rollout/update/evaluation/checkpoint workloads have median {device['cpu']['median_total_seconds']:.5f}s on CPU and {device['mps']['median_total_seconds']:.5f}s on MPS. Automatic selection chooses CPU. Peak benchmark process RSS is {device['peak_process_rss_bytes']/2**30:.3f} GiB. This is process RSS, not whole-machine memory use.", '',
                  f"Actual runs collected {transitions:,} on-policy transitions. Initial 100k-transition runs took roughly two minutes each; conservative warm-start runs took roughly 84 seconds each. Those measured runs are more informative than the short CPU forecast of {device['cpu']['forecast_100k_transitions_seconds']:.1f}s per 100k transitions. Full training times, entropy, divergence, value errors and gradient norms are saved with each run.", '',
                  f"The shared revised-task teacher supplied {teacher_steps:,} public-history transitions in {teacher_seconds:.1f}s before duplicate-outcome optimization. Converged single-model training took about 38–44 seconds. Teacher cost is paid once for a shared demonstration set, not hidden inside every model's interaction count. The earlier admitted-task teacher/training costs are separately retained. Fine-tuning logs exclude critic warmup wall time, while the 500 updates per seed are explicit.", '',
                  'Hypothetical prediction caching integrates each intervention once, retaining both duration endpoints. Exact duplicate predictive outcomes are now integrated once while retaining all posterior hypotheses, without particle pruning or rounding. Regression checks preserve expected decision value and posterior marginals to 1e-12. This reduced planning cost without changing the scientific model.', '',
                  f"A separate matched CPU deployment benchmark includes environment construction/reset, baseline, every decision and final scoring: fixed {online['measurements']['fixed']['median_seconds']:.6f}s/episode; VOI {online['measurements']['voi']['median_seconds']:.6f}s; learned imitation {online['measurements']['learned_imitation']['median_seconds']:.6f}s. The learned route is {online['planner_to_learned_ratio']:.1f}x faster than planning in this measured workload, with a small final-quality gap. A quality-adjusted compute break-even claim is not established.", '',
                  '## Numerical audits and validation','',
                  'The supplied four-variable values reproduce: proxy 0.457044687/0.542955313, final outcome 0.409467428/0.590532572, sign error 6.20717e-10, informed MSE 2.83653e-7 and negative return 0.05652837. One fixed assay solves it. The separate routing environment enumerates 100% adaptive versus 75% fixed accuracy; neither diagnostic establishes scientific RL value.', '',
                  'The original external audit scripts/seeds were not attached. The generous-budget eight-variable audit is an explicitly labeled independent reconstruction: fixed four-control additive RMSE 0.02620 versus private reference 0.02456, with 97.66% versus 99.22% success. It preserves the conclusion that generous-budget fixed identification is powerful; exact original rows are not asserted.', '',
                  'Independent code review found two bounded configuration/provenance defects, both repaired with failing-then-passing tests: smaller experiment caps in fixed tuning, and checkpoints binding the wrong default manifest. Research checkpoints now identify their actual manifests; legacy metadata is preserved. Review found no concrete current-study simulator, private-data, recurrent-reset or padding defect. Test counts and fresh validation evidence are recorded in reports/validation.json.', '',
                  '## Remaining limits','',
                  '- The positive result applies to the revised synthetic family with zero culture-offset variation and familiar topology support. Broader biology, structural transfer and real experiments remain unvalidated.',
                  '- Three seeds provide limited evidence about the training-seed distribution. Primary learned-method intervals bootstrap both seeds and paired mechanisms; they do not prove PPO can never help.',
                  '- The planner is one-step, finite-quadrature and Monte Carlo approximate. Continuous posterior calibration beyond the checked prior remains a limitation.',
                  '- The 429-action flagship, paired interventions, full BO/multi-fidelity research matrix, DAD reproduction and broad world models remain gated extensions.',
                  '- CPU checkpoints resume between complete episodes. Partial-rollout resume and bitwise CPU/MPS identity are not promised.', '',
                  'No method was retuned after the locked results were opened. Freeze/checkpoint hashes, raw rows, paired statistics, source snapshots and all preserved negative studies are available under reports/. See the README for runnable commands.',''])
    where=lines.index('## Actual compute')
    lines[where:where]=['## Same evidence, common final interpretation','',
                       f"Replaying the GRU imitation policy's purchased assays with the common posterior final selector reduces RMSE from {im['aggregate_rmse']:.5f} to {interpretation['recurrent_imitation']['same_assays_common_selector_rmse']:.5f}, versus VOI {methods['voi']['aggregate_rmse']:.5f}. Full-history imitation with the common selector reaches {interpretation['history_imitation']['same_assays_common_selector_rmse']:.5f}. Much of the small remaining learned gap is evidence interpretation rather than experiment selection. These are final-estimator ablations; their replay timing is not a full learned-policy speed measurement.",'']
    (ROOT/'docs/feasibility.md').write_text('\n'.join(lines))
    print('Generated docs/feasibility.md, reports/final-summary.json and two scientific figures.')


if __name__=='__main__':main()
