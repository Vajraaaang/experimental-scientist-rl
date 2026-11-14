"""Measured follow-up report from immutable fresh locked results."""
from pathlib import Path
import json
import os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'reports/ppo-v4'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'reports/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def average_summaries(values):
    result={k:float(np.mean([v[k] for v in values])) for k in values[0]}
    result['aggregate_rmse']=float(np.sqrt(np.mean([v['aggregate_rmse']**2 for v in values])))
    return result


def interval_text(value):
    return f"{value['mean']:.6g} (95% CI [{value['ci95'][0]:.6g}, {value['ci95'][1]:.6g}])"


def main():
    data=json.loads((STUDY/'fresh-locked/results.json').read_text())
    differences=data['paired_differences'];methods=dict(data['baseline_summary']);stochastic={};shared={}
    for name,seeds in [('curriculum_ppo',range(20,25)),('imitation',range(20,25)),('imitation_ppo',range(20,25)),
                       ('legacy_ppo',range(3)),('legacy_imitation',range(3))]:
        models=[data['learned'][f'{name}-seed{s}'] for s in seeds]
        methods[name]=average_summaries([m['summary'] for m in models])
        shared[name]=average_summaries([m['shared_estimator_ablation'] for m in models])
        if not name.startswith('legacy_'):stochastic[name]=average_summaries([m['stochastic']['summary'] for m in models])
    accounting={'scratch_interactions':0,'finetune_interactions':0,'sampled_policy_decisions':0,'training_validation_seconds_sum':0.,
                'imitation_seconds_sum':0.,'critic_warmup_seconds_sum':0.,'seeds':{}}
    for seed in range(20,25):
        scratch=json.loads((STUDY/'main'/f'seed{seed}'/'scratch-result.json').read_text())
        fine=json.loads((STUDY/'main'/f'seed{seed}'/'finetune-result.json').read_text())
        imitation=json.loads((STUDY/'main'/f'seed{seed}'/'imitation/result.json').read_text())
        accounting['scratch_interactions']+=scratch['total_interactions'];accounting['finetune_interactions']+=fine['training']['transitions']
        accounting['sampled_policy_decisions']+=scratch['selector']['policy_decisions']+scratch['joint']['policy_decisions']+fine['training']['policy_decisions']
        accounting['training_validation_seconds_sum']+=scratch['selector']['training_and_validation_seconds']+scratch['joint']['training_and_validation_seconds']+fine['training']['training_and_validation_seconds']
        accounting['imitation_seconds_sum']+=imitation['seconds'];accounting['critic_warmup_seconds_sum']+=fine['critic_warmup_seconds']
        accounting['seeds'][seed]={'selector_interactions':scratch['selector']['transitions'],
                                  'joint_interactions':scratch['joint']['transitions'],'joint_selected_at':scratch['joint']['selected_at_transitions'],
                                  'finetune_interactions':fine['training']['transitions'],'finetune_selected_at':fine['training']['selected_at_transitions']}
    accounting['main_interactions']=accounting['scratch_interactions']+accounting['finetune_interactions']
    diagnostic=0
    for directory in (STUDY/'development').iterdir():
        if directory.is_dir() and not directory.name.startswith('main_'):
            for path in directory.glob('seed*/result.json'):diagnostic+=json.loads(path.read_text())['transitions']
    accounting['development_control_interactions']=diagnostic
    teacher=json.loads((STUDY/'teacher-compute.json').read_text());accounting['teacher']=teacher
    original_device=json.loads((STUDY/'device-benchmark.json').read_text())
    tensor_device=json.loads((STUDY/'device-benchmark-accelerated-128.json').read_text())
    summary={'scope':data['freeze']['scope'],'mechanisms':1024,'training_seeds':5,'methods':methods,
             'stochastic_methods':stochastic,'shared_final_estimator':shared,'paired_differences':differences,
             'compute':accounting,'gpu_cores':30,'batch128_benchmark':tensor_device,
             'fresh_checkpoint_hashes_unchanged':data['checkpoints_unchanged'],
             'previous_v3_results_unchanged':data['previous_v3_results_unchanged']}
    (STUDY/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    figures=STUDY/'figures';figures.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for name,label in [('selector_control','Normalized advantages'),('selector_raw','Reward-unit advantages')]:
        result=json.loads((STUDY/'development'/name/'seed10/result.json').read_text())
        axes[0].plot([r['transitions'] for r in result['curve']],[r['update']['entropy'] for r in result['curve']],label=label)
    axes[0].set(title='Matched selector control: exploration',xlabel='Physical training transitions',ylabel='Categorical entropy')
    axes[0].set_xticks([0,50000,100000,150000,200000],['0','50k','100k','150k','200k'])
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    for seed in range(20,25):
        result=json.loads((STUDY/'development/main_curriculum_ppo'/f'seed{seed}/result.json').read_text())
        axes[1].plot([e['transitions'] for e in result['evaluations']],[e['free']['mean_return'] for e in result['evaluations']],label=f'Seed {seed}')
    axes[1].axvspan(0,600000,color='gray',alpha=.12,label='Transient public bias')
    axes[1].axhline(-.64,color='black',linestyle=':',label='Prior stop')
    axes[1].set(title='Joint stage after reward-only pretraining',xlabel='Joint PPO transitions',ylabel='Unbiased greedy validation return')
    axes[1].legend(fontsize=7,ncol=2);axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(figures/'collapse-and-recovery.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    labels=['legacy_ppo','fixed','curriculum_ppo','imitation','imitation_ppo','voi']
    titles=['Old PPO','Fixed','Reward-only\ncurriculum PPO','Imitation','Imitation\nthen PPO','VOI planner']
    colors=['#9ca3af','#f59e0b','#60a5fa','#2563eb','#14b8a6','#8b5cf6']
    ax.bar(range(len(labels)),[methods[x]['aggregate_rmse'] for x in labels],color=colors)
    for i,label in enumerate(labels):
        if label in ('curriculum_ppo','imitation','imitation_ppo'):
            values=[data['learned'][f'{label}-seed{s}']['summary']['aggregate_rmse'] for s in range(20,25)]
            ax.scatter(np.linspace(i-.16,i+.16,5),values,color='black',s=17,zorder=3)
    ax.axhline(methods['reference']['aggregate_rmse'],color='gray',linestyle=':',label='Private nominal reference')
    ax.set_xticks(range(len(labels)),titles);ax.set(ylabel='Final target RMSE',ylim=(0,.09),title='Fresh locked test: 1,024 mechanisms; five new training seeds')
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'fresh-policy-comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.8,4.2))
    for method,label in [('fixed','Fixed'),('curriculum_ppo','Curriculum PPO'),('imitation','Imitation'),('imitation_ppo','Imitation + PPO'),('voi','VOI')]:
        r=methods[method];ax.scatter(r['mean_variable_cost'],r['aggregate_rmse'],s=55,label=label)
    ax.set(xlabel='Mean variable experimental cost',ylabel='Final target RMSE',title='Fresh locked quality and experimental cost')
    ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=9);fig.tight_layout();fig.savefig(figures/'fresh-cost-quality.png',dpi=180);plt.close(fig)
    ppo=methods['curriculum_ppo'];im=methods['imitation'];fine=methods['imitation_ppo']
    d=differences['curriculum_ppo_return_minus_fixed'];di=differences['curriculum_ppo_return_minus_imitation'];df=differences['imitation_ppo_return_minus_imitation']
    lines=['# PPO collapse follow-up: measured report','',
           '**Outcome:** collapse is avoided with a reward-only experimental curriculum and more training. The strengthened PPO beats the preserved exhaustive fixed baseline on a fresh locked test, but imitation remains more cost-effective. Additional PPO after imitation still has no established benefit.','',
           '## Fresh frozen comparison','',
           'Five independently initialized training seeds (20–24), 1,024 fresh mechanisms from generator seed 27401, unchanged unfiltered task physics/costs/actions, and uncached reference scoring. New mechanisms are hash-disjoint from training, validation and the previous test. All seeds are retained. Checkpoints are selected on development return before scoring.','',
           '| Method | Final RMSE | Variable cost | Mean return | Success within 0.04 |','|---|---:|---:|---:|---:|']
    for name in ('prior','fixed','fixed_quality','additive','random','bo','eig','voi','legacy_ppo','legacy_imitation','curriculum_ppo','imitation','imitation_ppo','reference'):
        r=methods[name];title=name.replace('_',' ')
        if name=='reference':title='private nominal reference (privileged)'
        lines.append(f"| {title} | {r['aggregate_rmse']:.5f} | {r['mean_variable_cost']:.3f} | {r['mean_return']:.5f} | {100*r['success_rate']:.2f}% |")
    lines.extend(['', 'All methods receive the setup baseline assay; its fixed cost 3 is outside the variable budget. The reference knows the mechanism and has privileged zero variable assay cost. Unreachable targets remain included. Primary quality is mean target MSE; tables display its aggregate square root. Cost and return are separately reported.','',
                  f"Reward-only curriculum PPO improves return over exhaustive fixed by {d['mean']:.5f}, hierarchical 95% CI [{d['ci95'][0]:.5f}, {d['ci95'][1]:.5f}]. Its paired raw MSE reduction is {interval_text(differences['curriculum_ppo_mse_reduction_vs_fixed'])}. Its cost reduction is {interval_text(differences['curriculum_ppo_cost_reduction_vs_fixed'])}.", '',
                  f"Against strengthened imitation, curriculum PPO changes return by {di['mean']:.5f}, CI [{di['ci95'][0]:.5f}, {di['ci95'][1]:.5f}]. It spends {ppo['mean_variable_cost']-im['mean_variable_cost']:.3f} more cost units on average. Raw-quality difference is not clearly established: MSE reduction {interval_text(differences['curriculum_ppo_mse_reduction_vs_imitation'])}.", '',
                  f"Imitation followed by another million PPO transitions per seed changes return by {df['mean']:.6f}, CI [{df['ci95'][0]:.6f}, {df['ci95'][1]:.6f}]. Its raw MSE reduction interval also includes zero: {interval_text(differences['imitation_ppo_mse_reduction_vs_imitation'])}. This does not establish PPO-specific added value over strong imitation.", '',
                  'Intervals resample both training seeds and paired mechanisms. Five seeds remain a limited sample of optimization randomness. No method was retuned after these results were opened.','',
                  f"The stochastic-policy secondary check has mean returns {stochastic['curriculum_ppo']['mean_return']:.5f} for curriculum PPO, {stochastic['imitation']['mean_return']:.5f} for imitation and {stochastic['imitation_ppo']['mean_return']:.5f} for imitation+PPO. The main ordering is therefore not an artifact of greedy deployment. These are descriptive secondary estimates; primary intervals above concern the frozen greedy protocol.",'',
                  '![Fresh policy comparison](../reports/ppo-v4/figures/fresh-policy-comparison.png)','',
                  '![Fresh cost and quality](../reports/ppo-v4/figures/fresh-cost-quality.png)','',
                  '## Why the old PPO collapsed','',
                  'No concrete reward-sign, recurrent-reset, padding, probability-ratio or GAE bug explained the collapse. Independent review and focused numerical tests found those paths consistent. The dominant issue was a learning cold start: experiments help only after final intervention choices can interpret them. Before that skill exists, assays add certain cost.', '',
                  'Under the declared prior, no intervention has expected terminal return -0.6400; dose 0.4 interventions average about -0.72490, dose 0.8 about -0.92834, and uniform commitment about -0.80589. If final choices ignore evidence, buying an assay does not improve those expectations. The initial policy therefore receives a real gradient toward stopping.', '',
                  'Old entropy fell from about 3.5 to 0.006–0.023 by 10k transitions; reset probabilities of buying evidence fell to roughly 0.000005–0.0033. Extending a nearly deterministic stop policy supplies almost no informative data.', '',
                  'A second amplifier was entropy scaling. With per-batch advantage normalization, the fixed entropy coefficient acts roughly like coefficient times advantage SD in return units. As the stop policy/critic reduce variance, that effective exploration pressure shrinks. In a matched fixed-assay control, removing advantage normalization preserves entropy around 1.3 rather than near zero, but 200k transitions were still insufficient to learn the final interpretation.', '',
                  'One million raw-advantage selector transitions did learn: return -0.4848 versus -0.6875 for choosing none after the same public assays; a shared posterior selector on those assays reaches -0.4636. The early 200k curriculum had released the joint problem before this competence developed. A paired single-assay/two-choice control also learned; its initial mixed-schedule evaluation was corrected to the proper matched control before protocol selection.', '',
                  'Larger batches or stronger entropy alone did not solve the task in the bounded screens. A one-million-transition joint raw-advantage run still greedily stopped. Learning interpretation first, then gradually releasing assay selection, provided a successful route. Critic weighting may affect stability, but both tested weights recovered, so it was not established as the sole cause.', '',
                  '![Collapse and recovery](../reports/ppo-v4/figures/collapse-and-recovery.png)','',
                  '## Stronger training, with the scientific task held fixed','',
                  'The 64-unit GRU actor–critic remains unchanged (32,814 parameters). The reward-only PPO arm uses no planner action labels. It first buys randomized public fixed designs and learns only terminal intervention choices from real terminal rewards. Forced assay positions are excluded from actor/entropy losses, while their public histories remain differentiable and their physical costs remain charged.', '',
                  'Selector training has a one-million-transition minimum, a two-million cap and a development competence threshold of -0.51. All five seeds graduated at about one million. The following two-million-transition joint phase uses stored, public exploration biases that are included in both old and new probabilities. The biases disappear after the first 30%; checkpoints are eligible only after at least one million joint transitions. Final evaluation restores the original masks and uses zero training bias.', '',
                  'Advantages remain in reward units; complete-episode MC returns use gamma 1 and GAE lambda 1. PPO uses clipping, gradient clipping, post-update nonnegative approximate KL, a KL stopping guard, advantage/entropy/value diagnostics and early actor-versus-critic gradient audits. Separate phase seed namespaces ensure new observation draws between phases.', '',
                  'The strengthened imitation baseline uses 2,048 public-history VOI demonstrations and 12,000 whole-episode minibatch updates, with only mathematically ancillary baseline/within-cohort-SD augmentation. The same GRU architecture is used. Fine-tuning freezes the policy for 500 critic-head warmup updates before another million PPO transitions. All checkpoint selection uses the same development mechanisms and never the new test.', '',
                  'The exhaustive fixed and quality-focused schedules remain byte-for-byte preserved from the 10,373-design/512-calibration search. Prior, random, additive single-control identification, direct-assay BO, EIG and VOI remain in the fresh comparison. The additive subset is tuned on calibration only. Fixed/random/BO/EIG/VOI share the same final estimator; additive and fully learned commitments are distinguished.', '',
                  f"Replaying curriculum PPO's assays with the common final selector gives RMSE {shared['curriculum_ppo']['aggregate_rmse']:.5f}, versus fully learned {ppo['aggregate_rmse']:.5f}. For imitation the corresponding figures are {shared['imitation']['aggregate_rmse']:.5f} and {im['aggregate_rmse']:.5f}. These are interpretation ablations; replay timing is not a full-policy latency measurement.", '',
                  '## Compute and the 30-core GPU','',
                  f"The main comparison consumed {accounting['main_interactions']:,} physical interactions: {accounting['scratch_interactions']:,} for reward-only selector/joint PPO and {accounting['finetune_interactions']:,} for imitation fine-tuning. {accounting['sampled_policy_decisions']:,} were sampled policy decisions; forced public assay positions are counted in physical interactions but not actor updates. Development controls used another {diagnostic:,} interactions.", '',
                  f"Summed main PPO training/validation wall time is {accounting['training_validation_seconds_sum']:.1f}s across parallel processes, not elapsed wall time. Imitation training/validation totals {accounting['imitation_seconds_sum']:.1f}s; critic warmup totals {accounting['critic_warmup_seconds_sum']:.1f}s. The shared teacher cost {teacher['seconds']:.1f}s for {teacher['episodes']:,} episodes and {teacher['transitions']:,} public-history decisions, paid once for the shared dataset. Actual per-seed counts and selected checkpoints are in summary.json.", '',
                  'Actual hardware inspection confirms an M3 Max with 30 GPU cores, 14 CPU cores and 36 GiB unified RAM, macOS 26.3.1(a), native arm64 Python 3.12.1 and PyTorch 2.14.1. Scientific reference calculations stay NumPy/SciPy float64 on CPU; neural models use float32. Native MPS works with fallback disabled.', '',
                  'Compute rework was limited to measured bottlenecks. An evaluator-owned zero-offset endpoint cache preserves every checked observation/reward exactly (maximum error 0.0), fresh noise and RNG consumption; unsupported settings use the original simulator. It gives about 9.25x rollout speedup. Policies never receive its hidden endpoints. The locked comparison uses the original uncached simulator.', '',
                  'Host-padded one-transfer-per-field packing and batch-vectorized GAE preserve fields/returns exactly in the checked CPU/MPS cases. A separate optimizer-step test preserves the result within float32 tolerance. These are optional performance paths; no larger architecture or changed scientific problem was introduced.', '',
                  f"At the actual 128-episode workload, the original measured CPU/MPS rates were {original_device['devices']['cpu']['median_transitions_per_second']:.0f}/{original_device['devices']['mps']['median_transitions_per_second']:.0f} transitions/s. Tensorization raised the measured MPS rate to {tensor_device['devices']['mps']['median_transitions_per_second']:.0f}; CPU remained faster at {tensor_device['devices']['cpu']['median_transitions_per_second']:.0f}. The latter benchmark shared the machine with imitation training. All main scientific runs therefore used CPU.", '',
                  'The GPU can help larger dense batches: a preloaded 4,096-episode GRU microstep measured about 5.1ms on MPS versus 46.0ms on one contended CPU thread. Including synthetic host packing/GAE gave about 67.1ms versus 103.4ms. Those microbenchmarks exclude real rollout/PPO overhead and do not justify enlarging the current model solely to use GPU cores. Higher-compute rework stops here unless a later measured need appears.', '',
                  '## Scope, preservation and validation','',
                  'The new test has 1,024 fresh continuous mechanisms within familiar motif support, no feasibility conditioning, budget 8 and culture-offset SD zero. Structural, range-shift, culture-variation and real-biology transfer remain unqualified. Greedy policy deployment is primary; stochastic rollout results are recorded separately in summary.json and the raw locked file.', '',
                  'The previous v3 locked results, 15 protected checkpoints, baseline schedules and original deliverable remain unchanged. Fresh protocol/checkpoint hashes were frozen before scoring, and source/checkpoint integrity is checked after evaluation. No difficult seed was removed and no fresh result selected a model or hyperparameter.', '',
                  'Fresh full-suite and native-device evidence is recorded in reports/ppo-v4/validation.json. Source snapshots, configs, training curves, selected checkpoints, demonstration data and raw paired rows are included. The old report remains a historical result; this follow-up report supersedes its conclusion that reward-only PPO could only stop.', '',
                  'Implementation references: [OpenAI Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) and [Stable Baselines3 PPO source](https://stable-baselines3.readthedocs.io/en/v2.7.1/_modules/stable_baselines3/ppo/ppo.html). These support the algorithmic controls; the simulator and measured conclusions are this project\'s experiments.',''])
    (ROOT/'docs/ppo_followup.md').write_text('\n'.join(lines))
    print('Wrote docs/ppo_followup.md, reports/ppo-v4/summary.json and three figures.')


if __name__=='__main__':main()
