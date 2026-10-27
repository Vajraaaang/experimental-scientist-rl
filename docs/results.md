# Reading the results

## Separate claims

1. **Information value:** experiments improve final interventions after cost in the declared synthetic tasks.
2. **Adaptive value:** the original admitted task has no extra final-quality gain over a strong fixed design. The unfiltered development revision and subsequent locked tests show an adaptive advantage against the original tuned designs and a stronger exhaustive affordable-design search.
3. **Learnability:** converged imitation provides a useful learned approximation to the public-history planner on independent continuous-parameter validation mechanisms within familiar topology support.
4. **RL-specific value:** PPO and imitation→PPO are implemented and trained. Reward-only curriculum PPO beats the exhaustive fixed design on the 1,024-mechanism locked test, but a robust added benefit over strong imitation remains unestablished. Do not label imitation's speed advantage as an RL achievement.

## Evidence map

| Location | Meaning |
|---|---|
| `reports/feasibility-v1/` | Original primary negative result, twelve budget/noise/cost/variation scenarios, diagnostics, and reconstructed generous-budget audit |
| `reports/research-v2/` | Stronger conditional-prior inference and fixed-design study that still failed the adaptive screen |
| `reports/unfiltered-v3/assessment.json` | Preregistered development revision retaining unreachable targets; 128 calibration and 128 validation mechanisms |
| `reports/unfiltered-v3/exhaustive-fixed.json` | 10,373 fixed-design candidates tuned on 512 calibration mechanisms, evaluated on the assessment slice |
| `reports/unfiltered-v3/matched-baselines.json` | Fixed/random/EIG/VOI/BO on a different paired validation slice used for learning comparisons |
| `reports/unfiltered-v3/converged-imitation/` | Three full-history and three recurrent imitation seeds, 512 teacher demonstrations and 5,000 augmented epochs |
| `reports/unfiltered-v3/strong-finetune/` | Three recurrent imitation→PPO runs with at least 100,000 on-policy transitions each |
| `reports/unfiltered-v3/initial-learning-completed.json` | All 12 completed initial learning records, including three full-history PPO seeds and three recurrent PPO seeds; preliminary undertrained warm starts are preserved |
| `reports/unfiltered-v3/locked-test/` | First frozen 512-mechanism test, preserving the original PPO collapse and imitation comparison |
| `reports/ppo-v4/` | Reward-only curriculum, five-seed training, failed controls, GPU profiles and fresh 1,024-mechanism frozen comparison |
| `reports/ppo-cost-v5/` | Development cost diagnosis/calibration, paired-return controls and another 2,048-mechanism frozen comparison |
| `reports/ppo-efficiency-v6/` | 50M completed new interactions, another 4,096-mechanism frozen test; PPO improves over its previous version but full efficiency parity with imitation remains unestablished |
| `reports/compute/benchmark.json` | Complete CPU/MPS workload measurements on the actual Mac |

The original external audit scripts and seeds were absent from the supplied specification. `generous-audit.json` is explicitly an independent reconstruction, not an exact reproduction of its original rows.

## Statistical and scientific limits

The validation mechanisms have independent continuous gain/rate values, but familiar topology support. Locked continuous-parameter evaluation is completed in four successive studies. Withheld structures, parameter-range extrapolation, culture variation and simulator misspecification remain unqualified. A per-model mechanism bootstrap does not measure uncertainty across training seeds. Three seeds in the initial study and five in each follow-up do not establish universal training reliability. Follow-up intervals resample both seeds and paired mechanisms.

The private informed reference has privileged mechanism access and no variable assay cost. Its condition differs from attainable methods. Unreachable targets remain in the revised study's main results; a success rate below 100% can reflect task reachability rather than agent error.

Timing fields have different scopes. Complete learned-policy timing includes simulator execution; shared-estimator replay timing omits the learned selection pass. Do not interpret replay timing as end-to-end learned speed. The published CPU/MPS benchmark includes transfers, sampling, updates, evaluation and checkpoint round trips. Its short-run forecast excludes teacher generation and sustained thermal effects.

## Publication snapshot

The snapshot includes completed evidence records and the complete training code. Active training output and weight files are excluded. Public setup uses the clone's absolute path rather than the author's machine path. A private handoff URL and machine-specific path prefixes were removed; the publication manifest records original and published hashes.

Publication-only portability changes make optional Mac hardware commands report unavailability on other platforms and use the platform's temporary directory for checkpoint round trips. They do not change simulator dynamics, observations, costs, or learning objectives. The local publication tests exercise these changes. Hosted CPU CI is configured separately and is not considered passing until GitHub runs it.

Measured records originated in the development environment before publication-only portability cleanup. Their original code hashes are provenance, not assertions that the cleaned snapshot is byte-identical. Reproduction commands write to `runs/` to preserve the supplied evidence.

## Latest cost-efficiency conclusion

The development-selected paired-return PPO plus public-price calibration spends 8.2% less than original curriculum PPO on the latest test. The cost interval is positive and the declared MSE noninferiority criterion passes. The net-return interval crosses zero, so the full utility-inclusive screen fails. Imitation remains cheaper. Original prices, raw MSE and strong baselines are unchanged. See [the complete cost report](ppo_cost_efficiency.md) for margins, intervals, controls and limitations.

The raw latest record is gzip-compressed to avoid an oversized Git blob. The report renderer accepts it directly. Checkpoint binaries and source archives are excluded from this public snapshot; their hashes and training configuration are retained. Original freeze hashes describe the measured development snapshot before publication-only redaction/portability edits. The previous publication manifest is historical; the update manifest records the current mapping.

## V6 continuation

The latest [locked comparison](ppo_efficiency_v6.md) improves PPO cost, target MSE and return over the previous efficient PPO with favorable paired intervals. The remaining cost premium over unchanged imitation is 0.1586 units, 95% interval [0.1050, 0.2101]. The declared quality margin passes, while cost/return bounds prevent an efficiency-gap closure claim. All five seeds and the original baselines remain retained. This uses more optimization and per-seed development selection; it is not an equal-compute PPO-versus-imitation experiment.

V6 raw shards and final rows are atomic, lossless gzip packets. Original hashes remain development provenance; the v6 publication manifest records redacted published hashes. Checkpoint binaries remain excluded. The curated final report clarifies fixed-design versus self-policy exploratory curricula without changing numerical outputs.
