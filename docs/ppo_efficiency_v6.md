# PPO efficiency v6: another fresh locked comparison

The selected policy did not meet the predeclared efficiency-gap closure screen against both strong imitation comparators. All previous locked studies and fixed designs remain unchanged. The result below was opened only after development selection and freezing.

## Independent final comparison

Five retained training seeds, 4,096 new mechanisms from generator seed 97401 and episode seed 105442. The unfiltered task, GRU64 (32,814 parameters), actual fees, reward, noise and budget 8 remain unchanged. Final scoring uses the original uncached float64 reference dynamics and independent final cohorts. Greedy deployment is primary.

| Method | Final RMSE | Variable cost | Mean return |
|---|---:|---:|---:|
| prior | 0.08000 | 0.000 | -0.64000 |
| fixed | 0.05521 | 6.400 | -0.39333 |
| fixed quality | 0.05512 | 7.400 | -0.40236 |
| additive | 0.06159 | 5.800 | -0.45772 |
| random | 0.06814 | 7.169 | -0.55352 |
| bo | 0.06124 | 5.093 | -0.44534 |
| eig | 0.05489 | 7.699 | -0.40298 |
| voi | 0.04823 | 4.651 | -0.30885 |
| curriculum ppo | 0.04892 | 5.422 | -0.32262 |
| v5 efficient ppo | 0.04912 | 4.969 | -0.31988 |
| v6 trained ppo | 0.04867 | 4.782 | -0.31401 |
| selected ppo | 0.04865 | 4.782 | -0.31384 |
| imitation | 0.04859 | 4.623 | -0.31211 |
| v5 calibrated imitation | 0.04864 | 4.616 | -0.31250 |
| calibrated imitation | 0.04858 | 4.621 | -0.31201 |
| imitation ppo | 0.04858 | 4.637 | -0.31210 |
| private nominal reference (privileged) | 0.04814 | 0.000 | -0.26155 |

Compared with each named comparator, positive cost/MSE differences mean the selected PPO is worse; positive return difference means PPO is better. Intervals resample both training seeds and paired mechanisms.

- Versus imitation: cost increase 0.158574 (95% CI [0.105014, 0.210075]); raw MSE increase 5.8018e-06 (95% CI [-4.91653e-05, 9.51148e-05]); return improvement -0.00173038 (95% CI [-0.00994558, 0.00336416]).
- Versus calibrated imitation: cost increase 0.160884 (95% CI [0.107423, 0.212486]); raw MSE increase 6.59531e-06 (95% CI [-4.75706e-05, 9.59206e-05]); return improvement -0.00182696 (95% CI [-0.00999521, 0.00324425]).
- Versus v5 efficient ppo: cost increase -0.187725 (95% CI [-0.350541, -0.0355336]); raw MSE increase -4.57511e-05 (95% CI [-9.85599e-05, -5.59087e-07]); return improvement 0.0060461 (95% CI [0.0017647, 0.0103618]).

Closure requires upper two-sided 95% bounds of at most 0.10 additional cost and 0.0001 raw MSE, plus a lower return-difference bound of at least -0.005, against both unchanged and newly calibrated imitation. These are predeclared engineering noninferiority margins, not proof of identical performance. Zero-margin differences remain visible above.

Screen components: `{'imitation': {'cost': False, 'quality': True, 'return': False}, 'calibrated_imitation': {'cost': False, 'quality': True, 'return': False}}`.

![Fresh cost and quality](../reports/ppo-efficiency-v6/figures/fresh-efficiency.png)

## Development findings and selected method

Previous paired continuation updates had near-zero policy KL and zero clipping. Direct stronger-update controls mostly changed probabilities without changing greedy assays. A new public-history belief audit separated excess spending from an interpretation deficit in one seed. Longer temporary-temperature exploration repaired some interpretation; all bounded controls and unsuccessful candidates are retained.

Development selected `main_calibrated` with deployment calibration {'duration_penalty': 0.0, 'dose_penalty': 16.0, 'stop_bonus': 0.0}. All five seeds remain retained. The main continuation compares two three-million-interaction settings per seed; each seed's checkpoint/setting is selected on original tuning return. Additional bounded pilots and any self-curriculum restarts are explicitly counted. This is development-selected continuation with more compute, not an equal-compute algorithm comparison. The selected paths and calibration are fully recorded; public calibration is distinguished from gradient learning.

Public-price calibration uses declared duration/dose prices and public experiment count only. If present, it is explicit post-processing and is not credited as a PPO gradient discovery. The same extended calibration search is offered to imitation, with unchanged imitation, previous calibration and imitation-plus-PPO preserved.

Temporary training temperature is used identically in old and new probabilities. Deployment returns to temperature 1. The fixed-design cheap-assay curriculum uses predetermined random public low-dose designs. A separate self-curriculum uses a frozen previous PPO policy on current purchased histories, with public low-dose projection. Both charge full scientific fees, learn terminal interventions only from actual rewards, and exclude forced assay positions from actor loss. Neither exploratory curriculum was promoted. No planner labels, true mechanisms, unpurchased observations or evaluation randomness enter the policy.

Training uses the original independent 4,096-mechanism pool. New tuning and confirmation each contain 512 mechanisms from development generator indices [2048,4096); confirmation remains development evidence. Checkpoint/candidate/calibration selection uses these development slices only. All five seeds remain retained.

A common final estimator on selected PPO's purchased assays has RMSE 0.04815, compared with its learned commitments at 0.04865. The estimator ablation separates evidence selection from interpretation; replay timing is not end-to-end policy latency.

## Compute, preservation and limits

Two final restart pilots were interrupted by disk exhaustion and excluded from promotion. Their uncompleted additional work is bounded between 1,500,852 and 2,003,412 interactions, separately from the exact completed-run totals. Disposable project install/cache directories were removed; completed studies and baselines were preserved. Final evaluation packets use atomic lossless gzip writes.

This continuation consumed 50,011,727 new physical interactions and 46,005,942 sampled policy decisions across completed controls and main runs. Summed training/validation time is 7645.9 seconds across parallel processes, not elapsed wall time. Evaluation shard wall times sum to 1695.3 seconds. The original pretraining/teacher costs are additional historical compute, not hidden inside these counts.

The current native complete256-episode workload measured CPU 5662 versus MPS 698 transitions/s, with CPU training running concurrently. Fallback was unset.

The exact evaluator-owned simulator cache accelerates training only. Native CPU execution remains complete; no larger network or GPU-only dependency was introduced. Full rollout/update measurements already favored CPU at this model size. The fresh test uses uncached scientific dynamics.

The exhaustive 10,373-design fixed frontier, quality schedule, additive single-control identification, prior, random, direct-assay BO, EIG and VOI remain in the comparison. Fixed/random/BO/EIG/VOI share the same final estimator. Setup cost 3 lies outside variable assay cost; the informed reference has privileged mechanism access and zero assay spending.

All earlier study artifacts are checked against pre-continuation hashes. The freeze includes model hashes and all existing evaluation/library helper sources. Source and model integrity is verified in each shard and at reduction. Completed results refuse overwrite. No fresh-test outcome was used for further tuning.

The task remains a continuous-parameter holdout within familiar motifs and original parameter ranges, with culture-offset SD zero. Structural extrapolation, culture variation and real biological usefulness remain unqualified. Five training seeds limit optimization uncertainty; more mechanisms do not replace more independent optimization seeds.


Publication note: checkpoints, demonstration binaries and source ZIP archives are excluded from Git. Recorded numerical rows, configurations, freeze hashes and unsuccessful controls are included. Original hashes describe development artifacts before machine-path redaction and publication edits. The final raw comparison is losslessly gzip-compressed.
