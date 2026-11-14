# Offline M2 development confirmation

This is development confirmation after tuning. The new final pool remains unopened. No hosted calls were made.

Pure PPO slot frozen from all-five tuning: **scratch_selected** (m2). Confirmation does not change this selection.

| Cohort | Completed / expected | MSE | Aggregate RMSE | Variable cost | Original return |
|---|---:|---:|---:|---:|---:|
| calibrated_imitation | 640 / 640 | 0.002206153 | 0.04696971 | 4.54625 | -0.2967653 |
| imitation | 640 / 640 | 0.002206153 | 0.04696971 | 4.547187 | -0.2967747 |
| imitation_ppo | 640 / 640 | 0.002206153 | 0.04696971 | 4.561094 | -0.2969138 |
| imitation_ppo_continued | 640 / 640 | 0.002206153 | 0.04696971 | 4.562344 | -0.2969263 |
| ppo | 640 / 640 | 0.002261146 | 0.04755151 | 4.85125 | -0.3045646 |
| scratch_selected | 640 / 640 | 0.002190276 | 0.04680039 | 4.861562 | -0.2982682 |
| v6_continued | 640 / 640 | 0.002261146 | 0.04755151 | 4.781875 | -0.3038709 |
| prior | 128 / 128 | 0.0064 | 0.08 | 0 | -0.64 |
| fixed | 128 / 128 | 0.002768225 | 0.05261392 | 6.4 | -0.3676975 |
| fixed_quality | 128 / 128 | 0.002768225 | 0.05261392 | 7.4 | -0.3776975 |
| additive | 128 / 128 | 0.003123295 | 0.05588645 | 5.8 | -0.3953295 |
| random | 128 / 128 | 0.004516209 | 0.06720275 | 7.057812 | -0.5403241 |
| bo | 128 / 128 | 0.003388201 | 0.05820826 | 4.961719 | -0.4103123 |
| eig | 128 / 128 | 0.002835482 | 0.05324925 | 7.7 | -0.3864857 |
| voi | 128 / 128 | 0.002190276 | 0.04680039 | 4.564063 | -0.2952932 |
| reference | 128 / 128 | 0.002190276 | 0.04680039 | 0 | -0.2496526 |

All five training seeds remain required. Incomplete cohorts have no inferential summary. Baselines are evaluated once per mechanism and tiled only for paired resampling; those copies are not independent replicates.

## Predeclared signed comparisons

Differences are first method minus comparator. The registered paired two-way bootstrap over training seed and mechanism uses 10,000 draws and RNG 991499. Intervals computed: True. Each completed primary comparison reports 95% descriptive and 98.333333% Bonferroni intervals for the three families. A conservative joint gate requires adjusted return lower bound > 0 and adjusted MSE-increase upper bound <= 0.0001.

- scratch_selected-minus-ppo: development joint gate not met.
  - final_mse: -7.086985e-05; 95% [-0.000225400528681053, 1.2874900798265365e-20]; 98.333333% [-0.0002705983598602884, 2.168404344971009e-20].
  - variable_cost: 0.0103125; 95% [-0.13438281250000003, 0.23328906249999992]; 98.333333% [-0.1678125, 0.30599609375000086].
  - total_assay_cost: 0.0103125; 95% [-0.1343828124999999, 0.23328906249999995]; 98.333333% [-0.16781249999999984, 0.30599609375000103].
  - return_: 0.00629636; 95% [-0.0016987109375000038, 0.020618547979694195]; 98.333333% [-0.0025671874999999922, 0.0244492136387707].
- v6_continued-minus-ppo: development joint gate not met.
  - final_mse: -5.082198e-22; 95% [-2.0667603913004927e-20, 1.6601845766184288e-20]; 98.333333% [-3.009931578068156e-20, 2.439454888092385e-20].
  - variable_cost: -0.069375; 95% [-0.22265625, 0.017812500000000002]; 98.333333% [-0.2740625, 0.03203125000000002].
  - total_assay_cost: -0.069375; 95% [-0.22265624999999992, 0.017812499999999995]; 98.333333% [-0.27406249999999993, 0.03203125].
  - return_: 0.00069375; 95% [-0.00017812500000000017, 0.002226562500000001]; 98.333333% [-0.00032031250000000094, 0.0027406250000000013].
- imitation_ppo_continued-minus-imitation_ppo: development joint gate not met.
  - final_mse: 0; 95% [0.0, 0.0]; 98.333333% [0.0, 0.0].
  - variable_cost: 0.00125; 95% [-0.00015624999999999943, 0.004218749999999977]; 98.333333% [-0.0004179687500000035, 0.005156249999999982].
  - total_assay_cost: 0.00125; 95% [-0.00015624999999999943, 0.004218749999999977]; 98.333333% [-0.0004179687500000035, 0.005156249999999982].
  - return_: -1.25e-05; 95% [-4.2187499999996926e-05, 1.5625000000000056e-06]; 98.333333% [-5.156249999999848e-05, 4.179687500002047e-06].

Descriptive comparisons against both imitation controllers, fixed/fixed-quality designs and VOI are in summary.json. The privileged reference uses the true nominal mechanism and zero assay cost; it is not a public controller.

Recorded episodes: 5632; step calls: 16814; failures: 0. The episode and step-call caps are 10,000 and 50,000; CPU work has a 600-second deadline. Partial traces and missing-case counts remain in the raw records.

Raw actions, rewards, public traces and per-case metrics are compressed JSONL. The mechanism manifest and source snapshot support credentials-free and weights-free replay. No binary checkpoint is copied into this report.

## Limits

- All outcomes are development confirmation after tuning; intervals are selection-affected.
- Scratch cohort: GRU512; preserved imitation: GRU64. This is a comparison of complete systems whose network capacities and training histories are not held equal across all cohorts. It does not isolate PPO algorithmic or training value; matched-capacity imitation and matched lifetime-compute comparisons remain necessary for that claim.
- Five training seeds and common mechanisms are crossed resampling units; baseline copies are not independent replicates.
- Aggregate RMSE is sqrt(mean MSE), never mean per-case RMSE.
- Parameter holdout within familiar motifs does not establish structural or biological generalization.
- Warm starts inherit prior compute outside the new-training budget; equal lifetime compute is not claimed.
- Latency is scoped CPU runtime, with allocated neural batch latency; no temperature or profiling claim.
