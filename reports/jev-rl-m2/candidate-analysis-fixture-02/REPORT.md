# M2 paired development analysis

Mode: fixture. Mechanisms: 1; planned training seeds: [20, 21, 22, 23, 24].
Fixture mode: descriptive synthetic numbers only; efficacy and calibration benefit claims are disabled.
All differences are the first method minus its comparator. Null means unavailable or blocked, never zero.
The three prespecified families use 10,000 paired bootstrap draws, RNG 991499, descriptive 95% and Bonferroni 98.333333% intervals. Learned-method intervals require all five registered seeds; the fixed probe requires eight mechanism clusters. Gates are exploratory development signals only.

| Primary family | Status / reason | Return or Brier delta (95%; adjusted) | MSE delta (95%; adjusted) | Exploratory gate |
|---|---|---|---|---|
| arbitration-minus-ppo | descriptive_only / insufficient_mechanisms | 0 (null; null) | 0 (null; null) | null |
| rerank-minus-arbitration | descriptive_only / insufficient_mechanisms | 0 (null; null) | 0 (null; null) | null |
| probe-jev-minus-posterior | descriptive_only / insufficient_mechanisms | 0.069806 (null; null) | null (null; null) | null |

Return benefit requires adjusted return lower > 0 and adjusted MSE upper <= 0.0001. Forecast benefit requires complete paired probe coverage and adjusted delta-Brier upper < 0. Incomplete cohorts are never reduced to successful seeds.

| Method | Completed / planned | Return | Raw MSE | sqrt(mean MSE) | Variable / total assay cost | Effort | Assays | Reset-inclusive seconds | Failed / not run / missing | Immediate stops |
|---|---|---|---|---|---|---|---|---|---|---|
| additive | 1/1 | -0.698 | 0.0064 | 0.08 | 5.8 / 8.8 | 0 | 2 | 0.00955108 | 0 / 0 / 0 | 0 |
| arbitration | 5/5 | -0.6952 | 0.0064 | 0.08 | 5.52 / 8.52 | 0 | 2.4 | 0.347321 | 0 / 0 / 0 | 0 |
| bo | 1/1 | -0.698 | 0.0064 | 0.08 | 5.8 / 8.8 | 0 | 2 | 0.234688 | 0 / 0 / 0 | 0 |
| calibrated_imitation | 5/5 | -0.704 | 0.0064 | 0.08 | 6.4 / 9.4 | 0 | 3 | 0.0114074 | 0 / 0 / 0 | 0 |
| chance | 5/5 | -0.6952 | 0.0064 | 0.08 | 5.52 / 8.52 | 0 | 2.4 | 0.348205 | 0 / 0 / 0 | 0 |
| eig | 1/1 | -0.717 | 0.0064 | 0.08 | 7.7 / 10.7 | 0 | 3 | 0.0772841 | 0 / 0 / 0 | 0 |
| fixed | 1/1 | -0.704 | 0.0064 | 0.08 | 6.4 / 9.4 | 0 | 3 | 0.0118451 | 0 / 0 / 0 | 0 |
| fixed_quality | 1/1 | -0.714 | 0.0064 | 0.08 | 7.4 / 10.4 | 0 | 3 | 0.011959 | 0 / 0 / 0 | 0 |
| forecast_only | 5/5 | -0.6952 | 0.0064 | 0.08 | 5.52 / 8.52 | 0 | 2.4 | 0.135976 | 0 / 0 / 0 | 0 |
| hidden | 5/5 | -0.6952 | 0.0064 | 0.08 | 5.52 / 8.52 | 0 | 2.4 | 0.347501 | 0 / 0 / 0 | 0 |
| imitation | 5/5 | -0.704 | 0.0064 | 0.08 | 6.4 / 9.4 | 0 | 3 | 0.0113892 | 0 / 0 / 0 | 0 |
| imitation_ppo | 5/5 | -0.704 | 0.0064 | 0.08 | 6.4 / 9.4 | 0 | 3 | 0.0113491 | 0 / 0 / 0 | 0 |
| incumbent_m1 | 5/5 | -0.6992 | 0.0064 | 0.08 | 5.92 / 8.92 | 0 | 2.6 | 0.345645 | 0 / 0 / 0 | 0 |
| jev_only | 1/1 | -0.704 | 0.0064 | 0.08 | 6.4 / 9.4 | 0 | 3 | 0.371193 | 0 / 0 / 0 | 0 |
| ppo | 5/5 | -0.6952 | 0.0064 | 0.08 | 5.52 / 8.52 | 0 | 2.4 | 0.0116735 | 0 / 0 / 0 | 0 |
| ppo_v6 | 5/5 | -0.6992 | 0.0064 | 0.08 | 5.92 / 8.92 | 0 | 2.6 | 0.010666 | 0 / 0 / 0 | 0 |
| prior | 1/1 | -0.64 | 0.0064 | 0.08 | 0 / 3 | 0 | 0 | 0.0043 | 0 / 0 / 0 | 1 |
| probe | 1/1 | -0.704 | 0.0064 | 0.08 | 6.4 / 9.4 | 0 | 3 | 0.161425 | 0 / 0 / 0 | 0 |
| public | 5/5 | -0.6952 | 0.0064 | 0.08 | 5.52 / 8.52 | 0 | 2.4 | 0.345471 | 0 / 0 / 0 | 0 |
| random | 1/1 | -0.787826 | 0.00701826 | 0.083775 | 6.6 / 9.6 | 0.4 | 3 | 0.011525 | 0 / 0 / 0 | 0 |
| rerank | 5/5 | -0.6952 | 0.0064 | 0.08 | 5.52 / 8.52 | 0 | 2.4 | 0.30872 | 0 / 0 / 0 | 0 |
| rerank_shadow | 5/5 | -0.6952 | 0.0064 | 0.08 | 5.52 / 8.52 | 0 | 2.4 | 0.307502 | 0 / 0 / 0 | 0 |
| voi | 1/1 | -0.704 | 0.0064 | 0.08 | 6.4 / 9.4 | 0 | 3 | 0.0829973 | 0 / 0 / 0 | 0 |

Fixed common probe: 6/6 paired valid events; 0 missing Jev forecasts. Conditional delta Brier: 0.069806. Whole-probe Jev Brier bounds: [0.25, 0.25]; delta-Brier bounds: [0.069806, 0.069806].

| Role | Attempted calls | Known input / output tokens | Unknown usage calls | API seconds | Estimated USD | Known-usage USD subtotal |
|---|---|---|---|---|---|---|
| additive | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| arbitration | 32 | 0 / 0 | 0 | 0.00489675 | 0 | 0 |
| bo | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| calibrated_imitation | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| chance | 32 | 0 / 0 | 0 | 0.0047875 | 0 | 0 |
| eig | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| fixed | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| fixed_quality | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| forecast_only | 12 | 0 / 0 | 0 | 0.000843293 | 0 | 0 |
| hidden | 32 | 0 / 0 | 0 | 0.00445521 | 0 | 0 |
| imitation | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| imitation_ppo | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| incumbent_m1 | 33 | 0 / 0 | 0 | 0.00366217 | 0 | 0 |
| jev_only | 7 | 0 / 0 | 0 | 0.000937752 | 0 | 0 |
| ppo | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| ppo_v6 | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| prior | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| probe | 3 | 0 / 0 | 0 | 0.000218917 | 0 | 0 |
| public | 32 | 0 / 0 | 0 | 0.00466029 | 0 | 0 |
| random | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| rerank | 29 | 0 / 0 | 0 | 0.00224438 | 0 | 0 |
| rerank_shadow | 29 | 0 / 0 | 0 | 0.00231158 | 0 | 0 |
| voi | 0 | 0 / 0 | 0 | 0 | 0 | 0 |

Repeated exact request hashes: 62 groups among 241 actual attempts. Per-group actual counts, model labels, answer hashes and maximum Noul probability spreads are recorded in summary.json; these describe reply variability, not proof of weight drift.
Secondary rerank-minus-hidden: Both decision inputs omit forecast values. This compares restricted proposals with the full menu, but prompts and forecast-call order, coverage, and failure rules also differ; the menu effect is not isolated.
Study stop reason: null. Detailed attempted costs, API stages, local guard/failure counts, secondary pairs, and input/source hashes are in summary.json.

## Limits

- Development only: intervals and gates are exploratory; no calibration qualification, final-test, or SOTA claim.
- Parameter holdout within one simulator family does not establish structural or biological generalization.
- At most eight development mechanisms and five training seeds limit precision and generalization; one mechanism has no intervals or inferential success claim.
- Learned-method comparisons require all five registered training seeds (20 through 24) for intervals or benefit gates; smaller seed cohorts are descriptive only. The fixed probe resamples mechanisms only.
- PPO architecture/checkpoint selection and prior development/fixture reuse can affect results; these comparisons are not an untouched final evaluation.
- Seed-independent baselines, Jev-only, and the fixed probe run once per mechanism. Tiling for pairing does not create independent replicates.
- Adaptive assay forecasts are selection-affected. The forecast primary uses only the fixed common probe and requires all six paired events per mechanism.
- Missing forecasts are never replaced by chance. Conditional paired Brier scores do not establish unconditional whole-probe improvement.
- Recorded end-to-end timing includes reset in both engines, reference computations, and synchronous journals; historical native inference speed is not compared.
- Estimated API USD is conditional on reported usage and the manifest price; unknown usage is not free. Simulator costs and API dollars are separate units.
- Repeated exact requests describe observed reply variability, not proof of model-weight drift; model labels alone do not certify fixed weights.
- Prospective registration requires independently recording the analysis source and contract hashes before hosted dispatch; an analysis output alone cannot establish chronology.
