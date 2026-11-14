# M2 paired development analysis

Mode: hosted. Mechanisms: 8; planned training seeds: [20, 21, 22, 23, 24].
Hosted mode: benefit gates remain exploratory development signals.
All differences are the first method minus its comparator. Null means unavailable or blocked, never zero.
The three prespecified families use 10,000 paired bootstrap draws, RNG 991499, descriptive 95% and Bonferroni 98.333333% intervals. Learned-method intervals require all five registered seeds; the fixed probe requires eight mechanism clusters. Gates are exploratory development signals only.

| Primary family | Status / reason | Return or Brier delta (95%; adjusted) | MSE delta (95%; adjusted) | Exploratory gate |
|---|---|---|---|---|
| arbitration-minus-ppo | complete / null | -0.0463827 ([-0.124407, 0.0038]; [-0.149567, 0.006175]) | 0.000544577 ([0, 0.00136843]; [0, 0.00164527]) | False |
| rerank-minus-arbitration | complete / null | 0.0206599 ([-0.0670308, 0.120781]; [-0.0914546, 0.14744]) | -0.000271599 ([-0.00135711, 0.00068169]; [-0.0016369, 0.000952686]) | False |
| probe-jev-minus-posterior | complete / null | 0.0403035 ([0.0100203, 0.0705055]; [0.00443669, 0.0765234]) | null (null; null) | False |

Return benefit requires adjusted return lower > 0 and adjusted MSE upper <= 0.0001. Forecast benefit requires complete paired probe coverage and adjusted delta-Brier upper < 0. Incomplete cohorts are never reduced to successful seeds.

| Method | Completed / planned | Return | Raw MSE | sqrt(mean MSE) | Variable / total assay cost | Effort | Assays | Reset-inclusive seconds | Failed / not run / missing | Immediate stops |
|---|---|---|---|---|---|---|---|---|---|---|
| additive | 8/8 | -0.517412 | 0.00441912 | 0.0664764 | 5.8 / 8.8 | 0.35 | 2 | 0.00937127 | 0 / 0 / 0 | 0 |
| arbitration | 40/40 | -0.357236 | 0.00285211 | 0.0534052 | 4.6025 / 7.6025 | 0.52 | 1.85 | 1.86904 | 0 / 0 / 0 | 0 |
| bo | 8/8 | -0.434368 | 0.00367243 | 0.0606005 | 4.7125 / 7.7125 | 0.4 | 1.625 | 0.0555834 | 0 / 0 / 0 | 0 |
| calibrated_imitation | 40/40 | -0.304528 | 0.00230753 | 0.0480368 | 4.3775 / 7.3775 | 0.6 | 1.875 | 0.0084562 | 0 / 0 / 0 | 0 |
| chance | 40/40 | -0.404231 | 0.00339856 | 0.0582972 | 4.2375 / 7.2375 | 0.44 | 1.625 | 1.72658 | 0 / 0 / 0 | 0 |
| eig | 8/8 | -0.337753 | 0.00230753 | 0.0480368 | 7.7 / 10.7 | 0.6 | 3 | 0.0668282 | 0 / 0 / 0 | 0 |
| fixed | 8/8 | -0.451366 | 0.00367366 | 0.0606107 | 6.4 / 9.4 | 0.4 | 3 | 0.0113911 | 0 / 0 / 0 | 0 |
| fixed_quality | 8/8 | -0.461366 | 0.00367366 | 0.0606107 | 7.4 / 10.4 | 0.4 | 3 | 0.011741 | 0 / 0 / 0 | 0 |
| forecast_only | 40/40 | -0.310853 | 0.00230753 | 0.0480368 | 5.01 / 8.01 | 0.6 | 2.075 | 0.572504 | 0 / 0 / 0 | 0 |
| hidden | 40/40 | -0.439813 | 0.00372388 | 0.0610236 | 4.6425 / 7.6425 | 0.42 | 1.8 | 1.8223 | 0 / 0 / 0 | 3 |
| imitation | 40/40 | -0.304528 | 0.00230753 | 0.0480368 | 4.3775 / 7.3775 | 0.6 | 1.875 | 0.00867972 | 0 / 0 / 0 | 0 |
| imitation_ppo | 40/40 | -0.304578 | 0.00230753 | 0.0480368 | 4.3825 / 7.3825 | 0.6 | 1.875 | 0.00846049 | 0 / 0 / 0 | 0 |
| incumbent_m1 | 40/40 | -0.519062 | 0.00459412 | 0.06778 | 4.065 / 7.065 | 0.38 | 1.525 | 1.58391 | 0 / 0 / 0 | 0 |
| jev_only | 8/8 | -0.6625 | 0.0064 | 0.08 | 2.25 / 5.25 | 0 | 0.875 | 1.12658 | 0 / 0 / 0 | 3 |
| ppo | 40/40 | -0.310853 | 0.00230753 | 0.0480368 | 5.01 / 8.01 | 0.6 | 2.075 | 0.0111985 | 0 / 0 / 0 | 0 |
| ppo_v6 | 40/40 | -0.309503 | 0.00230753 | 0.0480368 | 4.875 / 7.875 | 0.6 | 2.05 | 0.0100953 | 0 / 0 / 0 | 0 |
| prior | 8/8 | -0.64 | 0.0064 | 0.08 | 0 / 3 | 0 | 0 | 0.00392407 | 0 / 0 / 0 | 8 |
| probe | 8/8 | -0.451366 | 0.00367366 | 0.0606107 | 6.4 / 9.4 | 0.4 | 3 | 0.853118 | 0 / 0 / 0 | 0 |
| public | 40/40 | -0.391093 | 0.00326343 | 0.0571264 | 4.175 / 7.175 | 0.46 | 1.625 | 1.75082 | 0 / 0 / 0 | 0 |
| random | 8/8 | -0.626863 | 0.00533863 | 0.0730659 | 7.3 / 10.3 | 0.4 | 3.25 | 0.0120885 | 0 / 0 / 0 | 0 |
| rerank | 40/40 | -0.336576 | 0.00258051 | 0.0507987 | 5.0525 / 8.0525 | 0.56 | 2.1 | 1.52793 | 0 / 0 / 0 | 0 |
| rerank_shadow | 40/40 | -0.310853 | 0.00230753 | 0.0480368 | 5.01 / 8.01 | 0.6 | 2.075 | 1.51318 | 0 / 0 / 0 | 0 |
| voi | 8/8 | -0.304753 | 0.00230753 | 0.0480368 | 4.4 / 7.4 | 0.6 | 1.875 | 0.0626063 | 0 / 0 / 0 | 0 |

Fixed common probe: 48/48 paired valid events; 0 missing Jev forecasts. Conditional delta Brier: 0.0403035. Whole-probe Jev Brier bounds: [0.265108, 0.265108]; delta-Brier bounds: [0.0403035, 0.0403035].

| Role | Attempted calls | Known input / output tokens | Unknown usage calls | API seconds | Estimated USD | Known-usage USD subtotal |
|---|---|---|---|---|---|---|
| additive | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| arbitration | 219 | 2061969 / 193586 | 0 | 57.1488 | 0.0866027 | 0.0866027 |
| bo | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| calibrated_imitation | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| chance | 204 | 1899885 / 179276 | 0 | 52.8043 | 0.0797952 | 0.0797952 |
| eig | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| fixed | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| fixed_quality | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| forecast_only | 83 | 287881 / 3632 | 0 | 16.9501 | 0.012091 | 0.012091 |
| hidden | 212 | 1823854 / 174712 | 0 | 56.1113 | 0.0766019 | 0.0766019 |
| imitation | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| imitation_ppo | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| incumbent_m1 | 196 | 1324737 / 170844 | 0 | 48.1677 | 0.055639 | 0.055639 |
| jev_only | 29 | 250318 / 25606 | 0 | 6.86582 | 0.0105134 | 0.0105134 |
| ppo | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| ppo_v6 | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| prior | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| probe | 24 | 86101 / 1040 | 0 | 5.01675 | 0.00361624 | 0.00361624 |
| public | 205 | 1936226 / 183720 | 0 | 53.6538 | 0.0813215 | 0.0813215 |
| random | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| rerank | 208 | 902007 / 9949 | 0 | 45.3049 | 0.0378843 | 0.0378843 |
| rerank_shadow | 206 | 893325 / 9853 | 0 | 45.1235 | 0.0375197 | 0.0375197 |
| voi | 0 | 0 / 0 | 0 | 0 | 0 | 0 |

Repeated exact request hashes: 351 groups among 1586 actual attempts. Per-group actual counts, model labels, answer hashes and maximum Noul probability spreads are recorded in summary.json; these describe reply variability, not proof of weight drift.
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
