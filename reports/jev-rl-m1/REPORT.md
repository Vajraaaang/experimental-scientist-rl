# Jev integration validation

The first study checks whether the simulator, controllers, and trace replay agree.
A deterministic fixture supplies responses for forecasts and legal action choice.
The fixture results measure integration behavior. Hosted performance is measured
in the separate paired development study. [Hosted research note](../../docs/m2-research-note.md)

## Validation results

- All **100 fixture episodes** completed and replayed across 18 conditions, two development mechanisms, and five learned-policy seeds.
- All **40 learned-policy fixture cases** matched the original collector's action sequence and final score.
- The replay verified **236 fixture calls** and 12 scored fixed-probe events. The fixtures used no hosted calls.
- Five additional traces cover immediate commitment, a timeout after one purchased assay, model drift, invalid usage, and rate limiting. The timeout retained its assay cost and scored observations.

Each trace records the public history, forecast, action, eventual measurement, and
API receipt. Local spans connect forecasts and actions to outcomes. Receipts also
retain model version, usage, and provider request IDs when available.
[Fixture manifest](fixture-04/manifest.json) · [Results](fixture-04/results.json) · [Trace index](fixture-04/trace-index.json) · [Replay verification](fixture-04/verification.json) · [Failure fixtures](failure-examples/summary.json)

## Fixture comparison

Matching PPO and hybrid fixture outcomes are expected because the fixture follows
the supplied PPO probabilities. Variable cost excludes the common setup cost of
three units.

| Condition | Episodes | Final MSE | Variable cost | Return | Effort | Assays |
|---|---:|---:|---:|---:|---:|---:|
| additive | 2 | 0.00679037 | 5.800 | -0.747037 | 0.200 | 2.00 |
| bo | 2 | 0.00352560 | 5.800 | -0.430560 | 0.400 | 2.00 |
| calibrated_imitation | 10 | 0.00067857 | 4.600 | -0.153857 | 0.800 | 2.00 |
| eig | 2 | 0.00067857 | 7.700 | -0.184857 | 0.800 | 3.00 |
| fixed | 2 | 0.00640000 | 6.400 | -0.704000 | 0.000 | 3.00 |
| fixed_quality | 2 | 0.00640000 | 7.400 | -0.714000 | 0.000 | 3.00 |
| forecast_probe | 2 | 0.00640000 | 6.400 | -0.704000 | 0.000 | 3.00 |
| imitation | 10 | 0.00067857 | 4.600 | -0.153857 | 0.800 | 2.00 |
| imitation_ppo | 10 | 0.00067857 | 4.630 | -0.154157 | 0.800 | 2.00 |
| jev_only | 2 | 0.00640000 | 6.400 | -0.704000 | 0.000 | 3.00 |
| jev_rl | 10 | 0.00067857 | 4.400 | -0.151857 | 0.800 | 1.80 |
| jev_rl_chance | 10 | 0.00067857 | 4.400 | -0.151857 | 0.800 | 1.80 |
| jev_rl_hidden | 10 | 0.00067857 | 4.400 | -0.151857 | 0.800 | 1.80 |
| jev_rl_public | 10 | 0.00067857 | 4.400 | -0.151857 | 0.800 | 1.80 |
| ppo | 10 | 0.00067857 | 4.400 | -0.151857 | 0.800 | 1.80 |
| prior | 2 | 0.00640000 | 0.000 | -0.640000 | 0.000 | 0.00 |
| random | 2 | 0.00390034 | 7.250 | -0.492534 | 0.600 | 3.50 |
| voi | 2 | 0.00067857 | 4.600 | -0.153857 | 0.800 | 2.00 |

## Reproduction

Use the frozen snapshot to check receipts and recompute simulator outcomes in the
reference runtime. Public replay requires neither an API key nor policy weights.
The public hosted study also supplies a recorded-action replay that runs on
Linux and macOS. [Reproduction guide](../../docs/m2-reproduction.md)

```bash
PROJECT="$(pwd -P)"
"$PROJECT/.venv/bin/python" -B \
  "$PROJECT/reports/jev-rl-m1/fixture-04/source_snapshot/scripts/jev_rl_study.py" \
  verify --run "$PROJECT/reports/jev-rl-m1/fixture-04"
```
