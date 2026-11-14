# Experimental Scientist RL

A benchmark for choosing measurements under a budget. A policy observes noisy readings from a synthetic dynamical system, buys up to four measurements from 36 assay actions, and commits to one of nine final actions. The evaluator measures target error, assay cost, and intervention effort. We compare PPO, imitation learning, fixed designs, Bayesian planning, and typed decisions from TypeSafe Jev. [Task and scoring protocol](docs/protocol.md)

## Main results

The hosted development study completed **600 cases across eight synthetic mechanisms and five PPO seeds** with pinned `jev-1.14.0`. All three registered benefit criteria failed. Return differences below are first method minus comparator. Lower Brier error is better. [Research note](docs/m2-research-note.md) · [Registered analysis](docs/jev_rl_m2_analysis_registration.json)

| Comparison | Result |
|---|---|
| Full Jev arbitration vs. standalone PPO | Return **−0.04638** (adjusted 98.33% CI [−0.14957, +0.00618]). |
| Jev reranking vs. full arbitration | **56.3% lower estimated API spend**. Return **+0.02066** (adjusted 98.33% CI [−0.09145, +0.14744]). |
| Jev forecasts vs. Bayesian posterior | Brier error **0.26511 vs. 0.22480** on 48 matched events. Difference **+0.04030** (adjusted 98.33% CI [+0.00444, +0.07652]). |
| Selected PPO vs. frozen v6 PPO | Return **+0.00630** on 128 development mechanisms (paired 95% CI [−0.00170, +0.02062]). |

The results cover the declared synthetic mechanism families. The final split remains unopened. [Full results and limitations](docs/results.md)

![Paired return, target-error, and forecast-error effects with confidence intervals. Zero of three registered benefit criteria passed.](reports/jev-rl-m2/figures/paired-effects.png)

[Return and API spend by policy](reports/jev-rl-m2/figures/return-and-spend.png) · [Forecast error by mechanism](reports/jev-rl-m2/figures/forecast-error.png)

## Reproduce the results

From a clone of this repository, run with Python 3.12:

```bash
PROJECT="$(pwd -P)"
python3.12 "$PROJECT/scripts/bootstrap.py"
"$PROJECT/.venv/bin/python" -B "$PROJECT/scripts/replay_jev_m2_recorded_actions.py" \
  --root "$PROJECT"
```

This checks artifact hashes, regenerates the analyses, and replays all 600 hosted cases from saved actions on CPU. The public replay requires neither an API key nor model weights. [Reproduction guide](docs/m2-reproduction.md) · [Linux and macOS checks](https://github.com/Vajraaaang/experimental-scientist-rl/actions/workflows/tests.yml)

## Evidence and code

- [Hosted analysis](reports/jev-rl-m2/hosted-analysis-01/REPORT.md) and [per-case traces](reports/jev-rl-m2/hosted-study-01/episodes)
- [PPO configuration study](reports/jev-rl-m2/REPORT.md) and [compute and performance tables](reports/jev-rl-m2/summary-01/COMPARISONS.md)
- [Policy card](docs/m2-policy-card.md), [evidence card](docs/m2-evidence-card.md), and [simulator and policy code](src/experimentlab)

Cite the commit containing the evidence you use. The repository currently specifies no reuse license. [Citation metadata](CITATION.cff)
