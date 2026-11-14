# Experimental Scientist RL

Experimental Scientist RL is a benchmark for **choosing measurements under a budget**. A controller selects up to four assays, observes noisy readings from a hidden synthetic system and commits to one of nine final actions. The evaluator measures target error, assay cost and intervention effort. The benchmark compares PPO with fixed designs, imitation learning, Bayesian planning and typed decisions from TypeSafe Jev. [Simulator and scoring protocol](docs/protocol.md)

The **M2 development study** compared full Jev arbitration, reranking three legal PPO proposals, and forecasting without changing PPO's choices. All 600 planned cases completed across eight mechanisms and five PPO training seeds. All three registered benefit gates failed. The intervals below are clustered paired bootstrap intervals. [Research note](docs/m2-research-note.md) · [Prospective analysis registration](docs/jev_rl_m2_analysis_registration.json)

| Question | Measured result |
|---|---|
| Did full Jev arbitration improve PPO return? | Mean return difference **−0.04638**, adjusted 98.333% CI **[−0.14957, +0.00618]**. |
| Did a smaller proposal menu help? | Reranking used **56.3% less estimated API spend** than full arbitration. Its return difference was **+0.02066**, adjusted CI **[−0.09145, +0.14744]**, and its mean return remained below PPO's. |
| Did Jev improve forecasts of the same purchased measurements? | Across **48 matched events**, Brier error was **0.26511** for Jev and **0.22480** for the Bayesian posterior. The difference was **+0.04030**, adjusted CI **[+0.00444, +0.07652]**. Lower is better. |
| What did the local PPO search establish? | CPU led the measured full loop on the M3 Max. On 128 development mechanisms, return for the selected 512-unit GRU minus v6 was **+0.00630**, paired 95% CI **[−0.00170, +0.02062]**. |

[Hosted analysis](reports/jev-rl-m2/hosted-analysis-01/REPORT.md) · [Per-case traces](reports/jev-rl-m2/hosted-study-01/episodes) · [Compute and PPO results](reports/jev-rl-m2/REPORT.md) · [Policy card](docs/m2-policy-card.md) · [Evidence card](docs/m2-evidence-card.md)

## Replay the study

In a checkout containing the M2 evidence, run the following from the repository root with Python 3.12:

```bash
PROJECT="$(pwd -P)"
python3.12 "$PROJECT/scripts/bootstrap.py"
env -u TYPESAFE_API_KEY "$PROJECT/.venv/bin/python" -B "$PROJECT/scripts/export_jev_m2_public.py" \
  --destination "$PROJECT" --verify-published
```

Replay runs on CPU without an API key or checkpoints. It reconstructs the recorded decisions and reproduces the paired analyses. The original learned weights are held locally, and the repository currently specifies no reuse license. Evidence covers the declared simulator families. The final split remains unopened. [Full reproduction guide](docs/m2-reproduction.md) · [Artifact hashes and checks](reports/jev-rl-m2/publication-export.json)

## Episode walkthrough

![Conceptual animation of a scientific agent choosing an experiment](docs/media/experimental-scientist.gif)

<sub>Conceptual workflow. [Watch the MP4](docs/media/experimental-scientist.mp4) · [Download the still cover](docs/media/experimental-scientist.png)</sub>

1. **Get the goal.** The agent receives target values, baseline measurements, available actions, prices, and a budget.
2. **Choose a test.** It selects a control to suppress (or a no-intervention control), a dose, a response duration, and which pair of variables to measure.
3. **Pay and observe.** The simulator runs the experiment on fresh cultures. The agent sees the measurements it purchased.
4. **Adapt the next choice.** Based on those results, it can run another test or stop. The main pilot allows at most four experiments within a budget of eight virtual cost units.
5. **Commit and evaluate.** It chooses a final intervention. A separate evaluator tests it on independent cultures and scores target error, assay spending, and intervention effort.

**Choose → experiment → observe → adapt → commit.**

For example, the agent might first measure downstream outcomes after suppressing one control. Its observations can lead it to investigate another control with a cheaper measurement, or to stop because it already has enough evidence. This is what makes the strategy *adaptive*. A fixed schedule preselects its tests before observing any results.

## What can the agent see and do?

| Part | In this project |
|---|---|
| Hidden system | Eight simulated variables connected by an unknown regulatory mechanism |
| Evidence | Baseline measurements and the noisy results of purchased experiments |
| Experiment choices | 36 combinations of intervention, duration, and measurement panel |
| Final choices | Nine interventions, including leaving the system unchanged |
| Resources | Budget 8; at most four purchased experiments in the main pilot |
| Fresh samples | Every experiment uses fresh cultures; final scoring uses an independent batch |
| Private information | True mechanism parameters, unpurchased observations, oracle rankings, and final-evaluation randomness stay outside the policy input |

The cost units are part of the simulator, not real laboratory prices. The policy receives public observations and action masks; trusted simulator/evaluator code holds the hidden system.

## What is being learned?

The agent must learn both **how to gather useful evidence** and **how to turn that evidence into an intervention**. Buying more tests is worthwhile only when their benefit justifies their cost.

| Approach | How it chooses |
|---|---|
| Fixed experimental design | Runs a schedule selected on development data |
| Bayesian decision-value planner | Updates a belief over possible mechanisms and estimates which test could improve the final decision |
| Imitation learning | Learns to reproduce the planner's decisions from public experiment histories |
| PPO reinforcement learning | Learns from experiments and rewards, using a reward-only curriculum to develop interpretation and experiment-selection skills |

The historical v4–v6 learned comparisons use the same 64-unit recurrent neural architecture. Training budgets differ and are reported. Public-price logit calibration is reported separately from PPO gradient learning. The newer M2 development study also tests larger policies and explicitly reports the capacity difference from imitation.

## Earlier locked evaluations

- **Strong fixed designs matter.** The original feasibility-conditioned task was already solved by a fixed schedule. That result is preserved.
- **Adaptive experiments can help.** A documented development revision retains unreachable targets. On fresh tests of that task, useful adaptive PPO policies beat the preserved strong fixed design.
- **PPO has improved, but imitation remains cheaper.** More training reduced PPO's spending and target error and improved its return. A PPO advantage over strong imitation has not been established.

The historical v6 frozen comparison uses **4,096 new mechanisms and all five retained training seeds**:

| Method | Final target error (RMSE ↓) | Experimental cost ↓ |
|---|---:|---:|
| Strong fixed design | 0.05521 | 6.400 |
| Previous efficient PPO | 0.04912 | 4.969 |
| Updated PPO | 0.04865 | 4.782 |
| Strong imitation | 0.04859 | 4.623 |

RMSE measures how far final outputs are from their targets. Experimental cost sums the purchased assays. The common baseline setup costs three additional units, outside both the budget of eight and this cost column. Lower is better for both metrics, and the full evaluation also accounts for intervention effort.

Updated PPO spends **3.8% less than its previous version**, with favorable paired 95% intervals for cost, target MSE, and return. It removes **54.2% of the previous cost gap** at the point estimate, but still spends **3.4% more than imitation**. The predeclared full efficiency-parity screen failed.

These are **synthetic-system results within familiar mechanism families**. They do not establish wet-lab usefulness, structural generalization, or therapeutic efficacy. Training/development and final tests are separated; methods are frozen before final scoring.

Read [the complete measured report](docs/ppo_efficiency_v6.md) for confidence intervals, failed controls, compute accounting, and limitations. [The results guide](docs/results.md) keeps the earlier studies and negative findings separate.

## M2 compute and hosted accounting

The M3 Max study completed 42 bounded training runs in 49.4 minutes, using 49.65 million new training transitions. CPU beat native MPS and a CPU/MPS split for the measured full loop. GRU128 led the short 240-second architecture screen; GRU512 led at approximately two million interactions and was selected for the five-seed continuation comparison.

Strong imitation and Bayesian value-of-information planning retained lower assay cost and higher mean return in the PPO development confirmation. The earlier locked evaluation and the M2 confirmation use different mechanism samples. [Comparable results within each study](docs/results.md)

The M2 hosted comparison used 1,586 successful calls to pinned `jev-1.14.0` for an estimated $0.48. Forecast-only Jev reproduced all 40 PPO action sequences. The 23-condition analysis includes all measured controls, observation counts, stops, failures, usage and latency. [Hosted results](reports/jev-rl-m2/hosted-analysis-01/REPORT.md)

The recorded evidence includes provider request IDs, local spans, forecasts and legal choices. The original local checkpoints reproduced all 1,497 logged policy distributions exactly. [Capability audit](docs/jev_capabilities_audit.md) · [Checkpoint verification](reports/jev-rl-m2/hosted-checkpoint-advice-audit.json) · [PPO tables](reports/jev-rl-m2/summary-01/COMPARISONS.md)

## Run a small demonstration

Clone the repository, open its directory in a terminal, and use native Python 3.12. CPU execution is supported; CUDA is not required.

```bash
PROJECT="$PWD"
python3.12 "$PROJECT/scripts/bootstrap.py"
"$PROJECT/.venv/bin/experimentlab" doctor
"$PROJECT/.venv/bin/experimentlab" validate-science --suite diagnostic
"$PROJECT/.venv/bin/experimentlab" feasibility --quick --output "$PROJECT/runs/quick-check"
```

The quick command checks the simulator and bounded comparison pipeline. It does not reproduce the full published training study. A full GUI is not included.

To inspect a recorded episode, use [a public replay packet](reports/ppo-v4/replays/success.json) with the replay command:

```bash
PROJECT="$PWD"
"$PROJECT/.venv/bin/experimentlab" replay \
  --episode "$PROJECT/reports/ppo-v4/replays/success.json"
```

The packet separates purchased observations/actions from post-episode evaluation. Additional replays show [a policy failure](reports/ppo-v4/replays/policy-failure.json) and [an unreachable target](reports/ppo-v4/replays/unreachable-target.json).

## Explore the implementation and evidence

| Start here | What it explains |
|---|---|
| [Scientific protocol](docs/protocol.md) | Dynamics, actions, budgets, noise, and information boundaries |
| [Development revisions](docs/revisions.md) | Why the initial fixed-design success was preserved and how the task was revised |
| [Results guide](docs/results.md) | Which claims each study supports and what remains uncertain |
| [M2 research note](docs/m2-research-note.md) | Hosted results, protocol chronology, uncertainty and limitations |
| [Reproduction guide](docs/m2-reproduction.md) | Exact CPU replay and report regeneration commands |
| [M2 policy card](docs/m2-policy-card.md) and [evidence card](docs/m2-evidence-card.md) | Intended use, provenance, split boundaries, schemas and reuse scope |
| [PPO collapse investigation](docs/ppo_followup.md) | Why early training stopped buying experiments and how the curriculum helped |
| [Latest efficiency report](docs/ppo_efficiency_v6.md) | The new locked comparison, uncertainty, and remaining imitation gap |
| [Source package](src/experimentlab) | Simulator, environment, inference, baselines, and learning code |
| [Study workflows](scripts) | Bounded training, selection, evaluation, reporting, and benchmarking scripts |
| [Tests](tests) | Scientific calculations, masks, recurrent histories, padding, checkpoints, and device paths |

For an initial learning smoke run:

```bash
PROJECT="$PWD"
"$PROJECT/.venv/bin/experimentlab" train --smoke --device cpu \
  --output "$PROJECT/runs/learning-smoke"
"$PROJECT/.venv/bin/python" -m pytest "$PROJECT/tests" -c "$PROJECT/pyproject.toml"
```

The generic CLI retains the original pilot's gates. Later revised studies use the workflows under `scripts/`, including `ppo_efficiency_v6.py`, `ppo_v6_select.py`, and `ppo_v6_locked.py`. Read their protocols before launching larger runs. Use a separate project/output copy: the default research paths contain recorded evidence, and completed locked results refuse overwrite. Never use saved test outcomes to select a new method.

The public release contains checkpoint hashes, configurations, training curves, and the recorded outcomes. Neural inference can be checked with the original checkpoints. Gzip-compressed raw records preserve the numerical evidence without oversized Git blobs. [Verification scope](docs/m2-reproduction.md)

The reference calculations use NumPy/SciPy float64 on CPU; the neural models use PyTorch float32. Native CPU and Apple MPS paths have been measured. Current complete workloads favor CPU at this model size; benchmarks and their scopes are reported alongside the studies.

[Build specification](autonomous_scientist_build_spec.md) · [Original publication manifest](docs/publication-manifest.json) · [V3–V5 update manifest](docs/publication-update-2026-10-26.json) · [V6 publication manifest](docs/publication-update-v6.json) · [Documentation and cover update](docs/documentation-update-2026-10-29.json)

Original development hashes are provenance for the measured artifacts. Publication redactions and portability edits are recorded separately. The cover animation illustrates experiment selection; recorded results come from the simulator and frozen evaluation files.

Use the repository citation and the commit containing the evidence you use. The repository currently specifies no reuse license. [Citation metadata](CITATION.cff)
