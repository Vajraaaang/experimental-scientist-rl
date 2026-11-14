# Replay and reproduce the M2 development evidence

The public workflow verifies the archive, recomputes the measured analyses, and reexecutes the hosted simulator outcomes using the recorded actions. The original snapshot verifiers also provide exact decision replay in the reference environment. Both paths use CPU and need neither a TypeSafe key nor policy checkpoints. [TypeSafe autoresearch cookbook](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery) · [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist)

## Quick start

Clone the repository and check out a commit containing the M2 hosted evidence. To reproduce the recorded-outcome checks, run:

```bash
git clone --branch vajraang/jev-rl-m2 https://github.com/Vajraaaang/experimental-scientist-rl.git
PROJECT="$(pwd -P)/experimental-scientist-rl"
python3.12 "$PROJECT/scripts/bootstrap.py"
env -u TYPESAFE_API_KEY "$PROJECT/.venv/bin/python" -B "$PROJECT/scripts/replay_jev_m2_recorded_actions.py" \
  --root "$PROJECT"
```

`bootstrap.py` installs the pinned dependencies and runs the tests inside the clone's virtual environment. Recorded-action replay checks the exact hashes of the eight synthetic mechanism parameter sets against the locked plan. It then checks action legality, observations, rewards and final MSE. Float comparisons allow absolute error up to `1e-12` in these numerical outcomes. Hashes, actions and status fields retain exact equality. The checker uses the saved decisions, forecasts and API receipts. [Dependency lock](../requirements.lock) · [CI workflow](../.github/workflows/tests.yml)

To verify the archive and regenerate its M2 analyses from stored cases, including on Linux:

```bash
env -u TYPESAFE_API_KEY "$PROJECT/.venv/bin/python" -B "$PROJECT/scripts/export_jev_m2_public.py" \
  --destination "$PROJECT" --verify-published --archive-only
```

This mode checks each receipt-listed file and the plan, journal and analysis bindings. It recomputes the PPO confirmation intervals and both M2 paired analyses using the registered analysis code. The report marks simulator dynamics as unreexecuted in this mode.

## Exact reference replay

The original M3 Max environment used macOS 26.3.1, CPython 3.12.1 and the pinned packages. Its strict verifier also recomputes controller and Bayesian reference decisions. That run reproduced all 6,422 archived episodes. Linux exact replay stopped at forecast or task reconstruction checks. On GitHub's Apple Silicon runner, exact replay reproduced the 5,632 PPO confirmation episodes and stopped at the baseline-action check in the Jev runs. These results delimit exact replay to the tested reference environment. [Linux diagnostic run](https://github.com/Vajraaaang/experimental-scientist-rl/actions/runs/43630568248) · [Apple Silicon diagnostic run](https://github.com/Vajraaaang/experimental-scientist-rl/actions/runs/43631793675)

```bash
env -u TYPESAFE_API_KEY "$PROJECT/.venv/bin/python" -B "$PROJECT/scripts/export_jev_m2_public.py" \
  --destination "$PROJECT" --verify-published
```

## Rebuild the PPO tables and figure

```bash
OUTPUT="$(mktemp -d)"
"$PROJECT/.venv/bin/python" -B "$PROJECT/scripts/summarize_ppo_m2.py" \
  --root "$PROJECT" --output "$OUTPUT/ppo"
```

The output contains `COMPARISONS.md`, `summary.json` and `m2-comparison.png`. Table values come from registered race and confirmation records. Compare the regenerated summary and Markdown to the published files. Figure rendering can vary with fonts or the graphics backend. [Measured PPO tables and figure](../reports/jev-rl-m2/summary-01)

## Inspect a decision

Start with the run's `trace-index.json` or an individual case listed in `results.json`. Each case JSON points to its JSONL journal. Follow the sequence from public state through a typed forecast and legal action choice to the purchased observation. Each forecast records the public cutoff and links to its later outcome. The snapshot verifier also checks API usage, stopping rules, and the recorded plan.

## What each reproduction mode covers

| Mode | Inputs | Result |
|---|---|---|
| Reference simulator replay | Published traces, hashes, source snapshots and reference Apple Silicon CPU dependencies | Reconstructs decisions from recorded advice and responses, runs the simulator, checks receipts and recomputes paired statistics |
| Archive verification | Published traces, summaries, hashes, analysis code and pinned CPU dependencies | Verifies file and protocol bindings and recomputes the statistics from recorded outcomes |
| Hosted recorded-action replay | Archive plus the hash-checked synthetic task parameters | Reruns simulator observations, costs, rewards and final outcomes for recorded actions |
| Checkpoint evaluation | Original checkpoint files, their recorded hashes and frozen policies | Recomputes policy advice from the trained tensors. The public bundle includes hashes, traces and the reference audit receipt. |
| Training reproduction | Registered training configurations, warm-start artifacts, native dependencies and the declared compute budget | Reruns optimization. Exact timing and trained results can differ with device, package versions and scheduling. |
| New hosted replication | Fresh TypeSafe access, current pricing, declared budget, and a prospectively frozen design | Obtains newly sampled responses from the provider. Stored responses support independent replay even if later service behavior differs. |

The original checkout with the registered checkpoints and original hosted plan can repeat the checkpoint advice audit using the same CPU environment:

```bash
OUTPUT="$(mktemp -d)"
env -u TYPESAFE_API_KEY "$PROJECT/.venv/bin/python" -B "$PROJECT/scripts/verify_jev_m2_checkpoint_advice.py" \
  --output "$OUTPUT/checkpoint-advice.json"
```

The public clone verifies the stored decisions. The command above additionally requires the original checkpoint files and pre-export evidence named by the receipt. [Checkpoint audit](../reports/jev-rl-m2/hosted-checkpoint-advice-audit.json) · [Research note](m2-research-note.md) · [Evidence card](m2-evidence-card.md)
