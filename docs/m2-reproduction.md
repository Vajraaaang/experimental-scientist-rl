# Replay and reproduce the M2 development evidence

The public workflow verifies artifact hashes, replays recorded decisions, and recomputes the reported analyses. It uses the same frozen simulator snapshots and stored Jev responses as the public evidence. It runs on CPU without a TypeSafe key or policy checkpoints. This verification level follows TypeSafe's example of replay from saved answers and the NeurIPS guidance on documenting exact environments, commands and reproducibility limits. [TypeSafe autoresearch cookbook](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery) · [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist)

## Quick start

Clone the repository and check out a commit containing the M2 hosted evidence. To replay the current M2 branch, run:

```bash
git clone --branch vajraang/jev-rl-m2 https://github.com/Vajraaaang/experimental-scientist-rl.git
PROJECT="$(pwd -P)/experimental-scientist-rl"
python3.12 "$PROJECT/scripts/bootstrap.py"
env -u TYPESAFE_API_KEY "$PROJECT/.venv/bin/python" -B "$PROJECT/scripts/export_jev_m2_public.py" \
  --destination "$PROJECT" --verify-published
```

`bootstrap.py` installs the pinned dependencies and runs the tests inside the clone's virtual environment. GitHub Actions runs the tests, simulator diagnostics, and publication replay on Linux CPU. The original measurement environment used arm64 macOS, CPython 3.12.1, and the package versions recorded in the lock file. [Dependency lock](../requirements.lock) · [CI workflow](../.github/workflows/tests.yml)

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
| Public replay | Published traces, hashes, source snapshots and CPU dependencies | Reconstructs decisions from recorded advice and responses, runs the simulator, checks receipts and recomputes paired statistics |
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
