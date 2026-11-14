# Jev/RL first research milestone implementation plan

> Use subagent-driven development for the isolated API adapter and independent
> reviews; the primary agent owns integration and evidence. No public release,
> new training, or new final test is part of this milestone.

**Goal:** Execute and replay the full frozen-policy/Jev research loop on bounded
development fixtures, with a ready but separately gated authenticated pilot.

**Architecture:** Public-packet forecast/decision adapters surround the existing
unchanged simulator and full policies. Append-only event receipts precede each
observation. Comparison manifests and deterministic replay bind evidence to code.

**Tech stack:** Existing Python 3.12 virtual environment, NumPy/SciPy, CPU PyTorch,
standard-library HTTPS, JSON and SHA256. New files live outside `src` so historical
`code_hash()` and frozen source files remain intact.

- [x] Verify both checkouts/remotes, read Section 0, preserve 695 preexisting files.
- [x] Run pre-change suite: 86 passed. Start science diagnostic/inference checks.
- [x] Write protocol before new pilot observations; retain all original null results.
- [x] Add `integrations/jev_client.py` and `tests/test_jev_client.py`: validate typed
  responses, pin models, retain usage, bound calls/bytes/reservations, sanitize
  failures, journal request before dispatch. Test invalid actions/probabilities,
  model drift, timeout, missing usage, TLS and exhausted limits before implementation.
- [x] Add `integrations/jev_research.py` and `tests/test_jev_research.py`: public
  serializers, legal candidate forecasts, exact conditional Gaussian reference,
  public heuristic, full-policy logits, controllers, journal and episode loop.
  Tests must observe a frozen forecast event before `env.step`, reject masked
  choices, retain immediate commitment and failure, and demonstrate hybrid inputs.
- [x] Add `scripts/jev_rl_study.py`: development-only manifest, adapters for all
  five checkpoints/seeds and fixed/VOI/imitation baselines, fixture/hosted modes,
  summary with coverage, raw results, source/checkpoint hashes, no overwrite.
- [x] Add credential-free replay that regenerates tasks, checks requests/menus and
  public observations, recomputes forecast outcomes and local final score, and
  rejects changed hash links or dishonest summaries. Exercise tampering regressions.
- [x] Run the two-mechanism/all-five-seed fixture pilot and replay it; verify current
  PPO against its original collector on the same development tasks. Run only a
  short hosted pilot if the current process provides a fresh credential.
- [x] Map each spec section to implementation/evidence/gap, audit local/public
  differences, record exact v6 numbers, write reproducibility and submission blockers.
- [x] Independent spec review, then code-quality review; resolve actionable findings.
- [x] Run final tests, science checks and preservation audit; deliver evidence and
  commands. Do not claim authenticated Jev performance from fixtures.

Work occurs in the primary checkout on `vajraang/jev-rl-m1`. It has an unborn Git
history, so there is no source commit from which to make a normal worktree. The
separate publication checkout stays at `ae6cdc8`; before/after content hashes
provide preservation evidence for local work. No original evidence is overwritten.

Verified closeout: local M1 accepted; hosted validation remains pending because no process key was supplied. Final evidence: reports/jev-rl-m1/REPORT.md. No final evaluation or public release occurred.
