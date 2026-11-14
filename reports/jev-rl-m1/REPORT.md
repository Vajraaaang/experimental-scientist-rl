# Jev/RL milestone 1: measured engineering report

Date: 2026-11-13 (America/Los_Angeles).

**Local engineering milestone passed. Authenticated hosted-pilot acceptance is pending.**
The current system includes prospective Jev forecasts, a frozen full-policy PPO
comparator, Jev-only and Jev+RL controllers, matched forecast-information controls,
strong recorded baselines, traceable calls, failure accounting and independent
credential-free replay. This is not a publication-ready scientific result or a
Jev effectiveness comparison.

## Verified outcome

- Pre-change primary suite: 86 passed. Final primary suite: **241 passed** in 5.73s.
- Untouched public checkout: **90 passed** in 6.50s using the available native environment.
- Final fixture study: **100 episodes, 18 conditions, two new development mechanisms,
  all five retained learned-policy seeds**. Every episode completed and replayed.
- **40/40 full learned-policy evaluations** matched the original collector's
  action sequence and final score. Native learned terminal commitments are retained.
- **236 fixture calls** have one study trace ID and 236 distinct call spans.
  No authenticated hosted calls occurred; simulated token counts/cost are zero.
- Fixed forecast probe: 432 candidate events requested/validated; 12 purchased
  measurement events and 12 scored events, matching the predeclared probe count.
- Five additional fault examples replay: immediate commitment; timeout after one
  purchased assay; changed model; oversized usage; rate limit. These are injected
  fixtures, not measured provider failures. The timeout retains its 1.8 assay cost
  and two scored measurements while final MSE remains missing.
- **695/695 preexisting source/config/report files are byte-identical**, including
  the standalone Jev demo and all checked historical artifacts/checkpoints.

The diagnostic and inference suites were run before implementation. The supplied
proxy diagnostic is reproduced; dt=.05 versus DOP853 maximum discrepancy is
0.00039487073007493656, improving to 0.00019559277551289833 at dt=.025. Mean-packet
95% coverage was .96875 in each 256-case check at offset SD 0 and .03; the latter
uses the declared approximate likelihood. Twelve versus 64 planner draws selected
the same first action in all 16 checked cases. These are implementation checks,
not broad posterior or biological qualification.

## What the new traces contain

Official TypeSafe documentation was checked online. The implementation preserves
provider `x-typesafe-request-id` when supplied, safe HTTP status/retry metadata,
resolved model, native full Choice distributions/confidence and token usage.
Local trace/span IDs, UTC times, request/response hashes and exact wire question/
candidate order connect forecasts, choices and later scores. `trace-index.json`
is a compact cross-episode index. The provider ID is distinct from our local span;
no provider-side trace backend or native OpenTelemetry integration is claimed.

Typed validation rejects masked/invalid actions, nonfinite probabilities,
inconsistent choices, malformed usage and unexpected model versions. Caps apply
before dispatch. Unknown token usage retains a reservation; failed calls are not
silently free. Automatic retries are disabled and replay rejects retries hidden
inside an episode. Raw headers and HTTP error bodies are excluded from artifacts.

Read [the online capability audit](../../docs/jev_capabilities_audit.md) for sources,
selected capabilities and deliberate boundaries. The existing standalone demo and
its forced-first-assay diagnostic remain unchanged; this comparison always permits
an immediate lawful commitment.

## Fixture measurements, not Jev results

The table below tests metrics and comparator execution. Only two development
mechanisms were used. The fixture emits 0.5 forecasts and follows a deterministic
schedule or the supplied PPO argmax. Thus matching hybrid/PPO results are expected
by construction and provide no evidence that Jev forecasts or arbitration help.
No significance intervals or calibration claims are justified by these fixtures.
Setup cost 3 must be added to variable cost; raw records also include total costs,
all-attempt denominators, instrumented timings, Brier scores and reliability bins.

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

## Historical claims remain unchanged

The fourth locked v6 comparison used 4,096 mechanisms and all five retained seeds.
Updated PPO reduced spending by 3.78% versus prior efficient PPO, with favorable
paired intervals for MSE, cost and return. It still spent 0.158574 additional cost
units versus strong imitation (95% CI [0.105014, 0.210075]); the full declared
parity screen failed. The exhaustive fixed-design success, Bayesian planner,
strong imitation and imitation-then-PPO evidence remain intact. No comparative
Jev efficacy has been established.

## Evidence index

| Artifact | Purpose |
|---|---|
| [Final manifest](fixture-04/manifest.json) | Frozen configuration, conditions, source/model hashes, development splits, caps and trace ID |
| [Raw results](fixture-04/results.json) | Per-episode outcomes, full call receipts, all metrics, coverage and summaries |
| [Call trace index](fixture-04/trace-index.json) | Episode/method/request/provider-ID/span/version/usage/latency correspondence |
| [Replay receipt](fixture-04/verification.json) | 100 independently regenerated episodes; no credentials or checkpoint loading |
| [Final tests](tests-final.log) | 241 passed |
| [Diagnostic checks](baseline-checks/diagnostics.json) | Numerical fixture and branching controls |
| [Inference checks](baseline-checks/inference-checks.json) | Predictive mean coverage and planner-draw check |
| [Fault examples](failure-examples/summary.json) | Five retained positive/negative fixture traces; adjacent JSONL journals and driver |
| [Preservation proof](preservation-after.json) | All 695 preexisting files unchanged |
| [Checkout audit](checkout-audit.json) | Exact local/public differences and original Jev demo hash |
| [Hosted status](hosted-pilot-status.json) | No process key; bounded live command rejected before any network call |
| [Protocol](../../docs/jev_rl_protocol.md) | Hypotheses, splits, selection, uncertainty, cost and stopping rules |
| [Requirement map](../../docs/jev_rl_requirements.md) | Every specification section mapped to code, evidence or a documented gap |

Earlier fixture-01/02/03 directories remain available with their own source
snapshots; they were engineering iterations before review fixes. Fixture-04 is
the final acceptance run. None was used to tune scientific method quality.

## Reproduce and inspect

Run from the primary `experimental-scientist` checkout. Every destination resolves
to an absolute path. Choose a new output directory for new execution; existing
study directories refuse overwrite.

```bash
PROJECT="$(pwd)"
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/jev_rl_study.py" verify \
  --run "$PROJECT/reports/jev-rl-m1/fixture-04"
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/jev_rl_study.py" run \
  --output "$PROJECT/runs/jev-rl-fresh-fixture"
```

The preserved source snapshot can verify this run after subsequent development:

```bash
PROJECT="$(pwd)"
"$PROJECT/.venv/bin/python" \
  "$PROJECT/reports/jev-rl-m1/fixture-04/source_snapshot/scripts/jev_rl_study.py" \
  verify --run "$PROJECT/reports/jev-rl-m1/fixture-04"
```

Verification does not load pretrained policy weights or contact TypeSafe. It
regenerates uncached simulator observations and final outcomes and validates the
recorded call/order/forecast/score evidence. It cannot regenerate proprietary
hosted weights or independently prove a remotely reported timestamp/request ID.
The hash chain is local tamper detection, not external timestamp attestation.

A short hosted pilot is ready through the same runner:

```bash
PROJECT="$(pwd)"
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/jev_rl_study.py" run --live \
  --output "$PROJECT/runs/jev-rl-hosted-pilot"
```

Use `TYPESAFE_API_KEY` in the process environment or the interactive hidden prompt.
Never paste a key into chat or a report. The live route is fixed to one development
mechanism/seed20 and at most 32 attempts, with byte/token/estimated-cost checks.
No future final split is generated or scored by this CLI.

## Review and reproduction limits

Independent specification and code reviews found and helped resolve missing
hybrid-advice verification, omitted failure-resource denominators, duplicate-call
and retry acceptance in replay, unchecked receipt/global accounting, oversized
usage integers and zero-attempt accounting. Regression tests cover the reproduced
cases. Reviewers reported no remaining local-M1 code blockers after the fixes.
This is engineering review, not external scientific peer review.

No new training was performed. Retained v6 checkpoint costs remain historical and
additional to the Jev pilot. Instrumented loop timing includes reference forecasts,
I/O synchronization and transport overhead; reset-inclusive timing is separate.
These timings should not be compared directly with the historical neural-only
inference benchmark. Independent-machine/clean-environment reproduction, full
retraining and authenticated hosted behavior have not been verified here.

## Publication and scientific blockers

1. Supply a fresh process credential and complete the capped hosted pilot; inspect
   actual model resolution, traces, coverage, token cost, failures and immediate stops.
2. Freeze hosted repeat/version handling after the pilot, develop only on declared
   tuning/confirmation splits, then freeze methods before any new final comparison.
3. Run adequately powered, matched Jev-only/RL/hybrid/ablation comparisons and the
   common-event forecast probe, including the predefined uncertainty analysis.
4. Resolve or explicitly limit the spec gaps: structural/parameter-range shifts,
   revised-task culture variation, complete permutation tests, varied baselines,
   missing ablations and omitted literature baselines. Familiar motifs are not
   structural generalization; none of this establishes biological usefulness.
5. Prepare a selective public release and validate it independently. The public
   branch is clean at ae6cdc8 and has portability/compressed-evidence changes not
   present locally. Local Jev files and README content must be carried forward;
   copying one checkout over the other would lose work. The local source snapshot
   preserves the original specification, including its private handoff pointer;
   public release redaction remains necessary. No release, push or paper submission
   was performed.

Primary checkout remains an unborn Git repository on `vajraang/jev-rl-m1`; no
initial commit containing preexisting untracked work was created. Content hashes
and source snapshots provide the local evidence chain. Public checkpoint binaries
remain excluded, so new learned-policy evaluation from the public repository still
requires reproducing training; receipt verification does not.
