# M2: bounded M3 Max PPO study and Jev capability implementation

2026-11-14. This is a development report. New M2 hosted results are pending.

CPU was the fastest measured backend for this complete PPO workload. The bounded
42-run search selected a scratch-trained GRU512 with batch 512, learning rate 0.0003
and two update epochs. Five-seed confirmation did **not** establish a return gain
over v6, and strong imitation and the Bayesian VOI planner remained cheaper with
higher mean return. The Jev client, three distinct Jev roles, matched controls,
failure accounting and credential-free replay are implemented. Their larger hosted
comparison has a frozen analysis contract but requires the proposed allowance and
a fresh process credential. The new final split remains unopened.

## What was preserved and frozen

The original simulator, observation masks, rewards, learning modules, fixed designs,
five-seed registries, checkpoints and locked v6 evidence were preserved. Historical
failures remain in the original reports. The M1 report is a historical pre-hosted
closeout; the subsequent authenticated pilot below supersedes its pending-pilot
status without rewriting that report.

The [task-derived protocol](../../docs/jev_rl_m2_protocol.md) and
[source-to-design rationale](../../docs/jev_rl_m2_sources_and_design.md) were published
before registered training in commit
[`9043afc`](https://github.com/Vajraaaang/experimental-scientist-rl/commit/9043afc0721f5613c0c13c607e7189751cc08e81).
The selected 35-checkpoint confirmation freeze was published in commit
[`cdda216`](https://github.com/Vajraaaang/experimental-scientist-rl/commit/cdda216)
before generating confirmation mechanisms. Selection used tuning generator 991311;
confirmation used 991321. Final generator 991401 was not opened. No PPO tuning was
performed after confirmation results were observed.

## Hardware and finite compute

The profiler measures environment reset/rollout, tensor packing and transfers,
advantages, PPO updates, validation and checkpoint round trips. Native MPS and
CPU-rollout/MPS-update were actually exercised, with no implicit device fallback.
The split implementation keeps learner/Adam state on MPS and synchronizes a CPU
policy mirror. Action masks, forced-policy positions, old log probabilities,
optimizer ownership and cached/reference simulator agreement are checked.

| Backend, GRU64/batch 1024 | Training transitions/cycle-second | Training transitions/whole-run-second | Sustained run |
|---|---:|---:|---:|
| CPU | 8,603.65 | 8,522.06 | 60.09 s |
| MPS | 6,137.59 | 6,080.59 | 60.54 s |
| CPU rollout / MPS update | 7,410.92 | 7,342.97 | 60.14 s |

Whole-run timing includes setup and finalization; cycle timing does not. CPU also
won the short history/GRU capacity probes through 512 units. Four single-threaded
CPU seed workers were selected for the race. Their profiling group stopped at its
transition cap after approximately 28 seconds, so it is not a full 45-second sustained
concurrency result. Summed per-process rates are not group wall-clock throughput.

| Work | Declared cap | Observed |
|---|---|---|
| Profiling, including resource-only selected-model probe | 3,000,000 simulator steps; bounded profiling windows | 2,998,784 steps |
| Registered race | 60,000,000 training transitions; 3,600 s active orchestration; 2,000,000 validation transitions | 49,652,308 training; 166,142 validation; 2,963.51 s (49.4 min) |
| Fresh development confirmation | 128 mechanisms, all five seeds; 600 s | 5,632 episodes; 16,814 step calls; 38.47 s including finalization; zero errors |

Engineering fixtures, unit tests and replay are separate from training counts.
Inherited v6/imitation pretraining is recorded separately and is not treated as
new training or equal-compute evidence.

CPU throughput in the first and second halves of the sustained run was about 8617
and 8590 transitions/cycle-second (−0.31%). No OS thermal/performance warning was
reported. Later 15-second resource samples covered only the last approximately 126
seconds of the race: thermal state stayed nominal, and sampled aggregate owned
training-process RSS peaked at 2,343,567,360 bytes (2.18 GiB). The machine has 36 GiB
unified memory; its nonfree pool includes unrelated apps and caches.

A separate post-selection resource probe measured 625,623,040 bytes peak process
RSS (0.583 GiB) with `/usr/bin/time -l`. That short probe never promoted a checkpoint.
These are scoped resource measurements, **not** an exact full-race unified-memory
peak or temperature measurement. Sampling can miss shorter peaks.

## Capacity, recipes and budget-dependent rankings

History and recurrent policies at 64/128/256/512 units were screened on seeds 20/21
with approximately 1 M public-selector and 1 M joint interactions. Four batch/update
recipes then continued the selected architecture. The selected scratch and v6
continuations retained all five seeds 20–24; imitation→PPO continuation stayed a
separate cohort. All 42 jobs completed correctly with valid primary snapshots.

The [generated tables](summary-01/COMPARISONS.md) show every two-seed architecture,
recipe and v6 screen at equal interaction and equal wall budgets, including phase,
actual transitions and elapsed time. At approximately 2 M interactions, GRU512 had
the best two-seed endpoint mean return (−0.334584). At 240 seconds, GRU128 led
(−0.345770); GRU512 was still in its selector curriculum and its unforced policy
stopped immediately (−0.640000). Thus 512 is not best at every wall budget.

Recipe wall comparisons are 70-second **continuations** from common trained parents
for each seed. They exclude inherited pretraining time. The frozen selection rule
can choose an eligible intermediate checkpoint; endpoint tables do not silently
substitute that selected checkpoint. The race is a finite screen, not an exhaustive
search over PPO architectures or all possible training recipes.

## Five-seed confirmation and null result

All learned methods used the same 128 confirmation mechanisms and five retained
seeds. Seed-independent planners were evaluated once per mechanism; tiling for
pairing did not invent replicates. The separate privileged reference knows the
mechanism and has no assay cost, so it is not an attainable comparator.

| Method | Mean target MSE ↓ | Variable assay cost ↓ | Original return ↑ |
|---|---:|---:|---:|
| Selected scratch PPO, GRU512 | 0.00219028 | 4.86156 | −0.298268 |
| Original v6 PPO, GRU64 | 0.00226115 | 4.85125 | −0.304565 |
| Continued v6 PPO | 0.00226115 | 4.78188 | −0.303871 |
| Strong imitation, GRU64 | 0.00220615 | 4.54719 | −0.296775 |
| Calibrated imitation | 0.00220615 | 4.54625 | −0.296765 |
| Imitation→PPO | 0.00220615 | 4.56109 | −0.296914 |
| Continued imitation→PPO | 0.00220615 | 4.56234 | −0.296926 |
| Bayesian VOI | 0.00219028 | 4.56406 | −0.295293 |
| Strong fixed design | 0.00276822 | 6.40000 | −0.367697 |

Selected PPO minus v6 return was+0.00629636, with paired 95% interval
[−0.00169871,+0.02061855] and multiplicity-adjusted 98.333% interval
[−0.00256719,+0.02444921]. Continued-v6 return gain was+0.00069375,95%
[−0.00017813,+0.00222656]. Continued-imitation→PPO return difference was−0.00001250,
95%[−0.00004219,+0.00000156]. **All three prespecified gain screens failed.**

Intervals use 10,000 paired crossed seed/mechanism bootstrap draws, RNG 991499.
GRU512 matched the privileged reference's MSE on this slice, while paying more
assay cost than VOI and imitation. Capacity and training history differ between
GRU512 PPO and GRU64 imitation; this does not isolate the value of PPO learning.
Familiar motifs and parameter ranges remain a generalization limitation. Do not
compare these absolute errors to the historical 4,096-mechanism v6 test as a gain:
the mechanism samples differ. The frozen scoring configuration is the same.

![Hardware, budget and confirmation comparison](summary-01/m2-comparison.png)

In the confirmation panel, thick intervals are 95% and thin intervals are 98.333%.
Screen means have only two training seeds and no inferential superiority claim.

## Jev capabilities and actual hosted evidence

The [capability audit](../../docs/jev_capabilities_audit.md) maps current official
documentation to implemented behavior. Calls retain provider request IDs, local
trace/span IDs, UTC/monotonic timings, canonical request/response hashes, HTTP
status, bounded retry hints, complete typed outputs, token usage and dated cost
estimates. Forecast/action/observation events join to the originating call span.
Model drift, invalid probabilities, oversized requests, unsafe usage values and
budget exhaustion fail closed. Unknown usage retains a reservation; failures and
missing forecasts remain visible. Automatic retries are disabled.

No native Jev telemetry backend or W 3 C propagation contract was found in reviewed
official docs/source. Local journals do not expose provider internals or supply an
independent timestamp attestation. No extra telemetry service receives packets.
Choice confidence is preserved separately from Noul outcome probabilities.

The authenticated M1 pilot made 11 successful HTTP 200 calls to pinned`jev-1.14.0`,
with 11 unique provider request IDs,77,860 input/11,811 output tokens, estimated
input cost$0.00327012, and 2.6853 seconds transport time. Its 15 episodes replayed.
On six fixed-probe events, Jev Brier was 0.24938 versus posterior-reference 0.18235
and chance 0.25. Jev-only and hybrid both had MSE 0.0064, with costs 2.7 and 2.9.
This tiny one-mechanism/one-seed pilot establishes neither calibration nor benefit.

The implemented M2 roles are full legal-action arbitration, prediction-only
selected-assay forecasts, and restricted PPO-proposal reranking. The preserved M1
incumbent uses its original v6 policy. Matched hidden/chance/public forecasts,
rerank-shadow, pure PPO, v6, strong imitation, fixed and Bayesian baselines remain.
Forecast-only and post-reranking prediction failures do not veto already selected
actions. Decision-stage failures remain failures without covert fallback.

The 75-episode M2 fixture used all five selected PPO seeds, made zero hosted calls,
and passed independent replay. Its analysis is explicitly fixture-only and cannot
enable efficacy or calibration claims. Learned-method intervals require all five
registered seeds; an 8-mechanism/one-seed plan remains descriptive. Fixed-probe
inference clusters six events per mechanism and retains missingness bounds.

Primary-source research examined TypeSafe documentation/SDK code, an independent
academic Jev benchmark and released code, and community application code. Targeted
official-domain searches found no verified public Jev deployment at OpenAI,
Anthropic, DeepMind or Google Research. That is a bounded search result, not proof
of non-use. Other typed models and developer affiliations are not Jev adoption.

## Reproduction and remaining work

The export preserves numerical values and API payloads while normalizing known
machine paths and removing the private handoff link. Original-to-public hashes,
relinked journals/seals and independent replay results are recorded in
`publication-export.json`. This public repository omits binary weights and private
credentials. Replay validates recorded actions/advice and outcomes; without weights
it cannot independently establish the advice's neural origin or reproduce exact
training. Frozen execution snapshots preserve original runtime source bytes.

From a configured clone, these commands replay existing data without hosted calls:

```bash
PROJECT="$PWD"
"$PROJECT/.venv/bin/python" "$PROJECT/reports/jev-rl-m1/hosted-20261113-204144/source_snapshot/scripts/jev_rl_study.py" verify --run "$PROJECT/reports/jev-rl-m1/hosted-20261113-204144"
"$PROJECT/.venv/bin/python" "$PROJECT/reports/jev-rl-m2/candidate-fixture-02/source_snapshot/scripts/jev_m2_study.py" verify --run "$PROJECT/reports/jev-rl-m2/candidate-fixture-02"
"$PROJECT/.venv/bin/python" "$PROJECT/reports/jev-rl-m2/confirmation-01/source_snapshot/scripts/ppo_m2_confirm.py" replay --output "$PROJECT/reports/jev-rl-m2/confirmation-01"
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/summarize_ppo_m2.py" --root "$PROJECT" --output "$PROJECT/runs/m2-report-reproduction"
```

The [prospective analysis registration](../../docs/jev_rl_m2_analysis_registration.json)
binds the analyzer, contract and concrete hosted-plan hashes before new M2 dispatch.
The full proposal has 8 mechanisms, all five seeds,600 cases, at most 2,776 required
attempts within a 2,800-call/$1 estimated input-cost cap and a 1,200-second dispatch
window. The smaller engineering proposal has 23 cases and at most 79 attempts within
80 calls/$0.10. Neither prepared proposal increases the existing 32-call/$0.10
authorization, of which 11 calls were used. A running socket operation can outlast
the dispatch window; its latency remains recorded. Byte/token/cost limits may end
the study before the call limit. Missing outcomes block comparisons.

Fresh verification passed 473 tests in the development checkout and 477 tests in
the public checkout. Public snapshot replay passed all 100 M1 fixture, 15 M1 hosted,
75 M2 fixture and 5,632 confirmation episodes. The copied analyzer reproduced its
fixture summary and report exactly. Final preservation checks found all 695 original
files and all 1,268 M1 evidence files unchanged. Public CI status is reported separately
from these local results.

New hosted execution, prospective analysis and its GitHub results update remain
pending. The final split stays closed because the local gain screens failed and
no new M2 hosted benefit has been demonstrated.
