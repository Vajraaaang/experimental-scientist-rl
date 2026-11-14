# Prospective Jev and RL research protocol, milestone 1

Protocol date: 2026-11-13 (America/Los_Angeles). Status: development engineering
milestone; no new final evaluation authorized or opened. This extends the bounded
Section 0 pilot. It does not promote the flagship catalogue or claim RL necessity.

## Architecture and estimands

The trusted, unchanged `ScientistEnv` owns the mechanism, random streams, legal
mask, charging, and independent final scoring. Controllers receive only public
observations and the common action catalogue. No task seed, mechanism, private
scores, unpurchased measurements, or forecast-reference answers enter a Jev
request. Forecasts and action receipts are flushed to a hash-linked event journal
before `step`. The journal provides local ordering and tamper detection, not an
external timestamp authority or a sandbox against malicious Python introspection.

Jev forecasts, for every currently affordable experiment and its measured
variables, P(next measured sample mean > this episode's observed baseline mean).
The threshold is frozen, including baseline measurement noise. These binary
probabilities do not estimate effect magnitude, causal identification, or action
confidence. All forecasts precede observations. Only purchased outcomes are
revealed; unchosen forecasts remain unscored. A separate fixed-schedule forecast
probe provides common events for primary forecast comparisons.

The current complete v6 PPO policy supplies probabilities for every legal action,
including all nine commitments. The hybrid gives these probabilities and the
prospective forecast table to Jev's decision call. Jev chooses from the same full
legal menu. Thus the hybrid is Jev decision arbitration informed by an existing
RL policy, not new PPO training or a forecast-augmented network. Directly training
a new forecast-conditioned policy would confound new training and hosted data
cost at this stage; a forecast entropy bonus would impose an unvalidated link
between uncertainty and decision value. Both are deferred.

## Conditions fixed before the pilot

| Condition | Decision rule | Forecast information shown to decision rule |
|---|---|---|
| prior | common posterior commitment | none |
| fixed / fixed_quality | preserved exhaustive schedules, common posterior final estimator | none |
| voi | preserved Bayesian decision-value planner, 12 draws | internal public-history belief |
| ppo | full v6 selected policy and recorded public-price calibration | none |
| imitation / calibrated_imitation / imitation_ppo | preserved full learned policies | none |
| jev_only | Jev full legal-menu choice | Jev forecasts |
| jev_rl | Jev full legal-menu choice plus v6 PPO probabilities | Jev forecasts |
| jev_rl_hidden | same two call stages as hybrid, forecast results withheld from decision | none |
| jev_rl_chance | same two call stages, Jev forecast call retained for compute control | all 0.5 |
| jev_rl_public | same two call stages, Jev forecast call retained for compute control | public repeat-measurement heuristic |
| forecast_probe | preserved fixed schedule and posterior commitment | choice is independent of forecasts |

For the public heuristic, predict the next mean using the last purchased result
for the identical assay, with Gaussian uncertainty from two independent sample
means; before such evidence use 0.5. This is deliberately limited. Strong forecast
references are the existing 128-hypothesis quadrature posterior and its prior,
plus 0.5. Reference forecasts are computed on exactly the same public histories
and scored on the same events, but never supplied to Jev. They are exact Gaussian
conditionals within each hypothesis at offset SD zero, not exact continuous-truth
posteriors. Baseline uncertainty is conditioned on, not added a second time.

The common physical budget is 8, with at most four assays, eight cultures per
assay, readout SD 0.02, offset SD 0, 128 independent final cultures, and original
reward weights. Setup cost 3 is reported separately and in total expenditure.
Stopping immediately is always legal. The standalone demo's forced-first-assay
mode is preserved but is not used in these policy comparisons.

## Splits, selection, and hypotheses

M1 uses new development mechanisms from `continuous_family(991301, n)` and episode
seeds `991302000 + i`. Local fixture coverage uses two mechanisms and all retained
training seeds 20–24. A hosted connectivity pilot, when a fresh process credential
is supplied, uses only the first development mechanism and training seed 20.
This is explicitly development data, never an independent efficacy result.

For subsequent development, reserve generator seed 991311 for tuning (128
mechanisms) and 991321 for confirmation (128). Any prompt or rule revisions must
be versioned and selected on these slices, retaining all failures. Selection is
maximum mean original return subject to MSE increase <= 0.0001 relative to PPO;
ties within 0.001 return favor fewer hosted calls, then lexicographic prompt ID.
Do not select on M1 fixture quality, historical final outcomes, or future test.
Before expansion, pilot failure rates, resolved model availability, forecast
coverage, and actual usage must be reviewed and an explicit total API cap set.

Reserve seed 991401 for 512 future final mechanisms and episode seeds 991402000+i;
do not generate or score them in M1. Before final use, freeze methods, prompts,
prices, menu, source, dependencies, and checkpoints and assert mechanism-hash
disjointness against development and every historical split manifest. Keep all
five retained RL seeds. Hosted stochastic repeat handling must be frozen after
the pilot; five RL seeds are not five independent Jev training seeds.

Future confirmatory hypotheses, all allowed to fail:

1. Jev improves mean Brier score over the quadrature posterior on the fixed probe.
2. Jev+RL improves original return over the unchanged full PPO policy while its
   MSE increase upper confidence bound is <= 0.0001.
3. Jev+RL improves return over the same-call hidden-forecast condition.
4. Jev+RL improves return over Jev-only, supporting value of the RL information.

Use paired mechanism bootstrap for the fixed probe and a paired two-way bootstrap
over mechanisms and retained training seeds for learned conditions. Resample
mechanisms with all their events together, never individual correlated forecasts.
Use 10,000 replicates, RNG seed 991499, simultaneous Bonferroni 98.75% intervals
for the four primary hypotheses; report ordinary 95% descriptive intervals too.
Do not claim calibration from a small pilot: report ten predeclared equal-width
reliability bins, counts, mean probability, event rate, Brier score, and coverage.
For continuous event ties, use strictly greater-than. No post-hoc bin selection.

Record MSE, RMSE, variable and total assay cost, effort, original return, assay
count, immediate-stop rate, decision/end-to-end latency, hosted attempted/successful
calls, input/output tokens, estimated bill, and every failure. Missing final scores
remain missing with failure counts; never average them silently as successes.
Comparisons across adaptive event sets are descriptive; primary forecast inference
uses the common fixed probe. Budget-exhausted, malformed, timed-out, and model-drift
episodes are retained. There is no automatic retry or unreported fallback.

## Hosted boundary and reproducibility

Official API contract inspected at `https://api.typesafe.ai/openapi.json` and
`https://api.typesafe.ai/redoc`: `POST /v1/systemone`, typed `noul` and `choice`
answers, response model, and input/output usage. Pricing inspected at
`https://typesafe.ai/`: $42 per billion input tokens; OpenAPI says output tokens
are currently free. Freeze this dated estimate; an invoice is authoritative.

Default hosted pilot: pinned requested model `jev-1.14.0`, at most 32 calls,
64 KiB serialized request limit, 1 MiB cumulative serialized requests, 20-second
timeout per call, and 256 KiB response limit. Reserve request bytes plus 4096
tokens per call before dispatch against a 1,200,000 input-token allowance and
$0.10 estimated-input-cost ceiling. These are conservative client accounting
limits, not a guarantee about undocumented server tokenization/billing. Record
actual reported usage; a reservation overrun terminates further calls. Model drift
terminates the episode and preserves the receipt. No API credential is present in
the initial process; hosted validation is therefore pending, not a fixture result.

Credentials come only from `TYPESAFE_API_KEY` in the process or an interactive
hidden prompt. Never read credentials from files, browser state, prior chats, or
Git. HTTP errors and exceptions are reduced to fixed codes; no raw error body,
headers, or credential enters artifacts. Verified TLS and the macOS CA fallback
are retained. Fixture responses are conspicuously labeled and are not Jev data.

The output directory refuses overwrite. The manifest is written before execution,
records reproducible task seeds only for the trusted evaluator, source/checkpoint
hashes, protocol hash, conditions, and caps. Verification replays recorded actions
through uncached CPU simulator dynamics and checks public observations, legal
menus, prediction ordering, scores, results, and event hashes without credentials.
Replay verifies a receipt; it cannot independently reproduce hosted model weights.
Pretrained weights remain absent from the public checkout; public users can verify
recorded episodes without them, but new learned-policy runs require training.

## Gates and submission blockers

M1 acceptance: existing tests/science checks pass; all comparator adapters execute;
prospective forecasts and same-event references score correctly; failures and
drift are recorded; credential-free replay detects tampering; old evidence hashes
remain unchanged. Authenticated hosted-pilot acceptance is separately pending
until it actually runs. Do not expand from fixtures to final scoring.

Remaining submission blockers include authenticated forecasting/decision results,
adequately powered independent Jev comparisons, uncertainty analysis on real
hosted data, version/repeat stability, clean public-release reproduction, training
cost completeness, and the requirement gaps in `jev_rl_requirements.md`. Familiar
motif and parameter holdouts establish neither structural extrapolation nor
biology. Revised-task culture variation, topology/range shifts, full permutation
tests, and broader specification ablations remain separate gated work.
