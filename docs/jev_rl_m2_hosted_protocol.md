# M2 hosted roles and execution boundary

This protocol defines the hosted evaluation roles, failure handling, and dispatch
limits. [Registered analysis](jev_rl_m2_analysis_registration.json)

## Fixed roles

- `incumbent_m1`: unchanged M1 full Jev/PPO arbitration and prompt.
- `arbitration`, `jev_only`: full lawful-menu v2 decisions, with or without PPO
  probabilities. Forecasts precede the decision. V2 adds the declared public
  generator description and explicit action descriptions in forecast questions.
- `hidden`, `chance`, `public`: v2 arbitration with the same forecast-call stage;
  decision inputs respectively hide forecasts, replace them with0.5, or replace
  them with the previously specified public repeat-assay heuristic.
- `forecast_only`: PPO selects its action; Jev forecasts only that selected assay,
  before observation, and cannot modify the decision.
- `rerank`: Jev chooses among the fixed three-proposal PPO menu, then forecasts
  only the selected assay. The prediction does not feed back into that choice.
- `rerank_shadow`: identical reranking call stage, but its valid choice is ignored
  and PPO's argmax is executed. This is the extra-compute control for using the
  rerank reply. Per-decision stages match; trajectories and total calls can differ.
- `probe`: the preserved fixed schedule supplies common measurement events for
  prospective prediction scoring. It is independent of forecast answers.

The top-two-plus-best-commitment proposal rule and actual canonical JSON wire
order are both recorded. The ranking array and wire order are explicitly distinct.
All roles use the same simulator action mask, physical budget and evaluator.
No action confidence is reinterpreted as a measurement-outcome probability.

## Failure semantics

Decision-call failures and required predecision-forecast failures end arbitration
or reranking episodes as failed. There is no retry or undocumented fallback.
Shadow reranking retains the same decision-call failure behavior as reranking.

Forecast-only, fixed-probe, and post-reranking predictions have no authority to
veto an already selected action. A forecast API failure is recorded, the fixed
action proceeds, and no Jev prediction is invented for that event. References may
still be scored. Scientific episode completion and forecast completeness are
separate fields. Unknown usage remains unknown; all attempted cost is retained.

Primary forecast comparisons use common probe events with both valid predictions,
and report missingness/coverage and failure counts alongside that conditional
score. Report best/worst Brier bounds for missing forecasts if making a whole-probe
comparison; do not call a lower conditional score an unconditional improvement.

## Development cases and limits

Candidate development mechanisms use generator991331 and episode seeds991332000+i.
These are separate from PPO training7401, tuning991311, confirmation991321 and
future final991401. Before execution, freeze source, prompts, task hashes,
checkpoint records, model, roles/order and limits in the manifest. No final data
is constructed by this runner.

One mechanism with one PPO seed and all roles needs at most79 additional hosted
attempts:9 for the unchanged incumbent,70 across v2 roles including probe and
shadow reranking. A full five-PPO-seed cohort reuses the seed-independent
Jev-only/probe cases; it needs at most347 calls per mechanism. Eight mechanisms
therefore require at most2776 attempts. These are upper bounds, not expected cost.

The engineering plan caps attempts at 80 and estimated input cost at $0.10.
The eight-mechanism development study caps attempts at 2,800 and estimated input
cost at $1. The larger study starts with the first complete paired case. Source,
prompt, or model drift and invalid receipts stop further dispatch. The results
retain all attempted and skipped cases. The eight-mechanism design limits
generalization and calibration inference. [Registered plans](jev_rl_m2_analysis_registration.json) · [Hosted results](m2-research-note.md)
