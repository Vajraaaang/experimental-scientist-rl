# M2 requirements-led candidate derivation

Date: 2026-11-13. This note derives candidates from the task and Section 0 before
mapping them to reusable incumbent code. The incumbent is a comparator, not the
design objective. Frozen checkpoints and every completed study stay immutable.

## Task properties and consequences

The agent receives a short history of public noisy observations, pays for up to
four experiments, and chooses one of nine terminal interventions. The evaluator
owns hidden dynamics and independently evaluates the final latent target MSE.
Three different questions must remain separate: predictive skill, useful
experimental adaptation, and added value of RL training. There is no requirement
that a hosted model arbitrate every action.

1. The complete history has at most five observation packets. Both a recurrent
   network and an explicitly history-conditioned feedforward network can represent
   this information. Compare them empirically; do not presume a recurrent model
   or a larger model is better. Retain full action selection and terminal control.
2. Accurate forecasts need not improve experimental decisions. A forecast-only
   sidecar can isolate Jev's predictive skill while leaving the scientific policy
   untouched. Its cost is extra, and identical underlying policy outcomes are
   expected by construction.
3. Letting a general-purpose model choose among every action may discard a useful
   learned policy. A restricted reranker can retain the PPO proposal distribution
   while allowing a small, measurable correction. Its restricted menu is an
   explicit algorithmic choice, not a hidden change to simulator legality.
4. Exact lawful actions, accounting, stopping, numerical arithmetic and scoring
   belong in code. A confidence statistic is not evidence of biological or
   numerical correctness.
5. The Bayesian reference has a declared generator prior. Matching information
   requires documenting that prior and deciding prospectively how the hosted
   request receives the same public family description. It must never receive the
   actual episode mechanism, seed, unpurchased data, or reference answers.

## Candidate families to assess

| Candidate | Distinct scientific role | What would count as useful |
|---|---|---|
| Full PPO, history MLP and GRU capacity variants | Entire assay/stop/intervention policy | Better held-out original return, respecting target MSE and assay cost, at declared equal-interaction/equal-time budgets |
| Incumbent Jev arbitration | Chooses full legal menu with prospective forecasts and PPO advice | Paired improvement over full PPO and strong baselines after added call cost |
| PPO plus Jev selected-assay forecasts | PPO alone selects action; Jev forecasts only its impending bought assay | Better same-event Brier/calibration than strong forecast references; no automatic policy-quality credit |
| Jev reranking of legal PPO proposals | Chooses within a predeclared small PPO proposal menu | Paired benefit over PPO and a matched non-Jev reranking control |
| Strong imitation, fixed schedule, Bayesian decision value | Credible non-PPO winners | Allowed to outperform every new candidate; null results retained |

Candidate definitions, menu size, prompt/prior packet, selection grids and final
split rules will be frozen in the M2 protocol after primary-source inspection.
No final split is opened by this derivation or hardware profiling. The M1 hosted
pilot is development evidence only, never a selection-independent test.

## Hardware evidence before configuration selection

The initial profiling cap is 600 wall-clock seconds, with at most 3,000,000 local
simulator transitions, no hosted calls and no writes to existing studies. Measure
CPU rollout/update, MPS policy/update and CPU-rollout/MPS-update separately.
Time complete environment setup/reset, rollout, packing/transfer, advantages,
optimization, validation and checkpoint roundtrip. Include CPU batch/thread and
seed-concurrency controls. Profile a sustained repeated workload and record OS
thermal status when available; elapsed-time changes alone are not proof of thermal
throttling. Report process peak RSS and currently allocated MPS memory distinctly.

The finite training wall-clock and interaction budgets will be computed from
measured correct throughput, written before any configuration race, and applied
to all candidate arms. Equal wall-clock and equal-interaction comparisons are
separate estimands. Existing v6 pretraining is a disclosed inherited cost; do not
pretend warm-start continuation and scratch training have equal lifetime compute.
