# M2 frozen development protocol

Declared 2026-11-13, after primary-source inspection and hardware profiling and
before the registered configuration race. This protocol does not authorize a new
final-test opening or increase hosted-call limits.

## Evidence and resource decision

The recorded hardware study used 2,996,045 local simulator transitions. CPU was
faster than native MPS and CPU-rollout/MPS-update for every tested history/GRU
capacity through hidden size 512. At the tested GRU64 batch1024 workload, sustained
cycle rates were about 8604 CPU, 6138 MPS and 7411 split training transitions/s.
Four CPU seeds achieved about31k aggregate cycle transitions/s. These include the
declared cycle's validation/checkpoint work; raw stage times and startup costs are
separate. The OS reported no recorded thermal warning, but no temperature or
synchronized whole-machine unified-memory peak was measured. Process RSS peaks
and sampled MPS driver allocations are labeled by scope.

Use CPU, one intra-op thread per process, at most four simultaneous training
seeds. No MPS fallback, cloud compute or API calls in the training race.

**New-training cap: 60,000,000 physical training transitions and 3,600 active
orchestrator wall-clock seconds.** Validation has a separate2,000,000-transition
cap. Profiling's3M cap is separate and already exhausted. Each job also has a
finite wall/interaction ceiling. Complete-operation shutdown overhead is measured;
the orchestrator terminates its own remaining jobs at its deadline. All skipped,
failed and incomplete-budget runs remain in the ledger. Existing checkpoints and
raw locked results are never overwritten.

## Data and estimands

Training: the preserved 4,096-mechanism generator pool7401, disclosed as historical
training data. New tuning:128 mechanisms from991311, episode seeds991312000+i.
Confirmation:128 from991321, episode seeds991322000+i; construct only after model,
architecture, update recipes and checkpoint hashes are frozen. The future final
pool991401 remains unopened. Hash-check all new assessment mechanisms against
known training/development/final manifests before scoring.

All conditions retain original dynamics, prices, masks, budget8, maximum4 assays,
eight assay cultures, readout SD.02 and offset SD0. Original return remains
`-100*MSE - .01*variable_cost - .05*effort`. Native CPU final confirmation uses128
fresh final cultures and uncached dynamics. Unreachable targets remain included.

Primary training comparison: quality achieved within the same new physical
interaction budget. Secondary: a checkpoint at the first complete-operation
boundary after a common elapsed wall time, including setup/selector/evaluation/
checkpoint work. Overshoot is reported. Primary selection never uses extra work
after its interaction limit. Warm-start comparisons report inherited training
separately; they are not equal-lifetime-compute comparisons with scratch models.

## Finite configuration race

1. **Architecture screen:** history MLP and GRU, each hidden64/128/256/512; seeds20
   and21 for every architecture. One million selector interactions followed by
   one million joint interactions; batch256; joint Adam LR.0003, two epochs.
   Secondary wall snapshot240s; joint hard cap1.25M; per-job hard wall900s.
   Launch order balances ascending capacities for seed20 with descending
   capacities for seed21. This is a bounded grid, not a proof of a global optimum.
2. **Update/batch screen on the selected scratch architecture:** both screen seeds,
   starting from their hash-verified eligible primary checkpoint. Four recipes:
   `(batch, LR, epochs) = (128,.0003,2), (256,.0003,2), (512,.0003,2),
   (1024,.0001,4)`. Each has500k primary interactions,70s secondary snapshot,
   750k hard interaction cap and360s hard wall. Extra training starts from the
   inherited episode offset in namespace91303; controls share draws prospectively.
3. **Complete the scratch cohort:** run the selected base architecture and chosen
   continuation recipe for seeds22,23,24, retaining seeds20,21. All five must be
   reported, including failures. No seed is discarded because it performs badly.
4. **V6 continuation screen:** seeds20,21, registered selected v6 checkpoints and
   their existing public-price calibration. Recipes `(512,.0001,2)` and
   `(512,.0003,4)`;500k primary,70s secondary,750k hard cap,360s hard wall.
   Complete the selected recipe for22–24. Initial incumbent is eligible, so
   “additional training did not help” is an allowed result.
5. **Historical imitation-then-PPO continuation:** all five registered IM→PPO
   checkpoints, batch512, LR.0001, two epochs, entropy.001,500k primary/hard cap,
   no wall-snapshot extension,300s hard wall. This is further continuation of an
   already trained IM→PPO policy, not a fresh imitation-to-PPO experiment.

Maximum scheduled training reservations total58.75M, leaving margin under60M.
The exact executed count and each unused reservation are reported. A time cap can
leave later stages incomplete; that is not a scientific failure assigned a fake
score. Primary promotion requires correct, completed equal-interaction snapshots
and every required seed. A missing secondary wall snapshot excludes only that
secondary comparison; it must not censor a faster valid primary run.

Scratch selector uses public randomized assay schedules, Adam LR.001, four epochs,
entropy.05. Forced assay positions have zero actor/entropy loss but retain critic
training. Joint curriculum bias6 and initial-outcome bias3 decay to zero by30% of
the primary joint budget; primary selection requires >=50% and zero biases.
All gradients use unchanged scientific reward, gamma1, lambda1, raw uncentered
advantages, value coefficient.1, clip.15 and target KL.02. Adam is reset at the
joint boundary. Joint entropy goes from.03 to.001 by70%; continuation starts.01
and ends.001, except the explicit constant-.001 IM→PPO arm. Deployment has no
training bias or temperature. Local initial checkpoints only receive an incumbent
exemption if they carry validated eligible-parent provenance.

Architecture/recipe selection uses mean original greedy tuning return across both
screen seeds, then lower mean MSE, lower assay cost and lexicographic identifier
for exact ties. Per-seed checkpoint selection is the best eligible tuning return
within the primary interaction budget. Publish every screened configuration and
all its compute. Confirmation outcomes do not trigger new tuning.

## Jev designs and controls

Treat M1 arbitration as a frozen incumbent. Candidate v2 requests receive a
declared public generator description, never the realized mechanism. The change
in prior exposure is labeled and separated from the original prompt.

- Full standalone PPO, strongest imitation, exhaustive fixed/quality designs,
  prior and Bayesian decision-value planner retain their own decision rules.
- Jev-only and full Jev+PPO arbitration get the full lawful menu. Matched hidden,
  chance and public-forecast conditions retain the same hosted call stages.
- Forecast-only Jev cannot change a PPO action. It forecasts only the selected
  bought assay before observation. Identical underlying policy outcome is expected
  by construction; its separate estimands are Brier/calibration, cost and latency.
- Reranking uses exactly three legal proposals: top two PPO actions plus the best
  legal commitment, deduplicated and filled by PPO rank; ties use lower action ID.
  Jev cannot expand this menu. Compare the same proposals without Jev.
- The common fixed-probe schedule supplies the primary same-event forecast
  comparison. Adaptive event-set scores are descriptive and cannot substitute for
  that probe. Same public history/threshold is used for Jev, posterior/prior,
  chance and public-history heuristic forecasts.

The original authenticated pilot used11 calls,77,860 input tokens and no reported
failures. It was one mechanism/seed20, with six probe events: insufficient for
calibration or comparative efficacy. Its receipts and source snapshot are frozen.
The current process has no key. The existing maximum32-call/.10 estimated-cost
authorization remains; do not silently multiply it by looping new runs. If the
complete paired design exceeds available calls, prepare its exact bounded Terminal
command and request the extra allowance while continuing local work. No key is
read from files, prior chats, browser state or logs.

## Hypotheses, uncertainty and conclusions

Predeclared primary questions for subsequent sufficiently powered locked testing:

1. Does the selected scratch/continued PPO improve original return over frozen v6
   while its target-MSE increase upper bound is <=.0001?
2. Does a narrower Jev role improve end-to-end return or reduce hosted cost versus
   full arbitration while preserving the relevant quality endpoint?
3. Does Jev improve common-probe Brier over the strong Bayesian reference, and do
   forecasts improve choices beyond matched hidden/chance/public controls?

Report signed paired changes in raw MSE, variable/total assay cost, effort,
original return, assay count, stops, hosted usage, failures and latency. For five
training seeds and common mechanisms, use paired two-way bootstrap over seed and
mechanism, 10,000 draws, RNG991499. Report95% descriptive and multiplicity-adjusted
intervals for predeclared confirmatory families; development intervals remain
selection-affected. Brier/calibration resampling clusters all events by mechanism.
Ten fixed reliability bins include counts and forecast coverage; tiny pilots get
no calibration-success claim. Missing final outcomes remain missing, with failure
and all-attempt cost denominators exposed.

No superiority claim is required. The winner may remain the incumbent, imitation,
fixed design or Bayesian planner. Parameter holdout within familiar motifs is not
structural generalization. Stronger shifts and biology remain unqualified. A new
final split is opened only after completed development, independent correctness
review, a frozen design and adequate hosted authorization; otherwise the report
explicitly records that final inference is pending.
