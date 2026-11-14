# Specification traceability and remaining gates

This audit maps the benchmark requirements to evidence available before the
hosted study. The subsequent evaluation reports its findings and remaining
limitations. [Benchmark specification](../autonomous_scientist_build_spec.md) · [Hosted evaluation](m2-research-note.md)

## Gate decisions

The original admitted pilot and stronger admitted v2 failed scientific adaptive
Gate C: fixed designs already achieved the available quality. Those results stay
in `reports/feasibility-v1` and `reports/research-v2`. The documented unfiltered
revision retained unreachable targets and passed development Gate C; its later
locked 512-mechanism comparison recovered 18.4% additional raw-error gap. This
does not retroactively change the admitted-task conclusion. M0–M7 in
`docs/milestones.md` describe the restricted benchmark, not all flagship features.

Latest historical v6: five retained seeds, 4,096 independent test mechanisms.
Selected PPO RMSE 0.04865092088617208, variable cost 4.7815625, return
-0.3138388665572575. Previous efficient PPO: RMSE 0.04911886780730731, cost
4.969287109375, return -0.319884969810923. Strong imitation: RMSE
0.04859125744102652, cost 4.62298828125, return -0.31210849090751147. Paired PPO
minus imitation cost is 0.15857421875000005, CI [0.10501428222656257,
0.2100749511718751]; return difference -0.0017303756497460416, CI
[-0.009945576456604974, 0.003364161225235835]. The full declared parity screen
failed. Source: `reports/ppo-efficiency-v6/summary.json` and
`docs/ppo_efficiency_v6.md`; none of this is comparative Jev evidence.

## Requirement-to-code/evidence/gap map

| Requirement | Runnable implementation and evidence | Qualification or documented gap |
|---|---|---|
| §0.1 numerical diagnostic and generous fixed controls | `worlds.py`, `diagnostics.py`, `audits.py` under `src/experimentlab`; `reports/feasibility-v1` | Supplied external audit scripts were absent; independent reconstruction is labeled. Time-3/full-panel cost arithmetic exists, but that schedule's estimator/performance is unqualified. |
| §0.2 information/adaptation/RL claims distinct | `docs/feasibility.md`, `docs/revisions.md`, v3–v6 reports | Information and restricted adaptation supported historically. PPO superiority to strong imitation is unestablished. |
| §0.3 bounded 25/45-action pilot, four assays, budgets 8/16, noise grid | `envs.py`, `tasks.py`, `configs/pilot.yaml`, `test_diagnostic_environments.py`, original scenario reports | Full 429-action catalogue remains gated. New M1 stays at budget 8; it does not requalify every noise/culture slice. |
| §0.4 A numerical/information validity | `test_science.py`, `test_environment.py`, `test_inference.py`, `test_evaluation_contracts.py`, `test_private_cache.py`; new baseline science receipts | Revised-task culture variation remains unqualified. |
| §0.4 B information after cost and negative noise control | `reports/feasibility-v1/feasibility-report.md`; SD 10 correctly stops | Restricted generator result; no authenticated Jev result. |
| §0.4 C adaptive value | `research-v2/training-decision.json`, `unfiltered-v3/assessment.json`, `unfiltered-v3/locked-test` | v2 failure retained; v3 pass restricted to documented revised family. |
| §0.4 D learnability and E added RL value | `learning.py`, `ppo_v4.py`, v4/v5/v6 same-architecture comparisons | Learned policies are useful; full RL-specific superiority screen remains false. |
| §0.5 runtime and break-even | Teacher cost, interaction counts, inference/training timing in v3–v6 | No qualified quality-adjusted deployment break-even; do not count imitation speed as PPO achievement. |
| §1 independent project, contribution and limits | `pyproject.toml`, `docs/protocol.md`; standalone `experimentlab` | Evidence covers synthetic mechanisms and tasks. |
| §2 public reset, priced fresh experiments, independent final cultures, setup cost | `envs.py`, `worlds.py`, `tasks.stream`; M1 records variable and total cost | Sequential design then commitment, not ongoing feedback control. At offset SD 0, final latent replicas are identical. |
| §3.1 roles, allowed controls, neutral names | `envs.catalogue`, public masks, `tasks.finite_family` | Dynamics permutation test exists; complete action/role/observation/policy permutation invariance is unqualified. |
| §3.2 suppression dynamics and contraction | `worlds.Mechanism`, `integrate`, row-L1 checks, `test_science.py` | Contractive synthetic model only. |
| §3.3 matched/varied baselines, independent targets | `worlds.matched_mechanism`, `tasks.make_task` | Benchmark generators use baseline .5. A varied-baseline study is absent despite the helper accepting a baseline argument. |
| §3.4 integrator precision and solver checks | float64 exponential Euler, DOP853/dt-half tests | No implied exact ODE integration or biological validity. |
| §3.5 fresh culture offsets and noisy raw replicates | `worlds.measure`, `envs._measure`, `checks.predictive_check` | Culture-mixture likelihood is approximate; revised main study offset SD 0. |
| §4.1–4.2 experiment/commit menu and stable indexing | `envs.Action`, `catalogue`, 36 assays + 9 commitments | Paired controls, time 3 and four-panel flagship deferred by §0. |
| §4.3 costs, masks, invalid-run handling | integer ticks, bounded penalty, decision cap in `envs.step`; environment tests | New adapter rejects illegal hosted choices before stepping and records failed runs; original simulator policy unchanged. |
| §5 public observations, missing masks, hidden truth | `envs._obs`, `learning.token`, private-cache tests; M1 `public_state` allowlist | Architectural boundary, not protection against malicious local introspection. Public request packets omit seeds/hashes/truth. |
| §6.1 target priors, feasibility conditioning, unreachable cases | `PredictionBank.admitted`, `Belief`, `research_stage.UnfilteredBank`; revisions and unreachable replay | Revised study uses unfiltered prior. Broad difficulty-stratified benchmark incomplete. |
| §6.2 latent final MSE/effort | `envs.step` final result, 128-culture later locks | Privileged nominal reference is not an attainable policy or rigorous global upper bound. |
| §6.3 reward accounting and undiscounted return | `envs.step`, `learning.returns_gae`, cost sensitivity studies | Brier is a separate forecast endpoint, never an undocumented reward change. |
| §6.4 optional protected-state constraints | No core implementation | Deferred; no safety/viability qualification. |
| §7.1 informative proxy fixture | `diagnostics.numerical_audit`; original values reproduced in new diagnostic receipt | Fixed threshold solves it; not evidence for RL necessity. |
| §7.2 branching fixture | `diagnostics.branching_audit`, exact enumeration tests | 100% adaptive vs 75% fixed is a toy correctness result. |
| §7.3 history without action-memory shortcut | `learning.delayed_memory_check`, `test_learning.py` | History check does not establish main-family adaptation. |
| §8 prior/random/fixed/additive/BO/EIG/VOI and strong imitation | `baselines.py`, `bo.py`, `evaluation.py`, exhaustive 10,373-design study; all retained | Restricted declared baseline suite; no global optimality claim. M1 runs these alongside all five learned seeds. |
| §8.1 common final estimator and separation of selection/interpretation | Historical common-selector ablations; fixed/VOI share `Belief.commitment` | M1 preserves native learned commitments. Replacing them requires a labeled secondary ablation. |
| §8.2 exact/approximate posterior and calibration | `inference.py`, `checks.py`, finite Gaussian tests, quadrature refinement | No broad posterior rejuvenation or general continuous-posterior qualification; coverage of means alone is insufficient. |
| §8.3 decision value versus information gain | `DecisionPlanner` with VOI/EIG and MC draw checks | One-step planning; two-step scientific planner deferred because one-step reached development reference. |
| §8.4 fair BO | `bo.DiscreteBO`, negative variance-corrected observations retained, `test_bo.py` | No multi-fidelity BO; nonzero culture variation explicitly rejected. |
| §8.5 DAD/Step-DAD comparison | Brief scope statements in existing docs | No reproduction or substantive matched literature comparison; submission blocker if claiming broad state of the art. |
| §9.1 actor–critic/history/GRU | `learning.ActorCritic`, `ppo_v4.py`, `tensor_batch.py` | GRU64 is documented pilot refinement of proposed 128. Latest-only exists but no main measured comparison. |
| §9.2 PPO masks/padding/GAE/logp/entropy/gradients | learning, tensor, PPO and review regression tests | Complete-episode collector rejects unsupported external truncation; no silent bootstrap claim. |
| §9.3 curriculum | Reward-only selector-to-joint curriculum, v4 controls, v5/v6 continuation | Broad family curriculum with persistent easy-task mixture remains unqualified. |
| §9.4 optional learned belief/world model | None required for bounded milestone | Deferred. M1 hybrid is frozen PPO plus hosted arbitration, not new PPO training. |
| §10 native CPU/MPS, cache, measured throughput | `compute.py`, `private_cache.py`, `benchmark_ppo_v6.py`, equivalence tests | CPU wins measured workloads. Historical contended timings are not a current runtime guarantee. |
| §11 architecture boundaries | Compact independent simulator, inference, baselines, learning, evaluation modules | New integrations remain outside frozen `src`; broad proposed package hierarchy not required for bounded pilot. |
| §12 typed contracts and CLI | Frozen dataclasses, `cli.py`, stable command names | Public observations remain dictionaries. Generic train is smoke-only; locked studies use dedicated runners. Existing replay only displays JSON; M1 adds actual simulator verification. |
| §13 resolved/strict configurations | `PilotConfig`, strict YAML loader, `test_splits_and_cli.py`; M1 manifest config/caps | No general flagship/baselines YAML. M1 CLI permits only declared fixture or short hosted development modes. |
| §14 split discipline and generalization | `tasks.split_manifests`, historical manifests/freeze hashes; M1 new seed namespace and hash checks | Parameter holdout within familiar motifs/ranges only. Structural/group canonicalization, out-of-range and misspecification shifts remain absent. |
| §15 independent evaluation and uncertainty | v3/v4/v5/v6 paired/hierarchical bootstrap, all seeds retained; `jev_rl_protocol.md` future hypotheses/intervals | M1 fixture has no efficacy CI. More mechanisms do not replace independent optimization seeds. Future hosted repeat design still needs pilot evidence before final freeze. |
| §16 ablations | fixed/adaptive, EIG/VOI, history/recurrent, imitation/PPO/combined, learned/common selector, cost/stop controls | Measured latest-only, proxy/outcome-only, diversity, single/combined and broader ablations incomplete. M1 adds forecast hiding/chance/public replacement. |
| §17 seeds, checkpoints, dependencies, provenance | Stable stream namespaces, saved optimizer/RNG/config hashes, source snapshots; M1 journals/manifest/snapshot | Complete-episode resume only, no partial-environment snapshot. No cross-platform bitwise guarantee or released Jev weights. |
| §18 scientific/engineering acceptance tests | Existing 86 tests plus new transport, forecast, hybrid and replay tests | Passing tests do not qualify biology, all spec shifts, or authenticated Jev performance. |
| §19 milestones M0–M7 | `docs/milestones.md` historical restricted accounting | New Jev M1 is separately reported; full flagship/submission not complete. |
| §20 demonstrations and inspectability | Historical success/failure/unreachable traces and figures; new full request/response journals and replay | No interactive replay GUI. Conceptual animation is not a measured experiment. |
| §21 optional extensions | None needed for bounded pilot | Continuous actions, other physics, retrospective real data, world models deferred. |
| §22 references | Original source list retained verbatim in spec | Literature-positioning review and matched omitted-baseline justification needed for submission. |

## Jev evaluation requirements

| Jev/RL requirement | Implemented route | Evidence needed beyond M1 |
|---|---|---|
| Prospectively measurable Jev role | All affordable assay event forecasts before choice/observation; `jev_research.py` | Authenticated Brier/calibration on common fixed probe. |
| Full learned RL, Jev-only, hybrid | Hash-verified five-seed registry; full PPO probabilities and native commitments | Independent efficacy and RL-information contribution, not just input wiring. |
| Information/extra-compute ablations | Hidden/chance/public tables; same forecast and choice stages | Real hosted results; calls matched per decision, not exact total tokens or trajectories. |
| Matched budgets and local scoring | Same environment, targets/seeds, budget 8, max 4, original reward | Larger primary and sensitivity studies after gates. |
| Strong forecast references | Prior and posterior predictive mixtures, chance and public repeat heuristic | Continuous-family calibration and model-misspecification investigation. |
| Freeze version/prompts/menus/predictions | Pinned model, request/response journal before `step`, source snapshot | Hosted model repeat/drift characterization; journal is not external timestamp attestation. |
| Cost/latency/failure accounting | Client preflight caps, reported usage, unknown-usage reservations, per-condition receipts | Actual token bill/latencies; request-byte token reservation is not a billing guarantee. |
| Manifest, replay and public reproducibility | Source snapshot verification regenerates uncached outcomes without keys or model checkpoints | Clean publication export and independent-machine reproduction. Public checkpoint retraining remains required for new policy runs. |
| API access | Verified TLS, bounded failures, and offline replay | A live run requires TypeSafe API access. |
| Scope of the evidence | Protocol, requirements audit, measured M1 report | Compare learned and hosted policies on matched mechanisms, document uncertainty, and test transfer separately. |

## Validation entry points

Use a fresh output directory for every run. Do not invoke old locked runners to
obtain mutable new results.

```bash
PROJECT="$(pwd)"
"$PROJECT/.venv/bin/python" -m pytest "$PROJECT/tests" -c "$PROJECT/pyproject.toml"
"$PROJECT/.venv/bin/experimentlab" validate-science --suite diagnostic --output "$PROJECT/runs/new-diagnostics"
"$PROJECT/.venv/bin/experimentlab" validate-science --suite inference --output "$PROJECT/runs/new-inference"
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/jev_rl_study.py" run --output "$PROJECT/runs/new-jev-fixture"
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/jev_rl_study.py" verify --run "$PROJECT/runs/new-jev-fixture"
```

The complete original 12-scenario feasibility command, fresh learning smoke and
CPU/MPS benchmark remain runnable, but rerunning their expensive historical studies
is unnecessary to verify this integration. No separate literal `check-spec` CLI
exists: this traceability audit plus the scientific/configuration/tests comprise
the relevant specification checks.
