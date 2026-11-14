# Active M2 goal state

User expanded the goal to a bounded M3 Max PPO configuration comparison, paired
Jev study, primary-source/task-first design, reproducible evidence and GitHub
update. The goal remains active; do not mark it complete from M1 engineering work.

## Latest state — 2026-11-14

The earlier sections below are chronological context. Current state:

- Race completed all 42 jobs: 49,652,308 training interactions, 166,142 validation steps, 2963.5s; all primary snapshots correct and all selected five-seed cohorts retained.
- Selected pure PPO is scratch GRU512 with the selected batch512/LR0.0003/two-epoch continuation. Selection used tuning only.
- Confirmation freeze-02 (freeze-01 preserved) was published as public compact checkpoint freeze in GitHub commit cdda216 before constructing confirmation. Real confirmation-01 completed 5632 episodes / 16814 step calls / zero errors in38.5s. Return-gain screens all failed; selectedPPO mean return delta vs v6 is+0.006296, 95%CI[-0.001699,+0.020619]. It matches privileged-reference MSE on this128-mechanism slice but is costlier/lower-return than strongimitation/VOI. Final991401 remains unopened; do not retune on confirmation.
- M2 candidate engine integrations/jev_m2_episode.py has13 tests and independent review clearance. Forecast-only/non-veto predictions and restricted reranking/shadow/full controls implemented.
- Study CLI scripts/jev_m2_study.py now schema4: separate frozen v6 incumbent, all registered-source bindings, fresh admission checks, fixture reuse disclosure;20 tests+independent review clearance. Full selected-PPO75-case fixture candidate-fixture-02 running or complete; checkits verification.json. Earlier22-case fixture01 preserved. No newhostedcalls.
- New scripts/analyze_jev_m2.py has22 synthetic tests, fixed3primaryfamilies,10000draws/991499; source hash5fb8741bc88c8494a38c9e65f9ae332a3079545d50101ea0102af67f7865618a, contracthash efc988090eafe87ae68995ff259e5bf4f4c917aa1ab71d0ec11b17a17cf2f6cf. Needs independent review/fullfixture analysis, then public freeze beforehostedapproval.
- Agentprocess still has no TypeSafe key. Existingauthorization32calls/.10; originalpilot11. Prepare concrete 80-call/.10 single-case or2800-call/1.00 eight-mechanism/five-seed plans and ask explicitapproval/keyviaTerminal afterallcode/fixturechecks. Do not execute largerplan withoutapproval; do notaskforsecretinchat.
- GitHub featurebranch vajraang/jev-rl-m2 hasprotocol+checkpointfreeze commits9043afc andcdda216 only. Fullcode/results/publicreplayexport andPR/defaultbranch update remain. Releaseworktree experimental-scientist-release-m2. CLIgit/gh writable, connectorread-only.
- Resource monitor startedlate; inspectcoverage beforeclaimingmemory/thermalpeaks. A postselection resource-only shortselected-memory-profile isrunning to capture nativepeakRSS within remaining3Mprofilingcap (<=3260additionalsteps). Neverpromoteitscheckpoint.
- Remaining: strictconfirmationreplay; finalreports/figures;75-casefixture+snapshotreplay; hostplan approval/userTerminalexecution+analysis; publicexport preservingportability/redactingprivatehandoffpointer/paths; freshtests andpreservation; goalstatus remainsactive.

## Completed inputs

- Primary checkout: `experimental-scientist`, unborn Git branch
  `vajraang/jev-rl-m1`; original695 files unchanged. Public checkout remains clean
  at ae6cdc8 on `vajraang/publication`.
- M1: prospective forecast/decision loop, same-call controls, strong baselines,
  full frozen PPO/imitation/IM→PPO, hash-linked journals, strict credential-free
  replay, safe provider request IDs/HTTPmetadata/localspan tracing. Final fixture04
  has100episodes/18conditions,236fixturecalls,40original-policy equivalences.
  Primary suite at that checkpoint241passed; public90passed. Five saved fault
  examples replay. Earlier fixture01/02/03 snapshots are retained.
- A user-run authenticated pilot appeared at
  `reports/jev-rl-m1/hosted-20261113-204144`; replayed15episodes,11real calls,
  jev-1.14.0,11unique provider IDs, allHTTP200,77860input/11811output tokens,
  estimatedinputcost.00327012USD, no failures. Fixedprobe6events: JevBrier.2493833,
  Bayes.1823470. This is one mechanism/seed20, not efficacy/calibration evidence.
  Current agent process has no TYPESAFE_API_KEY. Never recover it from files/logs.
- Root personally inspected official docs, independent Jev research/repro source,
  OpenAI PPO2, DeepMind Acme PPO/R2D2, DAD code and relevant primary papers.
  `docs/jev_rl_m2_sources_and_design.md` and source receipts record evidence.
  No verified frontier-lab Jev deployment was found in targeted official-domain
  searches; do not turn this into proof of non-use or count other typed models.
- Task-derived candidates: incumbent full arbitration; narrower PPO-selected
  assay forecasts; restricted3-proposal reranking. Public prior fairness is
  explicit. `integrations/jev_candidates.py` implements pure builders;41focused
  tests,189withrelated tests passed. Episode runner/comparison is still needed.
- Hardware profiles completed under3Mtransition/600s cap:2,996,045transitions.
  CPU wins every tested history/GRU64..512configuration. SustainedGRU64batch1024:
  CPU8604, MPS6138, split7411transitions/cycle-second. CPU4seed aggregate~31k/s.
  CPU60s drift minimal; pmset records no thermal/performance warning, no measured
  temperature. ProcessRSS peaks measured, not exacttotalunified-memory peak.
  Hardware artifacts: hardware-sweep, profile-capacity512, profile-initial-*.
- Frozen M2 protocol: `docs/jev_rl_m2_protocol.md`, `study-protocol.json`.
  Beforedata amendment clarifies primary eligibility independent of secondarywall
  snapshot completion; oldhash is retained in study-protocol.pre-review.json.
- Protocol published BEFORE registeredtraining on GitHub branch
  `vajraang/jev-rl-m2`, commit9043afc0721f5613c0c13c607e7189751cc08e81.
  Receipt: public-protocol-freeze.json. Connected GitHub app is read-only; local
  gh/git haspushpermission. Release worktree sibling experimental-scientist-release-m2.
  Only4protocol/source-rationale docs have been published so far; noPR yet.

## Implementation/review at this checkpoint

- Profiler: integrations/m3_ppo_profile.py, scripts/profile_m3_ppo.py,
  tests/test_m3_ppo_profile.py;44focused tests inclnativeMPS/split passed.
- Trainer: integrations/ppo_m2_train.py, scripts/train_ppo_m2.py,
  tests/test_ppo_m2_train.py. Worker `ppo_m2_trainer` owns these until final handoff.
  It implemented64..512history/GRU, completeepisodes, CPUmirrors/MPSlearner,
  immutableversioned checkpoints, exactcounts, separateprimary/wall snapshots,
  rawtuningrows, native/cacheproof, continuationnamespace91303/inheritedoffsets.
  Original76combinedtests passed. Independent reviewer `quality_review` found
  local selector/biased checkpoints could receive the zero-step incumbent
  exemption. Worker is fixing and adding regressiontests; wait for final recheck
  before longruns. No registeredtraining hasstarted yet.
- Race orchestrator: scripts/run_ppo_m2_race.py, tests/test_ppo_m2_race.py.
  Five tests pass. Reviewer cleared it after fixing exclusion of validprimary
  results missing only the secondarywall snapshot. Allfilters use primary_eligible.
  HandlestruncatedJSON asfailedreceipt; reservesunknownfailedjobcomputeconservatively.

## Frozen race plan and next action

After trainer gatefix is re-reviewed, run the race once in a fresh directory:

```bash
PROJECT="$(pwd)"
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_ppo_m2_race.py" \
  --output "$PROJECT/reports/jev-rl-m2/race-01"
```

CPU4workers,1thread each. Totalactiveorchestrator wallcap3600s; trainingcap60M,
validation2M. Maximumplannedreservation58.75M. Architecture8variants ×seeds20/21:
history/GRUhidden64,128,256,512,1Mselector+1Mjoint primary,1.25Mjointhard,
batch256,wall240s,jobcap900s. Bestarchitecturethen4update recipes ×2seeds,
500kprimary/750khard/wall70s. Complete selectedscratchrecipe22–24. Two v6
continuationrecipes20/21 then selected22–24; separate historicalIM→PPO
continuationall5. Preserveall failures, incompletepanels, costs and seeds.
Primaryselection usesfresh tuning991311; confirmation991321 is not yet constructed,
futurefinal991401 notopened. Protocol is already public; any necessarychange must
be a disclosed pre-result amendment, never silently tuned onconfirmation/final.

## Remaining work

1. Complete race, independently verify checkpoints/rawrows, freeze selectedcohorts,
   then evaluate uncached freshconfirmation with paired seed/mechanism intervals.
   Report equalinteraction andwall panels separately and inheritedcompute honestly.
2. Implement M2 candidateepisode runner + deterministic replay and study manifests,
   retain incumbent andsamecallhidden/chance/publiccontrols plusstrongbaselines.
   Forecast-only cannot changePPOactions; reranker cannotexpand3legalproposals.
3. Hosted authorization stays32calls/.10estimatedUSD (pilot used11). A complete
   newpairedstudy likelyneedsmore calls and a freshkey. Prepare exactbounded
   Terminalcommand and askifnecessary; continue localwork meanwhile. No secrets
   inchat/files/Git; no automaticunbounded retries/runs. No finaltest withoutgates.
4. Export code/results/replay to releaseworktree without overwritingpublicportability
   changes. Redactprivate originalspec handoffpointer from publicsnapshots while
   documentingprovenance. Considerportableweights-onlyexports fornewselectedmodels,
   but do not publish arbitraryfullcheckpoints/privatepaths without review.
5. Publictests, replay, finalsource/preservation checks, measuredreport andGitHubPR/
   branchupdate. Goalcomplete only when required verifiedworkdone; current broad
   goal notcomplete and no need to markblocked while localprogress remains.
