# Primary sources and their consequences for M2

These references informed the study design and implementation. [Candidate derivation](jev_rl_m2_task_derivation.md)

## Jev and typed decision models

| Evidence | Verified scope | Transfer and limits |
|---|---|---|
| [TypeSafe API/SDK](https://docs.typesafe.ai/sdk/python/api/types/responses), [Choice](https://docs.typesafe.ai/primitives/choice), [Noul](https://docs.typesafe.ai/primitives/noul) | Vendor contract: typed outputs, complete Choice distributions, Noul probabilities, response model/usage and provider request IDs | Implement exact parsing/tracing. Vendor calibration descriptions are hypotheses to test on this simulator, not imported evidence of forecast quality. |
| [Deußer et al., arXiv v2](https://arxiv.org/abs/2609.37647v2), [actual repository](https://github.com/AppliedMachineLearning-Lab/jev-benchmarking) | Independent research evaluation of hosted Jev 1.13.0; 37 datasets and a sequential track, released harness and response archive. Authors include Bonn/Lamarr/Fraunhofer affiliations. | Frozen prompts, identical requests, legal menus and offline cache replay transfer. Reported sequential weaknesses/repeated ineffective actions motivate a narrower Jev role. Different version, natural-language tasks and environment horizons do not establish our expected performance. |
| [Sequential runner inspected](https://github.com/AppliedMachineLearning-Lab/jev-benchmarking/blob/main/jev_benchmarking/sequential/runner.py) | Actual backend calls, legal-action checks, per-step distributions, environment feedback, separate reference labels | Keep reference answers out of model input; separate decision/forecast metrics from task outcome. Our no-retry bounded policy and pre-observation journal remain explicit differences. |
| [Will Kelly evaluation](https://github.com/willkelly/jev-evaluation), [prospective plan](https://github.com/willkelly/jev-evaluation/blob/main/jev-evaluation-plan.md) | Independent developer evaluation with a prewritten plan, raw logs, strong formal-task controls and negative results | Test calibration by condition; do not infer usefulness from well-typed output or aggregate confidence. The simulator uses its own acceptance rules. |
| [Actual Pokémon integration](https://github.com/valentynkit/jev-plays-pokemon-red), [inspected call path](https://github.com/valentynkit/jev-plays-pokemon-red/blob/main/src/jpp/policy.py) | Developer code constructs lawful branches and asks Choice/Noul, supports recordings and independent outcome labeling. README explicitly says its labeled sample is too small for a calibration claim. | Separate deterministic action constraints from model judgment; measure a prospective event against later state. Do not treat the small demo or its code-default fallbacks as autonomous planning evidence. |

## RL and experimental-design sources

| Source / inspected implementation | What transfers | What does not transfer automatically |
|---|---|---|
| [OpenAI PPO2](https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py), [rollout runner](https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py) | Store behavior action probabilities/values; align actions/rewards/termination; preserve recurrent sequences and reconstruct advantages from the actual rollout. | TensorFlow vectorized long-rollout defaults, reward clipping and discounting are not our five-step undiscounted objective. Do not port framework/hardware claims. |
| [DeepMind Acme PPO learner](https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/ppo/learning.py), [R2D2 learner](https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/r2d2/learning.py) | Explicit learner/optimizer/random state, behavior probabilities, normalization choices and sequence boundaries. A separated actor/learner motivates measuring transfer overhead. | Distributed JAX/pmap and off-policy prioritized recurrent replay are different workloads. R2D2 burn-in does not justify adding replay complexity to complete five-step PPO episodes. |
| [Andrychowicz et al., Google Brain study](https://arxiv.org/html/2006.05990v1) | Architecture, learning rate, batch/update and normalization choices materially affect on-policy results; measure a finite, declared configuration grid rather than attribute every gain to PPO itself. | Their 250,000-agent continuous-control study and particular hyperparameter recommendations are not results for masked categorical experiment design in this benchmark. |
| [Engstrom et al., Implementation Matters](https://arxiv.org/abs/2005.12729), [authors' code](https://github.com/MadryLab/implementation-matters) | Implementation changes must be separated from claimed algorithm improvements and backed by controlled comparisons. | A PPO/TRPO continuous-control result does not establish a preferred optimizer/device for this task. |
| [ICLR PPO implementation audit](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), [inspected PPO implementation](https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py) | Behavior-logp checks, clipped likelihood ratio, optimizer initialization, advantage normalization and full accounting are explicit experimental choices. | Flattened transition minibatching is inappropriate for our recurrent histories unless sequence context is maintained; unchanged reward units and legal masks remain mandatory. |
| [Ni et al., recurrent POMDP baselines](https://proceedings.mlr.press/v162/ni22a.html) | Strong tuned model-free memory baselines deserve a fair chance before adding complex model-based components. | Their many POMDP workloads do not prove a GRU beats explicit full history when there are at most five packets. |
| [DAD](https://proceedings.mlr.press/v139/foster21a.html), [authors' code](https://github.com/ae-foster/dad), [inspected model](https://github.com/ae-foster/dad/blob/main/location_finding.py) | Amortize design computation; compare against strong static designs; maintain histories and disclose offline costs. | Original EIG objective differs from priced final-control loss. This study uses discrete five-step episodes. |
| [Step-DAD](https://proceedings.mlr.press/v267/hedman25a.html), [full method](https://arxiv.org/html/2507.14057v1) | Account for any test-time adaptation and compare its benefit to extra online compute. | Online posterior-conditioned refinement is an additional algorithm and cost. It is not equivalent to asking Jev for another opinion. No Step-DAD implementation or matched reproduction is claimed in M2. |
| [Blau et al., RL for sequential design](https://proceedings.mlr.press/v162/blau22a.html) | Policies can optimize sequential design in discrete/black-box settings; compare against explicit inference and planning. | Their SAC/information objective, discount choices and experiment models do not replace our original reward, PPO comparator or simulator. |
| [Agarwal et al., statistical reliability](https://arxiv.org/abs/2108.13264) | Retain seeds, publish uncertainty and avoid conclusions from a single best seed/point estimate. | Across-game aggregate scores are not interchangeable with paired mechanism and optimization-seed uncertainty in one motif family. |
| [PyTorch MPS reference](https://docs.pytorch.org/docs/stable/mps.html) | Synchronize device work for timing; distinguish allocated GPU memory from process peak memory; record backend availability and fallback. | Compare complete rollout, update, validation, and checkpoint cycles on the benchmark workload. |

## Source-to-design decisions, fixed before the new comparison

1. **Keep full standalone PPO and strong non-RL winners.** Compare explicit full
   history and GRU capacities, with a declared small optimizer/batch grid. Use
   reward-only curriculum for scratch arms, and account for all pretraining in any
   v6 continuation. No hidden privileged teacher information enters scratch PPO.
2. **Forecast-only Jev:** PPO selects the action first; Jev forecasts only the
   selected assay, before observation, and has no authority to change the action.
   The primary estimand is forecast quality versus same-event Bayesian/prior/
   chance references. Policy quality is identical by construction; any latency or
   monetary cost is additional. This is a materially narrower role than arbitration.
3. **Restricted proposal reranking:** Jev may choose among a frozen small menu of
   high-probability legal PPO proposals. The menu construction, commitment handling,
   order and maximum size must be frozen and logged. Compare the same menu without
   Jev and account for additional calls. A useful result must improve original
   control return, not only agreement with a model's preferred action.
4. **Retain the current arbitration incumbent and information controls.** Hidden,
   chance and public-forecast controls expose whether benefits come from forecast
   content, PPO advice or additional hosted computation. Do not tune on locked
   historical or future outcomes. Independent common fixed-probe events are the
   primary forecast comparison.
5. **Explicit public prior fairness:** candidate prompts receive the same declared
   family/physics description available to the tailored Bayesian reference, not
   the true mechanism. Freeze this packet separately from the incumbent prompt;
   changes to public prior exposure must be labeled as an input ablation.
6. **Avoid unearned scale:** no world model, unrestricted test-time optimization,
   cluster infrastructure or GPU-only dependency is justified at this milestone.
   The scientifically strongest method may remain fixed, Bayesian or imitation.

## Evaluation scope

The protocol requires paired held-out outcomes, all retained seeds, cost accounting,
and uncertainty estimates. The M2 study reports development evidence. Final
evaluation requires a separate freeze and passing the registered criteria.
[Protocol](jev_rl_m2_protocol.md) · [Results](m2-research-note.md)
