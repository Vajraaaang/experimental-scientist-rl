
This specification defines a new, standalone reinforcement learning project for a MacBook Pro with an M3 Max: an autonomous experimental scientist that purchases informative experiments on unfamiliar simulated regulatory systems, then chooses a final intervention. The deliverable is a reproducible research result about experimental decisions under uncertainty.

The first scientific domain is a deliberately transparent biological-inspired dynamical system. The learning system is a small recurrent policy. Its weights are trained across mechanisms; at evaluation it adapts to a new mechanism by accumulating experimental evidence in memory. All numerical parameters below are initial design choices to validate, not measured M3 Max performance or established biological constants.

# 0 Feasibility contract before committing to RL

This project is feasible as a small simulated experimental-design benchmark on an M3 Max. That is an engineering assessment of the proposed scale, not a measured runtime guarantee. Whether adaptive experimentation helps, and whether RL training improves on simpler learned or planned strategies, are separate empirical questions.

The user's priority is a project whose RL environment is useful. Therefore these gates take precedence over expanding the feature list below. Build a bounded feasibility pilot first. Do not interpret a functioning simulator or a rising PPO reward curve as evidence that RL is necessary.

## 0.1 What has already been checked

The four-variable diagnostic in Section 7 has been numerically evaluated with the proposed dynamics. A short proxy assay reveals which final intervention reaches the target. With no culture variability, Gaussian readout standard deviation 0.02, and 8 independent cultures, its exact two-hypothesis sign-classification error is approximately 6.2 × 10^-10. No intervention has target MSE 0.0081; a fixed proxy assay followed by the appropriate intervention has expected target MSE approximately 2.84 × 10^-7. Including the stated intervention and assay penalties gives expected total negative return approximately 0.05653, versus 0.81 for no intervention.

This is strong evidence of the value of a measurement in that constructed example. It also demonstrates that the example is much too easy to justify adaptive RL: a fixed assay followed by a threshold rule solves it. Keep it as a correctness fixture and negative control for elaborate planning.

A second concrete concern is the flagship budget. At budget 16, a fixed schedule can assay each of the four controls at dose 0.8:

- Measuring both outcomes at time 6 costs 4 × 2.9 = 11.6 units.

- Measuring all eight variables at time 3 costs 4 × 3.8 = 15.2 units.

Those schedules are affordable and potentially powerful. Include them explicitly. Do not remove them or change costs after seeing that they beat RL.

An exploratory numerical audit also tested one explicit eight-variable family before any RL training. Each node had two or three non-self parents, balanced interaction signs, magnitudes Uniform(0.4, 1.5) rescaled to the row-norm bound, and rates Uniform(0.5, 1.5). All noiseless baselines were 0.5, and each goal was independently 0.5 ± 0.08. Feasibility meant that a nominal noiseless intervention attained outcome RMSE at most 0.04. There were 119 admitted calibration mechanisms out of 300 and 237 admitted held-out mechanisms out of 600; the first 128 admitted held-out mechanisms were evaluated without selecting on policy results.

The fixed designs used 8 cultures per assay, culture-offset standard deviation 0.03, and readout standard deviation 0.02. Final interventions were scored on 128 fresh cultures per mechanism, paired across methods and independent of exploratory data. A simple additive predictor selected a final intervention from the full 33-action menu using measured single-control responses.

| Method in this exploratory audit | Variable assay cost | Aggregate final target RMSE | Success within 0.04 |
| - | -: | -: | -: |
| No experiments and no intervention | 0 | 0.08038 | 0% |
| Tuned fixed two-control assay schedule | 5.8 | 0.05631 | 31.25% |
| Fixed four-control assay schedule | 11.6 | 0.02755 | 94.53% |
| Private nominal-model reference | 0 | 0.02628 | 96.09% |

Aggregate RMSE is the square root of mean squared error over mechanisms, fresh cultures, and both outcomes. The reference knows the mechanism, chooses using nominal zero-offset dynamics, then is evaluated on fresh cultures; it is not a rigorous optimum under the culture distribution. Its zero assay cost is a privileged reference condition, not an attainable budget-matched baseline. The two-control subset was chosen on calibration data only.

These results show that a fixed identification strategy nearly reaches this reference's final quality at the generous budget. They do not establish adaptive value at the tighter budget. This was one exploratory generator with parameter-held-out seeds, not a locked multi-family benchmark or structural-generalization result. No BO, calibrated continuous-posterior planner, imitation policy, or RL method was evaluated in this audit. Audit scripts and captured outputs accompany the Codex handoff.

No recurrent policy has been trained for this specification, no scientific adaptive advantage has yet been established for the eight-variable generator, and no M3 Max throughput has been measured. The bounded pilot must resolve those questions.

## 0.2 Three distinct claims

1. Information value: purchased experiments improve the final intervention relative to the best prior-informed commitment.

2. Adaptation value: selecting or stopping experiments based on observations improves the cost-quality frontier relative to tuned fixed schedules using the same final estimator.

3. RL training value: PPO or PPO fine-tuning improves on a history-conditioned policy trained by imitation, after accounting for training interactions and teacher computation.

Demonstrating claim 1 does not demonstrate claim 2. Demonstrating claim 2 does not demonstrate claim 3. A fast learned policy may be useful even if imitation is sufficient, but that is an amortized planning result, not proof of an RL-specific advantage.

## 0.3 Bounded first release

Implement two diagnostic fixtures and one restricted scientific pilot before the full catalogue:

- Calibration configuration: four variables, two controls, two doses, two experimental durations, two panels, and no paired interventions. This yields 20 experimental actions plus 5 commitments.

- Scientific pilot configuration: eight variables, four controls, two doses, durations 1 and 6, one proxy panel containing both intermediates and one outcome panel containing both outcomes, with no paired interventions. This yields 36 experimental actions plus 9 commitments.

- Start with a maximum of four purchased assays. Evaluate tight and generous budgets separately, including 8 and 16.

- Use a small declared motif family with a handful of uncertain parameters. Begin with a finite-hypothesis case where posterior calculations can be checked exactly, then introduce held-out continuous parameters.

- Initially disable culture-offset variation to validate the likelihood; retain readout noise. Add declared culture variation after inference checks pass.

- Use 4–8 cultures per assay, with a noise sensitivity grid established before judging the policy.

- Start with no-experiment, fixed, additive-response, exact or small-belief decision-value, and imitation baselines. Add expensive methods only after this comparison is informative.

The 429-action catalogue, broad random networks, large particle banks, two-step planning over all candidates, and optional world models belong to later stages. The flagship YAML in Section 13 is a target configuration, not the initial command to run.

## 0.4 Proceed or revise criteria

Gate A — numerical and observational validity:\
The dynamics, intervention semantics, noise likelihood, information boundary, and independent final evaluation must pass their scientific checks.

Gate B — actual decision value:\
On a declared scientific pilot family, experiments must improve final decision quality after cost relative to the prior-informed baseline. Check several noise levels, including an uninformative-noise negative control.

Gate C — scientific adaptive advantage:\
A credible adaptive belief planner must improve over tuned fixed designs on the scientific simulator, not only on the separate routing toy. Compare error versus cost curves, allow both methods the same evidence interpretation, and include fixed single-control identification schedules and additive response models. A provisional engineering threshold is recovery of at least 10 percent of the prior-to-full-information loss gap beyond the best fixed design on a preregistered pilot slice, with a positive paired confidence interval. Treat that threshold as a screening choice, not a universal scientific standard.

If Gate C fails across reasonable, predeclared budgets and noise settings, retain the negative result. Examine whether the task is solved by one assay, nearly linear response identification, uninformative measurements, or excessive noise. Make a scientifically motivated, documented revision or frame the project as an experimental-design benchmark. Do not scale PPO merely to make RL appear necessary.

Gate D — learnability and compute:\
A small history-conditioned imitation policy should learn a useful approximation to the planner on held-out pilot tasks. A recurrent learning smoke test must work. Benchmark complete simulator, inference, rollout, and update workloads on the actual Mac. Forecast a full run from measured throughput before launching it.

Gate E — added value of RL:\
Compare the same recurrent architecture trained by imitation, PPO, and imitation followed by PPO. Account for teacher planning cost and training interactions separately. If PPO adds no measurable benefit, report that result; do not label imitation's speed advantage as an RL achievement.

These gates are development decisions evaluated on training/validation pilot data. Once the generator and methods are frozen, do not revise them using locked test outcomes.

## 0.5 Useful outcomes and limits

A successful project can show either better final decisions at a fixed experimental cost or comparable decision quality with substantially cheaper online planning. In the latter case compare against imitation too, and report offline training cost. If comparing repeated deployments, report the approximate compute break-even number of episodes as offline training cost divided by per-episode planning-cost savings, when those quantities are comparable and savings are positive.

The environment provides controllable ground truth, repeated independent mechanisms, priced evidence, and hidden final evaluation. Those properties make experimental policies testable. It cannot by itself establish that a learned policy improves a real wet-lab process or guarantees a job. A defensible simulator, strong comparisons, reproducible software, and an honest research result are the intended portfolio deliverables.

# 1 Project objective and contribution

Working repository name: `experimental-scientist`. Python import package: `experimentlab`. Command name: `experimentlab`.

Build a fresh repository. Do not import from, modify, or depend on the user's RNA-seq harness, jev-rl repository, Avenio implementation, or private employer data.

Primary question: Can a policy trained across hidden mechanisms choose experiments that yield better final interventions at the same experimental budget, while adapting to previously unseen mechanisms?

The intended contribution is decision-focused experimentation under partial observability, evaluated against strong explicit inference and planning methods. The benchmark must distinguish:

- An experiment that helps predict an observation.

- An experiment that helps identify a mechanism.

- An experiment that improves the final control decision.

Those objectives need not select the same experiment. The policy should be allowed to stop when additional evidence is not worth its cost.

The flagship release comprises a scientifically checked simulator, a versioned benchmark, reproducible non-RL baselines, a trained recurrent policy, held-out evaluation, and inspectable experimental histories. A negative result is legitimate if the comparisons and failure analysis are sound.

The model is not a validated virtual cell. Time, activity, dose, cost, and constraints are abstract benchmark quantities. No result establishes actual therapeutic efficacy, real gene-network discovery, or the validity of interventions in observational RNA-seq data.

# 2 The experiment protocol

One episode contains one hidden mechanism. That mechanism stays fixed until the final intervention is submitted. Each purchased experiment uses fresh matched simulated cultures governed by that mechanism.

At reset:

1. Sample a task from the appropriate manifest or training generator.

2. Construct the private mechanism and public target.

3. Provide one standard baseline assay, the permitted intervention targets, assay menus, public costs, and remaining experimental budget.

4. Reset the policy's recurrent memory.

At each decision:

1. The policy either buys an experiment or commits to a final intervention.

2. An experiment specifies an intervention, incubation duration, and measurement panel.

3. The simulator creates fresh cultures, applies that intervention, advances their dynamics, and returns only the purchased noisy measurements.

4. The experimental cultures are discarded. Their physical state does not carry into the next experiment.

5. Observations and remaining budget accumulate in the episode.

At commitment:

1. Freeze the submitted intervention.

2. Evaluate it on fresh cultures whose random draws are independent of every exploratory assay.

3. Compute terminal utility privately.

4. End the episode and reset memory before another mechanism.

This is sequential experimental design followed by control on fresh replicates. It is not continuous feedback control of one repeatedly measured living cell. If continuous control is added later, give it a separate environment and evaluation protocol.

All methods receive the same baseline assay. State explicitly that this initial assay is included as a fixed setup cost outside the variable exploration budget; report total assay expenditure as well as variable expenditure.

# 3 Proposed scientific system

## 3.1 Variables and roles

The flagship v1 uses eight normalized state variables:

- Four directly controllable variables.

- Two intermediate regulators or proxy readouts.

- Two outcome variables that define the goal.

Intervention is not allowed directly on the outcome variables. This prevents a shortcut in which the agent ignores the mechanism and directly forces the answer.

Represent roles with public masks. A later node-permutation evaluation should permute the complete system, actions, role masks, and observations consistently. Names should be neutral identifiers such as X0 through X7, not real gene names.

The baseline configuration may use indices 0–3 for controls, 4–5 for intermediate variables, and 6–7 for outcomes. Scientific tests may use smaller fixtures.

## 3.2 Dynamics

Let x be an eight-dimensional normalized activity vector. The hidden parameters are a signed interaction matrix W, production offsets b, and positive response rates gamma.

```math
\frac{dx_i}{dt}
=
\gamma_i\left[
(1-u_i)\sigma\left(b_i+\sum_j W_{ij}x_j\right)-x_i
\right],
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
```

An intervention u suppresses production. A dose of 0.8 means multiplying production by 0.2 during the selected exposure. It does not instantly replace the variable's state with 0.2.

For the stable v1 family require:

```math
\max_i\sum_j |W_{ij}| \le 3.2.
```

Because the logistic derivative is at most 1/4, the production map is a contraction in the infinity norm with constant at most 0.8. Its fixed point is unique for a fixed intervention. This is a deliberate restriction of the benchmark, not a claim that all biological networks are contractive.

Use rates initially sampled in [0.5, 1.5], sparse signed interactions, and a recorded distribution over motifs. Validate that permitted controls can influence the outcomes in the informative task families.

## 3.3 Baseline matching

To build mechanisms with indistinguishable unperturbed observations, choose a baseline x_base strictly inside (0, 1), then set:

```math
b=\operatorname{logit}(x_{\mathrm{base}})-Wx_{\mathrm{base}}.
```

This makes x_base the noiseless baseline equilibrium. Different W matrices can therefore share the same baseline.

Use a dedicated matched-baseline family in the benchmark. Also include a varied-baseline family, and report their results separately. Matching baselines is an intentional experimental construction, not evidence that real systems have that distribution.

Targets must be drawn from a public target distribution before private feasibility screening. Do not construct each target by revealing the output of its hidden optimal intervention.

## 3.4 Numerical integration

Use float64 CPU calculations for the reference implementation. Start with the exponential-Euler update:

```math
\alpha_i=1-e^{-\gamma_i\Delta t},
```

```math
x_i^{k+1}
=
(1-\alpha_i)x_i^k
+
\alpha_i(1-u_i)\sigma(b_i+W_i x^k).
```

Start with dt = 0.05. The update is a convex combination of values in [0,1], so it preserves bounds without clipping.

This remains a numerical approximation. Compare selected cases with dt/2 and an independently configured high-accuracy ODE solver. If differences are material relative to assay noise or target tolerance, reduce dt and regenerate affected reference results.

Do not fix an unstable or inaccurate implementation by silently clipping states. Record numerical method, dtype, timestep, tolerances, and simulator version in every result.

## 3.5 Replicate variability and measurement noise

An episode has one shared W, b, and gamma. Each fresh culture has a small, independent production-offset perturbation:

```math
b^{(r)}=b+\epsilon^{(r)},
\qquad
\epsilon^{(r)}\sim\mathcal N(0,s_{\mathrm{rep}}^2 I).
```

Start with s_rep = 0.03. Hold this perturbation fixed throughout that culture's experiment. Initialize each culture at its own unperturbed fixed point.

Use an initial cohort size of 8 cultures per purchased assay. For each measured variable, add independent readout noise with initial standard deviation 0.02. These are abstract continuous sensor readings, not simulated UMI counts.

Provide the policy and all baselines the same evidence packet: masked replicate readouts, their summaries, cohort size, panel identity, and declared readout-noise model. Do not provide the actual replicate offsets or noiseless values.

Noisy readings may slightly exceed [0,1]. Latent activities remain bounded. Avoid clipping readouts unless a clipped likelihood is explicitly modeled by every relevant inference baseline.

# 4 Experimental and final action catalogues

Use a finite action catalogue first. A continuous action policy is a later experiment.

## 4.1 Intervention menu

Permitted doses are 0.4 and 0.8. Include:

- No intervention.

- Each single control at either dose.

- Each pair of distinct controls at each allowed dose combination.

With four controls, this gives 1 + 8 + 24 = 33 interventions.

Implement the singles-only subset first for debugging. The flagship catalogue includes pairs so that interaction effects can be studied. Paired actions do not by themselves force sophisticated reasoning: the exploratory audit found that an additive response predictor can perform very well. Measure whether interactions actually change optimal decisions before making that claim.

## 4.2 Assays

Durations are 1, 3, and 6 dimensionless time units.

The four panels are:

- First intermediate regulator.

- Second intermediate regulator.

- Both outcome variables.

- All eight variables.

An experimental action chooses an intervention, duration, and panel. The full catalogue has 33 × 3 × 4 = 396 experimental actions.

The final action chooses one of the 33 interventions, applied for a fixed duration of 6. It provides no new observation before ending the episode.

A flat implementation therefore has 429 actions. Keep a stable, versioned mapping from action IDs to typed action specifications. Include interpretable action features so a later action-embedding policy can share information across related choices.

## 4.3 Costs and budget

Use the initial cost function:

```math
c(a)=1+0.2T+0.25|P|+0.25\lVert u\rVert_1.
```

Costs represent abstract setup, incubation, assay breadth, and intervention effort. They are not laboratory prices.

Store costs as integer ticks to avoid floating-point budget drift. With 20 ticks per cost unit, the proposed doses, durations, and panel sizes yield integral costs.

Initial variable budgets are 8, 16, and 24 units. Use 16 for the development configuration. Allow at most 8 purchased experiments even when a larger budget would permit more.

Actions are masked only using public information: available targets, resource costs, remaining budget, and experimental-count limits. Never use hidden effectiveness or a hidden oracle ranking to mask actions.

Commitment is always available. When no more experiments are affordable, expose only final commitments. Do not automatically submit the hidden best action.

Invalid actions need defined behavior even when masks are used: return a public error and consume a decision slot. Deduct a specified invalid-action charge up to the remaining resource budget; never make the budget negative. Preserve the hidden mechanism. If the resource charge is zero because the budget is exhausted, the decision slot still advances. At the decision cap, terminate with a declared invalid-run score, not a privileged commitment. A legal commitment always remains available.

# 5 Observations and information boundary

The policy observation contains:

- Target values and target-variable mask.

- Public control and panel masks.

- Remaining budget and experiment count.

- Initial baseline assay.

- Latest purchased replicate readouts and measurement mask.

- Latest action features and its public cost.

- A flag distinguishing reset from subsequent evidence.

- The current public action mask.

Use fixed-sized arrays with explicit masks; a missing measurement is not a zero-valued measurement.

The complete history is reconstructable from the public trace. The recurrent policy accumulates it in its hidden state. All baseline policies may access the same public history.

The policy must never receive:

- W, b, gamma, or replicate perturbations.

- Mechanism or family labels that reveal hidden structure.

- Oracle action rankings, true uncertainty, or hidden distances to the goal.

- Feasibility margins or private difficulty labels.

- Unpurchased measurements.

- Final-evaluation random seeds.

- An oracle reward before commitment.

Keep privileged fields out of observations, public info dictionaries, training features, filenames visible to an agent, and policy-side logging callbacks.

The training process may receive the terminal score to compute gradients. It must not insert that score into another exploratory observation. Reset recurrent memory when the mechanism changes.

# 6 Goal and reward definition

## 6.1 Targets

Start with outcome baselines near 0.5 and independent public target offsets, such as ±0.08. The exact distribution and tolerance must be calibrated on the training distribution.

Generate targets independently of the private best intervention. Characterize feasibility and difficulty with an offline privileged reference.

For the main informative benchmark, admit tasks where:

- At least one permitted final intervention improves meaningfully over no intervention.

- The target is attainable to the declared tolerance.

- Initial observations alone do not trivially reveal the best intervention.

Document every admission rule, rejection rate, and resulting distribution. Admission based on hidden feasibility changes the mechanism prior conditional on the public target. All Bayesian baselines must sample or reweight according to this admitted distribution, rather than the original unfiltered prior. The learned agent and planning baselines must face the same task prior. Freeze the generator before locked evaluation.

Keep separate unfiltered and unreachable-target stress suites. Do not silently remove difficult test episodes or give the agent a private attainable/unattainable flag.

## 6.2 Terminal loss

For submitted intervention u, evaluate K fresh cultures. Let G be the public set of outcome variables.

```math
L_{\mathrm{target}}
=
\frac{1}{K|G|}
\sum_{r=1}^{K}
\sum_{g\in G}
\left(x_g^{(r)}(T_f;u)-g_g^\star\right)^2.
```

Score the latent simulated outcomes, not the readout noise added to observations. This evaluates the intervention's effect rather than accidental assay noise.

Initial K proposals:

- Training: 8.

- Validation: 32.

- Locked evaluation: 128.

These are starting points. Use pilot Monte Carlo uncertainty and measured runtime to choose final values before the test run.

## 6.3 Reward

Use public experimental costs as the only intermediate reward:

```math
r_{\mathrm{experiment}}=-\lambda_c c(a).
```

At commitment:

```math
r_{\mathrm{commit}}
=
-100L_{\mathrm{target}}-\lambda_u\lVert u\rVert_1.
```

Initial coefficients: lambda_c = 0.01 and lambda_u = 0.05. The factor 100 is a fixed public scale for normalized-state squared error. These coefficients are engineering choices to calibrate on validation tasks, not biological facts.

Charge each experimental cost once. Do not subtract it again at termination.

Report raw target error, intervention effort, and experimental cost separately. A higher scalar reward can otherwise hide a worse intervention.

Use gamma = 1 for the finite budgeted objective. GAE may use a separate lambda parameter; do not confuse it with the cost coefficient.

Run a declared cost-weight sensitivity analysis. Choose the primary setting before the locked test. If always stopping or always exhausting the budget is optimal, first determine whether that follows from the intended task economics before diagnosing a learning failure.

## 6.4 Optional constrained-control family

After the core comparison works, add publicly specified acceptable bands for protected non-target variables. Report violation probabilities across fresh cultures.

Keep these as synthetic state constraints. Do not call them real toxicity or viability predictions.

If included in reward, apply the private outcome penalty only at commitment. Public action masks must still not encode hidden risk. Define whether constraints concern the final intervention, exploratory interventions, or both; changing that rule changes the task.

# 7 Calibration fixtures that must work before a large training run

## 7.1 An informative proxy assay

Implement a four-variable fixture A, B, M, Y with hidden h equal to +1 or −1 with equal probability.

Production terms:

- A and B have unperturbed production 0.5.

- M has production sigmoid(1.5 h (A − B)).

- Y has production sigmoid(3(M − 0.5)).

Use response rates 1, 1, 1.2, and 0.5. All baseline activities equal 0.5. Set replicate variation to zero for the deterministic check. Production suppression uses the same intervention semantics as the main simulator.

The following values were numerically checked during specification design using float64 exponential Euler with dt = 0.05. They are a simulator sanity check, not a trained-policy result.

| Measurement | h = +1 | h = −1 |
| - | -: | -: |
| Suppress A at dose 0.8 for time 1, measure M | 0.457045 | 0.542955 |
| Same experiment, measure Y | 0.494860 | 0.505140 |
| Suppress A at dose 0.8 for time 6, measure Y | 0.409467 | 0.590533 |
| Suppress B at dose 0.8 for time 6, measure Y | 0.590533 | 0.409467 |

Set the target for Y to 0.41. No intervention has squared error 0.0081. The appropriate intervention depends on h. A short proxy measurement reveals which choice is useful while Y has barely moved.

Test that the signal survives a declared measurement-noise level and that its expected benefit exceeds its experimental cost. Derive the posterior and Bayes decision for the two-hypothesis case.

This fixture establishes that information can improve a final decision. It does not by itself establish that an adaptive sequence beats every fixed experimental design.

## 7.2 A branching design fixture

Add a separate finite diagnostic environment to validate adaptive experiment selection and planning:

- A hidden route selects which of two assays is informative.

- A cheap routing assay reveals that route.

- The informative assay reveals a hidden binary target; the other produces an independent fair bit.

- The experimental budget permits two assays.

- The final action predicts the hidden target.

Under the balanced noiseless-informative construction, routing followed by the appropriate assay reaches perfect target identification. A predetermined two-assay design is limited to 75 percent expected accuracy. Verify this by exact enumeration.

This is a unit-level planning fixture, not the biological benchmark.

## 7.3 History-dependent training check

Use a separate delayed-evidence sequence test to verify that the recurrent implementation can integrate observations from nonadjacent steps. Check recurrent-state resets, sequence masks, and padding independently.

Do not assume a feedforward policy must fail a task merely because it is described as history-dependent: previous actions can sometimes serve as external memory. Inspect the actual observation/action interface.

# 8 Baseline suite

Build and evaluate baselines before substantial RL training.

| Baseline | Required interpretation |
| - | - |
| No experiments | Best prior-informed final decision using the common baseline assay and public target |
| Fixed schedule | A tuned nonadaptive schedule, with the same final estimator used by other design baselines; include affordable single-control identification schedules |
| Additive response model | Estimate each control's effect from fixed outcome assays and predict combinations; report its validity under nonlinear interactions |
| Random feasible schedule | A sanity baseline, with a competent final estimator |
| Direct-assay Bayesian optimization | Queries the outcome panel at final duration and optimizes the same final objective |
| Greedy expected information gain | Chooses informative experiments under an explicit posterior, with cost accounting |
| One-step decision value of information | Chooses an experiment by its expected improvement to the final decision |
| Bounded two-step belief planner | Adds limited lookahead on small instances and selected benchmark subsets |
| Full-history feedforward PPO | Receives the complete short public history with padding and masks; simpler equal-information learned baseline |
| Latest-observation PPO | Diagnostic memory ablation that intentionally discards older evidence |
| Recurrent imitation | Same recurrent architecture trained on a public-history belief planner's demonstrations; report teacher compute |
| Recurrent PPO | Main RL-trained agent |
| Imitation followed by PPO | Tests whether reward-based fine-tuning improves the same learned planner |

## 8.1 Shared final estimator

For experiment-selection comparisons, use the same posterior-based final intervention selector after fixed, random, EIG, and decision-value designs.

Then evaluate the fully learned recurrent policy, including its learned commitment choice, separately. This distinguishes better experiment selection from better interpretation of the collected evidence.

A useful diagnostic ablation replays the recurrent agent's chosen assays and lets the shared estimator make its final choice.

## 8.2 Particle belief model

For the exact finite-hypothesis diagnostic, use the complete declared prior support; it includes the possible true mechanism but never reveals its hidden index. This diagnostic does not establish generalization to unseen mechanisms. For the continuous-parameter scientific benchmark, start with a bank of plausible mechanisms drawn independently from the training prior; never insert the episode's true parameters into that bank.

Given observations, update particle weights with a declared likelihood. Integrate or approximate replicate variability consistently. Monitor effective sample size, posterior collapse, and predictive calibration.

The planner may know the generative family and prior, but must not receive the episode's actual parameters. Record this structural assumption. Forward simulations of hypothetical particles are computational work, not purchased experiments; report that compute separately.

Start with the exact two-mode posterior fixture before approximate eight-variable inference. Independent prior particles can collapse under informative observations in a high-dimensional parameter space. Keep the pilot's uncertain parameterization small, check posterior predictive coverage and effective sample size, and use justified posterior rejuvenation or another suitable inference method when necessary. A failed particle approximation is not evidence against Bayesian experimental design. Tune inference on validation tasks.

## 8.3 Decision value of information

For each affordable experiment:

1. Draw possible outcomes from the posterior predictive distribution.

2. Form the corresponding hypothetical posterior.

3. Select the best final intervention under that posterior.

4. Estimate the resulting utility.

5. Subtract the experiment's resource penalty.

Choose the experiment with the highest expected downstream value, or commit if further experimentation is not worthwhile. Replan after every purchased observation.

Keep Monte Carlo sample counts and candidate screening budgets explicit. A numerical approximation is not an exact optimal policy. Likewise, describe a fixed schedule as optimal only when its relevant search space and expectation have been solved accurately; otherwise say tuned fixed schedule and document the search budget.

## 8.4 Bayesian optimization fairness

A direct-assay BO baseline may use the outcome panel at the final deployment duration, which produces evidence about the actual objective. It must pay the same cost.

With noisy replicate readouts, estimate the target error using the declared measurement model; do not give BO noiseless latent scores. When correcting known additive measurement variance, retain and explain the resulting noisy estimate rather than silently clipping it into a different objective.

If a later multi-fidelity BO method uses proxy panels and shorter durations, model the relation between those measurements and the final objective. Do not treat a cheap proxy readout as though it were the final objective.

## 8.5 Literature baselines

Deep Adaptive Design and Step-DAD are relevant learned experimental-design methods. A generic EIG heuristic is not a reproduction of either.

For a research-oriented release, implement an appropriately matched version of one of these methods, or document a precise comparison scope and why the explicit decision-value planner is the relevant comparator. Preserve their different objectives and adaptation assumptions.

# 9 Primary learning method

## 9.1 Recurrent meta-RL

Use a small actor–critic with:

- An observation encoder.

- One GRU or GRUCell with an initial hidden size of 128.

- A categorical policy over the finite action catalogue.

- A value head.

Start with a simple flat action head. An action-embedding head or factorized policy is an optional ablation after the baseline works.

The policy updates its recurrent state after each experimental observation. Model weights remain frozen while adapting to a held-out mechanism. Reset memory at the beginning of each new mechanism.

Use this same recurrent architecture for imitation, PPO, and imitation followed by PPO. The demonstration teacher may use public histories and the declared prior, but not the current hidden mechanism. Treat privileged-teacher supervision as a separate setting. A fixed-schedule, history-conditioned supervised final selector is another useful cheap baseline.

This is recurrent meta-RL: slow training across tasks learns a fast, history-based adaptation strategy. It does not require a language model.

## 9.2 PPO implementation contract

First establish a full-history, padded-observation PPO baseline using a tested masked-discrete implementation where possible. A short experiment history can be represented explicitly, avoiding a custom recurrent training loop during the first learning check. Latest-observation-only PPO is a weaker memory ablation, not the main equal-information comparator.

For the recurrent model, prefer a small audited implementation over a large new training framework. Verify any chosen library's support for recurrence, discrete action masking, and MPS rather than assuming separately supported features can be combined. Current [SB3-contrib MaskablePPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) lists recurrent policies as unsupported; its [RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html) is separate. Do not assume combining the two is a supported configuration.

Because the experiment horizon is short, initially collect complete episodes and train on padded whole-episode sequences. This simplifies recurrent-state reconstruction.

Required details:

- Store observation masks, action masks, actions, old log probabilities, values, rewards, and termination flags.

- Reset memory with the exact episode boundaries.

- Recompute contiguous sequences during updates.

- Give padded timesteps a valid dummy action before constructing distributions so all-invalid masks cannot create NaNs. Exclude padding from policy, value, entropy, and advantage calculations.

- Use the same action-mask semantics when computing old and new probabilities.

- Stop-value bootstrapping at true commitment.

- Handle externally truncated runs separately.

- Clip gradients and record nonfinite losses.

- Log entropy, policy divergence, value error, gradient norms, experiment counts, and commitment timing.

- Keep normalizers fitted to training data only.

Do not randomly shuffle individual recurrent transitions. Do not assume checkpoints containing only weights can exactly resume a partially collected rollout.

## 9.3 Curriculum

First learn the diagnostic fixtures, then a restricted scientific family, then the full training distribution.

Keep a persistent fraction of easier tasks to detect regressions. Freeze a separate validation set. Do not use locked test failures to choose the curriculum.

If the curriculum itself becomes a learned component, treat that as a later research question with its own baselines and data boundary.

## 9.4 Optional belief-model agent

After recurrent PPO is working, add an explicit latent-inference or dynamics-ensemble policy.

Train prediction using visible experiment histories. Evaluate predictive calibration and decision quality separately. Privileged supervision on W or hidden parameters must be labeled as a different training setting and excluded from the main comparison unless it is a declared research choice.

# 10 Apple Silicon execution plan

The user has an M3 Max MacBook Pro, but installed memory and OS version are unknown. Inspect the actual machine before choosing throughput settings.

Use:

- Native arm64 Python.

- NumPy/SciPy float64 on CPU for reference simulation and scoring.

- PyTorch float32 for policy/value networks.

- CPU and MPS training paths.

- Native local execution as the reference workflow.

PyTorch provides an MPS device for Apple GPU execution. MPS does not support float64 tensors; computations requiring double precision stay on CPU. See the official [MPS documentation](https://docs.pytorch.org/docs/main/notes/mps.html#double-precision-types).

Do not make CUDA, NVIDIA-only libraries, LLM fine-tuning, or a cloud GPU prerequisites for the core project.

## 10.1 Device selection

Support explicit cpu, mps, and auto modes.

The auto mode must run the actual workload benchmark:

- Simulator stepping.

- Observation transfer.

- Recurrent forward pass.

- Masked action sampling.

- Full PPO loss and backward pass.

- Gradient clipping and optimizer update.

- Evaluation and checkpoint save/load.

Time complete work, including synchronization and transfers. Select MPS only if the required operations work and measured end-to-end performance justifies it.

CPU is a fully supported operating mode. Unsupported MPS operations should produce a clear diagnostic. Do not silently enable CPU fallback and describe the result as pure GPU training. See [MPS environment variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html).

## 10.2 Parallelism

Start with synchronous environments and vectorize across cultures. Add process workers only after profiling.

On macOS, use spawn, top-level picklable worker functions, and a guarded main entry point. Keep the policy and MPS context in the parent; simulator workers stay on CPU. Limit numerical-library threads to avoid oversubscription. These choices follow Python's [multiprocessing guidance](https://docs.python.org/3/library/multiprocessing.html).

The number of environments is not the number of OS processes.

## 10.3 Proposed local presets

| Preset | Starting configuration | Meaning |
| - | - | - |
| Smoke | CPU, 1–2 environments, hidden size 64, about 2,000 transitions | Checks execution and finite losses; not evidence of learning |
| Development | CPU/MPS benchmark-selected, about 8 environments, hidden size 128, about 100,000 transitions as an initial cap | Profiles learning behavior and runtime |
| Research | Reuse the validated model, 3–5 seeds run sequentially, larger explicit transition budget chosen from pilot measurements | Produces the main comparison |

These are starting configurations, not predicted training durations or memory guarantees. Measure peak memory, transitions per second, episode throughput, and time spent in baselines versus training.

A short empirical pilot should estimate the cost of a full run. Thermal throttling, context switching, and unified-memory pressure can make long runs differ from a brief microbenchmark.

## 10.4 Planning cost and caching

One environment transition is a complete experiment. At T = 6 and dt = 0.05, each assay takes 120 batched integration iterations. Eight cultures correspond to 960 culture-state updates, plus equilibrium initialization.

Naively predicting all 396 assays for 256 particles and 8 predictive cultures requires about 54 million culture-state updates per planning decision before nested value-of-information calculations. This arithmetic is a workload estimate, not measured runtime. The planner can dominate the small neural network's cost.

For a given hypothetical culture and intervention, integrate once to the longest duration and retain endpoints at all shorter durations. Measurement panels select coordinates from those endpoints and should not repeat the dynamics solve. Cache reusable particle predictions without giving them access to actual unpurchased measurements. With 33 interventions, this reduces that endpoint-precomputation example to about 8.1 million culture-state updates.

Begin with the 45-action scientific pilot, small exact hypotheses or validated inference, and a declared subset of planning candidates. Limit Monte Carlo outcomes and lookahead depth explicitly. Record how candidate pruning affects decisions on cases where exhaustive calculation is possible. Report hypothetical forward-model calls as well as wall time.

Do not precompute candidate results using the episode's true mechanism and expose them to a baseline. Only the private full-information evaluator may do that.

# 11 Repository architecture

| Path | Responsibility |
| - | - |
| `src/experimentlab/worlds/` | Hidden mechanisms, dynamics, cohorts, assays, reference integration |
| `src/experimentlab/tasks/` | Public task definitions, private generation, manifests, difficulty diagnostics |
| `src/experimentlab/envs/` | Budgets, action catalogues, observations, episode lifecycle |
| `src/experimentlab/inference/` | Particle beliefs, likelihoods, prediction, posterior checks |
| `src/experimentlab/baselines/` | Fixed/random designs, BO, EIG, decision-value planners |
| `src/experimentlab/policies/` | Feedforward and recurrent actor–critic models |
| `src/experimentlab/training/` | Rollouts, PPO, curriculum, checkpoints |
| `src/experimentlab/evaluation/` | Frozen suites, private scoring, paired statistics, ablations |
| `src/experimentlab/records/` | Public trajectories, private evaluator records, provenance |
| `src/experimentlab/cli/` | Doctor, benchmark, baseline, train, evaluate, replay, report |
| `configs/` | Versioned smoke, development, research, and baseline configurations |
| `tests/` | Scientific, information-boundary, and training-mechanics checks |
| `docs/` | Scientific assumptions, benchmark protocol, methods, limitations |
| `reports/` | Generated results and figures from recorded runs |

Keep dependencies small. A reasonable initial stack is Python, NumPy, SciPy, PyTorch, Gymnasium, a configuration library or dataclasses, pytest, and Matplotlib. Add libraries only for an implemented need.

Pin a compatible dependency set after testing on the Mac. Record the resolved lock and hardware report with experiments.

# 12 Data structures and interfaces

Use typed, immutable task/action records where practical.

```python
@dataclass(frozen=True)
class Experiment:
    intervention: tuple[float, ...]
    duration: float
    panel: tuple[int, ...]

@dataclass(frozen=True)
class Commitment:
    intervention: tuple[float, ...]

@dataclass(frozen=True)
class PublicTask:
    target: tuple[float, ...]
    goal_mask: tuple[bool, ...]
    control_mask: tuple[bool, ...]
    budget_ticks: int
    max_experiments: int
    # Public costs, panel definitions, and measurement assumptions.

@dataclass(frozen=True)
class EvidencePacket:
    readouts: object
    measured_mask: object
    experiment: Experiment
    cohort_size: int
    cost_ticks: int
    # Only experimentally observable information.
```

Conceptual interfaces:

```python
world = World.sample(private_task_spec, mechanism_rng)
evidence = world.measure(experiment, experiment_rng)
obs, public_info = env.reset(seed=seed, options=public_options)
obs, reward, terminated, truncated, public_info = env.step(action_id)
action, memory = policy.act(obs, memory, action_mask)
metrics = evaluator.score(world, commitment, independent_evaluation_rng)
```

The private task specification is never passed to the policy. Keep the policy-facing API separate from evaluator-only diagnostics.

Expected commands are design requirements to implement, not commands that already exist:

```bash
experimentlab doctor
experimentlab validate-science --suite diagnostic
experimentlab feasibility --config configs/pilot.yaml
experimentlab benchmark --config configs/development.yaml
experimentlab baselines --suite validation --config configs/baselines.yaml
experimentlab train --config configs/development.yaml
experimentlab evaluate --checkpoint runs/selected/checkpoint.pt --suite test-id
experimentlab replay --episode runs/evaluation/episode_001.json
experimentlab report --run-directory runs/evaluation
```

# 13 Initial configuration contract

The exact schema may be refined before implementation, but preserve the meaning of these fields.

```yaml
project:
  name: experimental-scientist
  schema_version: 1

world:
  n_variables: 8
  n_controls: 4
  n_intermediates: 2
  n_outcomes: 2
  max_row_l1: 3.2
  rate_range: [0.5, 1.5]
  integration_dt: 0.05
  dtype: float64
  replicate_offset_sd: 0.03
  measurement_sd: 0.02
  assay_replicates: 8

actions:
  doses: [0.4, 0.8]
  max_controlled_variables: 2
  durations: [1.0, 3.0, 6.0]
  final_duration: 6.0
  include_no_intervention: true

episode:
  exploration_budget_units: 16
  ticks_per_unit: 20
  max_experiments: 8
  include_common_baseline_assay: true

reward:
  target_loss_scale: 100.0
  experiment_cost_weight: 0.01
  intervention_l1_weight: 0.05
  train_final_replicates: 8
  validation_final_replicates: 32
  test_final_replicates: 128

policy:
  kind: recurrent_ppo
  hidden_size: 128
  dtype: float32
  device: auto

training:
  gamma: 1.0
  environments: 8
  collect_complete_episodes: true
  initial_transition_cap: 100000
  seed: 0

evaluation:
  tune_on: validation
  locked_test: true
  paired_mechanisms: true
  bootstrap_unit: mechanism
  report_raw_target_error: true
  report_online_compute: true
```

The implementation must validate all configurations. Do not treat these initial values as tuned hyperparameters.

# 14 Split design and generalization

Create explicit manifests before training. Separate train, validation, and test by underlying mechanism groups.

At minimum include:

1. New parameterized mechanisms within familiar families.

2. Held-out graph structures or withheld motif compositions.

3. Held-out parameter ranges.

4. Observation shifts such as changed readout noise.

5. A declared simulator-misspecification suite introduced after v1 is stable.

A new observation-noise seed is not a new mechanism. A relabeling of the same graph is not an independent structural holdout.

Group topology-preserving permutations and closely related generated variants together. Use stable generator parent IDs and an appropriate canonical graph representation for split checks. Do not expose those group IDs to the policy.

For each test suite, state exactly what changed. Keep costs, action meanings, and other factors fixed when testing a specific scientific shift. A changed-cost suite is a different experiment.

Characterize feasibility and difficulty privately, using the same criteria across methods. Report difficulty strata and rejection rates. Avoid an apparent generalization failure that is entirely due to more unreachable targets.

All hyperparameter selection, reward coefficients, stopping rules, and baseline tuning use train and validation. Freeze them before the test run.

# 15 Evaluation and statistical reporting

Primary outcome: final target error versus experimental budget on held-out mechanisms.

Secondary outcomes:

- Experimental cost and number of assays.

- Final intervention effort.

- Success within a predeclared target tolerance.

- Optional state-constraint violations.

- Online planning time and peak memory.

- Training interactions and wall time.

- Performance under each declared shift.

- Sensitivity to readout noise and replicate variability.

Use the same hidden mechanisms and goals across policies. Final replicate draws must be fresh relative to exploratory assays. They may be paired across policies to reduce comparison variance; record this common-random-number design.

Aggregate at the mechanism level. Bootstrap mechanisms rather than treating every culture as an independent task. Account for multiple goals or replicate evaluation runs nested within a mechanism.

Train at least three independent seeds for the main learned-method comparison; five is preferable if measured runtime permits. Select test-set size from pilot uncertainty before looking at locked results.

Report both raw outcomes and differences with confidence intervals. Do not infer that an observed improvement is robust from a single seed or a single favorable mechanism.

The privileged reference enumerates the permitted finite final-intervention menu using access to the mechanism. With finite Monte Carlo evaluation it is an approximate full-information reference. Do not call it a rigorous upper bound unless its optimization and expected utilities are exact. Do not silently clip negative estimated regret.

A useful result may be comparable final quality with much lower online planning cost. Keep experimental cost, inference compute, and offline training compute separate.

# 16 Required ablations

Prioritize:

- Recurrent memory versus no recurrence.

- Adaptive versus fixed experiment schedules, including fixed single-control identification and additive response models.

- Imitation versus PPO versus imitation followed by PPO with the same architecture.

- Decision-value design versus information-gain design.

- Proxy panels versus outcome-only assays.

- Full training diversity versus restricted diversity.

- Learned commitment choice versus a shared posterior-based selector.

- Single interventions versus combined interventions.

- Cost penalties and stopping behavior.

Each ablation should answer a specific question. Do not expand the matrix until baseline reliability and primary evaluation are established.

# 17 Reproducibility and checkpoints

Use independent random streams for:

- Mechanism generation.

- Initial baseline assays.

- Exploratory cultures.

- Measurement noise.

- Policy action sampling.

- PPO minibatches.

- Final evaluation cultures.

Derive streams from stable task/episode identifiers, not worker scheduling order. NumPy's [SeedSequence guidance](https://numpy.org/doc/stable/reference/random/parallel.html) provides a suitable pattern.

Save:

- Model and optimizer states.

- Normalizers.

- Random generator states.

- Training counters and curriculum state.

- Recurrent and environment state when resuming partial episodes.

- Resolved configuration.

- Code revision and dependency lock.

- Simulator, action-catalogue, and manifest hashes.

- Hardware and device benchmark results.

Test interruption and resume on the CPU reference path. PyTorch does not guarantee identical results across platforms, devices, or releases; document reproducibility within a declared configuration rather than promising bitwise CPU/MPS identity. See [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

# 18 Meaningful validation gates

Scientific gates:

- Bounded trajectories without post-hoc clipping.

- Fixed-point convergence.

- Correct zero-dose and production-suppression behavior.

- Timestep refinement and independent ODE comparison.

- Diagnostic-pair numerical values.

- Positive value of information after experimental cost.

- Branching fixture's adaptive advantage by exact enumeration.

- Adaptive advantage inside the scientific simulator against the strong fixed-design baselines, before scaling RL.

- Consistent behavior under node permutation.

- Independent exploratory and final random streams.

Environment gates:

- Exact budget accounting.

- Masks use only public constraints.

- Commitment remains possible at budget exhaustion.

- Invalid actions cannot produce an endless free loop.

- Correct terminated versus truncated semantics.

- No privileged fields in public observations or trace features.

Learning gates:

- Recurrent states reset at the correct boundaries.

- Padding and action masks are respected in losses.

- Whole-episode log-probability recomputation is consistent.

- Finite gradients and stable updates on diagnostic fixtures.

- Checkpoint resume produces the expected continuation on the reference path.

Comparison gates:

- Identical evidence access and experimental budgets.

- Baseline posterior predictive checks.

- Fair treatment of BO's noisy objective.

- Disjoint mechanism groups and locked test configuration.

- No reporting of simulator results as real biological findings.

# 19 Build milestones and acceptance criteria

## Milestone 0 Define and profile

Create the independent repository, package skeleton, specification, data contracts, configuration validation, and doctor command.

Acceptance: reproducible native installation, recorded hardware/dependencies, and a runnable empty CLI. Inspect actual RAM and macOS compatibility. Establish a bounded pilot runtime and output budget. Do not begin a large training run yet.

## Milestone 1 Validate diagnostic environments

Implement the two-mechanism scientific fixture, the branching design fixture, exact small-case baselines, and public/private data separation.

Acceptance: numerical checks and exact-enumeration comparisons pass. A purchased informative assay improves the final decision after cost. An adaptive schedule can improve on a fixed schedule in the branching fixture.

## Milestone 2 Build and assess the restricted scientific pilot

Implement stable dynamics, a restricted eight-variable motif family, fresh cultures, the 45-action pilot catalogue, costs, commitments, and reference scoring. Implement the cheap fixed and small-belief baselines needed to test feasibility.

Acceptance: scientific and environment gates pass, seeded episodes replay, and Section 0's information-value and scientific-adaptivity gates are assessed with measured results. If they fail, write the failure report and revise the smallest justified assumption on development data. Do not expand automatically to the 429-action catalogue.

## Milestone 3 Establish serious baselines

Implement no-experiment, fixed/random, additive response, BO, EIG, and decision-value methods, then the same-architecture imitation baseline. Add bounded two-step planning on small cases only when it addresses a demonstrated planning limitation. Expand the action catalogue only when the pilot supports the added complexity.

Acceptance: produce an actual baseline table, posterior predictive checks, teacher-compute accounting, and an empirical full-run cost estimate. If the task is uninformative or solved cheaply by fixed experiments, preserve that finding and revise the scientific claim before scaling RL.

## Milestone 4 Train the recurrent agent

Implement feedforward and recurrent PPO, sequence-level tests, curriculum, checkpoints, and validation. Compare imitation, PPO, and imitation followed by PPO.

Acceptance: the diagnostic learning checks work; training does not show obvious numerical failure or reward leakage; validation is evaluated against strong baselines. Improvement on every scientific setting is not an acceptance requirement.

## Milestone 5 Optimize for the Mac

Benchmark CPU and MPS end to end. Vectorize cohorts, profile the bottleneck, and add spawn workers only if useful.

Acceptance: device selection is measured, peak memory is recorded, and a bounded research run has an empirical cost estimate.

## Milestone 6 Run the locked comparison

Freeze versions, manifests, metrics, and hyperparameters. Train multiple seeds and evaluate paired mechanisms.

Acceptance: reproducible tables, confidence intervals, raw metrics, failure examples, and the full accounting of training and online compute.

## Milestone 7 Present the result

Create a replay interface, scientific figures, and a concise technical report.

Acceptance: another person can install the project, run a small benchmark, inspect an experimental sequence, and reproduce a reported result from the supplied configuration.

# 20 Demonstration and reporting

The replay should show:

- The public target and budget.

- Each chosen intervention, duration, panel, and cost.

- Purchased measurements with uncertainty.

- Remaining resources.

- The final commitment and private evaluation after the episode ends.

Show the true mechanism only in a clearly labeled post-episode evaluator view. If displaying an inferred mechanism or uncertainty, derive it from an actual fitted model. Do not draw an invented inferred graph for the recurrent policy.

Recommended figures:

1. Target error versus experimental cost.

2. Policy comparison on held-out mechanism groups.

3. Online planning time versus decision quality.

4. Proxy-assay selection and stopping behavior.

5. Ablations of memory and adaptive design.

6. Example histories for a success and a failure.

The report should state the hypothesis, scientific assumptions, methods, experimental protocol, results, uncertainty, failure analysis, and compute. Avoid claims of novel causal discovery unless the evaluation actually establishes them.

# 21 Later research extensions

Add one extension after the main comparison is complete:

- Explicit latent-belief inference and world-model planning.

- Transfer to new activation functions or controlled delay dynamics.

- Continuous intervention strength or experimental timing.

- A second physical system inspired by diffusion or source reconstruction.

- A restricted retrospective study using public perturbation data.

Real observational or perturbation data do not automatically provide outcomes for arbitrary sequences of unobserved interventions. Define a supported evaluation protocol before introducing a real-data adapter.

A physics extension should be a new environment sharing the experiment/commit interface. Treat cross-domain transfer as an experiment, not an assumption.

# 22 Literature and implementation references

These sources establish relevant methods and platform behavior. The specific simulator, costs, task construction, and proposed research hypothesis above are design choices.

- [Deep Adaptive Design](https://proceedings.mlr.press/v139/foster21a.html): learned sequential experimental design.

- [Step-DAD](https://proceedings.mlr.press/v267/hedman25a.html): semi-amortized experimental design with instance-specific adaptation.

- [RL squared](https://arxiv.org/abs/1611.02779): recurrent policies that adapt through their hidden state.

- [VariBAD](https://www.jmlr.org/papers/v22/21-0657.html): explicit uncertainty inference in meta-RL.

- [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html): Apple GPU backend.

- [MPS double precision limitation](https://docs.pytorch.org/docs/main/notes/mps.html#double-precision-types).

- [MPS environment variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html).

- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

- [Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html).

- [NumPy random streams](https://numpy.org/doc/stable/reference/random/parallel.html).

- [SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

- [SB3-contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) and [RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html): verify supported interfaces before choosing an implementation.

# 23 Codex starting instruction

```codex-prompt
Build a new standalone repository implementing this Autonomous Experimental Scientist specification. My development machine is a MacBook Pro with an M3 Max. Inspect its actual RAM, macOS version, Python architecture, and PyTorch CPU/MPS support; do not assume CUDA or a particular memory capacity.

The scientific objective is to learn cost-aware experiments on unfamiliar hidden regulatory mechanisms and select a final intervention using the evidence collected. Use fresh matched cultures for every assay, preserve the private evaluator boundary, and score the final intervention on independent cultures.

The feasibility contract in Section 0 takes precedence over the flagship feature list. Begin with the package and configuration contracts, two diagnostic fixtures, exact small-case baselines, and the CPU reference simulator. Then implement only the restricted 45-action scientific pilot and the baselines needed to test it. Prove that experiments help and that adapting the scientific experiment sequence improves over strong fixed schedules before expanding the action space or starting long training.

Use NumPy/SciPy float64 for scientific reference calculations and a small PyTorch float32 recurrent actor–critic for the learned policy. Compare CPU and MPS on the actual rollout-and-update workload. Provide a complete CPU path.

Build serious baselines: prior-informed commitment; fixed single-control assays; an additive response model; validated belief-based decision-value planning; a fair direct-assay Bayesian optimization baseline when the pilot warrants it; and same-architecture imitation. Compare imitation, PPO, and imitation followed by PPO. Do not call faster learned inference an RL-specific benefit without this comparison. Keep training, validation, and test mechanisms separate. Compare learned and model-based decisions at matched experimental budgets, and report computation separately.

Implement the specified reward and observation boundaries. Do not leak hidden mechanisms, oracle utility, unpurchased measurements, or final-evaluation noise to the policy. Do not replace the scientific model with an arbitrary action-to-reward lookup table.

Make the implementation decisions explicit in docs. If an assumption makes the task trivial, unidentifiable, numerically unreliable, or too expensive, demonstrate the problem and revise the smallest relevant component before scaling. Record every revision that changes the benchmark.

The first deliverable is a measured feasibility report stating which of information value, adaptive value, and RL value has actually been established. Record workload and memory measurements on my Mac and forecast full-run cost before a long experiment. If the scientific adaptive advantage is absent, preserve the negative result and revise the smallest scientifically justified assumption; never weaken a baseline to manufacture a win.

After each milestone, report what was implemented, the meaningful checks run, actual results, unresolved issues, and the next milestone. Produce runnable code and measured experiments, not only architecture or a dashboard. Do not fabricate training outcomes or promise real biological validity.
```
