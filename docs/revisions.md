# Development revisions and preserved results

No locked-test outcomes have been evaluated or used to select these changes.

## v1: admitted, matched-baseline pilot

The original 45-action pilot samples independent public targets and conditions on nominal attainability. It uses four control signs and one output-route sign, plus a separate continuous gain/rate variation check. Strong fixed schedules solve the primary low-noise setting. That result is preserved in reports/feasibility-v1. The diagnostic fixture's single assay and the generous-budget four-control audit are also preserved as negative controls for RL necessity.

## v2: stronger inference and baselines

Increase calibration from24 to128 distinct continuous mechanisms, evaluate128 separate mechanisms, cover all32 sign modes with128 gain/rate quadrature hypotheses, and add a quality-focused fixed design. First measure cache construction and planning cost:32 hypotheses require34,560 culture steps and about0.0047s;128 require138,240 steps and about0.0152s. Eight first decisions cost about0.134s versus1.628s. Full128-particle VOI episodes cost about0.487s, so planning remains the main compute cost.

Fixed utility and quality tuning examine1,837 and1,819 schedules. The cost-tuned design attains RMSE0.03442, the quality-focused design0.03402, and adaptive VOI0.03320. Against the cost-tuned design, adaptive return difference is+0.01533 with95% CI[-0.01275,+0.05198]; raw error-gap recovery is1.5%. Strong fixed designs already recover approximately95–96% of the prior-to-reference raw error gap. Scientific adaptivity remains unqualified. These results are preserved in reports/research-v2.

Larger supervised planning runs use512 public-history demonstrations,1,000 epochs and3 seeds each for full-history and recurrent models. This is an amortized-planning study, not an RL-specific result; results and teacher cost are recorded in reports/amortization-v2. Main scientific PPO remains gated for this admitted task.

## v3: retain unreachable targets and remove admission conditioning

Diagnosis: in the admitted prior, if the route says a target depends on one control pair and a measured control has the wrong sign, nominal attainability can force the unmeasured sibling to have the right sign. A measurement of the sibling is then unnecessary. This is legitimate prior inference, but makes the benchmark easier than an experimental setting where attainability is unknown.

The development revision keeps **every generated mechanism/target pair**, including unreachable targets. It removes the attainable-task condition from both generation and the belief prior. It preserves dynamics, gain/rate ranges, readout SD0.02, targets0.5+/-0.08, all45 actions, budget8, assay/intervention penalties, final estimator, strong fixed search and independent final cohorts. No hidden attainable/unattainable flag reaches any policy. Original admitted results remain separate.

The revised study declares128 calibration and128 validation mechanisms before measurement, evaluates both cost-tuned and quality-focused fixed schedules, and uses the same strict raw-error/utility confidence-interval and10% gap screen. Exact methods and script snapshots are in reports/unfiltered-v3. Larger three-seed learning runs may proceed only if this revised scientific screen passes. Unreachable cases remain in every table; no favorable subset is selected for the main comparison.

This is a change to the synthetic task prior, not evidence of real biological validity. It must eventually receive its own frozen locked evaluation and independent structural/range checks. Development success cannot replace those tests.
