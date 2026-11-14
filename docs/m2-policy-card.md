# M2 experimental policy card

The selected policy is a 512-unit GRU actor–critic trained with proximal policy optimization (PPO). Five checkpoints, one for each training seed from 20 through 24, represent the selected configuration. The model chooses among a masked set of abstract simulator actions, including committing to a final decision.

| Property | Specification |
|---|---|
| Inputs | Target, initial and purchased observations, action costs, budget and legal action mask. The policy uses at most five public observation packets. |
| Outputs | Masked logits over 36 experiment actions and nine final decisions, plus a value estimate used during training. |
| Training | In the selector phase, public randomized assays provide histories for learning the terminal decision from simulator rewards. Joint training then learns the full policy. Forced actions contribute to critic training without contributing to the actor loss. |
| Selection | An architecture and update screen on seeds 20 and 21, followed by all five seeds for the selected configuration. Tuning uses original return, with target MSE and assay cost as tie-breakers. |
| Selected continuation recipe | CPU, batch 512, Adam learning rate 0.0003, two epochs per PPO update. The per-seed checkpoint records identify the actual selected training step and any inherited training. |
| Comparator policies | Frozen v6 PPO, strong imitation, calibrated imitation and imitation followed by PPO. These retain their own architectures and training histories. |
| Confirmation | 128 fresh mechanisms for each learned seed, with each seed-independent comparator evaluated once per mechanism. |
| Checkpoint identity | The frozen cohort lists seed, original checkpoint hash, architecture and calibration. The public artifact supplies recorded advice and action traces for replay. |

The five-seed development confirmation estimated a return difference of +0.00630 for selected PPO minus v6, with a paired 95% interval of [−0.00170, +0.02062]. Imitation and Bayesian value-of-information planning achieved higher mean return at lower assay cost. The capacity and training differences between PPO and imitation limit conclusions about the training algorithm. [Confirmation freeze](../reports/jev-rl-m2/confirmation-freeze-02/freeze.json) · [Measured comparison](../reports/jev-rl-m2/summary-01/COMPARISONS.md)

The intended use is research on budgeted decisions in this numerical simulator. Performance estimates cover familiar mechanism families and the declared noise, action and cost settings. Transfer to other families, real measurements or real interventions requires separate evidence. [Scientific protocol](protocol.md) · [M2 protocol](jev_rl_m2_protocol.md)

Jev is a separate hosted component, pinned to `jev-1.14.0` in live comparisons. Jev receives legal actions and public observations. Its forecasting and decision roles use typed Noul and Choice outputs, respectively. The local PPO weights are trained on simulator rewards. [Hosted roles and controls](jev_rl_m2_hosted_protocol.md) · [TypeSafe response types](https://docs.typesafe.ai/sdk/python/api/types/responses)

All 1,497 recorded policy steps across 520 hosted-study cases matched the original local checkpoint probabilities and argmax exactly in the reference CPU environment. The public repository supplies code, configuration, checkpoint identities, replay artifacts and that audit receipt. Inference from the original trained parameters requires the saved checkpoints. The repository currently specifies no reuse license for the code, checkpoints or evidence. [Checkpoint audit](../reports/jev-rl-m2/hosted-checkpoint-advice-audit.json) · [Reproduction guide](m2-reproduction.md)
