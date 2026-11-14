# M2 synthetic evaluation evidence card

This collection records agents buying noisy measurements and committing to a final action in a numerical simulator. Its units are synthetic systems and virtual assay costs. The simulator and evaluator retain the realized mechanism and final evaluation randomness. Hosted requests contain public observations, legal actions, and role-specific advice or a declared prior.

| Partition | Purpose | Size and pairing |
|---|---|---|
| PPO training pool | Shared mechanism family for the registered race | 4,096 generated mechanisms, with simulator interactions charged to a separate training ledger |
| PPO tuning | Select architecture, update recipe and checkpoint | 128 mechanisms, with two screening seeds and five retained seeds for the selected cohorts |
| PPO development confirmation | Compare the frozen checkpoints and fixed/planner baselines | 128 fresh mechanisms, five retained learned seeds, 5,632 recorded episodes |
| M2 engineering fixture | Exercise all controller roles, receipts and replay with deterministic fake API answers | One mechanism, five PPO seeds and 75 cases |
| M2 hosted development | Compare pinned Jev roles and local baselines on paired synthetic conditions | Eight mechanisms and five PPO seeds. The first mechanism overlaps the disclosed engineering fixture. Jev-only and the fixed probe run once per mechanism. |
| Future final | Test a separately admitted design | Unopened. The published results cover development and the separately identified historical v6 evaluation. |

See the frozen protocol and registration for split seeds, task digests, the fixed probe schedule, and allowed comparisons. [M2 protocol](jev_rl_m2_protocol.md) · [Hosted roles](jev_rl_m2_hosted_protocol.md) · [Analysis registration](jev_rl_m2_analysis_registration.json)

| Public file | Unit and fields |
|---|---|
| `plan.json`, `manifest.json` | Frozen case order, mechanism digests, training seeds, checkpoint identities, scoring conditions, hosted limits and runtime hashes |
| `episodes/<case>.jsonl` | Ordered and hash-linked observations, legal proposals, forecasts, committed actions, simulator results and API receipts |
| `episodes/<case>.json`, `results.json` | Case outcomes and attempt status, return, target MSE, assay costs, effort, failure/stop reasons, summary and API accounting |
| `trace-index.json` | Provider request IDs and local spans joined to study cases |
| `summary.json` in an analysis folder | Paired effects and intervals, forecast coverage, spending, latency, missingness bounds and source hashes |
| `hosted-recorded-action-tasks.json` | Eight synthetic development tasks with the exact parameter hashes from the hosted plan. Used by the evaluator to regenerate outcomes from recorded actions. |
| `publication-export.json` | Per-file original/public SHA-256 values and the transformations applied to the public derivative |

Success, failure and skipped cases stay in the planned denominator. Terminal outcomes can be missing. A forecast probability applies to its defined future measured event. The same-event result uses the fixed probe, which supplies up to six forecast events per mechanism. Those events are clustered by mechanism for uncertainty. [Prospective analysis registration](jev_rl_m2_analysis_registration.json)

Known machine paths in metadata are normalized for release, and the exporter rebuilds the affected public hashes. Requests, responses, observations and numerical outcomes are compared with the original evidence. Published traces support exact policy-decision replay in the reference environment and replay of recorded simulator outcomes on other verified platforms. The release carries checkpoint metadata and hashes for inference that used original weights. [Export receipt](../reports/jev-rl-m2/publication-export.json) · [Reproduction guide](m2-reproduction.md)

The collection supports research on constrained decisions in this simulator. Performance on new mechanism structures or biological experiments is outside its coverage. The repository currently specifies no reuse license for the evidence. [Hugging Face dataset card guidance](https://huggingface.co/docs/hub/datasets-cards)
