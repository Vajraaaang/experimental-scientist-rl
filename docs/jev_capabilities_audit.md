# Jev tracing and capability audit

Checked online on 2026-11-13 and rechecked on 2026-11-14, using TypeSafe's official
documentation, OpenAPI and linked SDK source. The implemented scope is the bounded
synthetic research loop.
"Best" is not an experimentally established ranking of tracing implementations.

## Implemented and locally verified

| Capability | What the research integration records or enforces |
|---|---|
| Provider request correlation | Optional `x-typesafe-request-id` is retained on success, HTTP errors and invalid-response paths. It is distinct from our locally generated trace/span IDs. |
| Call-level traces | Study trace ID, unique call span, UTC start/finish, monotonic elapsed time, request/validated-response hashes, call number, actual/proposed request bytes, validation outcome, model and usage. |
| Transport diagnostics | HTTP status and fixed auth/rate-limit/overload/server/timeout codes; bounded numeric retry hints; bounded response-body digest/size and completeness when bytes were read. Missing metadata stays missing. |
| Scientific event linkage | Forecast freezing, action commitment and later Brier scoring point to the original call span. Hash-linked journals preserve pre-observation order; `trace-index.json` joins calls to episodes and methods. |
| Full Choice output | Candidate mapping and actual canonical wire order, selected action, all option probabilities and unchanged native confidence. Nonfinite, missing, illegal, nonnormalized or nonmaximal selections fail closed. |
| Prospective Noul events | Explicit future measured-mean event, observed baseline threshold, public evidence cutoff, cohort size, assay duration, probability, and later outcome/Brier score. Noul does not get mislabeled as action confidence. |
| Model reproducibility | Requested version pinned to `jev-1.14.0`, resolved version retained, drift receipt preserved and further dispatch blocked. Fixtures use the explicit name `jev-fixture`. |
| Usage and budget | Actual input/output tokens; unknown usage explicitly retains reservations; dated cost estimates; per-plan call, byte, token, estimated-cost and dispatch-time checks. The original authorized pilot was capped at 32 attempts. Larger M2 plans remain proposals until authorized. |
| Failure and replay accounting | Failures remain in known spending/latency/assay denominators; missing final outcomes remain missing. Replay rejects duplicate decisions, retry-after-failure, false usage envelopes, changed predictions/results and exceeded caps. |
| Task-specific Jev roles | Full legal-action arbitration, prospective selected-assay forecasts without action authority, and reranking of a bounded PPO proposal menu, with a shadow control and matched hidden/chance/public-forecast controls. |
| Analysis integrity | Fixed common probe, paired seed/mechanism comparisons, complete-cohort requirements, missing-forecast bounds and explicit failure origins. Learned-method intervals require all five registered seeds; fixture data cannot establish benefit. |

The official SDK exposes provider IDs as `request_id` and exposes raw HTTP
metadata. The integration retains only an allowlist and hashes instead of saving
all headers/error bodies. [Official response metadata](https://docs.typesafe.ai/sdk/python/api/types/responses),
[official exceptions](https://docs.typesafe.ai/sdk/python/api/exceptions).

The vendor documents complete Choice distributions and Noul probabilities. Choice
confidence is a derived statistic of its distribution; it is not an independent
forecast about simulator success. Forecasting future assays is our application
of Noul, evaluated locally, rather than a dedicated vendor forecasting endpoint.
[Choice](https://docs.typesafe.ai/primitives/choice), [Noul](https://docs.typesafe.ai/primitives/noul),
[confidence](https://docs.typesafe.ai/confidence).

## Deliberate boundaries

No native Jev trace/span backend, OpenTelemetry exporter or W3C propagation
contract was found in the reviewed official documentation and SDK transport.
Our IDs and event journal are local instrumentation; they do not expose provider
internals. An external telemetry exporter could be added later, but no research
packets are sent to an extra telemetry service. [SDK client configuration](https://docs.typesafe.ai/sdk/python/api/clients/sync),
[reviewed official transport source](https://github.com/typesafe-ai/typesafe-sdk-python/blob/f36e68549ceca3191e5791e5f99ea1baa1a258b5/src/typesafe_sdk/_core/transport.py).

The SDK's ordinary automatic retry behavior is useful for applications, but this
study deliberately makes one network attempt per recorded call. HTTP 429 and 529
are measured failures; hints are recorded, not silently retried. Changing that
policy would require freezing its accounting and matched controls first.
[Official error guidance](https://docs.typesafe.ai/api/errors),
[SDK retry implementation](https://github.com/typesafe-ai/typesafe-sdk-python/blob/f36e68549ceca3191e5791e5f99ea1baa1a258b5/src/typesafe_sdk/_core/retry.py).

Model discovery is an authenticated inventory endpoint, not an exhaustive
validator of versioned pins: the current documentation says discovery lists aliases
while versioned IDs may still be accepted. This pilot preserves its explicit pin
and validates the returned version; it does not silently switch to an alias.
[Models](https://docs.typesafe.ai/models), [OpenAPI](https://api.typesafe.ai/openapi.json).

Score, image/Box and long-context opt-ins do not serve this pilot's binary-event
forecast and finite legal-action task. Companion Noul is useful when no candidate
fits; here code always supplies legal final commitments. An extra subjective
"scientific defensibility" gate would alter the studied controller without a
validated criterion, so it is not added. Candidate-order sensitivity remains a
separate empirical check before larger hosted comparisons. [Companion Noul](https://docs.typesafe.ai/patterns/companion-noul),
[fan-out](https://docs.typesafe.ai/patterns/fan-out), [model limitations](https://docs.typesafe.ai/model-jaggedness).

SDK DEBUG logging can include request/response bodies, so it is not enabled.
Connection pooling, compression and concurrent requests are documented throughput
options, but this sequential, capped pilot has no authenticated performance data
to justify adding them. Current timings explicitly include instrumentation and
per-call transport overhead. [Advanced SDK usage](https://docs.typesafe.ai/sdk/python/usage/advanced).

No ordinary account zero-retention claim is made; the vendor describes zero
retention as an enterprise arrangement. Only public synthetic packets belong in
this integration. [Official legal overview](https://docs.typesafe.ai/legal).

## Verification boundary

Offline tests use mocked HTTP responses to exercise provider IDs on success and
errors, case-insensitive headers, usage overflow, drift, cap exhaustion, sanitized
metadata, candidate ordering and trace correlation. The fixture study tests the
same controller/replay machinery with the actual frozen RL checkpoints.

The user-run `hosted-20261113-204144` pilot supplies limited authenticated evidence:
11 successful calls, 11 distinct provider request IDs, resolved `jev-1.14.0`,
77,860 input and 11,811 output tokens, and approximately 2.685 seconds of measured
transport time. Credential-free replay verifies all 15 pilot episodes. Its six
common-probe events produced Jev Brier 0.24938 versus posterior-reference 0.18235;
this one-mechanism, one-seed pilot does not establish calibration or hybrid benefit.
The current rate, rechecked on the official model page, is $0.042 per million input
tokens and no output-token charge; the pilot's calculated input cost is $0.00327012.
[Models and pricing](https://docs.typesafe.ai/models).

The M2 fixture verifies and replays 75 episodes across all five selected PPO seeds,
including the narrower Jev roles and the unchanged v6 incumbent. It used no hosted
calls. The separate PPO confirmation replays 5,632 episodes and does not establish
a statistically supported return gain over v6. New M2 hosted efficacy remains
unmeasured; larger allowance and a fresh process/hidden-prompt key are required.
No credential is saved in artifacts or source files.

The journal proves internal ordering and detects altered records; it is not an
independent timestamp attestation. The configured network timeout is a socket/
operation timeout, not a proven hard deadline for the entire HTTP call. Dispatch
guards stop subsequent calls after the study wall budget; a call already in progress
can finish after that boundary. Replay validates recorded PPO advice and simulator
outcomes without weights, but cannot independently establish the advice's neural
origin without the original checkpoint.
