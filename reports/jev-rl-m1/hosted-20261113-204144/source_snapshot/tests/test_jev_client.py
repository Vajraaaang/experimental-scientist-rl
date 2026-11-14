"""Offline contract tests for the bounded TypeSafe adapter; never use credentials."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import hashlib
import importlib
import json
import re
import socket
import ssl

import pytest


MODEL = "jev-fixture"


def adapter():
    return importlib.import_module("integrations.jev_client")


def request():
    return {
        "model": MODEL,
        "state": {"history": [{"measurement": 0.5}]},
        "questions": {
            "next_action": {
                "type": "choice",
                "instructions": "Choose a legal action.",
                "criteria": {"action_0": "Stop", "action_1": "Measure"},
            },
            "forecast": {"type": "noul", "instructions": "Will it exceed 0.5?"},
        },
    }


def response():
    return {
        "model": MODEL,
        "answers": {
            "next_action": {
                "type": "choice",
                "choice": "action_1",
                "confidence": 0.7,
                "probabilities": {"action_0": 0.3, "action_1": 0.7},
            },
            "forecast": {"type": "noul", "noul": 0.6},
        },
        "usage": {"input_tokens": 123, "output_tokens": 10},
    }


class FixtureTransport:
    def __init__(self, result=None, error=None):
        self.result = response() if result is None else result
        self.error = error
        self.requests = []

    def __call__(self, payload):
        self.requests.append(deepcopy(payload))
        if self.error is not None:
            raise self.error
        return deepcopy(self.result)


def make_client(transport=None, **kwargs):
    return adapter().JevClient(
        transport if transport is not None else FixtureTransport(),
        expected_model=MODEL,
        **kwargs,
    )


def request_bytes(payload):
    return len(json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def payload_sha256(payload):
    body = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def test_valid_call_sanitizes_response_and_accounts_reported_usage():
    supplied = response()
    supplied["private_debug"] = "DO_NOT_PERSIST"
    supplied["usage"]["private_debug"] = "DO_NOT_PERSIST"
    supplied["answers"]["next_action"]["reasoning"] = "DO_NOT_PERSIST"
    supplied["answers"]["forecast"]["reasoning"] = "DO_NOT_PERSIST"
    transport = FixtureTransport(supplied)
    client = make_client(transport)
    payload = request()

    assert client(payload) == response()
    assert payload == request()
    assert transport.requests == [request()]
    stats = client.stats()
    assert stats["attempted_calls"] == stats["successful_calls"] == 1
    assert stats["input_tokens"] == 123
    assert stats["output_tokens"] == 10
    assert stats["unknown_usage_calls"] == 0
    assert stats["usage_complete"] is True
    assert stats["serialized_request_bytes"] == request_bytes(payload)
    assert stats["estimated_usd"] == pytest.approx(123 * 0.042 / 1_000_000)
    assert stats["latency_seconds"] >= 0
    assert stats["failures"] == {}
    assert "DO_NOT_PERSIST" not in json.dumps(client.receipts)
    assert client.receipts[0]["response"] == response()


def test_pure_validator_handles_all_forecasts_and_all_legal_choices():
    payload = request()
    result = response()
    criteria = {f"action_{index}": "Public action" for index in range(45)}
    payload["questions"]["next_action"]["criteria"] = criteria
    result["answers"]["next_action"]["probabilities"] = {name: 1 / 45 for name in criteria}
    for index in range(72):
        name = f"forecast_{index}"
        payload["questions"][name] = {"type": "noul", "instructions": "Exceed baseline?"}
        result["answers"][name] = {"type": "noul", "noul": 0.5}
    assert adapter().validate_response(payload, result, MODEL) == result


@pytest.mark.parametrize("probability", [0.699, 0.701])
def test_probability_sum_tolerance_includes_boundary(probability):
    result = response()
    result["answers"]["next_action"]["probabilities"]["action_1"] = probability
    assert adapter().validate_response(request(), result, MODEL) == result


def test_mutated_transport_error_code_cannot_leak_arbitrary_text():
    error = adapter().JevError("network_error")
    error.code = "PRIVATE_TEXT"
    client = make_client(FixtureTransport(error=error))
    with pytest.raises(adapter().JevError, match="unexpected_error"):
        client(request())
    assert "PRIVATE_TEXT" not in json.dumps([client.receipts, client.stats()])


@pytest.mark.parametrize("change", [
    lambda data: data["answers"]["next_action"].update(choice="action_999"),
    lambda data: data["answers"]["next_action"].update(choice=True),
    lambda data: data["answers"]["next_action"].update(confidence=True),
    lambda data: data["answers"]["next_action"].update(confidence=float("nan")),
    lambda data: data["answers"]["next_action"].update(confidence=-0.01),
    lambda data: data["answers"]["next_action"].pop("probabilities"),
    lambda data: data["answers"]["next_action"]["probabilities"].pop("action_0"),
    lambda data: data["answers"]["next_action"]["probabilities"].update(action_999=0.0),
    lambda data: data["answers"]["next_action"]["probabilities"].update(action_0=0.1),
    lambda data: data["answers"]["next_action"]["probabilities"].update(action_0=True),
    lambda data: data["answers"]["next_action"]["probabilities"].update(action_0=float("nan")),
    lambda data: data["answers"]["next_action"]["probabilities"].update(action_0=float("inf")),
    lambda data: data["answers"]["next_action"]["probabilities"].update(action_0=-0.1),
    lambda data: data["answers"]["forecast"].update(noul=True),
    lambda data: data["answers"]["forecast"].update(noul=float("nan")),
    lambda data: data["answers"]["forecast"].update(noul=1.1),
    lambda data: data["answers"]["forecast"].update(type="choice"),
    lambda data: data["answers"].pop("forecast"),
    lambda data: data["answers"].update(unrequested={"type": "noul", "noul": 0.5}),
    lambda data: data.update(answers=[]),
    lambda data: data.update(model="private text that is not a model"),
])
def test_malformed_or_masked_answers_fail_closed(change):
    result = response()
    change(result)
    client = make_client(FixtureTransport(result))
    with pytest.raises(adapter().JevError) as caught:
        client(request())
    assert caught.value.code == "invalid_response"
    assert client.stats()["successful_calls"] == 0
    assert client.stats()["input_tokens"] == 123
    assert client.stats()["unknown_usage_calls"] == 0
    assert "response" not in client.receipts[0]


@pytest.mark.parametrize("usage,code", [
    (None, "missing_usage"),
    ({"input_tokens": 1}, "invalid_usage"),
    ({"input_tokens": True, "output_tokens": 0}, "invalid_usage"),
    ({"input_tokens": -1, "output_tokens": 0}, "invalid_usage"),
    ({"input_tokens": 1.0, "output_tokens": 0}, "invalid_usage"),
    ({"input_tokens": 1, "output_tokens": False}, "invalid_usage"),
])
def test_unknown_usage_is_not_reported_as_free(usage, code):
    result = response()
    if usage is None:
        result.pop("usage")
    else:
        result["usage"] = usage
    client = make_client(FixtureTransport(result))
    with pytest.raises(adapter().JevError, match=code):
        client(request())
    stats = client.stats()
    assert stats["unknown_usage_calls"] == 1
    assert stats["usage_complete"] is False
    assert stats["estimated_usd"] is None
    assert stats["reserved_input_tokens"] == request_bytes(request()) + 4096
    assert stats["reserved_estimated_usd"] > 0
    assert client.receipts[0]["usage"] is None


def test_model_drift_retains_safe_response_and_disables_future_dispatch():
    result = response()
    result["model"] = "jev-other"
    result["private_debug"] = "DO_NOT_PERSIST"
    transport = FixtureTransport(result)
    events = []
    client = make_client(transport, event_sink=events.append)
    with pytest.raises(adapter().JevError, match="model_drift"):
        client(request())
    with pytest.raises(adapter().JevError, match="model_drift"):
        client(request())
    assert len(transport.requests) == 1
    assert client.stats()["input_tokens"] == 123
    assert client.receipts[0]["response"]["model"] == "jev-other"
    assert events[1]["kind"] == "api_failure"
    assert events[1]["response"]["model"] == "jev-other"
    assert "DO_NOT_PERSIST" not in json.dumps(events)
    with pytest.raises(adapter().JevError, match="model_drift"):
        adapter().validate_response(request(), result, MODEL)


@pytest.mark.parametrize("change,code", [
    ({"max_calls": 0}, "call_limit"),
    ({"max_request_bytes": 1}, "request_too_large"),
    ({"max_total_request_bytes": 1}, "total_request_limit"),
    ({"max_input_tokens": 4096}, "input_token_limit"),
    ({"max_estimated_usd": 0}, "cost_limit"),
])
def test_caps_are_checked_before_dispatch(change, code):
    transport = FixtureTransport()
    limits = replace(adapter().CallLimits(), **change)
    client = make_client(transport, limits=limits)
    with pytest.raises(adapter().JevError, match=code):
        client(request())
    assert transport.requests == []
    assert client.stats()["attempted_calls"] == 0
    assert client.stats()["serialized_request_bytes"] == 0
    assert client.stats()["failures"] == {code: 1}


def test_cumulative_call_and_byte_caps_are_checked_before_second_dispatch():
    payload = request()
    for limits, code in [
        (replace(adapter().CallLimits(), max_calls=1), "call_limit"),
        (replace(adapter().CallLimits(), max_total_request_bytes=request_bytes(payload)), "total_request_limit"),
    ]:
        transport = FixtureTransport()
        client = make_client(transport, limits=limits)
        client(payload)
        with pytest.raises(adapter().JevError, match=code):
            client(payload)
        assert len(transport.requests) == 1


def test_unknown_usage_retains_reservation_against_future_token_cap():
    transport = FixtureTransport(error=TimeoutError("PRIVATE_KEY"))
    limits = replace(adapter().CallLimits(), max_input_tokens=request_bytes(request()) + 4096)
    client = make_client(transport, limits=limits)
    with pytest.raises(adapter().JevError, match="timeout"):
        client(request())
    with pytest.raises(adapter().JevError, match="input_token_limit"):
        client(request())
    assert len(transport.requests) == 1


def test_actual_usage_over_reservation_is_retained_and_stops_further_calls():
    result = response()
    result["usage"]["input_tokens"] = request_bytes(request()) + 4097
    transport = FixtureTransport(result)
    client = make_client(transport)
    assert client(request()) == result
    assert client.stats()["input_tokens"] == result["usage"]["input_tokens"]
    assert client.receipts[0]["reservation_overrun"] is True
    with pytest.raises(adapter().JevError, match="reservation_overrun"):
        client(request())
    assert len(transport.requests) == 1


@pytest.mark.parametrize("change", [
    lambda data: data["state"].update(value=float("nan")),
    lambda data: data.update(model="jev-other"),
    lambda data: data.update(questions={}),
    lambda data: data["questions"]["forecast"].update(type="unsupported"),
    lambda data: data["questions"]["next_action"].update(criteria={}),
    lambda data: data.update(extra_private_field="should not be sent"),
])
def test_invalid_requests_never_dispatch(change):
    payload = request()
    change(payload)
    transport = FixtureTransport()
    client = make_client(transport)
    with pytest.raises(adapter().JevError, match="invalid_request"):
        client(payload)
    assert transport.requests == []


def test_nonstring_json_object_keys_cannot_be_silently_coerced():
    payload = request()
    payload["state"] = {1: "public measurement"}
    transport = FixtureTransport()
    client = make_client(transport)
    with pytest.raises(adapter().JevError, match="invalid_request"):
        client(payload)
    assert transport.requests == []


@pytest.mark.parametrize("error,code", [
    (TimeoutError("PRIVATE_KEY and secret error body"), "timeout"),
    (RuntimeError("PRIVATE_KEY and secret error body"), "transport_error"),
])
def test_arbitrary_transport_exceptions_are_sanitized_without_retry(error, code):
    events = []
    transport = FixtureTransport(error=error)
    client = make_client(transport, event_sink=events.append)
    with pytest.raises(adapter().JevError) as caught:
        client(request())
    assert str(caught.value) == code
    assert len(transport.requests) == 1
    assert "PRIVATE_KEY" not in json.dumps([events, client.receipts, client.stats()])
    assert caught.value.__context__ is None
    assert client.stats()["unknown_usage_calls"] == 1
    assert [event["kind"] for event in events] == ["api_request", "api_failure"]


def test_request_event_is_synchronously_written_before_dispatch():
    events = []

    def transport(payload):
        assert len(events) == 1
        assert events[0]["kind"] == "api_request"
        assert events[0]["request"] == request()
        assert events[0]["call_index"] == 1
        return response()

    client = make_client(transport, event_sink=events.append)
    assert client(request()) == response()
    assert events[1]["kind"] == "api_response"
    assert events[1]["response"] == response()
    assert events[1]["call_index"] == 1


def test_event_sink_failure_before_dispatch_prevents_network_and_is_sanitized():
    def broken_sink(event):
        raise RuntimeError("PRIVATE_KEY")

    transport = FixtureTransport()
    client = make_client(transport, event_sink=broken_sink)
    with pytest.raises(adapter().JevError, match="event_sink_error") as caught:
        client(request())
    assert caught.value.__context__ is None
    assert transport.requests == []
    assert client.stats()["attempted_calls"] == 0


def test_error_rejects_arbitrary_text_as_a_public_code():
    error = adapter().JevError("PRIVATE_KEY")
    assert error.code == "unexpected_error"
    assert "PRIVATE_KEY" not in repr(error)


@pytest.mark.parametrize("code", ["illegal_action", "simulator_error", "decision_limit"])
def test_controller_guard_codes_are_preserved(code):
    assert adapter().JevError(code).code == code


def test_changing_event_sink_routes_next_episode_without_resetting_caps():
    first_events, second_events = [], []
    client = make_client(event_sink=first_events.append)
    client(request())
    client.event_sink = second_events.append
    client(request())
    assert client.event_sink == second_events.append
    assert [event["kind"] for event in first_events] == ["api_request", "api_response"]
    assert [event["kind"] for event in second_events] == ["api_request", "api_response"]
    assert first_events[0]["call_index"] == 1
    assert second_events[0]["call_index"] == 2
    assert client.stats()["attempted_calls"] == 2


def test_invalid_event_sink_assignment_fails_without_replacing_sink():
    events = []
    client = make_client(event_sink=events.append)
    with pytest.raises(adapter().JevError, match="invalid_configuration"):
        client.event_sink = "PRIVATE_TEXT"
    client(request())
    assert len(events) == 2


def test_sink_mutation_does_not_change_dispatched_request_or_returned_response():
    def mutating_sink(event):
        if event["kind"] == "api_request":
            event["request"]["questions"].clear()
        elif event["kind"] == "api_response":
            event["response"]["answers"].clear()

    transport = FixtureTransport()
    client = make_client(transport, event_sink=mutating_sink)
    assert client(request()) == response()
    assert transport.requests == [request()]
    assert client.receipts[0]["response"] == response()


def test_response_journal_failure_retains_usage_and_blocks_future_dispatch():
    def sink(event):
        if event["kind"] == "api_response":
            raise RuntimeError("PRIVATE_TEXT")

    transport = FixtureTransport()
    client = make_client(transport, event_sink=sink)
    with pytest.raises(adapter().JevError, match="event_sink_error"):
        client(request())
    assert client.stats()["input_tokens"] == 123
    assert client.stats()["successful_calls"] == 0
    assert client.receipts[0]["status"] == "failed"
    with pytest.raises(adapter().JevError, match="event_sink_error"):
        client(request())
    assert len(transport.requests) == 1


@pytest.mark.parametrize("settings", [
    {"limits": "not limits"},
    {"expected_model": "private text"},
    {"input_usd_per_million": float("nan")},
    {"input_usd_per_million": True},
    {"limits": {"max_calls": -1}},
])
def test_invalid_configuration_fails_with_fixed_code(settings):
    if type(settings.get("limits")) is dict:
        settings = {"limits": replace(adapter().CallLimits(), **settings["limits"])}
    with pytest.raises(adapter().JevError, match="invalid_configuration"):
        adapter().JevClient(FixtureTransport(), **settings)


class FakeHTTPResponse:
    def __init__(self, body=None, status=200, headers=()):
        self.body = json.dumps(response()).encode() if body is None else body
        self.status = status
        self.headers = list(headers)
        self.read_sizes = []

    def getheaders(self):
        return self.headers

    def read(self, size):
        self.read_sizes.append(size)
        return self.body[:size]


class FakeHTTPSConnection:
    def __init__(self, reply=None, error=None):
        self.reply = reply or FakeHTTPResponse()
        self.error = error
        self.requests = []
        self.closed = False

    def request(self, *args, **kwargs):
        self.requests.append((args, kwargs))
        if self.error:
            raise self.error

    def getresponse(self):
        return self.reply

    def close(self):
        self.closed = True


def install_fake_https(monkeypatch, connection):
    module = adapter()
    contexts = []
    context = object()
    monkeypatch.setattr(module, "tls_context", lambda: context)

    def factory(host, *, timeout, context):
        contexts.append((host, timeout, context))
        return connection

    monkeypatch.setattr(module, "HTTPSConnection", factory)
    return contexts, context


def test_hosted_transport_uses_verified_tls_bounded_https_and_header_only_key(monkeypatch):
    connection = FakeHTTPSConnection()
    contexts, context = install_fake_https(monkeypatch, connection)
    transport = adapter().HostedTransport("TEST_KEY_DO_NOT_LOG")
    result = transport(request())
    assert adapter().validate_response(request(), result, MODEL) == response()
    assert contexts == [("api.typesafe.ai", 20, context)]
    args, kwargs = connection.requests[0]
    assert args == ("POST", "/v1/systemone")
    assert kwargs["headers"]["Authorization"] == "Bearer TEST_KEY_DO_NOT_LOG"
    assert b"TEST_KEY_DO_NOT_LOG" not in kwargs["body"]
    assert "TEST_KEY_DO_NOT_LOG" not in repr(transport)
    assert connection.reply.read_sizes == [256 * 1024 + 1]
    assert connection.closed is True


@pytest.mark.parametrize("error,code", [
    (socket.timeout("PRIVATE_KEY"), "timeout"),
    (OSError("PRIVATE_KEY"), "network_error"),
    (RuntimeError("PRIVATE_KEY"), "network_error"),
    (ssl.SSLCertVerificationError("PRIVATE_KEY"), "tls_error"),
])
def test_hosted_transport_sanitizes_network_errors(monkeypatch, error, code):
    connection = FakeHTTPSConnection(error=error)
    install_fake_https(monkeypatch, connection)
    with pytest.raises(adapter().JevError) as caught:
        adapter().HostedTransport("TEST_KEY")(request())
    assert str(caught.value) == code
    assert caught.value.__context__ is None
    assert connection.closed is True


@pytest.mark.parametrize("reply,code", [
    (FakeHTTPResponse(b"PRIVATE_KEY", 401), "http_auth_error"),
    (FakeHTTPResponse(b"PRIVATE_KEY", 403), "http_auth_error"),
    (FakeHTTPResponse(b"PRIVATE_KEY", 500), "server_error"),
    (FakeHTTPResponse(b"x" * (256 * 1024 + 1)), "response_too_large"),
    (FakeHTTPResponse(b"PRIVATE_KEY"), "invalid_json"),
    (FakeHTTPResponse(b'{"noul":NaN}'), "invalid_json"),
    (FakeHTTPResponse(b'{"model":"jev-fixture","model":"jev-other"}'), "invalid_json"),
])
def test_hosted_transport_rejects_error_bodies_and_invalid_json(monkeypatch, reply, code):
    connection = FakeHTTPSConnection(reply)
    install_fake_https(monkeypatch, connection)
    with pytest.raises(adapter().JevError) as caught:
        adapter().HostedTransport("TEST_KEY")(request())
    assert str(caught.value) == code
    assert "PRIVATE_KEY" not in repr(caught.value)
    assert connection.closed is True
    if reply.status != 200:
        assert reply.read_sizes == []


@pytest.mark.parametrize("name", ["input_tokens", "output_tokens"])
def test_unrepresentable_usage_is_rejected_without_losing_failure_receipt(name):
    result = response()
    result["usage"][name] = 10 ** 400
    client = make_client(FixtureTransport(result))
    with pytest.raises(adapter().JevError, match="invalid_usage"):
        client(request())
    assert len(client.receipts) == 1
    assert client.receipts[0]["code"] == "invalid_usage"
    assert client.receipts[0]["usage"] is None
    assert client.stats()["unknown_usage_calls"] == 1
    assert client.stats()["reserved_input_tokens"] == request_bytes(request()) + 4096


@pytest.mark.parametrize("count,accepted", [(2 ** 53 - 1, True), (2 ** 53, False)])
def test_usage_counts_have_explicit_safe_integer_boundary(count, accepted):
    result = response()
    result["usage"]["output_tokens"] = count
    if accepted:
        assert adapter().validate_response(request(), result, MODEL) == result
    else:
        with pytest.raises(adapter().JevError, match="invalid_usage"):
            adapter().validate_response(request(), result, MODEL)


def test_choice_must_maximize_returned_vector():
    result = response()
    result["answers"]["next_action"]["choice"] = "action_0"
    client = make_client(FixtureTransport(result))
    with pytest.raises(adapter().JevError, match="invalid_response"):
        client(request())
    assert client.receipts[0]["usage"] == response()["usage"]


@pytest.mark.parametrize("gap,accepted", [(0.0, True), (0.0000005, True), (0.000002, False)])
def test_choice_maximum_tolerance_preserves_native_confidence(gap, accepted):
    result = response()
    answer = result["answers"]["next_action"]
    answer["probabilities"] = {"action_0": 0.5 + gap / 2, "action_1": 0.5 - gap / 2}
    answer["confidence"] = 0.12345
    if accepted:
        assert adapter().validate_response(request(), result, MODEL) == result
    else:
        with pytest.raises(adapter().JevError, match="invalid_response"):
            adapter().validate_response(request(), result, MODEL)


def test_trace_ids_hashes_timestamps_and_order_correlate_request_and_receipt():
    events = []
    client = make_client(event_sink=events.append, trace_id="1234567890abcdef" * 2)
    assert client.trace_id == "1234567890abcdef" * 2
    client(request())
    client(request())
    spans = set()
    for index, receipt in enumerate(client.receipts):
        sent, received = events[2 * index:2 * index + 2]
        assert receipt["trace_id"] == sent["trace_id"] == received["trace_id"] == client.trace_id
        assert re.fullmatch(r"[0-9a-f]{16}", receipt["span_id"])
        assert receipt["span_id"] == sent["span_id"] == received["span_id"]
        spans.add(receipt["span_id"])
        assert receipt["request_sha256"] == sent["request_sha256"] == payload_sha256(request())
        assert receipt["response_sha256"] == payload_sha256(response())
        assert receipt["proposed_request_bytes"] == receipt["request_bytes"] == request_bytes(request())
        assert receipt["question_order"] == sent["question_order"] == sorted(request()["questions"])
        assert receipt["candidate_order"] == sent["candidate_order"] == {"next_action": ["action_0", "action_1"]}
        assert receipt["started_at_utc"] == sent["started_at_utc"]
        assert receipt["started_at_utc"].endswith("Z") and receipt["finished_at_utc"].endswith("Z")
        assert datetime.fromisoformat(receipt["finished_at_utc"]) >= datetime.fromisoformat(receipt["started_at_utc"])
    assert len(spans) == 2
    assert re.fullmatch(r"[0-9a-f]{32}", make_client().trace_id)


@pytest.mark.parametrize("trace_id", [True, "", "A" * 32, "g" * 32, "1" * 31, "1" * 33])
def test_malformed_trace_id_is_rejected(trace_id):
    with pytest.raises(adapter().JevError, match="invalid_configuration"):
        make_client(trace_id=trace_id)


def test_cap_failure_records_proposed_request_hash_without_dispatch_event():
    events = []
    client = make_client(limits=replace(adapter().CallLimits(), max_calls=0), event_sink=events.append)
    with pytest.raises(adapter().JevError, match="call_limit"):
        client(request())
    receipt = client.receipts[0]
    assert receipt["attempted"] is False
    assert receipt["request_bytes"] == 0
    assert receipt["proposed_request_bytes"] == request_bytes(request())
    assert receipt["request_sha256"] == payload_sha256(request())
    assert receipt["response_sha256"] is None
    assert receipt["question_order"] == sorted(request()["questions"])
    assert [event["kind"] for event in events] == ["api_failure"]
    assert events[0]["span_id"] == receipt["span_id"]


def test_invalid_request_receipt_has_trace_without_recording_malformed_payload():
    client = make_client()
    with pytest.raises(adapter().JevError, match="invalid_request"):
        client({"PRIVATE_TEXT": float("nan")})
    receipt = client.receipts[0]
    assert receipt["trace_id"] == client.trace_id
    assert receipt["request_sha256"] is None
    assert receipt["proposed_request_bytes"] == 0
    assert receipt["question_order"] == []
    assert receipt["candidate_order"] == {}
    assert "PRIVATE_TEXT" not in json.dumps(receipt)


def test_transport_metadata_sanitizer_is_pure_and_strips_unrecognized_fields():
    metadata = {
        "endpoint": "/v1/systemone", "provider_request_id": "req-id_123:abc.def",
        "http_status": 200, "retry_after": 1.25, "retry_after_ms": 1250,
        "response_bytes": 100, "response_body_sha256": "a" * 64,
        "response_body_complete": True,
    }
    supplied = {**metadata, "headers": {"Authorization": "PRIVATE_TEXT"}, "body": "PRIVATE_TEXT"}
    assert adapter().validate_transport_metadata(supplied) == metadata
    assert "headers" in supplied


@pytest.mark.parametrize("field,value", [
    ("endpoint", "https://secret.example/?key=PRIVATE_TEXT"),
    ("provider_request_id", "private text"),
    ("provider_request_id", "injected\r\nAuthorization: secret"),
    ("provider_request_id", "\u2603"),
    ("provider_request_id", "x" * 129),
    ("http_status", True), ("http_status", 600),
    ("retry_after", True), ("retry_after", -1), ("retry_after", 86401),
    ("retry_after", float("inf")), ("retry_after", "PRIVATE_TEXT"),
    ("retry_after_ms", 86400001),
    ("response_bytes", True), ("response_bytes", 256 * 1024 + 2),
    ("response_body_sha256", "PRIVATE_TEXT"),
    ("response_body_complete", "PRIVATE_TEXT"),
])
def test_transport_metadata_rejects_unsafe_or_unbounded_values(field, value):
    assert adapter().validate_transport_metadata({field: value}) == {}


def test_transport_metadata_is_sanitized_again_on_success_and_error_receipts():
    metadata = {"provider_request_id": "req-123", "http_status": 200, "body": "PRIVATE_TEXT"}
    result = {**response(), "_transport": metadata}
    client = make_client(FixtureTransport(result))
    assert client(request()) == response()
    assert client.receipts[0]["transport"] == {"provider_request_id": "req-123", "http_status": 200}
    assert client.receipts[0]["response_sha256"] == payload_sha256(response())
    error = adapter().JevError("network_error", transport=metadata)
    error.transport["body"] = "PRIVATE_TEXT"
    failed = make_client(FixtureTransport(error=error))
    with pytest.raises(adapter().JevError, match="network_error"):
        failed(request())
    assert failed.receipts[0]["transport"] == client.receipts[0]["transport"]
    assert "PRIVATE_TEXT" not in json.dumps([client.receipts, failed.receipts])


def test_hosted_response_retains_safe_case_insensitive_headers_and_body_digest(monkeypatch):
    reply = FakeHTTPResponse(headers=[
        ("X-TypeSafe-Request-ID", "req-123"), ("Retry-After", "1.25"),
        ("Retry-After-Ms", "1250"), ("Authorization", "PRIVATE_TEXT"),
    ])
    connection = FakeHTTPSConnection(reply)
    install_fake_https(monkeypatch, connection)
    client = make_client(adapter().HostedTransport("TEST_KEY"))
    assert client(request()) == response()
    assert client.receipts[0]["transport"] == {
        "endpoint": "/v1/systemone", "provider_request_id": "req-123", "http_status": 200,
        "retry_after": 1.25, "retry_after_ms": 1250,
        "response_body_sha256": hashlib.sha256(reply.body).hexdigest(),
        "response_bytes": len(reply.body), "response_body_complete": True,
    }
    assert "PRIVATE_TEXT" not in json.dumps(client.receipts)
    assert len(connection.requests) == 1


@pytest.mark.parametrize("status,code", [
    (401, "http_auth_error"), (403, "http_auth_error"), (408, "request_timeout"),
    (429, "rate_limit"), (529, "overloaded"), (500, "server_error"),
    (503, "server_error"), (404, "http_error"),
])
def test_http_failure_preserves_request_id_and_status_without_body_or_retry(monkeypatch, status, code):
    reply = FakeHTTPResponse(b"PRIVATE_TEXT", status, [("x-typesafe-request-id", "req-failure")])
    connection = FakeHTTPSConnection(reply)
    install_fake_https(monkeypatch, connection)
    events = []
    client = make_client(adapter().HostedTransport("TEST_KEY"), event_sink=events.append)
    with pytest.raises(adapter().JevError, match=code):
        client(request())
    assert client.receipts[0]["transport"] == {
        "endpoint": "/v1/systemone", "provider_request_id": "req-failure", "http_status": status,
    }
    assert events[-1]["transport"] == client.receipts[0]["transport"]
    assert client.stats()["attempted_calls"] == len(connection.requests) == 1
    assert reply.read_sizes == []
    assert "PRIVATE_TEXT" not in json.dumps([client.receipts, events])


@pytest.mark.parametrize("body,code", [
    (b"PRIVATE_TEXT", "invalid_json"),
    (b'{"model":"jev-fixture","usage":{"input_tokens":1,"output_tokens":0},"answers":{}}', "invalid_response"),
    (b"[]", "invalid_response"),
    (b"x" * (256 * 1024 + 1), "response_too_large"),
], ids=["invalid-json", "invalid-answers", "nonobject", "oversized"])
def test_invalid_hosted_response_keeps_body_digest_and_safe_metadata(monkeypatch, body, code):
    reply = FakeHTTPResponse(body, headers=[("x-typesafe-request-id", "req-bad-body")])
    connection = FakeHTTPSConnection(reply)
    install_fake_https(monkeypatch, connection)
    client = make_client(adapter().HostedTransport("TEST_KEY"))
    with pytest.raises(adapter().JevError, match=code):
        client(request())
    receipt = client.receipts[0]
    assert receipt["response_sha256"] is None
    assert receipt["transport"]["provider_request_id"] == "req-bad-body"
    assert receipt["transport"]["response_body_sha256"] == hashlib.sha256(body).hexdigest()
    assert receipt["transport"]["response_bytes"] == len(body)
    assert receipt["transport"]["response_body_complete"] is (len(body) <= 256 * 1024)
    assert "PRIVATE_TEXT" not in json.dumps(client.receipts)


@pytest.mark.parametrize("request_id", ["TEST_KEY", "prefix_TEST_KEY_suffix", "private text", "x" * 129])
def test_provider_id_cannot_echo_key_or_unbounded_text(monkeypatch, request_id):
    reply = FakeHTTPResponse(headers=[("x-typesafe-request-id", request_id)])
    connection = FakeHTTPSConnection(reply)
    install_fake_https(monkeypatch, connection)
    client = make_client(adapter().HostedTransport("TEST_KEY"))
    client(request())
    assert "provider_request_id" not in client.receipts[0]["transport"]
    assert "TEST_KEY" not in json.dumps(client.receipts)
