"""Bounded, credential-blind accounting and validation for the TypeSafe API.

The 2026-11-13 estimate is $0.042 per million input tokens; output is currently
free. Reported usage is retained even when a response fails validation. Unknown
usage stays unknown and retains its conservative pre-dispatch reservation.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from http.client import HTTPSConnection
import json
import math
import re
import ssl
from threading import Lock
from time import perf_counter
from uuid import uuid4

from integrations.jev_demo import tls_context


HOST = "api.typesafe.ai"
ENDPOINT = "/v1/systemone"
RESPONSE_LIMIT = 256 * 1024
MAX_USAGE_COUNT = 2 ** 53 - 1
_MODEL = re.compile(r"jev-[\w.-]{1,60}")
_CODES = frozenset({
    "invalid_configuration", "invalid_key", "invalid_request", "invalid_response",
    "missing_usage", "invalid_usage", "model_drift", "call_limit",
    "request_too_large", "total_request_limit", "input_token_limit", "cost_limit",
    "reservation_overrun", "event_sink_error", "timeout", "transport_error",
    "network_error", "tls_error", "http_auth_error", "http_error",
    "response_too_large", "invalid_json", "unexpected_error",
    "illegal_action", "simulator_error", "decision_limit",
    "rate_limit", "overloaded", "server_error", "request_timeout",
})


def validate_transport_metadata(metadata):
    """Return only bounded transport evidence; never return headers or bodies.

    Retry hints are numeric seconds/milliseconds, bounded to one day. Digests
    cover bytes actually read; ``response_body_complete`` distinguishes a
    bounded prefix from a complete body. Invalid or unknown fields are omitted.
    """
    if type(metadata) is not dict:
        return {}
    safe = {}
    if type(metadata.get("endpoint")) is str and metadata["endpoint"] == ENDPOINT:
        safe["endpoint"] = ENDPOINT
    identifier = metadata.get("provider_request_id")
    if type(identifier) is str and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", identifier):
        safe["provider_request_id"] = identifier
    status = metadata.get("http_status")
    if type(status) is int and 100 <= status <= 599:
        safe["http_status"] = status
    for field, maximum in (("retry_after", 86400), ("retry_after_ms", 86400000)):
        value = metadata.get(field)
        if type(value) in (int, float) and 0 <= value <= maximum and math.isfinite(value):
            safe[field] = value
    size = metadata.get("response_bytes")
    if type(size) is int and 0 <= size <= RESPONSE_LIMIT + 1:
        safe["response_bytes"] = size
    digest = metadata.get("response_body_sha256")
    if type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest):
        safe["response_body_sha256"] = digest
    complete = metadata.get("response_body_complete")
    if type(complete) is bool:
        safe["response_body_complete"] = complete
    return safe


class JevError(RuntimeError):
    """An error containing only a fixed public code, never transport text."""

    def __init__(self, code, *, transport=None):
        self.code = code if type(code) is str and code in _CODES else "unexpected_error"
        self.transport = validate_transport_metadata(transport)
        super().__init__(self.code)


@dataclass(frozen=True)
class CallLimits:
    max_calls: int = 32
    max_request_bytes: int = 65536
    max_total_request_bytes: int = 1048576
    max_input_tokens: int = 1200000
    max_estimated_usd: float = 0.10


def _model_name(value):
    return type(value) is str and _MODEL.fullmatch(value) is not None


def _probability(value):
    return type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)


def _json_value(value):
    if type(value) is dict:
        return all(type(key) is str and _json_value(item) for key, item in value.items())
    if type(value) is list:
        return all(_json_value(item) for item in value)
    if type(value) is float:
        return math.isfinite(value)
    return value is None or type(value) in (str, int, bool)


def _json_bytes(value):
    code = None
    try:
        if not _json_value(value):
            raise ValueError
        result = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        code = "invalid_request"
    if code:
        raise JevError(code)
    return result


def _validate_request(request, expected_model):
    body = _json_bytes(request)
    if (
        type(request) is not dict or set(request) != {"model", "state", "questions"}
        or not _model_name(expected_model) or request["model"] != expected_model
        or type(request["state"]) is not dict
        or type(request["questions"]) is not dict or not request["questions"]
    ):
        raise JevError("invalid_request")
    for name, question in request["questions"].items():
        if type(name) is not str or not name or type(question) is not dict:
            raise JevError("invalid_request")
        kind = question.get("type")
        required = {"type", "instructions", "criteria"} if kind == "choice" else {"type", "instructions"}
        if kind not in ("noul", "choice") or set(question) != required or type(question["instructions"]) is not str:
            raise JevError("invalid_request")
        if kind == "choice":
            criteria = question["criteria"]
            if type(criteria) is not dict or not criteria or any(
                type(key) is not str or not key or type(value) is not str
                for key, value in criteria.items()
            ):
                raise JevError("invalid_request")
    return body


def _usage(response):
    if type(response) is not dict or "usage" not in response:
        raise JevError("missing_usage")
    usage = response["usage"]
    if type(usage) is not dict or any(
        type(usage.get(name)) is not int or not 0 <= usage[name] <= MAX_USAGE_COUNT
        for name in ("input_tokens", "output_tokens")
    ):
        raise JevError("invalid_usage")
    return {name: usage[name] for name in ("input_tokens", "output_tokens")}


def _sanitize_response(request, response):
    if type(response) is not dict or not _model_name(response.get("model")):
        raise JevError("invalid_response")
    usage = _usage(response)
    answers = response.get("answers")
    questions = request["questions"]
    if type(answers) is not dict or set(answers) != set(questions):
        raise JevError("invalid_response")
    safe_answers = {}
    for name, question in questions.items():
        answer = answers[name]
        kind = question["type"]
        if type(answer) is not dict or answer.get("type") != kind:
            raise JevError("invalid_response")
        if kind == "noul":
            if not _probability(answer.get("noul")):
                raise JevError("invalid_response")
            safe_answers[name] = {"type": "noul", "noul": answer["noul"]}
        else:
            choice = answer.get("choice")
            confidence = answer.get("confidence")
            probabilities = answer.get("probabilities")
            criteria = question["criteria"]
            if (
                type(choice) is not str or choice not in criteria or not _probability(confidence)
                or type(probabilities) is not dict or set(probabilities) != set(criteria)
                or not all(_probability(value) for value in probabilities.values())
                or abs(math.fsum(probabilities.values()) - 1.0) > 0.001 + 1e-12
                or probabilities[choice] + 1e-6 < max(probabilities.values())
            ):
                raise JevError("invalid_response")
            safe_answers[name] = {
                "type": "choice", "choice": choice, "confidence": confidence,
                "probabilities": {key: probabilities[key] for key in criteria},
            }
    return {"model": response["model"], "answers": safe_answers, "usage": usage}


def validate_response(request, response, expected_model):
    """Pure replay validation; unknown response fields are never returned."""
    _validate_request(request, expected_model)
    safe_response = _sanitize_response(request, response)
    if safe_response["model"] != expected_model:
        raise JevError("model_drift")
    return safe_response


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


class JevClient:
    """Synchronous bounded client. The transport performs one call, with no retry.

    ``reserved_input_tokens`` in stats counts reservations for unknown usage;
    ``accounted_input_tokens`` adds those reservations to actual reported input.
    ``estimated_usd`` is unknown if any attempted call lacks valid usage.
    A response over its reservation is returned, but prevents any later call.
    """

    def __init__(self, transport, *, expected_model="jev-1.14.0", limits=None,
                 event_sink=None, input_usd_per_million=0.042, trace_id=None):
        self.limits = CallLimits() if limits is None else limits
        if (
            not callable(transport) or not _model_name(expected_model)
            or type(self.limits) is not CallLimits
            or (event_sink is not None and not callable(event_sink))
            or type(input_usd_per_million) not in (int, float)
            or not math.isfinite(input_usd_per_million) or input_usd_per_million < 0
            or any(type(getattr(self.limits, name)) is not int or getattr(self.limits, name) < 0
                   for name in ("max_calls", "max_request_bytes", "max_total_request_bytes", "max_input_tokens"))
            or type(self.limits.max_estimated_usd) not in (int, float)
            or not math.isfinite(self.limits.max_estimated_usd) or self.limits.max_estimated_usd < 0
            or (trace_id is not None and (type(trace_id) is not str or not re.fullmatch(r"[0-9a-f]{32}", trace_id)))
        ):
            raise JevError("invalid_configuration")
        self.expected_model = expected_model
        self._transport = transport
        self._event_sink = event_sink
        self._price = input_usd_per_million
        self._lock = Lock()
        self.receipts = []
        self._attempted = self._successful = self._input = self._output = 0
        self._unknown = self._bytes = self._reserved = 0
        self._latency = 0.0
        self._failures = Counter()
        self._blocked = None
        self._trace_id = uuid4().hex if trace_id is None else trace_id

    @property
    def trace_id(self):
        """Local instrumentation ID; unrelated to the provider request ID."""
        return self._trace_id

    @property
    def event_sink(self):
        return self._event_sink

    @event_sink.setter
    def event_sink(self, sink):
        if sink is not None and not callable(sink):
            raise JevError("invalid_configuration")
        self._event_sink = sink

    def _cost(self, tokens):
        return tokens / 1_000_000 * self._price

    def _emit(self, event):
        failed = False
        if self._event_sink is not None:
            try:
                self._event_sink(deepcopy(event))
            except Exception:
                failed = True
        if failed:
            raise JevError("event_sink_error")

    def _record(self, receipt, code=None):
        if code is not None:
            code = JevError(code).code
        receipt["finished_at_utc"] = _utc_now()
        receipt["transport"] = validate_transport_metadata(receipt["transport"])
        receipt["status"] = "failed" if code else "success"
        if code:
            receipt["code"] = code
            self._failures[code] += 1
        self.receipts.append(deepcopy(receipt))
        event = {"kind": "api_failure" if code else "api_response", **receipt}
        sink_failed = False
        try:
            self._emit(event)
        except JevError:
            sink_failed = True
        if sink_failed:
            self._blocked = "event_sink_error"
            if code != "event_sink_error":
                self._failures["event_sink_error"] += 1
            self.receipts[-1]["status"] = "failed"
            self.receipts[-1]["code"] = "event_sink_error"
            code = "event_sink_error"
        if code:
            raise JevError(code, transport=receipt["transport"])
        self._successful += 1

    def _check_caps(self, size, reservation):
        if self._blocked:
            raise JevError(self._blocked)
        checks = (
            (self._attempted >= self.limits.max_calls, "call_limit"),
            (size > self.limits.max_request_bytes, "request_too_large"),
            (self._bytes + size > self.limits.max_total_request_bytes, "total_request_limit"),
            (self._input + self._reserved + reservation > self.limits.max_input_tokens, "input_token_limit"),
            (self._cost(self._input + self._reserved + reservation) > self.limits.max_estimated_usd, "cost_limit"),
        )
        for exceeded, code in checks:
            if exceeded:
                raise JevError(code)

    def __call__(self, request):
        with self._lock:
            return self._call(request)

    def _call(self, request):
        receipt = {
            "call_index": len(self.receipts) + 1, "attempted": False,
            "request_bytes": 0, "usage": None, "latency_seconds": 0.0,
            "estimated_usd": 0.0, "reservation_overrun": False,
            "trace_id": self.trace_id, "span_id": uuid4().hex[:16],
            "started_at_utc": _utc_now(), "finished_at_utc": None,
            "request_sha256": None, "response_sha256": None,
            "proposed_request_bytes": 0, "question_order": [], "candidate_order": {},
            "transport": {},
        }
        code = None
        try:
            body = _validate_request(request, self.expected_model)
            size = len(body)
            reservation = size + 4096
            payload = json.loads(body)
            receipt.update(
                request_sha256=hashlib.sha256(body).hexdigest(), proposed_request_bytes=size,
                question_order=list(payload["questions"]),
                candidate_order={name: list(question["criteria"]) for name, question in payload["questions"].items()
                                 if question["type"] == "choice"},
            )
            self._check_caps(size, reservation)
            trace = {field: receipt[field] for field in (
                "call_index", "trace_id", "span_id", "started_at_utc", "request_sha256",
                "proposed_request_bytes", "question_order", "candidate_order",
            )}
            self._emit({"kind": "api_request", "request": payload, **trace})
        except JevError as error:
            code = error.code
            receipt["transport"] = validate_transport_metadata(error.transport)
        if code:
            if code == "event_sink_error":
                self._blocked = code
            self._record(receipt, code)

        # No dispatch can occur before every cap and the synchronous request event.
        self._attempted += 1
        self._bytes += size
        self._reserved += reservation
        receipt.update(attempted=True, request_bytes=size, reserved_input_tokens=reservation, estimated_usd=None)
        start = perf_counter()
        raw = None
        try:
            raw = self._transport(deepcopy(payload))
        except JevError as error:
            code = error.code
            receipt["transport"] = validate_transport_metadata(error.transport)
        except TimeoutError:
            code = "timeout"
        except Exception:
            code = "transport_error"
        elapsed = perf_counter() - start
        self._latency += elapsed
        receipt["latency_seconds"] = elapsed
        if type(raw) is dict:
            receipt["transport"] = validate_transport_metadata(raw.get("_transport"))

        usage = None
        if raw is not None:
            try:
                usage = _usage(raw)
            except JevError:
                pass
        if usage is None:
            self._unknown += 1
        else:
            self._reserved -= reservation
            self._input += usage["input_tokens"]
            self._output += usage["output_tokens"]
            receipt["usage"] = usage
            receipt["estimated_usd"] = self._cost(usage["input_tokens"])
            receipt["reservation_overrun"] = usage["input_tokens"] > reservation
            if receipt["reservation_overrun"]:
                self._blocked = "reservation_overrun"
        if code is None:
            try:
                safe_response = _sanitize_response(payload, raw)
                receipt["response"] = safe_response
                receipt["response_sha256"] = hashlib.sha256(_json_bytes(safe_response)).hexdigest()
                if safe_response["model"] != self.expected_model:
                    code = self._blocked = "model_drift"
            except JevError as error:
                code = error.code
        self._record(receipt, code)
        return deepcopy(safe_response)

    def stats(self):
        known_cost = self._cost(self._input)
        return {
            "attempted_calls": self._attempted, "successful_calls": self._successful,
            "input_tokens": self._input, "output_tokens": self._output,
            "unknown_usage_calls": self._unknown, "usage_complete": self._unknown == 0,
            "serialized_request_bytes": self._bytes, "latency_seconds": self._latency,
            "estimated_usd": known_cost if not self._unknown else None,
            "known_estimated_usd": known_cost,
            "reserved_input_tokens": self._reserved,
            "reserved_estimated_usd": self._cost(self._reserved),
            "accounted_input_tokens": self._input + self._reserved,
            "accounted_estimated_usd": self._cost(self._input + self._reserved),
            "blocked_reason": self._blocked, "failures": dict(self._failures),
        }


def _reject_constant(value):
    raise ValueError


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _http_transport_metadata(reply, key):
    metadata = {"endpoint": ENDPOINT, "http_status": reply.status}
    headers = {}
    for name, value in reply.getheaders():
        if type(name) is str and type(value) is str:
            headers.setdefault(name.lower(), []).append(value)
    for header, field in (
        ("x-typesafe-request-id", "provider_request_id"),
        ("retry-after", "retry_after"), ("retry-after-ms", "retry_after_ms"),
    ):
        values = headers.get(header, [])
        # Duplicate headers and any credential echo are not usable evidence.
        if len(values) != 1 or key in values[0]:
            continue
        value = values[0]
        if field == "provider_request_id":
            metadata[field] = value
        elif re.fullmatch(r"[0-9]{1,8}(?:\.[0-9]{1,6})?", value):
            metadata[field] = float(value) if "." in value else int(value)
    return validate_transport_metadata(metadata)


class HostedTransport:
    """One HTTPS attempt using a caller-supplied key, with no credential lookup."""

    __slots__ = ("_key",)

    def __init__(self, key):
        if type(key) is not str or not key or any(
            char.isspace() or not char.isascii() or not char.isprintable() for char in key
        ):
            raise JevError("invalid_key")
        self._key = key

    def __repr__(self):
        return "<HostedTransport>"

    def __call__(self, request):
        body = _json_bytes(request)
        code = None
        connection = None
        response_body = None
        metadata = {"endpoint": ENDPOINT}
        try:
            connection = HTTPSConnection(HOST, timeout=20, context=tls_context())
            connection.request(
                "POST", ENDPOINT, body=body,
                headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            )
            reply = connection.getresponse()
            metadata = _http_transport_metadata(reply, self._key)
            if reply.status in (401, 403):
                code = "http_auth_error"
            elif reply.status == 408:
                code = "request_timeout"
            elif reply.status == 429:
                code = "rate_limit"
            elif reply.status == 529:
                code = "overloaded"
            elif 500 <= reply.status <= 599:
                code = "server_error"
            elif reply.status != 200:
                code = "http_error"
            else:
                response_body = reply.read(RESPONSE_LIMIT + 1)
                metadata.update(
                    response_body_sha256=hashlib.sha256(response_body).hexdigest(),
                    response_bytes=len(response_body),
                    response_body_complete=len(response_body) <= RESPONSE_LIMIT,
                )
                if len(response_body) > RESPONSE_LIMIT:
                    code = "response_too_large"
        except ssl.SSLCertVerificationError:
            code = "tls_error"
        except TimeoutError:
            code = "timeout"
        except Exception:
            code = "network_error"
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    code = code or "network_error"
        if code:
            raise JevError(code, transport=metadata)
        try:
            data = json.loads(response_body, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        except (TypeError, ValueError, UnicodeError, RecursionError):
            code = "invalid_json"
        if code:
            raise JevError(code, transport=metadata)
        if type(data) is not dict:
            raise JevError("invalid_response", transport=metadata)
        # The reserved envelope is ours, never a provider-supplied body field.
        data["_transport"] = validate_transport_metadata(metadata)
        return data
