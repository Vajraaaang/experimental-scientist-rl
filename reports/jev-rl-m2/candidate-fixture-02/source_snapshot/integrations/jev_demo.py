"""Optional TypeSafe Jev policy for a standalone synthetic demonstration."""
import argparse
from getpass import GetPassWarning, getpass
from http.client import HTTPException, HTTPSConnection
import json
import os
import re
import ssl
import sys
import warnings

from experimentlab.envs import PilotConfig, ScientistEnv
from experimentlab.tasks import continuous_family, make_task

HOST = "api.typesafe.ai"
RESPONSE_LIMIT = 256 * 1024


def describe(action):
    intervention = next(
        (f"suppress X{i} by dose {dose:g}" for i, dose in enumerate(action.intervention) if dose),
        "leave controls unchanged",
    )
    if action.kind == "commit":
        return f"Stop testing and {intervention} as the final intervention."
    panel = ", ".join(f"X{i}" for i in action.panel)
    return (
        f"Experiment: {intervention}, then measure {panel} after "
        f"{action.duration:g} time units. Cost {action.cost_ticks / 20:g}."
    )


def build_request(observation, catalogue, model):
    history = []
    for packet in observation["history"]:
        action_id = int(packet["action_id"])
        measurements = {
            f"X{i}": packet["readouts"][:, i].tolist()
            for i, measured in enumerate(packet["measured_mask"])
            if measured
        }
        history.append({
            "action": "baseline" if action_id == -1 else describe(catalogue[action_id]),
            "measurements": measurements,
        })
    return {
        "model": model,
        "state": {
            "task": "Simulated regulatory system with uncertain control effects. Match the target outcomes.",
            "targets": {
                f"X{i}": float(target)
                for i, target in zip(observation["goal_mask"].nonzero()[0], observation["target"])
            },
            "remaining_budget": observation["budget_ticks"] / 20,
            "history": history,
        },
        "questions": {
            "next_action": {
                "type": "choice",
                "instructions": (
                    "Choose the next action. Buy information when it can improve the final "
                    "intervention enough to justify the cost. Otherwise commit to an intervention "
                    "that best matches the target outcomes with low effort."
                ),
                "criteria": {
                    f"action_{i}": describe(action)
                    for i, action in enumerate(catalogue)
                    if observation["action_mask"][i]
                },
            }
        },
    }


def pilot_request(observation, catalogue, model, predict):
    request = build_request(observation, catalogue, model)
    if predict and observation["experiment_count"] == 0:
        criteria = request["questions"]["next_action"]["criteria"]
        for i, action in enumerate(catalogue):
            if action.kind == "commit":
                criteria.pop(f"action_{i}", None)
        if not criteria:
            raise RuntimeError("Prediction mode requires an affordable first experiment.")
    return request


def tls_context():
    context = ssl.create_default_context()
    # python.org's macOS installer may leave the OpenSSL CA bundle unset.
    system_bundle = "/etc/ssl/cert.pem"
    if sys.platform == "darwin" and not context.get_ca_certs() and os.path.isfile(system_bundle):
        context.load_verify_locations(cafile=system_bundle)
    return context


def call_jev(request, key):
    try:
        context = tls_context()
    except OSError:
        raise RuntimeError("Could not load Python's trusted HTTPS certificates.") from None
    connection = HTTPSConnection(HOST, timeout=20, context=context)
    try:
        connection.request(
            "POST", "/v1/systemone",
            body=json.dumps(request, allow_nan=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status in (401, 403):
            raise RuntimeError(f"TypeSafe denied access (HTTP {response.status}). Check the key and account access.")
        if response.status != 200:
            raise RuntimeError(f"TypeSafe returned HTTP {response.status}.")
        body = response.read(RESPONSE_LIMIT + 1)
    except ssl.SSLCertVerificationError:
        raise RuntimeError("Python could not verify TypeSafe's HTTPS certificate. Check the trusted CA bundle.") from None
    except (OSError, HTTPException):
        raise RuntimeError("Could not complete the TypeSafe HTTPS request.") from None
    finally:
        connection.close()
    if len(body) > RESPONSE_LIMIT:
        raise RuntimeError("TypeSafe response exceeded the demo size limit.")
    try:
        data = json.loads(body)
        model = data["model"]
        if not isinstance(model, str) or not re.fullmatch(r"jev-[\w.-]{1,60}", model):
            raise ValueError
        answers = data["answers"]
        if not isinstance(answers, dict):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("TypeSafe returned an invalid response.") from None
    return model, answers


def choose(request, key):
    model, answers = call_jev(request, key)
    try:
        answer = answers["next_action"]
        choice = answer["choice"]
        confidence = answer["confidence"]
        options = request["questions"]["next_action"]["criteria"]
        if (
            answer["type"] != "choice" or not isinstance(choice, str) or choice not in options
            or type(confidence) not in (float, int) or not 0 <= confidence <= 1
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("TypeSafe returned an invalid action or response.") from None
    return int(choice.split("_", 1)[1]), model, confidence


def forecast(request, observation, action, key):
    baseline = observation["history"][0]["readouts"]
    thresholds = {f"X{i}": float(baseline[:, i].mean()) for i in action.panel}
    forecast_request = {
        "model": request["model"],
        "state": {**request["state"], "planned_experiment": describe(action)},
        "questions": {
            name: {
                "type": "noul",
                "instructions": (
                    f"Will the mean of the upcoming measured {name} values "
                    f"be strictly above its measured baseline mean of {threshold!r}?"
                ),
            }
            for name, threshold in thresholds.items()
        },
    }
    resolved_model, answers = call_jev(forecast_request, key)
    try:
        probabilities = {}
        for name in thresholds:
            answer = answers[name]
            probability = answer["noul"]
            if answer["type"] != "noul" or type(probability) not in (float, int) or not 0 <= probability <= 1:
                raise ValueError
            probabilities[name] = probability
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("TypeSafe returned an invalid prediction.") from None
    return resolved_model, thresholds, probabilities


def run_episode(environment, model, key, *, predict=False):
    observation, _ = environment.reset()
    decisions = []
    predictions = []
    api_calls = 0
    total_reward = 0.0
    for _ in range(environment.config.max_experiments + 1):
        request = pilot_request(observation, environment.actions, model, predict)
        action, resolved_model, confidence = choose(request, key)
        api_calls += 1
        selected = environment.actions[action]
        prediction = None
        if predict and selected.kind == "experiment":
            prediction = forecast(request, observation, selected, key)
            api_calls += 1
        decisions.append({
            "action_id": action,
            "action": describe(environment.actions[action]),
            "model": resolved_model,
            "confidence": confidence,
        })
        observation, reward, done, truncated, info = environment.step(action)
        total_reward += reward
        if truncated or info.get("error"):
            raise RuntimeError("The simulated episode ended unexpectedly.")
        if prediction is not None:
            prediction_model, thresholds, probabilities = prediction
            readouts = observation["history"][-1]["readouts"]
            for i in selected.panel:
                name = f"X{i}"
                observed = float(readouts[:, i].mean())
                above = observed > thresholds[name]
                predictions.append({
                    "action_id": action,
                    "measurement": name,
                    "model": prediction_model,
                    "baseline_mean": thresholds[name],
                    "probability_above_baseline": probabilities[name],
                    "observed_mean": observed,
                    "observed_above_baseline": above,
                    "brier_score": (probabilities[name] - int(above)) ** 2,
                })
        if done:
            return {
                "scope": "single synthetic Jev prediction demonstration" if predict else "single synthetic Jev demonstration",
                "api_calls": api_calls,
                "decisions": decisions,
                **({"predictions": predictions} if predict else {}),
                "simulator_result": environment.private_result,
                "simulator_return": total_reward,
            }
    raise RuntimeError("The simulated episode exceeded its decision limit.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call TypeSafe for one synthetic episode.")
    parser.add_argument("--predict", action="store_true", help="Require a first test and forecast each purchased measurement before running it.")
    parser.add_argument("--model", default="jev-latest")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"jev-[\w.-]{1,60}", args.model):
        parser.error("expected a Jev model name such as jev-latest")
    config = PilotConfig()
    environment = ScientistEnv(make_task(continuous_family(96119, 1)[0], 96120, config), config)
    if not args.live:
        observation, _ = environment.reset()
        print(json.dumps(pilot_request(observation, environment.actions, args.model, args.predict), indent=2, allow_nan=False))
        return 0
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        if not sys.stdin.isatty():
            parser.error("Set TYPESAFE_API_KEY or run interactively for a hidden prompt.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", GetPassWarning)
                key = getpass("TypeSafe API key: ")
        except (EOFError, GetPassWarning):
            parser.error("A hidden prompt is unavailable. Set TYPESAFE_API_KEY.")
    if not key or any(char.isspace() or not char.isascii() or not char.isprintable() for char in key):
        parser.error("Expected a nonempty TypeSafe key without whitespace.")
    try:
        result = run_episode(environment, args.model, key, predict=args.predict)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
