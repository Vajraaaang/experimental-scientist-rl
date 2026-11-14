"""Replay completed hosted actions across CPU platforms using frozen synthetic tasks.

The policy and forecast decisions come from the journal. This check reruns the
simulator, not the policy or Bayesian planner. The reference snapshot also offers
strict decision replay on the original numerical environment.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN = 'reports/jev-rl-m2/hosted-study-01'
TASKS = 'reports/jev-rl-m2/hosted-recorded-action-tasks.json'
TOLERANCE = 1e-12


def check_value(actual, expected):
    """Allow absolute roundoff only for floats, including measurement arrays."""
    if type(actual) is not type(expected):
        raise ValueError('recorded value type differs from simulator')
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise ValueError('recorded fields differ from simulator')
        return max((check_value(value, expected[k]) for k, value in actual.items()), default=0.)
    if isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError('recorded sequence length differs from simulator')
        return max((check_value(a, b) for a, b in zip(actual, expected, strict=True)), default=0.)
    if isinstance(actual, float):
        if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual - expected) > TOLERANCE:
            raise ValueError('recorded numerical outcome exceeds simulator tolerance')
        return abs(actual - expected)
    if actual != expected:
        raise ValueError('recorded discrete outcome differs from simulator')
    return 0.


def replay(root):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    from scripts.export_jev_m2_public import verify_published

    checked = verify_published(root, archive_only=True, replay_recorded_actions=True)
    if checked['status'] != 'passed' or not checked['hosted_pair_included']:
        raise ValueError('hosted archive verification failed')
    return checked


def replay_hosted_actions(root):
    """Reexecute the bound hosted evidence after receipt verification."""
    run_path = root / RUN
    source = run_path / 'source_snapshot'
    spec = importlib.util.spec_from_file_location('hosted_simulator_snapshot', source / 'scripts/jev_m2_study.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    from experimentlab.tasks import PrivateTask
    from experimentlab.worlds import Mechanism
    from integrations.jev_research import episode_result, public_state, read_journal

    manifest = json.loads((run_path / 'manifest.json').read_text())
    results = json.loads((run_path / 'results.json').read_text())
    saved = json.loads((root / TASKS).read_text())
    if saved['schema'] != 'jev-m2-recorded-action-tasks-v1':
        raise ValueError('unknown recorded task schema')
    tasks = []
    for record, planned in zip(saved['tasks'], manifest['tasks'], strict=True):
        mechanism = Mechanism(**record['mechanism'])
        if {k: v for k, v in record.items() if k != 'mechanism'} != planned:
            raise ValueError('recorded task metadata differs from frozen plan')
        if study.mechanism_hash(mechanism) != planned['mechanism_hash']:
            raise ValueError('recorded task parameters do not match frozen hash')
        tasks.append(PrivateTask(mechanism, tuple(planned['target']), planned['episode_seed']))

    config = study.PilotConfig(**manifest['config'])
    error = 0.
    steps = 0
    started = 0
    for case, row in zip(manifest['cases'], results['cases'], strict=True):
        recorded = row['result']
        if row['journal'] is None:
            if not recorded['status'].startswith('not_run_'):
                raise ValueError('attempted case lacks journal')
            continue
        events = read_journal(run_path / row['journal'])
        env = study.ScientistEnv(tasks[case['task_index']], config)
        obs, _ = env.reset()
        started += 1
        error = max(error, check_value(public_state(obs, env.actions), events[0]['state']))
        actions, rewards = [], []
        for index, event in enumerate(events):
            if event['kind'] not in ('action', 'candidate_action'):
                continue
            action = event['action_id']
            if type(action) is not int or not 0 <= action < len(env.actions) or not obs['action_mask'][action]:
                raise ValueError('recorded action is illegal in simulator replay')
            obs, reward, done, truncated, info = env.step(action)
            if truncated or info:
                raise ValueError('recorded action causes a simulator error')
            seen = events[index + 1]
            observation_kind = 'candidate_observation' if event['kind'] == 'candidate_action' else 'observation'
            actual = {'kind': observation_kind, 'step': len(actions), 'done': bool(done),
                      'state': public_state(obs, env.actions), 'reward': float(reward)}
            error = max(error, check_value(actual, seen))
            actions.append(action)
            rewards.append(float(reward))
            steps += 1
        if (recorded['status'] == 'completed') != env.done:
            raise ValueError('recorded terminal status differs from simulator')
        actual = episode_result(env, recorded['method'], actions, recorded['forecasts'], rewards,
                                recorded['status'], recorded['failure'])
        error = max(error, check_value(actual, {key: recorded[key] for key in actual}))

    return {'passed': True, 'simulator_dynamics_reexecuted': True, 'actions_taken_from_archive': True,
            'cases_replayed': started, 'step_calls': steps, 'abs_float_tolerance': TOLERANCE,
            'max_absolute_error': error, 'mechanism_hashes_verified': len(tasks), 'hosted_calls_dispatched': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(replay(args.root.resolve()), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
