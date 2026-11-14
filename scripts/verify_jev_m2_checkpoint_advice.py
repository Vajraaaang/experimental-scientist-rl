"""Verify hosted development policy advice using hash-verified local checkpoints.

The bound snapshot verifies the complete run. This extra check reconstructs the
observations for every learned case and compares the original masked probabilities
and their argmax. It uses the frozen checkpoints, stored actions and local CPU.
"""
import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/jev-rl-m2/hosted-study-01'
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]

import numpy as np
import torch
from scripts import jev_m2_study as study


def verify_advice(output):
    run = RUN.resolve()
    output = output.resolve()
    if output.exists() or output.is_relative_to(run):
        raise ValueError('use a new receipt path outside the frozen run')
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['mode'] != 'hosted':
        raise ValueError('expected the completed hosted study')
    registration = json.loads((ROOT / 'docs/jev_rl_m2_analysis_registration.json').read_text())
    registered = [p for p in registration['plans'] if p['path'].endswith('/hosted-plan-study-01.json')]
    if len(registered) != 1 or study.file_hash(run / 'plan.json') != registered[0]['sha256']:
        raise ValueError('run plan is not the original registered plan')
    hashes = {name: study.file_hash(run / name) for name in ('manifest.json', 'plan.json', 'results.json')}
    replay = study.verify_run(run)
    if replay['failed_episodes'] or replay['not_run_episodes']:
        raise ValueError('checkpoint advice audit expects the completed cohort')
    result = json.loads((run / 'results.json').read_text())
    plan = study.read_plan(run / 'plan.json', files=False)
    tasks = study.validate_plan(plan, files=False)
    config = study.PilotConfig(**manifest['config'])
    models = {}
    counts = {'neural_cases': 0, 'policy_steps': 0}
    maximum_error = 0.0
    torch.set_num_threads(1)
    for case, row in zip(manifest['cases'], result['cases']):
        record = study.policy_record(manifest, case)
        if record is None:
            continue
        key = (record['path'], record['sha256'], study.canonical(record['calibration']))
        if key not in models:
            models[key] = study.load_policy(ROOT, record)
        policy = study.FrozenPolicy(models[key], record)
        env = study.ScientistEnv(tasks[case['task_index']], config)
        obs, _ = env.reset()
        counts['neural_cases'] += 1
        advice_seen = False
        steps = 0
        for event in study.read_journal(run / row['journal']):
            kind = event['kind']
            if kind in ('candidate_policy', 'rl_advice'):
                if advice_seen or event['step'] != steps:
                    raise ValueError(f'{case["case_id"]}: policy advice out of order')
                recorded = np.asarray(event['probabilities'], dtype=float)
                actual = np.asarray(policy.probabilities(obs), dtype=float)
                error = float(np.max(np.abs(actual - recorded)))
                maximum_error = max(maximum_error, error)
                if not np.array_equal(actual, recorded) or np.argmax(actual) != np.argmax(recorded):
                    raise ValueError(f'{case["case_id"]}: checkpoint advice differs at step {steps}')
                counts['policy_steps'] += 1
                advice_seen = True
            elif kind in ('candidate_action', 'action'):
                if not advice_seen or event['step'] != steps:
                    raise ValueError(f'{case["case_id"]}: action lacks reproduced policy advice')
                obs, _, _, _, _ = env.step(event['action_id'])
                steps += 1
                advice_seen = False
        if advice_seen or steps != len(row['result']['actions']):
            raise ValueError(f'{case["case_id"]}: checkpoint advice coverage is incomplete')
    study.verify_hashes(run, hashes)
    receipt = {
        'schema': 'jev-m2-checkpoint-advice-audit-v1',
        'run': 'hosted-study-01',
        'original_inputs_sha256': hashes,
        'audit_source_sha256': study.file_hash(Path(__file__)),
        'device': 'cpu',
        'hosted_calls_dispatched': 0,
        'verified_run_cases': replay['verified_episodes'],
        **counts,
        'distinct_policy_objects_loaded': len(models),
        'max_absolute_probability_error': maximum_error,
        'exact_probability_and_argmax_match': True,
        'scope': 'Original local checkpoints, stored hosted traces, frozen simulator, same software environment.',
    }
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_advice(args.output), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
