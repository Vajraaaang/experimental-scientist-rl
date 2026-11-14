"""Offline M2 planning, bounded fixture/hosted execution, and weight-free replay.

Preparing a larger allowance is a proposal, not authorization. There is no
automatic retry, resume, confirmation pool, final pool, or credential file lookup.
"""
import argparse
from collections import Counter
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import re
import shutil
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]

import numpy as np
import torch
from experimentlab.envs import PilotConfig, ScientistEnv
from experimentlab.learning import ActorCritic, TOKEN_SIZE
from experimentlab.tasks import continuous_family, make_task, mechanism_hash
from research_stage import UnfilteredBank, quadrature_family
from integrations.jev_client import CallLimits, HostedTransport, JevClient, JevError
from integrations.jev_m2_episode import ROLES, run_candidate, verify_candidate
from integrations.jev_research import (
    FixtureTransport, FrozenPolicy, Journal, canonical, digest, read_journal,
    run_episode, summarize_results, verify_episode,
)
from scripts.jev_rl_study import (
    BASELINES, NEURAL, file_hash, live_key, load_policy, model_records,
    previous_hashes, receipt_stats, trace_index, validate_api_accounting,
    verify_client_stats, verify_hashes, write_new,
)

SEEDS = [20, 21, 22, 23, 24]
INDEPENDENT = ('jev_only', 'probe')
TRAINED = tuple(role for role in ROLES if role not in INDEPENDENT) + ('incumbent_m1', 'ppo_v6')
IMITATION = tuple(name for name in NEURAL if name != 'ppo')
CALL_BOUNDS = dict.fromkeys(BASELINES + IMITATION + ('ppo', 'ppo_v6'), 0)
CALL_BOUNDS.update(forecast_only=4, rerank=9, rerank_shadow=9, arbitration=9,
                   hidden=9, chance=9, public=9, jev_only=9, probe=3, incumbent_m1=9)
SPEC = 'autonomous_scientist_build_spec.md'
REGISTRY_SOURCE = 'reports/ppo-efficiency-v6/selection.json'
SCHEDULE_SOURCE = 'reports/ppo-efficiency-v6/fresh-locked/freeze.json'
RUNTIME_FILES = tuple('src/experimentlab/' + name + '.py' for name in (
    '__init__', 'envs', 'tasks', 'worlds', 'learning', 'cost_policy', 'baselines',
    'inference', 'evaluation', 'bo', 'cli')) + (
    'scripts/research_stage.py', 'scripts/jev_rl_study.py', 'scripts/jev_m2_study.py',
    'integrations/jev_client.py', 'integrations/jev_demo.py', 'integrations/jev_research.py',
    'integrations/jev_replay.py', 'integrations/jev_candidates.py', 'integrations/jev_m2_episode.py',
    'docs/jev_rl_protocol.md', 'docs/jev_rl_m2_protocol.md', 'docs/jev_rl_m2_hosted_protocol.md',
    'requirements.lock',
)
NOT_RUN = {'not_run_wall', 'not_run_model_drift', 'not_run_source_change',
           'not_run_gate_failure', 'not_run_api_limit', 'not_run_runner_failure'}
FIXTURE_REUSE_SCOPE = 'disclosed reuse of development cases for fixture-to-hosted engineering validation; not an independent test'


def source_hashes(root):
    """Only the public runtime closure; the original handoff is hash-only metadata."""
    return {name: file_hash(Path(root) / name) for name in RUNTIME_FILES}


def _seeds(seeds):
    if seeds != [20] and seeds != SEEDS:
        raise ValueError('training seeds must be 20 or 20,21,22,23,24 in that order')
    if any(type(seed) is not int for seed in seeds):
        raise ValueError('training seeds must be integers')
    return list(seeds)


def case_plan(mechanisms, seeds):
    if type(mechanisms) is not int or mechanisms not in (1, 8):
        raise ValueError('mechanisms must be 1 or 8')
    seeds = _seeds(seeds)
    cases = []
    for index in range(mechanisms):
        ordered = [(method, None) for method in BASELINES]
        ordered += [(method, seeds[0]) for method in TRAINED + IMITATION]
        ordered += [(method, None) for method in INDEPENDENT]
        ordered += [(method, seed) for seed in seeds[1:] for method in TRAINED + IMITATION]
        for method, seed in ordered:
            cases.append({'case_id': f'm{index:03}-{method}-s{seed if seed is not None else "none"}',
                          'task_index': index, 'method': method, 'training_seed': seed,
                          'engine': 'm2' if method in ROLES else 'm1',
                          'engine_method': {'incumbent_m1': 'jev_rl', 'ppo_v6': 'ppo'}.get(method, method),
                          'max_calls': CALL_BOUNDS[method],
                          'stage': 'pilot' if index == 0 and seed in (None, 20) else 'expansion'})
    return cases


def planned_limits(max_calls, max_usd):
    if (type(max_calls) is not int or not 1 <= max_calls <= 2800 or
            type(max_usd) not in (int, float) or not math.isfinite(max_usd) or not 0 < max_usd <= 1):
        raise ValueError('planned allowance must be 1..2800 calls and 0 < USD <= 1.00')
    return asdict(CallLimits(max_calls=max_calls, max_request_bytes=65536,
                            max_total_request_bytes=min(max_calls * 65536, 200 * 1024 * 1024),
                            max_input_tokens=math.floor(max_usd / .042 * 1_000_000),
                            max_estimated_usd=float(max_usd)))


def fixture_limits(calls):
    # These synthetic responses report zero tokens. This allowance never reaches
    # HostedTransport and does not increase the proposed live allowance.
    return asdict(CallLimits(max_calls=calls, max_request_bytes=65536,
                            max_total_request_bytes=calls * 65536,
                            max_input_tokens=100_000_000, max_estimated_usd=10.))


def development_tasks(count, config):
    if type(count) is not int or count not in (1, 8):
        raise ValueError('development mechanism count must be 1 or 8')
    return [make_task(mechanism, 991332000 + index, config)
            for index, mechanism in enumerate(continuous_family(991331, count))]


def assert_disjoint(hashes, forbidden):
    if len(hashes) != len(set(hashes)):
        raise ValueError('duplicate development mechanisms')
    if set(hashes) & set(forbidden):
        raise ValueError('development mechanism overlap with prior data')


def exclusions(root):
    """Read metadata only; never generate confirmation/final mechanisms or outcomes."""
    root = Path(root)
    forbidden, provenance = previous_hashes(root)
    additional = set()
    inherited = set()
    confirmation = set()
    fixture_hashes = set()
    fixture_manifests = {}
    def visit(value, context=''):
        if isinstance(value, dict):
            for key, item in value.items():
                # Reuse disclosures are not independent assessment exclusions.
                if key != 'prior_fixture_overlap':
                    visit(item, context + '/' + str(key))
        elif isinstance(value, list):
            for item in value:
                visit(item, context)
        elif isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value):
            if 'historical_identifiers' in context.split('/') or context.endswith('/exclusions/hashes'):
                inherited.add(value)
            else:
                additional.add(value)
            if 'confirmation' in context:
                confirmation.add(value)
    for folder in ('jev-rl-m1', 'jev-rl-m2'):
        directory = root / 'reports' / folder
        paths = set(directory.rglob('*manifest*.json')) | set(directory.rglob('freeze.json'))
        for path in sorted(paths):
            if 'source_snapshot' in path.parts:
                continue
            data = json.loads(path.read_text())
            name, sha = str(path.relative_to(root)), file_hash(path)
            provenance[name] = sha
            if folder == 'jev-rl-m2' and data.get('mode') == 'fixture':
                fixture_manifests[name] = sha
                for task in data.get('tasks', []):
                    value = task.get('mechanism_hash')
                    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
                        raise ValueError('invalid prior fixture mechanism metadata')
                    fixture_hashes.add(value)
                # Keep all independent historical/train/tuning/final exclusions;
                # only this fixture's assessment task list is separated.
                visit({key: value for key, value in data.items() if key not in ('tasks', 'admission')})
            else:
                # Prior hosted M2 assessment overlap remains an exclusion.
                visit(data)
    train = {mechanism_hash(m) for m in continuous_family(7401, 4096)}
    tuning = {mechanism_hash(m) for m in continuous_family(991311, 128)}
    forbidden.update(additional | (inherited - fixture_hashes) | train | tuning)
    return {'hashes': sorted(forbidden), 'metadata_sha256': provenance,
            'prior_fixture_overlap': {'mechanism_hashes': sorted(fixture_hashes),
                'metadata_sha256': fixture_manifests, 'scope': FIXTURE_REUSE_SCOPE},
            'training_7401_hash_count': len(train), 'tuning_991311_hash_count': len(tuning),
            'confirmation_991321_metadata_hash_count': len(confirmation),
            'confirmation_check': 'metadata only; absent hashes are not generated',
            'confirmation_constructed': False, 'final_constructed': False}


def fixture_reuse(excluded, hashes):
    previous = excluded.get('prior_fixture_overlap', {
        'mechanism_hashes': [], 'metadata_sha256': {}, 'scope': FIXTURE_REUSE_SCOPE})
    reused = sorted(set(previous['mechanism_hashes']) & set(hashes))
    return {**previous, 'reused_mechanism_hashes': reused, 'reused_count': len(reused)}


def _record_metadata(root, record, seed, check_file):
    root = Path(root).resolve()
    if not isinstance(record, dict) or record.get('seed', seed) != seed or type(record.get('seed', seed)) is not int:
        raise ValueError('checkpoint training seed mismatch')
    sha = record.get('sha256')
    if type(sha) is not str or not re.fullmatch('[0-9a-f]{64}', sha):
        raise ValueError('checkpoint hash missing or invalid')
    raw = record.get('path')
    if type(raw) is not str:
        raise ValueError('checkpoint path must stay within repository reports')
    path = (root / raw).resolve()
    if not path.is_relative_to(root / 'reports') or path.suffix != '.pt':
        raise ValueError('checkpoint path must stay within repository reports')
    calibration = record.get('calibration', {})
    if not isinstance(calibration, dict):
        raise ValueError('checkpoint calibration must be an object')
    canonical(calibration)
    normalized = {**record, 'seed': seed, 'path': str(path.relative_to(root)), 'calibration': calibration}
    if check_file:
        verify_hashes(root, {normalized['path']: sha})
    return normalized


def normalize_ppo_records(root, records, seeds):
    if not isinstance(records, dict) or set(records) != {str(seed) for seed in seeds}:
        raise ValueError('PPO records must map exactly the requested seed strings to checkpoint records')
    result = {}
    for seed in seeds:
        record = _record_metadata(root, records[str(seed)], seed, True)
        record['checkpoint_binding'] = checkpoint_binding(root, record)
        result[str(seed)] = record
    return result


def _calibration(value):
    allowed = ('duration_penalty', 'dose_penalty', 'stop_bonus')
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError('invalid checkpoint calibration')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in value.values()):
        raise ValueError('invalid checkpoint calibration')
    return {name: float(value.get(name, 0.)) for name in allowed}


def checkpoint_binding(root, record):
    """Inspect only the explicitly selected, local, hash-verified M2 checkpoint."""
    verify_hashes(root, {record['path']: record['sha256']})
    torch.set_num_threads(1)
    state = torch.load(Path(root) / record['path'], map_location='cpu', weights_only=False)
    provenance = state.get('provenance', {}) if isinstance(state, dict) else {}
    if (provenance.get('seed') != record['seed'] or type(provenance.get('seed')) is not int or
            provenance.get('trainer_schema') != 'm2-ppo-train-v1' or
            provenance.get('scope') != 'm2_development_only'):
        raise ValueError('M2 checkpoint seed/trainer provenance mismatch')
    config = provenance.get('config', {})
    architecture = state.get('architecture', {})
    if (not isinstance(architecture, dict) or
            set(architecture) != {'input_size', 'actions', 'hidden', 'kind', 'horizon'} or
            architecture.get('input_size') != TOKEN_SIZE or architecture.get('actions') != 45 or
            architecture.get('horizon') != 5 or architecture.get('kind') not in ('history', 'recurrent') or
            architecture.get('hidden') not in (64, 128, 256, 512) or
            config.get('seed') != record['seed'] or config.get('hidden') != architecture['hidden'] or
            config.get('kind') != architecture['kind'] or
            ('architecture' in record and record['architecture'] != architecture)):
        raise ValueError('checkpoint architecture differs from M2 provenance')
    if 'calibration' not in provenance or _calibration(provenance['calibration']) != _calibration(record['calibration']):
        raise ValueError('checkpoint calibration differs from provenance')
    try:
        with torch.random.fork_rng(devices=[]):
            model = ActorCritic(**architecture)
            model.load_state_dict(state['model'], strict=True)
    except (RuntimeError, KeyError, TypeError) as error:
        raise ValueError('checkpoint architecture/state mismatch') from error
    return {'sha256': record['sha256'], 'seed': record['seed'], 'architecture': architecture,
            'calibration': _calibration(provenance['calibration']),
            'trainer_schema': provenance['trainer_schema'], 'scope': provenance['scope']}


def registered_inputs(root, seeds):
    """Read only the fixed original-v6 registry and preserved design artifact."""
    root = Path(root)
    registry_sha = file_hash(root / REGISTRY_SOURCE)
    schedule_sha = file_hash(root / SCHEDULE_SOURCE)
    records = model_records(root, seeds)
    for seed in seeds:
        records[f'ppo_v6-seed{seed}'] = copy.deepcopy(records[f'ppo-seed{seed}'])
    frozen = json.loads((root / SCHEDULE_SOURCE).read_text())
    schedules = {'fixed': frozen['fixed'], 'fixed_quality': frozen['fixed_quality'],
                 'additive': frozen['additive_schedule'], 'probe': frozen['fixed']}
    if file_hash(root / REGISTRY_SOURCE) != registry_sha or file_hash(root / SCHEDULE_SOURCE) != schedule_sha:
        raise ValueError('registered source artifacts changed while reading')
    return {'registry_source': {'path': REGISTRY_SOURCE, 'sha256': registry_sha},
            'schedule_source': {'path': SCHEDULE_SOURCE, 'sha256': schedule_sha},
            'models': records, 'schedules': schedules}


def verify_registered_inputs(plan, root=ROOT, *, files=False):
    registered = plan['registered_inputs']
    for field, path in (('registry_source', REGISTRY_SOURCE), ('schedule_source', SCHEDULE_SOURCE)):
        source = registered.get(field, {})
        if (source.get('path') != path or set(source) != {'path', 'sha256'} or
                type(source.get('sha256')) is not str or not re.fullmatch('[0-9a-f]{64}', source['sha256']) or
                plan[field] != source):
            raise ValueError('registered source path/hash binding mismatch')
    expected = {f'{method}-seed{seed}' for method in (*NEURAL, 'ppo_v6') for seed in plan['training_seeds']}
    if set(registered['models']) != expected:
        raise ValueError('registered model inventory mismatch')
    if plan['schedules'] != registered['schedules']:
        raise ValueError('preserved schedule binding mismatch')
    for name, record in registered['models'].items():
        seed = int(name.rsplit('seed', 1)[1])
        if set(record) != {'seed', 'path', 'sha256', 'calibration'} or record != _record_metadata(root, record, seed, False):
            raise ValueError('registered model metadata is not normalized')
        if name.startswith('ppo_v6-') and record != registered['models'][f'ppo-seed{seed}']:
            raise ValueError('incumbent v6 record differs from the original selected PPO')
        if not (name.startswith('ppo-') and plan['ppo_source'] == 'm2_checkpoint_records'):
            if canonical(plan['models'].get(name)) != canonical(record):
                raise ValueError('registered model method/seed/path/hash/calibration mismatch')
    if files and canonical(registered_inputs(root, plan['training_seeds'])) != canonical(registered):
        raise ValueError('registered inputs differ from the fixed registry/schedule source artifacts')


def prepare_plan(root=ROOT, *, mechanisms=1, seeds=None, max_calls=32, max_usd=.10,
                 ppo_records=None, wall_seconds=1200):
    root = Path(root).resolve()
    seeds = _seeds([20] if seeds is None else seeds)
    cases = case_plan(mechanisms, seeds)
    limits = planned_limits(max_calls, max_usd)
    if type(wall_seconds) not in (int, float) or not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 1200:
        raise ValueError('whole-study wall cap must be greater than zero and at most 1200 seconds')
    config = PilotConfig(final_cultures=128)
    tasks = development_tasks(mechanisms, config)
    hashes = [mechanism_hash(t.mechanism) for t in tasks]
    excluded = exclusions(root)
    assert_disjoint(hashes, excluded['hashes'])
    registered = registered_inputs(root, seeds)
    models = copy.deepcopy(registered['models'])
    if ppo_records is not None:
        raw = json.loads(Path(ppo_records).read_text()) if isinstance(ppo_records, (str, Path)) else ppo_records
        for seed, record in normalize_ppo_records(root, raw, seeds).items():
            models[f'ppo-seed{seed}'] = record
    schedules = copy.deepcopy(registered['schedules'])
    sources = source_hashes(root)
    worst = sum(case['max_calls'] for case in cases)
    plan = {'schema_version': 4, 'kind': 'offline-plan', 'scope': 'jev-rl-m2-development',
            'config': asdict(config), 'generator_seed': 991331,
            'tasks': [{'episode_seed': task.seed, 'mechanism_hash': sha,
                       'target': [float(v) for v in task.target]} for task, sha in zip(tasks, hashes)],
            'training_seeds': seeds, 'models': models, 'cases': cases, 'schedules': schedules,
            'ppo_source': 'registered_v6' if ppo_records is None else 'm2_checkpoint_records',
            'registry_source': registered['registry_source'], 'schedule_source': registered['schedule_source'],
            'registered_inputs': registered,
            'requested_model': 'jev-1.14.0', 'trace_id': uuid4().hex,
            'sources': sources, 'prompt_sources': {name: sha for name, sha in sources.items()
                if name in ('integrations/jev_candidates.py', 'integrations/jev_research.py',
                            'integrations/jev_m2_episode.py', 'integrations/jev_demo.py')},
            'original_spec': {'path': SPEC, 'sha256': file_hash(root / SPEC), 'copied': False},
            'exclusions': excluded, 'prior_fixture_overlap': fixture_reuse(excluded, hashes),
            'planned_api_limits': limits,
            'worst_case_api_calls': worst, 'complete_call_allowance': max_calls >= worst,
            'allowance_status': 'proposal only; larger caps require explicit user authorization',
            'allowance_note': 'Complete call allowance does not guarantee completion within byte, token, cost, or wall caps.',
            'wall_cap_seconds': float(wall_seconds), 'network_timeout_seconds': 20,
            'staged_scope': {'pilot_case_ids': [c['case_id'] for c in cases if c['stage'] == 'pilot'],
                'expansion_requires': ['complete first cohort', 'valid independent replay',
                    'valid receipts and usage', 'complete forecast coverage', 'unchanged model and sources']},
            'price_usd_per_million_input_tokens': .042, 'device': 'cpu', 'threads': 1,
            'confirmation_constructed': False, 'final_test_opened': False,
            'scope_note': 'Development only; fixture responses cannot establish Jev efficacy.'}
    validate_plan(plan, root=root, files=False)
    return plan


def prepare(output, **kwargs):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite a plan')
    plan = prepare_plan(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(output, plan)
    return {'plan': str(output), 'worst_case_api_calls': plan['worst_case_api_calls'],
            'planned_api_limits': plan['planned_api_limits'],
            'complete_call_allowance': plan['complete_call_allowance'],
            'allowance_status': plan['allowance_status'], 'offline': True}


def validate_plan(plan, *, root=ROOT, files=False):
    if (plan.get('schema_version') != 4 or plan.get('kind') != 'offline-plan' or
            plan.get('scope') != 'jev-rl-m2-development' or plan.get('generator_seed') != 991331 or
            plan.get('config') != asdict(PilotConfig(final_cultures=128)) or
            plan.get('requested_model') != 'jev-1.14.0' or plan.get('final_test_opened') is not False or
            plan.get('confirmation_constructed') is not False or plan.get('device') != 'cpu' or
            plan.get('threads') != 1 or plan.get('network_timeout_seconds') != 20):
        raise ValueError('unsupported M2 development plan')
    seeds = _seeds(plan['training_seeds'])
    cases = case_plan(len(plan['tasks']), seeds)
    if canonical(cases) != canonical(plan['cases']):
        raise ValueError('missing, extra, reordered, or relabelled planned cases')
    worst = sum(c['max_calls'] for c in cases)
    limits = plan['planned_api_limits']
    if limits != planned_limits(limits['max_calls'], limits['max_estimated_usd']):
        raise ValueError('unbounded or inconsistent planned API limits')
    if plan['worst_case_api_calls'] != worst or plan['complete_call_allowance'] is not (limits['max_calls'] >= worst):
        raise ValueError('worst-case call allowance mismatch')
    if (type(plan['wall_cap_seconds']) not in (int, float) or not math.isfinite(plan['wall_cap_seconds']) or
            not 0 < plan['wall_cap_seconds'] <= 1200 or not re.fullmatch('[0-9a-f]{32}', plan['trace_id'])):
        raise ValueError('invalid wall cap or trace ID')
    if plan['staged_scope']['pilot_case_ids'] != [c['case_id'] for c in cases if c['stage'] == 'pilot']:
        raise ValueError('pilot stage mismatch')
    if set(plan['sources']) != set(RUNTIME_FILES) or any(not re.fullmatch('[0-9a-f]{64}', sha) for sha in plan['sources'].values()):
        raise ValueError('runtime source inventory mismatch')
    if any(plan['sources'].get(name) != sha for name, sha in plan['prompt_sources'].items()):
        raise ValueError('prompt source mismatch')
    expected_models = {f'{method}-seed{seed}' for method in (*NEURAL, 'ppo_v6') for seed in seeds}
    if set(plan['models']) != expected_models:
        raise ValueError('model registry does not cover exactly the planned methods/seeds')
    for name, record in plan['models'].items():
        normalized = _record_metadata(root, record, int(name.rsplit('seed', 1)[1]), files)
        if canonical(normalized) != canonical(record):
            raise ValueError('checkpoint record must use normalized repository paths')
        if name.startswith('ppo-') and plan['ppo_source'] == 'm2_checkpoint_records':
            binding = record.get('checkpoint_binding', {})
            if (binding.get('sha256') != record['sha256'] or binding.get('seed') != record['seed'] or
                    binding.get('calibration') != _calibration(record['calibration']) or
                    binding.get('trainer_schema') != 'm2-ppo-train-v1' or binding.get('scope') != 'm2_development_only'):
                raise ValueError('checkpoint provenance binding mismatch')
            if files and canonical(checkpoint_binding(root, record)) != canonical(binding):
                raise ValueError('checkpoint provenance binding differs from actual weights')
    if plan['ppo_source'] not in ('registered_v6', 'm2_checkpoint_records'):
        raise ValueError('unknown PPO record source')
    verify_registered_inputs(plan, root, files=files)
    if set(plan['schedules']) != {'fixed', 'fixed_quality', 'additive', 'probe'}:
        raise ValueError('schedule inventory mismatch')
    for schedule in plan['schedules'].values():
        if not isinstance(schedule, list) or len(schedule) > 4 or any(type(a) is not int or not 0 <= a < 36 for a in schedule):
            raise ValueError('invalid fixed schedule')
    if plan['schedules']['probe'] != plan['schedules']['fixed'] or len(plan['schedules']['probe']) != 3:
        raise ValueError('three-assay common probe must match preserved fixed schedule')
    tasks = development_tasks(len(plan['tasks']), PilotConfig(**plan['config']))
    for task, record in zip(tasks, plan['tasks']):
        if record != {'episode_seed': task.seed, 'mechanism_hash': mechanism_hash(task.mechanism),
                      'target': [float(v) for v in task.target]}:
            raise ValueError('development task reconstruction mismatch')
    assert_disjoint([t['mechanism_hash'] for t in plan['tasks']], plan['exclusions']['hashes'])
    if plan['prior_fixture_overlap'] != fixture_reuse(plan['exclusions'], [t['mechanism_hash'] for t in plan['tasks']]):
        raise ValueError('prior fixture reuse disclosure mismatch')
    if files:
        verify_hashes(root, plan['sources'])
    return tasks


def read_plan(path, *, files=True):
    plan = json.loads(Path(path).read_text())
    if files:
        admit_plan(plan)
    else:
        validate_plan(plan, files=False)
    return plan


def admission_receipt(plan, current):
    """Compact receipt of an admission check, not a replacement for that check."""
    hashes = [task['mechanism_hash'] for task in plan['tasks']]
    checked = sorted(set(plan['exclusions']['hashes']) | set(current['hashes']))
    assert_disjoint(hashes, checked)
    return {'schema_version': 1, 'phase': 'before_credentials_output_and_dispatch',
            'checked_at_utc': datetime.now(timezone.utc).isoformat(),
            'plan_content_sha256': digest(plan),
            'registered_inputs_sha256': digest(plan['registered_inputs']),
            'models_sha256': digest(plan['models']), 'runtime_sources_sha256': digest(plan['sources']),
            'task_hashes': hashes, 'independent_overlap': [],
            'exclusion_identifier_count': len(checked), 'exclusion_identifiers_sha256': digest(checked),
            'current_exclusion_metadata_sha256': current.get('metadata_sha256', {}),
            'prior_fixture_overlap': fixture_reuse(current, hashes),
            'registered_checkpoint_hashes_verified': True,
            'selected_m2_provenance_verified': plan['ppo_source'] == 'm2_checkpoint_records',
            'checkpoint_inference_reproduced': False}


def admit_plan(plan, root=ROOT):
    """Recheck present registry, checkpoints, designs, sources and independent data."""
    validate_plan(plan, root=root, files=True)
    return admission_receipt(plan, exclusions(root))


def verify_admission(receipt, plan):
    expected = {'schema_version': 1, 'phase': 'before_credentials_output_and_dispatch',
                'plan_content_sha256': digest(plan),
                'registered_inputs_sha256': digest(plan['registered_inputs']),
                'models_sha256': digest(plan['models']), 'runtime_sources_sha256': digest(plan['sources']),
                'task_hashes': [task['mechanism_hash'] for task in plan['tasks']],
                'independent_overlap': [], 'registered_checkpoint_hashes_verified': True,
                'selected_m2_provenance_verified': plan['ppo_source'] == 'm2_checkpoint_records',
                'checkpoint_inference_reproduced': False}
    if any(canonical(receipt.get(key)) != canonical(value) for key, value in expected.items()):
        raise ValueError('admission receipt binding mismatch')
    try:
        timestamp = datetime.fromisoformat(receipt['checked_at_utc'])
        if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise ValueError
    except (TypeError, KeyError, ValueError):
        raise ValueError('invalid admission timestamp') from None
    count = receipt['exclusion_identifier_count']
    if type(count) is not int or count < len(set(plan['exclusions']['hashes'])):
        raise ValueError('admission lost frozen independent exclusions')
    sha = receipt['exclusion_identifiers_sha256']
    if type(sha) is not str or not re.fullmatch('[0-9a-f]{64}', sha):
        raise ValueError('invalid admission exclusion digest')
    for path, sha in receipt['current_exclusion_metadata_sha256'].items():
        if (Path(path).is_absolute() or '..' in Path(path).parts or
                type(sha) is not str or not re.fullmatch('[0-9a-f]{64}', sha)):
            raise ValueError('invalid admission metadata binding')
    reuse = receipt['prior_fixture_overlap']
    if reuse.get('scope') != FIXTURE_REUSE_SCOPE or reuse != fixture_reuse({'prior_fixture_overlap': reuse}, expected['task_hashes']):
        raise ValueError('admission fixture reuse disclosure mismatch')


def policy_record(manifest, case):
    method, seed = case['method'], case['training_seed']
    name = 'ppo_v6' if method in ('incumbent_m1', 'ppo_v6') else ('ppo' if method in TRAINED else method)
    return manifest['models'].get(f'{name}-seed{seed}') if seed is not None else None


def not_run_row(case, reason):
    if reason not in NOT_RUN:
        raise ValueError('unknown not-run reason')
    return {'case_id': case['case_id'], 'method': case['method'], 'journal': None,
            'journal_sha256': None, 'api_receipts': [],
            'result': {'method': case.get('engine_method', case['method']), 'status': reason,
                       'failure': reason, 'actions': [], 'forecasts': [],
                       'final_mse': None, 'return_': None, 'effort': None}}


def verify_case(output, manifest, case, row, task, config, bank):
    if row['case_id'] != case['case_id'] or row.get('method') != case['method']:
        raise ValueError('case label mismatch')
    if row['result']['status'] in NOT_RUN:
        if canonical(row) != canonical(not_run_row(case, row['result']['status'])):
            raise ValueError('not-run row contains invented evidence')
        return row['result']
    name = 'episodes/' + case['case_id'] + '.jsonl'
    if row['journal'] != name:
        raise ValueError('case journal path mismatch')
    verify_hashes(output, {name: row['journal_sha256']})
    path = Path(output) / name
    hosted = case['max_calls'] > 0
    binding = {'model': manifest['model'] if hosted else None,
               'trace_id': manifest['trace_id'] if hosted else None,
               'policy_metadata': policy_record(manifest, case),
               'schedule': manifest['schedules'].get(case['method'], [])}
    if case['engine'] == 'm2':
        replayed = verify_candidate(path, task, config, bank,
                                    expected_start={**binding, 'role': case['method']})
    else:
        start = read_journal(path)[0]
        expected = {**binding, 'method': case['engine_method'], 'planner_seed': 991333000 + case['task_index']}
        if any(canonical(start.get(key)) != canonical(value) for key, value in expected.items()):
            raise ValueError('incumbent/baseline manifest binding mismatch')
        replayed = verify_episode(path, task, config, bank)
    if canonical(replayed) != canonical(row['result']) or replayed['method'] != case['engine_method']:
        raise ValueError('replayed case result mismatch')
    receipts = [{k: v for k, v in event.items() if k != 'kind'}
                for event in read_journal(path) if event['kind'] in ('api_response', 'api_failure')]
    if canonical(receipts) != canonical(row['api_receipts']):
        raise ValueError('case API receipt mismatch')
    if sum(r['attempted'] for r in receipts) > case['max_calls']:
        raise ValueError('case exceeds its frozen call bound')
    return replayed


def _labelled(rows):
    labelled = []
    for row in rows:
        result = {**row['result'], 'method': row.get('method', row['result']['method'])}
        if 'end_to_end_seconds' in result.get('timing', {}):
            result['end_to_end_seconds'] = result['timing']['end_to_end_seconds']
        labelled.append({**row, 'result': result})
    return labelled


def study_summary(rows, mode, dispatch_guards=()):
    labelled = _labelled(rows)
    methods = {}
    for method in sorted({r['result']['method'] for r in labelled}):
        group = [r for r in labelled if r['result']['method'] == method]
        attempted = [r for r in group if r['result']['status'] not in NOT_RUN]
        results = [r['result'] for r in attempted]
        base = summarize_results(results).get(method, {'attempted': 0, 'completed': 0, 'failed': 0,
            'forecasts': {}, 'forecast_coverage': {}, 'mean_final_mse': None, 'mean_return_': None})
        receipts = [receipt for row in group for receipt in row['api_receipts']]
        base.update(planned=len(group), not_run=len(group) - len(attempted),
                    not_run_reasons=dict(Counter(r['result']['status'] for r in group if r not in attempted)),
                    forecast_failures=dict(Counter(code for r in attempted for code in r['result'].get('forecast_failures', []))),
                    api={**receipt_stats(receipts), 'hosted_calls': sum(r['attempted'] for r in receipts) if mode == 'hosted' else 0})
        case_ids = {r['case_id'] for r in group}
        base['local_dispatch_guard_failures'] = dict(Counter(g['reason'] for g in dispatch_guards if g['case_id'] in case_ids))
        base['attempted_api_failures'] = dict(Counter(r['code'] for r in receipts if r['attempted'] and r['status'] == 'failed'))
        latencies = [r['end_to_end_seconds'] for r in results
                     if r['status'] == 'completed' and r.get('end_to_end_seconds') is not None]
        base['mean_end_to_end_seconds'] = float(np.mean(latencies)) if latencies else None
        base['end_to_end_seconds_completed_count'] = len(latencies)
        methods[method] = base
    probe = [r for r in labelled if r['result']['method'] == 'probe']
    forecasts = [f for row in probe for f in row['result']['forecasts']]
    paired = [f for f in forecasts if 'jev' in f['probabilities']]
    expected = len(probe) * 6
    observed_loss = sum(f['brier']['jev'] for f in paired)
    missing = expected - len(paired)
    comparison = {}
    for reference in ('posterior', 'prior', 'chance', 'public_heuristic'):
        common = [f for f in paired if reference in f['brier']]
        comparison[reference] = {'paired_events': len(common),
            'mean_jev_minus_reference_brier': float(np.mean([f['brier']['jev'] - f['brier'][reference] for f in common])) if common else None}
    return {'scope': 'development integration only; no confirmatory inference', 'mode': mode,
            'response_provider': 'jev-fixture' if mode == 'fixture' else 'jev-1.14.0',
            'jev_efficacy_established': False, 'methods': methods,
            'local_guard_origin': 'local_dispatch_guard; these refusals are not provider outages',
            'forecast_comparison_primary': 'common fixed probe; adaptive event scores are descriptive',
            'common_probe': {'planned_events': expected, 'observed_events': len(forecasts),
                'valid_jev_events': len(paired), 'missing_jev_events': missing,
                'coverage': len(paired) / expected if expected else None,
                'whole_probe_brier_bounds': [observed_loss / expected, (observed_loss + missing) / expected] if expected else None,
                'paired_comparisons': comparison, 'calibration_success_claim': False},
            'timing_scope': 'end_to_end_seconds includes reset in both engines; online_seconds retains original engine semantics; saved numeric timing is not re-executed latency'}


class DispatchBoundary:
    """A synchronous request-sink guard: at most the in-flight call can overrun."""
    def __init__(self, manifest, started, clock=time.perf_counter):
        self.manifest, self.started, self.clock = manifest, started, clock
        self.reason = None
        self.case_id = None
        self.records = []

    def check(self):
        if self.reason:
            return self.reason
        if self.clock() - self.started >= self.manifest['wall_cap_seconds']:
            self.reason = 'not_run_wall'
        else:
            try:
                sources = dict(self.manifest['sources'])
                for field in ('registry_source', 'schedule_source'):
                    if field in self.manifest:
                        source = self.manifest[field]
                        sources[source['path']] = source['sha256']
                verify_hashes(ROOT, sources)
            except (OSError, ValueError):
                self.reason = 'not_run_source_change'
            if self.reason is None:
                try:
                    verify_hashes(ROOT, {r['path']: r['sha256'] for r in self.manifest['models'].values()})
                except (OSError, ValueError):
                    self.reason = 'not_run_model_drift'
            if self.reason is None and self.clock() - self.started >= self.manifest['wall_cap_seconds']:
                self.reason = 'not_run_wall'
        return self.reason

    def guard(self, sink, event):
        if event['kind'] == 'api_request' and self.check():
            self.record(event)
            raise RuntimeError('local dispatch boundary')
        sink(event)

    def record(self, receipt):
        if not any(row['span_id'] == receipt['span_id'] for row in self.records):
            self.records.append({'case_id': self.case_id, 'call_index': receipt['call_index'],
                'span_id': receipt['span_id'], 'reason': self.reason, 'origin': 'local_dispatch_guard',
                'receipt_code': 'event_sink_error', 'network_dispatched': False,
                'elapsed_seconds': self.clock() - self.started})


class BoundaryClient(JevClient):
    def __init__(self, *args, boundary, **kwargs):
        self.boundary = boundary
        super().__init__(*args, **kwargs)

    @JevClient.event_sink.setter
    def event_sink(self, sink):
        JevClient.event_sink.fset(self, None if sink is None else lambda event: self.boundary.guard(sink, event))

    def _record(self, receipt, code=None):
        if code == 'event_sink_error' and self.boundary.reason:
            self.boundary.record(receipt)
        return super()._record(receipt, code)


def expansion_gate(rows, expected_ids, receipts, stats, limits):
    problems = []
    if [row['case_id'] for row in rows] != expected_ids:
        problems.append('incomplete_first_cohort')
    if any(row['result']['status'] != 'completed' for row in rows):
        problems.append('incomplete_scientific_episode')
    if any(r['status'] != 'success' or r['usage'] is None or r['reservation_overrun'] for r in receipts):
        problems.append('invalid_or_incomplete_api_receipts')
    for row in rows:
        result = row['result']
        if result.get('forecast_failures'):
            problems.append('forecast_failure')
        if CALL_BOUNDS.get(row.get('method'), 0) > 0:
            coverage = result.get('forecast_coverage', {})
            if coverage.get('scored_jev_events', 0) != coverage.get('purchased_events', 0):
                problems.append('incomplete_forecast_coverage')
    verify_client_stats(stats, receipts, limits)
    return {'passed': not problems, 'problems': sorted(set(problems)), 'case_ids': [r['case_id'] for r in rows]}


def run_study(plan_path, output, *, live=False, key=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite an existing study directory')
    plan = read_plan(plan_path, files=False)
    if live and plan['planned_api_limits']['max_calls'] < plan['worst_case_api_calls']:
        raise ValueError(f"complete planned design requires at least {plan['worst_case_api_calls']} calls; prepare a sufficient explicitly authorized allowance before live execution")
    admission = admit_plan(plan)
    if digest(json.loads(Path(plan_path).read_text())) != digest(plan):
        raise ValueError('plan changed during admission')
    plan_sha = file_hash(plan_path)
    # Complete admission precedes credential acquisition, transport and output.
    transport = HostedTransport(live_key() if key is None else key) if live else FixtureTransport()
    torch.set_num_threads(1)
    started = time.perf_counter()
    mode = 'hosted' if live else 'fixture'
    limits = plan['planned_api_limits'] if live else fixture_limits(plan['worst_case_api_calls'])
    manifest = {**plan, 'kind': 'study-manifest', 'mode': mode,
                'model': 'jev-1.14.0' if live else 'jev-fixture',
                'response_provider': 'pinned hosted Jev' if live else 'jev-fixture',
                'api_limits': limits, 'plan_sha256': plan_sha, 'admission': admission,
                'fixture_caps_only': not live, 'python': platform.python_version(),
                'numpy': np.__version__, 'torch': torch.__version__}
    boundary = DispatchBoundary(manifest, started)
    client = BoundaryClient(transport, boundary=boundary, expected_model=manifest['model'],
                            limits=CallLimits(**limits), trace_id=manifest['trace_id'])
    output.mkdir(parents=True)
    (output / 'episodes').mkdir()
    shutil.copyfile(plan_path, output / 'plan.json')
    if file_hash(output / 'plan.json') != plan_sha:
        raise ValueError('plan changed after admission')
    write_new(output / 'manifest.json', manifest)
    for name in manifest['sources']:
        destination = output / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    verify_hashes(output / 'source_snapshot', manifest['sources'])
    config = PilotConfig(**manifest['config'])
    tasks = development_tasks(len(manifest['tasks']), config)
    bank = UnfilteredBank(quadrature_family(2), config)
    loaded = {}
    rows, stop_reason, gate = [], None, None
    pilot_ids = manifest['staged_scope']['pilot_case_ids']
    for case in manifest['cases']:
        boundary.case_id = case['case_id']
        stop_reason = stop_reason or boundary.check()
        if stop_reason:
            row = not_run_row(case, stop_reason)
        else:
            path = output / 'episodes' / (case['case_id'] + '.jsonl')
            before = len(client.receipts)
            try:
                with Journal(path) as journal:
                    record = policy_record(manifest, case)
                    policy = None
                    if record is not None:
                        cache_key = (record['path'], record['sha256'], canonical(record['calibration']))
                        if cache_key not in loaded:
                            loaded[cache_key] = load_policy(ROOT, record)
                        policy = FrozenPolicy(loaded[cache_key], record)
                    kwargs = {'policy': policy, 'client': client if case['max_calls'] else None,
                              'schedule': manifest['schedules'].get(case['method'], [])}
                    if case['engine'] == 'm2':
                        result = run_candidate(ScientistEnv(tasks[case['task_index']], config),
                                               case['method'], bank, journal, **kwargs)
                    else:
                        result = run_episode(ScientistEnv(tasks[case['task_index']], config),
                                             case['engine_method'], bank, journal,
                                             planner_seed=991333000 + case['task_index'], **kwargs)
            except Exception as error:
                # Preserve partial evidence, do not manufacture a scientific score.
                result = {'method': case['engine_method'], 'status': 'runner_failed',
                          'failure': type(error).__name__, 'forecasts': [], 'final_mse': None,
                          'return_': None, 'effort': None}
                stop_reason = 'not_run_runner_failure'
            row = {'case_id': case['case_id'], 'method': case['method'],
                   'journal': str(path.relative_to(output)), 'journal_sha256': file_hash(path),
                   'result': result, 'api_receipts': client.receipts[before:]}
            if stop_reason is None:
                try:
                    verify_case(output, manifest, case, row, tasks[case['task_index']], config, bank)
                    verify_client_stats(client.stats(), client.receipts, limits)
                except (ValueError, KeyError, TypeError):
                    stop_reason = 'not_run_gate_failure'
            if boundary.reason:
                stop_reason = boundary.reason
            elif client.stats()['blocked_reason'] == 'model_drift':
                stop_reason = 'not_run_model_drift'
            elif any(r.get('code') in ('call_limit', 'cost_limit', 'input_token_limit',
                     'total_request_limit', 'reservation_overrun') for r in row['api_receipts']):
                stop_reason = 'not_run_api_limit'
        rows.append(row)
        write_new(output / 'episodes' / (case['case_id'] + '.json'), row)
        print(f"{case['case_id']}: {row['result']['status']}", flush=True)
        if case['case_id'] == pilot_ids[-1]:
            gate = expansion_gate(rows, pilot_ids, client.receipts, client.stats(), limits)
            if boundary.check():
                gate['passed'] = False
                gate['problems'].append(boundary.reason)
                stop_reason = boundary.reason
            if not gate['passed']:
                stop_reason = stop_reason or 'not_run_gate_failure'
    index = trace_index(_labelled(rows), manifest['model'])
    write_new(output / 'trace-index.json', index)
    result = {'manifest_sha256': file_hash(output / 'manifest.json'), 'cases': rows,
              'summary': study_summary(rows, mode, boundary.records), 'client_stats': client.stats(),
              'trace_index_sha256': file_hash(output / 'trace-index.json'),
              'pilot_gate': gate, 'stop_reason': stop_reason,
              'dispatch_guards': boundary.records,
              'dispatch_guard_semantics': 'Listed event_sink_error receipts are local dispatch refusals, never provider outages.',
              'elapsed_seconds': time.perf_counter() - started}
    write_new(output / 'results.json', result)
    try:
        verification = verify_run(output)
    except (ValueError, KeyError, TypeError, OSError) as error:
        verification = {'verified': False, 'error_type': type(error).__name__,
                        'scope': 'raw attempted and skipped rows retained; replay did not validate this run'}
    write_new(output / 'verification.json', verification)
    return {'output': str(output), **verification, 'api': client.stats(), 'stop_reason': stop_reason}


def verify_run(output):
    output = Path(output).resolve()
    manifest = json.loads((output / 'manifest.json').read_text())
    result = json.loads((output / 'results.json').read_text())
    plan = json.loads((output / 'plan.json').read_text())
    if file_hash(output / 'manifest.json') != result['manifest_sha256'] or file_hash(output / 'plan.json') != manifest['plan_sha256']:
        raise ValueError('manifest or plan hash mismatch')
    tasks = validate_plan(plan, files=False)
    verify_admission(manifest['admission'], plan)
    mode = manifest['mode']
    if mode not in ('fixture', 'hosted') or manifest['kind'] != 'study-manifest':
        raise ValueError('unsupported run mode')
    for key, value in plan.items():
        if key != 'kind' and canonical(manifest[key]) != canonical(value):
            raise ValueError('manifest differs from frozen plan')
    expected_limits = plan['planned_api_limits'] if mode == 'hosted' else fixture_limits(plan['worst_case_api_calls'])
    if (manifest['api_limits'] != expected_limits or manifest['model'] != ('jev-1.14.0' if mode == 'hosted' else 'jev-fixture')
            or manifest['fixture_caps_only'] is not (mode == 'fixture')):
        raise ValueError('runtime model or caps differ from frozen mode')
    if mode == 'hosted' and not plan['complete_call_allowance']:
        raise ValueError('hosted execution used an incomplete planned call allowance')
    verify_hashes(ROOT, manifest['sources'])
    verify_hashes(output / 'source_snapshot', manifest['sources'])
    actual_sources = {str(p.relative_to(output / 'source_snapshot')) for p in (output / 'source_snapshot').rglob('*') if p.is_file()}
    if actual_sources != set(manifest['sources']):
        raise ValueError('extra or missing snapshot files')
    if [r['case_id'] for r in result['cases']] != [c['case_id'] for c in manifest['cases']]:
        raise ValueError('missing, extra, or reordered result cases')
    expected_files = {c['case_id'] + '.json' for c in manifest['cases']}
    expected_files.update(r['case_id'] + '.jsonl' for r in result['cases'] if r['journal'] is not None)
    if {p.name for p in (output / 'episodes').iterdir()} != expected_files:
        raise ValueError('extra or missing episode artifacts')
    config = PilotConfig(**manifest['config'])
    bank = UnfilteredBank(quadrature_family(2), config)
    skipped = False
    for case, row in zip(manifest['cases'], result['cases']):
        if canonical(json.loads((output / 'episodes' / (case['case_id'] + '.json')).read_text())) != canonical(row):
            raise ValueError('saved raw row differs from results')
        verify_case(output, manifest, case, row, tasks[case['task_index']], config, bank)
        if row['result']['status'] in NOT_RUN:
            skipped = True
            if row['result']['status'] != result['stop_reason']:
                raise ValueError('skipped case reason differs from study stop reason')
        elif skipped:
            raise ValueError('dispatch resumed after a stopped case')
    receipts = [r for row in result['cases'] for r in row['api_receipts']]
    verify_client_stats(result['client_stats'], receipts, manifest['api_limits'])
    pilot_count = len(manifest['staged_scope']['pilot_case_ids'])
    pilot_rows = result['cases'][:pilot_count]
    pilot_receipts = [r for row in pilot_rows for r in row['api_receipts']]
    gate_problems = expansion_gate(pilot_rows, manifest['staged_scope']['pilot_case_ids'], pilot_receipts,
                                  _stats(pilot_receipts, manifest['api_limits']), manifest['api_limits'])
    gate = result['pilot_gate']
    if gate is None or gate['case_ids'] != gate_problems['case_ids']:
        raise ValueError('missing pilot gate evidence')
    if gate['passed'] and not gate_problems['passed']:
        raise ValueError('pilot gate suppressed failures')
    if not gate['passed'] and any(r['result']['status'] not in NOT_RUN for r in result['cases'][pilot_count:]):
        raise ValueError('expanded after a failed pilot gate')
    if canonical(study_summary(result['cases'], mode, result['dispatch_guards'])) != canonical(result['summary']):
        raise ValueError('study summary mismatch')
    verify_hashes(output, {'trace-index.json': result['trace_index_sha256']})
    if canonical(json.loads((output / 'trace-index.json').read_text())) != canonical(trace_index(_labelled(result['cases']), manifest['model'])):
        raise ValueError('trace index mismatch')
    _verify_guards(result, receipts, manifest)
    return {'verified': True, 'verified_episodes': sum(r['result']['status'] not in NOT_RUN for r in result['cases']),
            'failed_episodes': sum(r['result']['status'] == 'failed' for r in result['cases']),
            'not_run_episodes': sum(r['result']['status'] in NOT_RUN for r in result['cases']),
            'mode': mode, 'hosted_calls': sum(r['attempted'] for r in receipts) if mode == 'hosted' else 0,
            'credential_free_replay': True, 'checkpoint_free_replay': True, 'source_hashes_match': True,
            'recorded_admission_binding_verified': True,
            'scope': 'stored receipts, recorded policy advice and simulator outcomes verified; weights and measured latency are not reproduced'}


def _stats(receipts, limits):
    value = {**receipt_stats(receipts), **validate_api_accounting(receipts, limits)}
    value.update(usage_complete=value['unknown_usage_calls'] == 0,
                 reserved_estimated_usd=value['reserved_input_tokens'] / 1_000_000 * .042,
                 accounted_estimated_usd=value['accounted_input_tokens'] / 1_000_000 * .042)
    return value


def _verify_guards(result, receipts, manifest):
    elapsed = result['elapsed_seconds']
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('invalid saved study timing')
    if result['stop_reason'] is not None and result['stop_reason'] not in NOT_RUN:
        raise ValueError('invalid study stop reason')
    if result['stop_reason'] == 'not_run_wall' and elapsed < manifest['wall_cap_seconds']:
        raise ValueError('wall stop predates frozen deadline')
    by_span = {r['span_id']: r for r in receipts}
    case_by_span = {r['span_id']: row['case_id'] for row in result['cases'] for r in row['api_receipts']}
    seen = set()
    for guard in result['dispatch_guards']:
        receipt = by_span.get(guard['span_id'])
        if (receipt is None or guard['span_id'] in seen or receipt['attempted'] or
                receipt.get('code') != 'event_sink_error' or guard['receipt_code'] != 'event_sink_error' or
                guard['origin'] != 'local_dispatch_guard' or guard['network_dispatched'] is not False or
                guard['call_index'] != receipt['call_index'] or guard['reason'] != result['stop_reason'] or
                guard['case_id'] != case_by_span[guard['span_id']]):
            raise ValueError('local dispatch guard receipt mismatch')
        seen.add(guard['span_id'])
        seconds = guard['elapsed_seconds']
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= elapsed:
            raise ValueError('invalid guard timing')
        if guard['reason'] == 'not_run_wall' and seconds < manifest['wall_cap_seconds']:
            raise ValueError('wall guard predates frozen deadline')
    if seen:
        first = min(by_span[span]['call_index'] for span in seen)
        expected = {r['span_id'] for r in receipts if r.get('code') == 'event_sink_error' and r['call_index'] >= first}
        if seen != expected:
            raise ValueError('local guard refusals missing from origin accounting')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Offline proposal only; no credentials or hosted calls.')
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--mechanisms', type=int, choices=(1, 8), default=1)
    prep.add_argument('--seeds', default='20')
    prep.add_argument('--max-calls', type=int, default=32)
    prep.add_argument('--max-usd', type=float, default=.10)
    prep.add_argument('--ppo-records', type=Path)
    prep.add_argument('--wall-seconds', type=float, default=1200)
    run = sub.add_parser('run', help='Uses jev-fixture unless --live is explicitly supplied.')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--live', action='store_true', help='Execute the explicitly authorized pinned hosted plan; acquire a fresh process/hidden-prompt key.')
    verify = sub.add_parser('verify', help='Credential-free and checkpoint-free; works from the copied source_snapshot script.')
    verify.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = prepare(args.output, mechanisms=args.mechanisms,
                             seeds=[int(v) for v in args.seeds.split(',')],
                             max_calls=args.max_calls, max_usd=args.max_usd,
                             ppo_records=args.ppo_records, wall_seconds=args.wall_seconds)
        elif args.command == 'run':
            result = run_study(args.plan, args.output, live=args.live)
        else:
            result = verify_run(args.run)
    except (ValueError, KeyError, TypeError, OSError, JevError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result.get('verified') is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
