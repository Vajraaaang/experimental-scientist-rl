"""Bounded M2 PPO runs on the unchanged eight-variable scientific task.

Each invocation owns a new output directory. Historical training mechanisms
are retained deliberately; only the fresh tuning pool is constructed. The
interaction-budget selection is sealed before any extra wall-budget training.
This module does not run an architecture race or qualify scientific learning.
"""
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import torch

from experimentlab.cost_policy import PublicCostPolicy
from experimentlab.envs import PilotConfig, ScientistEnv
from experimentlab.learning import ActorCritic, save_checkpoint
from experimentlab.ppo_v4 import collect_batch, update_ppo
from experimentlab.private_cache import CachedScientistEnv, PrivateDynamicsCache
from experimentlab.tasks import continuous_family, make_task, mechanism_hash
from experimentlab.tensor_batch import pack_episodes, whole_episode_advantages

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
# Import definitions only; the historical setup() also constructs other splits
# and must never be called by this trainer.
from ppo_followup import schedules_for, task_batch

SCHEMA = 'm2-ppo-train-v1'
REGISTRY_PATH = ROOT / 'reports/ppo-efficiency-v6/selection.json'
ZERO_CALIBRATION = {'duration_penalty': 0., 'dose_penalty': 0., 'stop_bonus': 0.}
TRAIN_POOL_SEED, TUNING_POOL_SEED, TUNING_EPISODE_SEED = 7401, 991311, 991312000
SELECTOR_NAMESPACE, JOINT_NAMESPACE, CONTINUATION_NAMESPACE = 91301, 91302, 91303
INITIAL_LOGP_TOLERANCE = 5e-5


@dataclass(frozen=True)
class TrainConfig:
    name: str = 'm2-ppo'
    seed: int = 20
    kind: str = 'recurrent'
    hidden: int = 64
    batch: int = 128
    threads: int = 1
    backend: str = 'cpu'
    initialization: str = 'scratch'
    selector_interactions: int | None = None
    joint_interactions: int = 1_000_000
    wall_snapshot_seconds: float = 120.
    joint_hard_interactions: int = 1_500_000
    hard_wall_seconds: float = 3600.
    eval_every_interactions: int = 100_000
    joint_lr: float = .0003
    joint_epochs: int = 2
    entropy_start: float = .03
    entropy_end: float = .001
    train_pool_size: int = 4096
    tuning_count: int = 128
    smoke: bool = False
    initial_checkpoint_record: str | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', self.name):
            raise ValueError('name must be a simple run identifier')
        if self.initialization not in ('scratch', 'v6', 'imitation_ppo'):
            raise ValueError('initialization must be scratch, v6, or imitation_ppo')
        if self.selector_interactions is None:
            object.__setattr__(self, 'selector_interactions',
                               1_000_000 if self.initialization == 'scratch' and not self.initial_checkpoint_record else 0)
        for name, minimum, maximum in (
            ('seed', 0, 2**32 - 1), ('batch', 1, 8192), ('threads', 1, 256),
            ('selector_interactions', 0, 20_000_000), ('joint_interactions', 5, 20_000_000),
            ('joint_hard_interactions', 5, 20_000_000), ('eval_every_interactions', 1, 20_000_000),
            ('joint_epochs', 1, 16), ('train_pool_size', 1, 4096), ('tuning_count', 1, 128)):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
        if self.kind not in ('history', 'recurrent') or type(self.hidden) is not int or self.hidden not in (64, 128, 256, 512):
            raise ValueError('architecture must use history/recurrent and hidden 64, 128, 256, or 512')
        if self.backend not in ('cpu', 'mps', 'split'):
            raise ValueError('backend must be cpu, mps, or split')
        for name, minimum, maximum, positive in (
            ('wall_snapshot_seconds', 0., 3600., False), ('hard_wall_seconds', 0., 3600., True),
            ('joint_lr', 0., .01, True), ('entropy_start', 0., 1., False), ('entropy_end', 0., 1., False)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < minimum or value > maximum or (positive and value == minimum)):
                raise ValueError(f'{name} must be finite and within its bounded range')
        if self.joint_hard_interactions < self.joint_interactions:
            raise ValueError('joint_hard_interactions must cover the primary joint_interactions target')
        if type(self.smoke) is not bool or (not self.smoke and (self.train_pool_size != 4096 or self.tuning_count != 128)):
            raise ValueError('changed pool sizes require explicit smoke=True')
        if self.initialization != 'scratch':
            if self.selector_interactions or self.kind != 'recurrent' or self.hidden != 64:
                raise ValueError('registered warm starts require recurrent hidden=64 and zero selector interactions')
        if self.initial_checkpoint_record is not None:
            if not isinstance(self.initial_checkpoint_record, (str, Path)) or self.selector_interactions:
                raise ValueError('a local initial checkpoint record requires a path and zero selector interactions')
            object.__setattr__(self, 'initial_checkpoint_record', str(Path(self.initial_checkpoint_record).expanduser().resolve()))


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def source_hashes():
    files = set((ROOT / 'src').rglob('*.py'))
    files.update(ROOT / path for path in ('integrations/ppo_m2_train.py', 'scripts/train_ppo_m2.py',
                                        'scripts/ppo_followup.py', 'scripts/research_stage.py',
                                        'requirements.lock', 'pyproject.toml'))
    return {str(path.relative_to(ROOT)): _digest(path) for path in sorted(files) if path.is_file()}


def _sync(device):
    if device == 'mps':
        torch.mps.synchronize()


@contextmanager
def _stage(timings, name, device='cpu'):
    _sync(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        _sync(device)
        timings[name] = timings.get(name, 0.) + time.perf_counter() - started


def _repository_path(value):
    path = Path(value).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('initial checkpoint and record must be inside the user repository')
    return path


def _incumbent_verification(state, origin):
    """Verify selection evidence; loading weights alone grants no exemption."""
    artifacts = {}

    def verified_file(value, expected):
        path = _repository_path(value)
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected) or _digest(path) != expected:
            raise ValueError('incumbent evidence hash mismatch')
        artifacts[str(path)] = expected
        return path

    def verify(current, link, seen):
        path = verified_file(link['path'], link['sha256'])
        if str(path) in seen or len(seen) >= 16:
            raise ValueError('incumbent lineage is cyclic or exceeds 16 ancestors')
        seen = seen | {str(path)}
        record_path = verified_file(link['record_path'], link['record_sha256'])
        record = json.loads(record_path.read_text())
        if link['source'] == 'fixed_v6_selection_registry':
            if record_path != REGISTRY_PATH.resolve():
                raise ValueError('incumbent historical registry is not the fixed registry')
            key = link['registry_key']
            if not re.fullmatch(r'(selected_ppo|imitation_ppo)-seed[0-9]+', key):
                raise ValueError('unsupported historical incumbent registry key')
            registered = record['registry'][key]
            if (_repository_path(registered['path']) != path or registered['sha256'] != link['sha256']
                    or registered['calibration'] != link['calibration']):
                raise ValueError('historical incumbent does not match its registry record')
            return 'registered_historical_policy'
        if link['source'] != 'local_m2_record':
            raise ValueError('unknown incumbent source')
        if _repository_path(record['path']) != path or record['sha256'] != link['sha256']:
            raise ValueError('local incumbent does not match its input record')
        provenance, counters = current['provenance'], current['counters']
        if (provenance.get('trainer_schema') != SCHEMA or provenance.get('checkpoint_role') != 'best_primary'
                or provenance.get('primary_selection_eligible') is not True
                or provenance.get('primary_selection_closed') is not False
                or counters.get('phase') != 'joint'):
            raise ValueError('parent checkpoint has no eligible primary-selection certificate')
        biases = provenance.get('training_biases')
        if (not isinstance(biases, dict) or set(biases) != {'behavior_strength', 'initial_outcome_bias'}
                or any(type(value) not in (int, float) or value != 0 for value in biases.values())):
            raise ValueError('parent checkpoint has missing or nonzero training biases')
        config = provenance['config']
        target = config['joint_interactions']
        counted = [counters.get(key) for key in ('joint_transitions', 'selector_transitions', 'training_transitions', 'updates')]
        if (type(target) is not int or not 5 <= target <= 20_000_000
                or any(type(value) is not int or value < 0 for value in counted)
                or counted[2] != counted[0] + counted[1] or counted[0] > target):
            raise ValueError('parent checkpoint has inconsistent primary-budget counters')
        source = provenance.get('source_sha256')
        if (not isinstance(source, dict) or source.get('integrations/ppo_m2_train.py') != _digest(Path(__file__))
                or source != source_hashes()):
            raise ValueError('parent checkpoint source is not the current verified trainer source')
        manifest_path = verified_file(provenance['manifest_path'], provenance['manifest_hash'])
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('schema') != SCHEMA or manifest.get('source_sha256') != source
                or manifest.get('config') != config or manifest.get('calibration') != provenance['calibration']):
            raise ValueError('parent checkpoint does not match its sealed manifest')
        basis = provenance.get('primary_selection_basis')
        if basis == 'joint_budget':
            if counted[0] < .5 * target:
                raise ValueError('parent checkpoint predates the minimum joint-training budget')
            return 'eligible_parent_joint_budget'
        if basis != 'verified_initial_incumbent' or any(counted):
            raise ValueError('parent zero-step exemption has no valid eligibility basis')
        ancestor = provenance['incumbent_origin']
        if provenance.get('initial_checkpoint_sha256') != ancestor['sha256']:
            raise ValueError('incumbent ancestor does not match initial checkpoint provenance')
        ancestor_path = verified_file(ancestor['path'], ancestor['sha256'])
        ancestor_state = torch.load(ancestor_path, map_location='cpu', weights_only=False)
        if (ancestor_state['architecture'] != current['architecture']
                or ancestor['calibration'] != provenance['calibration']
                or ancestor_state['model'].keys() != current['model'].keys()
                or any(not torch.equal(value, ancestor_state['model'][key]) for key, value in current['model'].items())):
            raise ValueError('zero-step incumbent changed its inherited policy')
        verify(ancestor_state, ancestor, seen)
        return 'verified_parent_incumbent_lineage'

    try:
        basis = verify(state, origin, set())
        return {'eligible': True, 'basis': basis, 'reason': None,
                'verified_artifacts': [{'path': path, 'sha256': digest} for path, digest in artifacts.items()]}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return {'eligible': False, 'basis': None, 'reason': str(error), 'verified_artifacts': []}


def _initial_model(config):
    """Load only a hash-verified registry artifact or a new M2 artifact record."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = ActorCritic(kind=config.kind, hidden=config.hidden, horizon=5)
    if config.initial_checkpoint_record is None and config.initialization == 'scratch':
        return model, None
    if config.initial_checkpoint_record is not None:
        record_path = _repository_path(config.initial_checkpoint_record)
        record = json.loads(record_path.read_text())
        origin = {'source': 'local_m2_record', 'record_path': str(record_path), 'record_sha256': _digest(record_path)}
    else:
        key = f"{'selected_ppo' if config.initialization == 'v6' else 'imitation_ppo'}-seed{config.seed}"
        registry = json.loads(REGISTRY_PATH.read_text())['registry']
        if key not in registry:
            raise ValueError(f'no registered warm start for {key}')
        record = registry[key]
        origin = {'source': 'fixed_v6_selection_registry', 'registry_key': key,
                  'record_path': str(REGISTRY_PATH), 'record_sha256': _digest(REGISTRY_PATH)}
    path = _repository_path(record['path'])
    expected = record.get('sha256')
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected) or _digest(path) != expected:
        raise RuntimeError('initial checkpoint hash mismatch')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['architecture'] != model.architecture:
        raise ValueError('initial checkpoint architecture mismatch')
    if config.initial_checkpoint_record and state.get('provenance', {}).get('trainer_schema') != SCHEMA:
        raise ValueError('local initial records must identify a parent-generated M2 checkpoint')
    calibration = record.get('calibration', state.get('provenance', {}).get('calibration', ZERO_CALIBRATION))
    if (not isinstance(calibration, dict) or calibration.keys() != ZERO_CALIBRATION.keys()
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 10000
                   for v in calibration.values())):
        raise ValueError('invalid deployment calibration in initial record')
    if config.initialization == 'scratch' and any(calibration.values()):
        raise ValueError('scratch continuation must retain zero deployment calibration')
    if config.initial_checkpoint_record and calibration != state['provenance']['calibration']:
        raise ValueError('local continuation must retain its checkpoint deployment calibration')
    model.load_state_dict(state['model'])
    information = {**origin, 'path': str(path), 'sha256': expected, 'calibration': dict(calibration),
                   'architecture': state['architecture'], 'inherited_counters': state.get('counters', {}),
                   'optimizer_loading': 'weights only; fresh Adam',
                   'initialization_description': ('continuation of historical IM->PPO, including prior imitation and PPO training'
                                                  if config.initialization == 'imitation_ppo' else
                                                  'continuation of historical selected v6 PPO' if config.initialization == 'v6' else
                                                  'continuation of a parent-generated M2 checkpoint'),
                   'inherited_provenance': {key: state.get('provenance', {}).get(key)
                                           for key in ('scope', 'seed', 'initial_checkpoint_sha256', 'config')}}
    verification = _incumbent_verification(state, information)
    information.update(incumbent_eligible=verification['eligible'], incumbent_verification=verification)
    return model, information


def _optimizer_devices(model, optimizer):
    expected = next(model.parameters()).device
    moments, steps = set(), set()
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                if name == 'step' and value.numel() == 1:
                    steps.add(str(value.device))
                else:
                    moments.add(str(value.device))
                    if value.device != expected:
                        raise RuntimeError('Adam moments left the learner device')
    if any(parameter.device != expected for parameter in model.parameters()):
        raise RuntimeError('learner parameters span devices')
    return {'parameter_devices': sorted({str(p.device) for p in model.parameters()}),
            'moment_devices': sorted(moments), 'scalar_step_devices': sorted(steps)}


class _CountedEnvironment:
    """Keep completed simulator-call accounting even if a collector later fails."""

    def __init__(self, environment, account, forced_steps=0):
        self.environment, self.account, self.forced_steps = environment, account, forced_steps
        self.steps = 0

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def reset(self):
        self.steps = 0
        return self.environment.reset()

    def step(self, action):
        result = self.environment.step(action)
        self.account(self.steps >= self.forced_steps, bool(result[2]))
        self.steps += 1
        return result


def evaluate_policy(model, tasks, simulator_config, cache, schedules=None, account=None):
    """Greedy CPU public-observation policy; original reward and per-task scores.

    Only trusted evaluator code accesses the environment's final private score.
    The policy receives the existing public token tensor, never these rows.
    """
    environments = [(ScientistEnv(task, simulator_config) if cache is None else
                     CachedScientistEnv(task, simulator_config, cache)) for task in tasks]
    if account is not None:
        environments = [_CountedEnvironment(environment, account, 0 if schedules is None else len(schedules[i]))
                        for i, environment in enumerate(environments)]
    records, metrics = collect_batch(model, environments, 'cpu', deterministic=True, public_schedules=schedules)
    rows = [{'task_index': i, 'task_seed': task.seed, 'mechanism_hash': mechanism_hash(task.mechanism),
             **environment.private_result, 'return': float(sum(record['rewards'])),
             'assays': environment.count, 'actions': record['actions'], 'rewards': record['rewards']}
            for i, (task, environment, record) in enumerate(zip(tasks, environments, records))]
    metrics.update(mean_mse=float(np.mean([row['final_mse'] for row in rows])),
                   mean_effort=float(np.mean([row['effort'] for row in rows])), rows=rows)
    return metrics


def _reference_comparison(cached, native):
    rows = []
    for left, right in zip(cached['rows'], native['rows']):
        errors = {key: abs(left[key] - right[key]) for key in
                  ('final_mse', 'variable_cost', 'return', 'effort', 'assays')}
        rewards_equal = len(left['rewards']) == len(right['rewards']) and bool(np.allclose(
            left['rewards'], right['rewards'], rtol=0, atol=1e-12))
        actions_equal = left['actions'] == right['actions']
        rows.append({'task_index': left['task_index'], 'cached': left, 'native': right,
                     'absolute_errors': errors, 'actions_equal': actions_equal,
                     'rewards_equal': rewards_equal,
                     'passed': actions_equal and rewards_equal and max(errors.values()) <= 1e-12})
    return {'cases': len(rows), 'absolute_tolerance': 1e-12,
            'passed': len(rows) == len(cached['rows']) == len(native['rows']) and all(row['passed'] for row in rows),
            'rows': rows}


def train(config, output_dir):
    """Train one declared candidate and write exact counts and both snapshots.

    ``mps`` and ``split`` both use a CPU rollout/evaluation mirror and an MPS
    learner: no backend is allowed to recompute old CPU behavior probabilities.
    Wall targets are observed at whole-operation boundaries, never by truncating
    episodes. Hard wall expiry stops new training/evaluation units; bounded
    final checkpoint/proof/provenance work is reported separately. A zero wall
    snapshot target disables that comparison. Full-run resume is not supported.
    """
    started = time.perf_counter()
    if not isinstance(config, TrainConfig):
        raise TypeError('config must be a TrainConfig')
    fallback = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '').strip().lower()
    if fallback not in ('', '0', 'false', 'off', 'no'):
        raise RuntimeError('PYTORCH_ENABLE_MPS_FALLBACK must be disabled')
    if config.backend != 'cpu' and not torch.backends.mps.is_available():
        raise RuntimeError('native MPS is unavailable')
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = 'cpu' if config.backend == 'cpu' else 'mps'
    scope = 'smoke_only' if config.smoke else 'm2_development_only'
    timings, checkpoints, evaluations, checkpoint_sequence = {}, {}, [], {}
    snapshots = {'equal_interaction': None, 'equal_wall': None}
    counts = dict.fromkeys(('selector_transitions', 'joint_transitions', 'training_transitions',
                           'training_policy_decisions', 'training_episodes', 'selector_episodes',
                           'joint_episodes', 'updates', 'validation_transitions',
                           'validation_policy_decisions', 'validation_episodes'), 0)
    phase = 'selector' if config.selector_interactions else 'joint'
    joint_started = None
    best_primary = selector_quality = initial = None
    optimizer = learner = mirror = learner_policy = mirror_policy = cache = None
    native_cache = None
    training_biases = {'behavior_strength': 0., 'initial_outcome_bias': 0.}
    primary_closed = False
    max_logp_error = 0.
    joint_episode_offset = 0
    joint_namespace = CONTINUATION_NAMESPACE if config.initial_checkpoint_record else JOINT_NAMESPACE
    scratch_curriculum = config.initialization == 'scratch'
    proof = {'passed': None, 'cases': 0, 'reason': 'no checkpoint available'}
    stop_reason = 'hard_wall_seconds'
    old_threads = torch.get_num_threads()
    old_torch_rng, old_numpy_rng = torch.get_rng_state(), np.random.get_state()
    old_mps_rng = torch.mps.get_rng_state() if device == 'mps' else None
    calibration = dict(ZERO_CALIBRATION)
    simulator_config = PilotConfig()
    hashes = {}

    def elapsed():
        return time.perf_counter() - started

    def expired():
        return elapsed() >= config.hard_wall_seconds

    def counters():
        return {**counts, 'phase': phase, 'phase_episodes': counts[f'{phase}_episodes'],
                'joint_episode_offset': joint_episode_offset, 'joint_namespace': joint_namespace,
                'elapsed_seconds': elapsed(),
                'joint_elapsed_seconds': None if joint_started is None else time.perf_counter() - joint_started}

    def event(value):
        with (output / 'learning_curve.jsonl').open('a') as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')

    def sync_mirror():
        with _stage(timings, 'mirror_copy', device):
            mirror.load_state_dict({key: tensor.detach().cpu() for key, tensor in learner.state_dict().items()})
            mirror.eval()

    def checkpoint(key, primary_eligible=False, selection_basis=None):
        with _stage(timings, 'checkpoint', device):
            checkpoint_sequence[key] = checkpoint_sequence.get(key, 0) + 1
            path = output / f'{key}-{phase}-u{counts["updates"]:07d}-{checkpoint_sequence[key]:04d}.pt'
            current = counters()
            save_checkpoint(path, learner, optimizer, current, {
                'trainer_schema': SCHEMA, 'scope': scope, 'seed': config.seed,
                'config': asdict(config), 'resolved_config': asdict(simulator_config),
                'manifest_path': str(output / 'manifest.json'), 'calibration': calibration,
                'initial_checkpoint_sha256': None if initial is None else initial['sha256'],
                'training_biases': dict(training_biases), 'source_sha256': hashes,
                'checkpoint_role': key, 'primary_selection_eligible': primary_eligible,
                'primary_selection_closed': primary_closed, 'primary_selection_basis': selection_basis,
                'incumbent_origin': ({name: initial[name] for name in
                    ('source', 'path', 'sha256', 'record_path', 'record_sha256', 'calibration', 'registry_key') if name in initial}
                    if selection_basis == 'verified_initial_incumbent' else None),
                'resume_supported': False, 'checkpoint_boundary': 'complete rollout and PPO update'})
            record = {'path': str(path), 'sha256': _digest(path), 'bytes': path.stat().st_size,
                      **current, 'calibration': dict(calibration),
                      'saved_elapsed_seconds': elapsed()}
            checkpoints[key] = record
        event({'event': 'checkpoint', 'checkpoint_key': key, 'checkpoint': record})
        return record

    def eligible(incumbent=False):
        verified_incumbent = (incumbent and initial is not None and initial['incumbent_eligible']
                              and counts['training_transitions'] == 0 and counts['updates'] == 0
                              and not any(training_biases.values()))
        return (phase == 'joint' and not primary_closed and counts['joint_transitions'] <= config.joint_interactions
                and (verified_incumbent or (counts['joint_transitions'] >= .5 * config.joint_interactions
                                   and not any(training_biases.values()))))

    def account_validation(sampled, terminated):
        counts['validation_transitions'] += 1
        counts['validation_policy_decisions'] += int(sampled)
        counts['validation_episodes'] += int(terminated)

    def account_training(sampled, terminated):
        counts[f'{phase}_transitions'] += 1
        counts['training_transitions'] += 1
        counts['training_policy_decisions'] += int(sampled)
        counts[f'{phase}_episodes'] += int(terminated)
        counts['training_episodes'] += int(terminated)

    def evaluate(reason, selection=True, incumbent=False, control=False):
        nonlocal native_cache, best_primary
        reuse = not control and native_cache is not None and native_cache[0] == counts['updates']
        if reuse:
            metrics = native_cache[1]
        else:
            sync_mirror()
            with _stage(timings, 'evaluation'):
                schedules = schedules_for(range(len(tuning)), 777) if control else None
                metrics = evaluate_policy(mirror_policy, tuning, simulator_config, cache, schedules, account_validation)
            if not control:
                native_cache = (counts['updates'], metrics)
        row = {**counters(), 'reason': reason, 'control': control, 'evaluation_reused': reuse,
               'selection_eligible': bool(selection and not control and eligible(incumbent)),
               'training_biases_zero': not any(training_biases.values()), 'metrics': metrics}
        if row['selection_eligible'] and (best_primary is None or metrics['mean_return'] > best_primary['metrics']['mean_return']):
            row['checkpoint'] = checkpoint('best_primary', primary_eligible=True,
                                           selection_basis='verified_initial_incumbent' if incumbent else 'joint_budget')
            best_primary = row.copy()
        evaluations.append(row)
        event({'event': 'evaluation', **row})
        return row

    def wall_snapshot():
        if (config.wall_snapshot_seconds and snapshots['equal_wall'] is None
                and elapsed() >= config.wall_snapshot_seconds and not expired()):
            captured = counters()
            row = evaluate('equal_wall', selection=False)
            record = checkpoint('equal_wall')
            snapshots['equal_wall'] = {**captured, 'selection_eligible': False,
                'target_seconds': config.wall_snapshot_seconds,
                'overshoot_seconds': captured['elapsed_seconds'] - config.wall_snapshot_seconds,
                'metrics': row['metrics'], 'checkpoint': record,
                'completed_elapsed_seconds': elapsed()}

    def train_batch(limit):
        nonlocal max_logp_error, training_biases
        phase_count = counts[f'{phase}_transitions']
        size = min(config.batch, (limit - phase_count) // 5)
        if size < 1:
            return False
        sync_mirror()
        selector = phase == 'selector'
        first = counts[f'{phase}_episodes'] + (0 if selector else joint_episode_offset)
        progress = counts['joint_transitions'] / config.joint_interactions
        decay = max(0., 1. - progress / .3) if scratch_curriculum else 0.
        training_biases = {'behavior_strength': 0. if selector else 6. * decay,
                           'initial_outcome_bias': 0. if selector else 3. * decay}
        entropy = .05 if selector else config.entropy_start + (config.entropy_end - config.entropy_start) * min(1., progress / .7)
        with _stage(timings, 'training_rollout'):
            tasks = task_batch(pool, simulator_config, config.seed, first, size,
                               SELECTOR_NAMESPACE if selector else joint_namespace)
            schedules = schedules_for(range(first, first + size), config.seed + 310) if selector else None
            environments = [_CountedEnvironment(CachedScientistEnv(task, simulator_config, cache), account_training,
                                                0 if schedules is None else len(schedules[i]))
                            for i, task in enumerate(tasks)]
            records, rollout = collect_batch(mirror_policy, environments, 'cpu', public_schedules=schedules,
                                            minimum_assays=2, maximum_assays=3, **training_biases)
        if counts[f'{phase}_transitions'] - phase_count != rollout['transitions']:
            raise RuntimeError('physical rollout accounting does not match complete records')
        if counts[f'{phase}_transitions'] > limit:
            raise RuntimeError('whole-episode interaction reservation violated')
        old_logp = np.concatenate([np.asarray(record['old_logp'], dtype=np.float32) for record in records])
        with _stage(timings, 'packing_advantages', device):
            batch = pack_episodes(records, device)
            advantages, targets = whole_episode_advantages(batch, gae_lambda=1.)
        with _stage(timings, 'training_update', device):
            update = update_ppo(learner_policy, optimizer, batch, epochs=4 if selector else config.joint_epochs,
                                clip=.15, entropy_coef=entropy, value_coef=.1, gae_lambda=1., target_kl=.02,
                                normalize_advantage=False, center_advantage=False,
                                prepared_advantages=advantages, prepared_targets=targets)
        counts['updates'] += 1
        max_logp_error = max(max_logp_error, update['initial_logp_error'])
        row = {'event': 'update', **counters(), 'phase_episode_start': first,
               'namespace': SELECTOR_NAMESPACE if selector else joint_namespace,
               'rollout': rollout, 'update': update, 'training_biases': dict(training_biases),
               'entropy_coef': entropy, 'old_logp_source': 'cpu_rollout',
               'old_logp_sha256': hashlib.sha256(old_logp.tobytes()).hexdigest(),
               'old_logp_recomputed': False, 'optimizer_devices': _optimizer_devices(learner, optimizer)}
        event(row)
        if update['initial_logp_error'] > INITIAL_LOGP_TOLERANCE:
            raise RuntimeError(f"initial rollout/update log probability mismatch: {update['initial_logp_error']}")
        return True

    try:
        with _stage(timings, 'setup_source_manifest'):
            hashes = source_hashes()
            manifest = {'schema': SCHEMA, 'scope': scope, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'output_dir': str(output), 'config': asdict(config), 'simulator_config': asdict(simulator_config),
                'source_sha256': hashes, 'confirmation_constructed': False, 'final_constructed': False,
                'hosted_calls': 0, 'resume_supported': False,
                'pools': {'train_seed': TRAIN_POOL_SEED, 'train_size': config.train_pool_size,
                          'tuning_seed': TUNING_POOL_SEED, 'tuning_size': config.tuning_count,
                          'tuning_episode_seed': TUNING_EPISODE_SEED, 'historical_training_pool_retained': True,
                          'selector_namespace': SELECTOR_NAMESPACE, 'joint_namespace': joint_namespace},
                'selection': {'criterion': 'original greedy CPU tuning mean return',
                              'minimum_joint_fraction': .5, 'all_training_biases_zero': True,
                              'warm_start_zero_eligible': 'verified historical registry policy or certified eligible parent lineage',
                              'post_primary_training_eligible': False},
                'budgets': {'interaction_scope': 'physical training transitions; validation counted separately',
                            'whole_episode_reservation': 5, 'phase_target_undershoot_max': 4,
                            'wall_snapshot_clock': 'train() entry, including setup, selector, evaluation, checkpoint',
                            'zero_wall_target': 'disabled',
                            'wall_boundary': 'first complete operation boundary at or after target',
                            'hard_wall_boundary': 'no new training/evaluation unit after expiry; final checkpoint/proof/source verification tail measured'},
                'optimizer': {'selector': {'lr': .001, 'epochs': 4, 'entropy': .05},
                              'joint': {'lr': config.joint_lr, 'epochs': config.joint_epochs},
                              'eps': 1e-5, 'reset_at_joint': True, 'value_coef': .1, 'clip': .15,
                              'gae_lambda': 1., 'normalize_advantage': False, 'center_advantage': False},
                'policy_contract': 'forced assays have no actor/entropy loss; original public inputs, action masks, and scientific reward',
                'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                            'platform': platform.platform(), 'learner_device': device,
                            'rollout_device': 'cpu', 'evaluation_device': 'cpu',
                            'backend_alias': 'mps and split both use explicit CPU rollout mirrors',
                            'wall_measurement_end': 'after final provenance check, before result serialization',
                            'mps_fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'unset')}}
            _write_json(output / 'manifest.json', manifest)
        if not expired():
            with _stage(timings, 'setup_model', device):
                torch.set_num_threads(config.threads)
                mirror, initial = _initial_model(config)
                calibration = dict(ZERO_CALIBRATION if initial is None else initial['calibration'])
                if config.initial_checkpoint_record:
                    inherited = initial['inherited_counters']
                    joint_episode_offset = inherited.get('joint_episode_offset', 0) + inherited.get('joint_episodes', 0)
                    scratch_curriculum = config.initialization == 'scratch' and inherited.get('phase') == 'selector'
                learner = copy.deepcopy(mirror).to(device)
                learner_policy = PublicCostPolicy(learner, **calibration).to(device)
                mirror_policy = PublicCostPolicy(mirror, **calibration)
                optimizer = torch.optim.Adam(learner.parameters(), lr=.001 if phase == 'selector' else config.joint_lr, eps=1e-5)
        if learner is not None and not expired():
            with _stage(timings, 'setup_pools'):
                pool = continuous_family(TRAIN_POOL_SEED, config.train_pool_size)
                tuning_pool = continuous_family(TUNING_POOL_SEED, config.tuning_count)
                tuning = [make_task(mechanism, TUNING_EPISODE_SEED + i, simulator_config)
                          for i, mechanism in enumerate(tuning_pool)]
                train_hashes = [mechanism_hash(mechanism) for mechanism in pool]
                tuning_hashes = [mechanism_hash(mechanism) for mechanism in tuning_pool]
                if set(train_hashes) & set(tuning_hashes):
                    raise RuntimeError('training/tuning mechanisms overlap')
                manifest['pools'].update(train_hashes=train_hashes, tuning_hashes=tuning_hashes)
            if not expired():
                with _stage(timings, 'setup_cache'):
                    cache = PrivateDynamicsCache(pool + tuning_pool, simulator_config)
        with _stage(timings, 'setup_final_manifest'):
            manifest['initial_checkpoint'] = initial
            manifest['calibration'] = calibration
            manifest['scratch_joint_curriculum'] = scratch_curriculum
            manifest['pools']['joint_episode_offset'] = joint_episode_offset
            manifest['setup_completed'] = cache is not None
            _write_json(output / 'manifest.json', manifest)

        if cache is not None and not expired():
            if phase == 'joint':
                joint_started = time.perf_counter()
                exempt = initial is not None and initial['incumbent_eligible']
                evaluate('initial_incumbent' if exempt else 'initial_nonexempt_checkpoint' if initial else 'initial_scratch',
                         incumbent=exempt)
            wall_snapshot()
            last_eval = {'selector': 0, 'joint': 0}
            while True:
                if expired():
                    stop_reason = 'hard_wall_seconds'
                    break
                wall_snapshot()
                if expired():
                    stop_reason = 'hard_wall_seconds'
                    break
                if phase == 'selector' and config.selector_interactions - counts['selector_transitions'] < 5:
                    free = evaluate('selector_quality_native', selection=False)
                    control = None if expired() else evaluate('selector_quality_public_schedule', selection=False, control=True)
                    selector_quality = {'native': free['metrics'], 'public_schedule_control': None if control is None else control['metrics'],
                                        'selector_transitions': counts['selector_transitions'],
                                        'learnability_established': False, 'graduation_asserted': False}
                    checkpoint('selector')
                    if expired():
                        stop_reason = 'hard_wall_seconds'
                        break
                    phase = 'joint'
                    joint_started = time.perf_counter()
                    training_biases = {'behavior_strength': 0., 'initial_outcome_bias': 0.}
                    with _stage(timings, 'joint_optimizer_reset', device):
                        optimizer = torch.optim.Adam(learner.parameters(), lr=config.joint_lr, eps=1e-5)
                    wall_snapshot()
                    continue
                if phase == 'joint':
                    if not primary_closed and config.joint_interactions - counts['joint_transitions'] < 5:
                        row = evaluate('equal_interaction')
                        record = checkpoint('equal_interaction')
                        snapshots['equal_interaction'] = {**counters(), 'target_interactions': config.joint_interactions,
                            'undershoot_interactions': config.joint_interactions - counts['joint_transitions'],
                            'selection_eligible': row['selection_eligible'], 'metrics': row['metrics'], 'checkpoint': record}
                        primary_closed = True
                    if primary_closed and (not config.wall_snapshot_seconds or snapshots['equal_wall'] is not None):
                        stop_reason = 'requested_snapshots_captured'
                        break
                    if config.joint_hard_interactions - counts['joint_transitions'] < 5:
                        stop_reason = 'joint_hard_interactions'
                        break
                if expired():
                    stop_reason = 'hard_wall_seconds'
                    break
                limit = config.selector_interactions if phase == 'selector' else (
                    config.joint_hard_interactions if primary_closed else config.joint_interactions)
                train_batch(limit)
                wall_snapshot()
                current_count = counts[f'{phase}_transitions']
                if not expired() and current_count // config.eval_every_interactions > last_eval[phase] // config.eval_every_interactions:
                    evaluate('periodic')
                    last_eval[phase] = current_count
                    if phase == 'joint':
                        checkpoint('latest')

        training_finished = elapsed()
        with _stage(timings, 'final_bookkeeping'):
            optimizer_devices = {} if learner is None else _optimizer_devices(learner, optimizer)
        if learner is not None:
            checkpoint('selector' if phase == 'selector' else 'latest')
        # Mandatory finite post-training audit; its simulator calls and tail time
        # remain in the run's totals, and it never selects another checkpoint.
        audit_record = (None if best_primary is None else best_primary['checkpoint'])
        if audit_record is None:
            audit_record = checkpoints.get('equal_interaction') or checkpoints.get('latest') or checkpoints.get('selector')
        if audit_record is not None and cache is not None:
            with _stage(timings, 'cache_reference_proof'):
                if _digest(audit_record['path']) != audit_record['sha256']:
                    raise RuntimeError('selected checkpoint changed before reference proof')
                state = torch.load(audit_record['path'], map_location='cpu', weights_only=False)
                audited = ActorCritic(**state['architecture'])
                audited.load_state_dict(state['model'])
                audited_policy = PublicCostPolicy(audited, **state['provenance']['calibration'])
                cached = evaluate_policy(audited_policy, tuning[:8], simulator_config, cache, account=account_validation)
                native = evaluate_policy(audited_policy, tuning[:8], simulator_config, None, account=account_validation)
                proof = {**_reference_comparison(cached, native), 'checkpoint': audit_record,
                         'checkpoint_role': 'best_eligible_primary' if best_primary else 'diagnostic_no_eligible_primary'}
                _write_json(output / 'cache_reference_proof.json', proof)
                if not proof['passed']:
                    raise RuntimeError('cached/native greedy trajectory or score mismatch')
        with _stage(timings, 'source_verification'):
            source_unchanged = hashes == source_hashes()
            initial_unchanged = initial is None or (_digest(initial['path']) == initial['sha256']
                and _digest(initial['record_path']) == initial['record_sha256']
                and all(_digest(item['path']) == item['sha256']
                        for item in initial['incumbent_verification']['verified_artifacts']))
            if not source_unchanged or not initial_unchanged:
                raise RuntimeError('source or original checkpoint provenance changed during training')
        wall = elapsed()
        finished = snapshots['equal_interaction'] is not None and (not config.wall_snapshot_seconds or snapshots['equal_wall'] is not None)
        result = {'schema': SCHEMA, 'name': config.name, 'seed': config.seed, 'scope': scope,
            'status': 'completed' if finished else 'incomplete_budget', 'stop_reason': stop_reason,
            'config': asdict(config), 'manifest': str(output / 'manifest.json'), 'runtime': manifest['runtime'],
            'totals': {**counts, 'all_transitions': counts['training_transitions'] + counts['validation_transitions'],
                       'wall_seconds': wall, 'training_stop_elapsed_seconds': training_finished,
                       'finalization_seconds': wall - training_finished,
                       'hard_wall_overrun_seconds': max(0., wall - config.hard_wall_seconds),
                       'measured_stage_seconds': sum(timings.values())},
            'stage_seconds': timings, 'snapshots': snapshots, 'checkpoints': checkpoints,
            'best_primary': best_primary, 'selected_checkpoint': None if best_primary is None else best_primary['checkpoint']['path'],
            'selection_status': 'eligible_primary_selected' if best_primary else 'no_eligible_primary_checkpoint',
            'evaluations': evaluations, 'selector_quality': selector_quality, 'initial_checkpoint': initial,
            'inherited_pretraining': {'included_in_new_training_budget': False,
                                     'counters': None if initial is None else initial['inherited_counters'],
                                     'complete_historical_compute_known': False if initial else None},
            'optimizer_devices': optimizer_devices, 'resume_supported': False,
            'correctness': {'max_initial_logp_error': max_logp_error, 'initial_logp_tolerance': INITIAL_LOGP_TOLERANCE,
                            'source_hashes_unchanged': source_unchanged, 'initial_checkpoint_unchanged': initial_unchanged,
                            'cache_reference': proof},
            'scientific_claim': 'bounded development training only; selector completion does not establish scientific learnability'}
        _write_json(output / 'result.json', result)
        return result
    except Exception as error:
        _write_json(output / 'failure.json', {'schema': SCHEMA, 'name': config.name, 'seed': config.seed,
            'scope': scope, 'config': asdict(config), 'error_type': type(error).__name__, 'message': str(error),
            'counts': counts, 'phase': phase, 'elapsed_seconds': elapsed(), 'stage_seconds': timings,
            'evaluations': evaluations, 'checkpoints': checkpoints, 'snapshots': snapshots,
            'source_sha256': hashes, 'initial_checkpoint': initial})
        raise
    finally:
        torch.set_num_threads(old_threads)
        torch.set_rng_state(old_torch_rng)
        np.random.set_state(old_numpy_rng)
        if old_mps_rng is not None:
            torch.mps.set_rng_state(old_mps_rng)
