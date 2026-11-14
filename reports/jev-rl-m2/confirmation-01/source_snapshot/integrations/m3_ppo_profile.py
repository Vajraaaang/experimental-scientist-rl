"""Bounded local full-PPO-loop profiling; no study runner or hosted API is used.

The fixed-prefix selector is a throughput control, not a scientific policy
result. Joint profiles retain the native policy, including early stopping.
Run each competing configuration in its own process for clean RSS accounting.
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
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from experimentlab.envs import PilotConfig
from experimentlab.learning import ActorCritic, load_checkpoint, save_checkpoint
from experimentlab.ppo_v4 import collect_batch, update_ppo
from experimentlab.private_cache import CachedScientistEnv, PrivateDynamicsCache
from experimentlab.tasks import continuous_family, mechanism_hash
from experimentlab.tensor_batch import pack_episodes, whole_episode_advantages

ROOT = Path(__file__).resolve().parents[1]
# ppo_followup imports its sibling research_stage. Importing either defines
# functions only: its historical setup()/training/evaluation runners are unused.
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
from ppo_followup import task_batch

SELECTOR_PREFIX = (4, 12, 31)
INITIAL_LOGP_TOLERANCE = 5e-5
TRAIN_POOL_SEED = 13112601
VALIDATION_POOL_SEED = 13112602
VALIDATION_EPISODE_SEED = 31813
TRAIN_NAMESPACE = 8113
VALIDATION_NAMESPACE = 8114
TRAIN_POOL_SIZE = 128
VALIDATION_POOL_SIZE = 16
# This local artifact/hash pair is recorded in the immutable M1 preservation
# record. There is deliberately no arbitrary checkpoint/path deserialization.
V6_CHECKPOINT = 'reports/ppo-efficiency-v6/development/extended_temperature/seed20/selected.pt'
V6_CHECKPOINT_SHA256 = 'c329c6b57945c78099f1af1a62d3d7dbd2e97c4140cc3657073c4189e5eac391'
STAGES = ('mirror_sync_before_rollout', 'environment_setup_reset', 'rollout',
          'packing_transfer', 'advantages', 'update', 'mirror_sync_before_validation',
          'validation_setup_reset', 'validation', 'checkpoint_roundtrip', 'diagnostics')


@dataclass(frozen=True)
class ProfileConfig:
    kind: str = 'recurrent'
    hidden: int = 64
    batch: int = 128
    threads: int = 1
    backend: str = 'cpu'
    cycles: int = 3
    seconds_cap: float = 30.
    seed: int = 20
    workload: str = 'selector'
    checkpoint_every: int = 1
    transition_cap: int = 3_000_000
    initialization: str = 'scratch'
    epochs: int = 2
    gae_lambda: float = .95

    def __post_init__(self):
        for name, maximum in (('hidden', 2048), ('batch', 16384), ('threads', 256),
                              ('cycles', 1_000_000), ('checkpoint_every', 1_000_000),
                              ('transition_cap', 3_000_000), ('epochs', 32)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f'{name} must be an integer in [1, {maximum}]')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        if (isinstance(self.seconds_cap, bool) or not isinstance(self.seconds_cap, (int, float))
                or not math.isfinite(self.seconds_cap) or not 0 < self.seconds_cap <= 600):
            raise ValueError('seconds_cap must be finite and in (0, 600]')
        if (isinstance(self.gae_lambda, bool) or not isinstance(self.gae_lambda, (int, float))
                or not math.isfinite(self.gae_lambda) or not 0 <= self.gae_lambda <= 1):
            raise ValueError('gae_lambda must be finite and in [0, 1]')
        if self.kind not in ('history', 'recurrent'):
            raise ValueError('kind must be history or recurrent')
        if self.backend not in ('cpu', 'mps', 'split'):
            raise ValueError('backend must be cpu, mps, or split')
        if self.workload not in ('selector', 'joint'):
            raise ValueError('workload must be selector or joint')
        if self.initialization not in ('scratch', 'v6'):
            raise ValueError('initialization must be scratch or v6')
        if self.initialization == 'v6' and (self.kind != 'recurrent' or self.hidden != 64 or self.workload != 'joint'):
            raise ValueError('v6 warm start is restricted to recurrent hidden=64 joint profiles')


class PreparedCachedEnv(CachedScientistEnv):
    """Time the real reset externally, then consume it once in collect_batch."""

    def prepare_reset(self):
        self._prepared_reset = super().reset()

    def reset(self):
        if hasattr(self, '_prepared_reset'):
            result = self._prepared_reset
            del self._prepared_reset
            return result
        return super().reset()


def _synchronize(device):
    if torch.device(device).type == 'mps':
        torch.mps.synchronize()


@contextmanager
def _stage(timings, name, device):
    _synchronize(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        _synchronize(device)
        timings[name] = timings.get(name, 0.) + time.perf_counter() - started


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def source_hashes():
    """Hash the scientific implementation and exact profiling entry points."""
    files = set((ROOT / 'src').rglob('*.py'))
    files.update(ROOT / relative for relative in (
        'integrations/m3_ppo_profile.py', 'scripts/profile_m3_ppo.py',
        'scripts/ppo_followup.py', 'scripts/research_stage.py',
        'docs/jev_rl_m2_task_derivation.md', 'requirements.lock', 'pyproject.toml'))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files) if path.is_file()}


def thermal_status():
    """Record the OS report verbatim; do not infer temperatures or throttling."""
    if platform.system() != 'Darwin':
        return {'available': False, 'reason': 'pmset is a macOS interface'}
    try:
        result = subprocess.run(['/usr/bin/pmset', '-g', 'therm'], capture_output=True,
                                text=True, timeout=1., check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'available': False, 'reason': type(error).__name__}
    return {'available': result.returncode == 0, 'command': 'pmset -g therm',
            'returncode': result.returncode, 'stdout': result.stdout[:4096],
            'stderr': result.stderr[:4096], 'temperature_measured': False}


def memory_status(device):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'process_peak_rss_bytes': int(rss if platform.system() == 'Darwin' else rss * 1024),
            'process_peak_rss_scope': 'process lifetime, including setup and checkpoint verification',
            'current_mps_allocated_bytes': int(torch.mps.current_allocated_memory()) if device == 'mps' else None,
            'current_mps_driver_allocated_bytes': int(torch.mps.driver_allocated_memory()) if device == 'mps' else None,
            'mps_memory_scope': 'current at sample time; not a peak'}


def optimizer_devices(model, optimizer):
    expected = next(model.parameters()).device
    parameter_devices = sorted({str(p.device) for p in model.parameters()})
    moments, steps = set(), set()
    for state in optimizer.state.values():
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if name == 'step' and tensor.numel() == 1:
                steps.add(str(tensor.device))
                if tensor.device.type not in ('cpu', expected.type):
                    raise RuntimeError('Adam scalar step is on an unrelated device')
            else:
                moments.add(str(tensor.device))
                if tensor.device != expected:
                    raise RuntimeError('Adam moments left the learner device')
    if any(p.device != expected for p in model.parameters()):
        raise RuntimeError('learner parameters span devices')
    return {'parameter_devices': parameter_devices, 'moment_devices': sorted(moments),
            'scalar_step_devices': sorted(steps)}


def _equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_equal_state(a, b) for a, b in zip(left, right))
    return left == right


def _adam(model):
    return torch.optim.Adam(model.parameters(), lr=.0003, eps=1e-5)


def checkpoint_roundtrip(path, model, optimizer, counters, manifest_path):
    """Verify state and one deterministic Adam continuation without training twice.

    Two throwaway copies receive an identical constant-gradient probe. The live
    learner is unchanged, no simulator transition is consumed, and this probe
    cost is included in the checkpoint stage. Only our newly saved file is read.
    """
    save_checkpoint(path, model, optimizer, counters, {
        'manifest_path': str(manifest_path), 'scope': 'm3_hardware_development_only'})
    restored = copy.deepcopy(model)
    restored_optimizer = _adam(restored)
    loaded = load_checkpoint(path, restored, restored_optimizer)
    parameters_equal = _equal_state(model.state_dict(), restored.state_dict())
    optimizer_equal = _equal_state(optimizer.state_dict(), restored_optimizer.state_dict())
    optimizer_devices(restored, restored_optimizer)
    reference = copy.deepcopy(model)
    reference_optimizer = _adam(reference)
    reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for candidate, candidate_optimizer in ((reference, reference_optimizer), (restored, restored_optimizer)):
        for parameter in candidate.parameters():
            parameter.grad = torch.full_like(parameter, .125)
        candidate_optimizer.step()
    continuation_equal = (_equal_state(reference.state_dict(), restored.state_dict()) and
                          _equal_state(reference_optimizer.state_dict(), restored_optimizer.state_dict()))
    if not (parameters_equal and optimizer_equal and continuation_equal and loaded == counters):
        raise RuntimeError('checkpoint roundtrip or optimizer continuation diverged')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size, 'parameters_equal': parameters_equal,
            'optimizer_equal': optimizer_equal, 'continuation_equal': continuation_equal,
            'continuation_probe': 'one constant-gradient Adam step on throwaway copies; no extra PPO update'}


def _prepared_environments(tasks, config, cache):
    environments = [PreparedCachedEnv(task, config, cache) for task in tasks]
    for environment in environments:
        environment.prepare_reset()
    return environments


def _sync_mirror(learner, mirror):
    # No learner .to('cpu') and no optimizer movement. The copy is explicit and
    # its device synchronization/cost is included in the mirror-sync stage.
    mirror.load_state_dict({key: tensor.detach().cpu() for key, tensor in learner.state_dict().items()})


def _load_initial_cpu_model(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = ActorCritic(kind=config.kind, hidden=config.hidden, horizon=5)
    if config.initialization == 'v6':
        path = ROOT / V6_CHECKPOINT
        if hashlib.sha256(path.read_bytes()).hexdigest() != V6_CHECKPOINT_SHA256:
            raise RuntimeError('known local v6 checkpoint hash mismatch')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['architecture'] != model.architecture:
            raise RuntimeError('known local v6 checkpoint architecture mismatch')
        model.load_state_dict(state['model'])
    return model


def _manifest(config, output, hashes):
    return {'schema': 'm3-ppo-profile-v1', 'scope': 'hardware_development_only',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'output_dir': str(output),
            'config': asdict(config), 'source_sha256': hashes,
            'final_split_opened': False, 'hosted_calls': 0,
            'pools': {'train_seed': TRAIN_POOL_SEED, 'validation_seed': VALIDATION_POOL_SEED,
                      'train_size': TRAIN_POOL_SIZE, 'validation_size': VALIDATION_POOL_SIZE,
                      'train_episode_seed': config.seed, 'validation_episode_seed': VALIDATION_EPISODE_SEED,
                      'train_namespace': TRAIN_NAMESPACE, 'validation_namespace': VALIDATION_NAMESPACE,
                      'scope': 'fresh independent profiling train/development pools; never final evaluation'},
            'selector_prefix': list(SELECTOR_PREFIX) if config.workload == 'selector' else None,
            'policy_contract': 'forced prefix has zero actor/entropy loss; all valid steps train the critic',
            'initial_checkpoint': {'path': V6_CHECKPOINT, 'sha256': V6_CHECKPOINT_SHA256,
                                   'inherited_pretraining_cost': 'not included in this profile',
                                   'optimizer': 'new Adam; weights-only warm start'} if config.initialization == 'v6' else None,
            'caps': {'wall_seconds': config.seconds_cap, 'all_simulator_transitions': config.transition_cap,
                     'semantics': 'check before each complete train/update/validation/checkpoint cycle; wall cap may overrun by one cycle; reserve worst-case transitions before starting'},
            'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                        'system': platform.system(), 'platform': platform.platform(),
                        'machine': platform.machine(), 'cpu_count': os.cpu_count(),
                        'intraop_threads_requested': config.threads,
                        'interop_threads': torch.get_num_interop_threads(),
                        'mps_available': torch.backends.mps.is_available(),
                        'mps_fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'unset')}}


def profile(config, output_dir):
    """Return/write a JSON-safe profile in an exclusively created directory.

    ``cpu`` uses CPU throughout; ``mps`` retains CPU environments with MPS
    inference and optimization; ``split`` keeps a CPU rollout/validation mirror
    and an MPS learner whose Adam moments never leave MPS. Validation uses 16
    independent development tasks greedily on every complete cycle.
    """
    if not isinstance(config, ProfileConfig):
        raise TypeError('config must be a ProfileConfig')
    fallback = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '').strip().lower()
    if fallback not in ('', '0', 'false', 'no', 'off'):
        raise RuntimeError('PYTORCH_ENABLE_MPS_FALLBACK must be disabled for honest native profiling')
    if config.backend != 'cpu' and not torch.backends.mps.is_available():
        raise RuntimeError('native MPS is unavailable')
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    setup = {}
    device = 'cpu' if config.backend == 'cpu' else 'mps'
    rollout_device = 'cpu' if config.backend == 'split' else device
    rows = []
    old_threads = torch.get_num_threads()
    old_torch_rng, old_numpy_rng = torch.get_rng_state(), np.random.get_state()
    try:
        with _stage(setup, 'source_manifest', 'cpu'):
            hashes = source_hashes()
            manifest = _manifest(config, output, hashes)
            _write_json(output / 'manifest.json', manifest)
        with _stage(setup, 'thermal_start', 'cpu'):
            thermal_before = thermal_status()
        with _stage(setup, 'model_setup', device):
            torch.set_num_threads(config.threads)
            initial_cpu = _load_initial_cpu_model(config)
            learner = copy.deepcopy(initial_cpu).to(device)
            mirror = initial_cpu if config.backend == 'split' else learner
            if config.backend != 'split':
                del initial_cpu
            optimizer = _adam(learner)
        with _stage(setup, 'pool_setup', 'cpu'):
            simulator_config = PilotConfig()
            train_pool = continuous_family(TRAIN_POOL_SEED, TRAIN_POOL_SIZE)
            validation_pool = continuous_family(VALIDATION_POOL_SEED, VALIDATION_POOL_SIZE)
            train_hashes = [mechanism_hash(m) for m in train_pool]
            validation_hashes = [mechanism_hash(m) for m in validation_pool]
            if set(train_hashes) & set(validation_hashes):
                raise RuntimeError('profiling training/development mechanisms overlap')
        with _stage(setup, 'cache_setup', 'cpu'):
            cache = PrivateDynamicsCache(train_pool + validation_pool, simulator_config)
        train_transitions = validation_transitions = train_policy = episodes = 0
        stop_reason = 'cycles'
        maximum_cycle_transitions = (4 if config.workload == 'selector' else 5) * (config.batch + VALIDATION_POOL_SIZE)
        for cycle in range(1, config.cycles + 1):
            if time.perf_counter() - started >= config.seconds_cap:
                stop_reason = 'seconds_cap'
                break
            if train_transitions + validation_transitions + maximum_cycle_transitions > config.transition_cap:
                stop_reason = 'transition_cap'
                break
            cycle_started = time.perf_counter()
            stages = dict.fromkeys(STAGES, 0.)
            if config.backend == 'split':
                with _stage(stages, 'mirror_sync_before_rollout', device):
                    _sync_mirror(learner, mirror)
            with _stage(stages, 'environment_setup_reset', device):
                tasks = task_batch(train_pool, simulator_config, config.seed, episodes, config.batch, TRAIN_NAMESPACE)
                environments = _prepared_environments(tasks, simulator_config, cache)
                schedules = [SELECTOR_PREFIX] * config.batch if config.workload == 'selector' else None
            with _stage(stages, 'rollout', device):
                records, training = collect_batch(mirror, environments, rollout_device, public_schedules=schedules)
            with _stage(stages, 'packing_transfer', device):
                batch = pack_episodes(records, device)
            with _stage(stages, 'advantages', device):
                advantages, targets = whole_episode_advantages(batch, gae_lambda=config.gae_lambda)
            with _stage(stages, 'update', device):
                update = update_ppo(learner, optimizer, batch, epochs=config.epochs,
                                    gae_lambda=config.gae_lambda, prepared_advantages=advantages,
                                    prepared_targets=targets)
                if update['initial_logp_error'] > INITIAL_LOGP_TOLERANCE:
                    raise RuntimeError(f"initial rollout/update log probability mismatch: {update['initial_logp_error']}")
            if config.backend == 'split':
                with _stage(stages, 'mirror_sync_before_validation', device):
                    _sync_mirror(learner, mirror)
            with _stage(stages, 'validation_setup_reset', device):
                validation_tasks = task_batch(validation_pool, simulator_config, VALIDATION_EPISODE_SEED,
                                              0, VALIDATION_POOL_SIZE, VALIDATION_NAMESPACE)
                validation_envs = _prepared_environments(validation_tasks, simulator_config, cache)
                validation_schedules = [SELECTOR_PREFIX] * VALIDATION_POOL_SIZE if config.workload == 'selector' else None
            with _stage(stages, 'validation', device):
                _, validation = collect_batch(mirror, validation_envs, rollout_device, deterministic=True,
                                              public_schedules=validation_schedules)
            train_transitions += training['transitions']
            train_policy += training['policy_decisions']
            validation_transitions += validation['transitions']
            episodes += training['episodes']
            checkpoint = None
            if cycle % config.checkpoint_every == 0:
                with _stage(stages, 'checkpoint_roundtrip', device):
                    checkpoint = checkpoint_roundtrip(output / f'checkpoint-{cycle:04d}.pt', learner, optimizer,
                                                      {'cycles': cycle, 'episodes': episodes,
                                                       'training_transitions': train_transitions,
                                                       'training_policy_decisions': train_policy}, output / 'manifest.json')
            with _stage(stages, 'diagnostics', device):
                devices = optimizer_devices(learner, optimizer)
                memory = memory_status(device)
            row = {'cycle': cycle, 'stage_seconds': stages,
                   'cycle_seconds': time.perf_counter() - cycle_started,
                   'elapsed_seconds': time.perf_counter() - started,
                   'training': training, 'validation': validation, 'update': update,
                   'checkpoint': checkpoint, 'optimizer_devices': devices, 'memory': memory}
            rows.append(row)
            with (output / 'cycles.jsonl').open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        with _stage(setup, 'thermal_end', 'cpu'):
            thermal_after = thermal_status()
        with _stage(setup, 'source_verification', 'cpu'):
            unchanged = hashes == source_hashes()
        if not unchanged:
            raise RuntimeError('profile source files changed during execution')
        stage_totals = {name: sum(row['stage_seconds'][name] for row in rows) for name in STAGES}
        cycle_seconds = sum(row['cycle_seconds'] for row in rows)
        wall_seconds = time.perf_counter() - started
        all_transitions = train_transitions + validation_transitions
        rates = [row['training']['transitions'] / row['cycle_seconds'] for row in rows]
        half = max(1, len(rows) // 2)

        def rate_for(group):
            return sum(r['training']['transitions'] for r in group) / sum(r['cycle_seconds'] for r in group) if group else None

        result = {'schema': 'm3-ppo-profile-v1', 'scope': 'hardware_development_only',
                  'status': 'completed', 'stop_reason': stop_reason, 'config': asdict(config),
                  'manifest': str(output / 'manifest.json'), 'rows': rows,
                  'setup_seconds': setup, 'stage_totals_seconds': stage_totals,
                  'totals': {'cycles': len(rows), 'training_episodes': episodes,
                             'training_transitions': train_transitions, 'training_policy_decisions': train_policy,
                             'validation_transitions': validation_transitions, 'all_transitions': all_transitions,
                             'cycle_seconds': cycle_seconds, 'wall_seconds': wall_seconds,
                             'measured_stage_seconds': sum(setup.values()) + sum(stage_totals.values()),
                             'training_transitions_per_cycle_second': train_transitions / cycle_seconds if cycle_seconds else None,
                             'training_transitions_per_wall_second': train_transitions / wall_seconds,
                             'all_transitions_per_wall_second': all_transitions / wall_seconds},
                  'sustained': {'per_cycle_training_transitions_per_second': rates,
                                'first_half_training_transitions_per_second': rate_for(rows[:half]),
                                'second_half_training_transitions_per_second': rate_for(rows[half:]),
                                'after_first_cycle_training_transitions_per_second': rate_for(rows[1:]),
                                'interpretation': 'measured repeated-cycle throughput; variation alone is not evidence of thermal throttling'},
                  'correctness': {'initial_logp_tolerance': INITIAL_LOGP_TOLERANCE,
                                  'max_initial_logp_error': max((r['update']['initial_logp_error'] for r in rows), default=None),
                                  'source_hashes_unchanged': unchanged,
                                  'pool_mechanism_hashes_disjoint': True,
                                  'train_mechanism_sha256': train_hashes, 'validation_mechanism_sha256': validation_hashes},
                  'memory': memory_status(device), 'thermal': {'before': thermal_before, 'after': thermal_after},
                  'scientific_claim': 'hardware profile only; neither selection nor biological/policy qualification'}
        _write_json(output / 'result.json', result)
        return result
    except Exception as error:
        _write_json(output / 'failure.json', {'error_type': type(error).__name__, 'message': str(error),
                                            'completed_cycles': len(rows), 'rows': rows,
                                            'elapsed_seconds': time.perf_counter() - started})
        raise
    finally:
        torch.set_num_threads(old_threads)
        torch.set_rng_state(old_torch_rng)
        np.random.set_state(old_numpy_rng)
