#!/usr/bin/env python3
"""Bounded kernel/packing microbenchmarks, never full training throughput.

Run natively for MPS access. CPU timings can be marked contended when other
training is active. No simulator, checkpoint, reward, or training data is read.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from experimentlab.learning import ActorCritic, episode_batch, returns_gae
from experimentlab.tensor_batch import pack_episodes, whole_episode_advantages


def synthetic_episodes(size):
    rng = np.random.default_rng(725)
    result = []
    for i in range(size):
        n = 1 + i % 5
        result.append({'tokens': rng.normal(size=(n, 75)).astype(np.float32),
                       'masks': np.ones((n, 45), dtype=bool),
                       'actions': rng.integers(0, 45, size=n),
                       'rewards': rng.normal(0, .2, size=n).astype(np.float32),
                       'old_logp': np.full(n, -np.log(45), np.float32),
                       'old_values': rng.normal(0, .2, size=n).astype(np.float32),
                       'biases': np.zeros((n, 45), np.float32),
                       'policy_steps': [False] * (n - 1) + [True]})
    return result


def legacy_pack(episodes, device):
    # Frozen packing reference: retain the original many-small-transfer cost
    # after ppo_v4.batch_episodes is changed to use the optimized implementation.
    batch = episode_batch(episodes, device)
    batch['biases'] = torch.zeros((*batch['actions'].shape, 45), device=device)
    batch['policy_valid'] = batch['valid'].clone()
    for i, episode in enumerate(episodes):
        n = len(episode['actions'])
        batch['biases'][i, :n] = torch.as_tensor(episode['biases'], dtype=torch.float32, device=device)
        batch['policy_valid'][i, :n] = torch.as_tensor(episode['policy_steps'], dtype=torch.bool, device=device)
    batch['policy_valid'] &= batch['valid']
    if not bool(batch['policy_valid'].any()):
        raise ValueError('batch has no sampled policy decision')
    return batch


def scalar_advantages(batch):
    advantages = torch.zeros_like(batch['rewards'])
    targets = torch.zeros_like(advantages)
    for i in range(len(batch['valid'])):
        n = int(batch['valid'][i].sum())
        advantages[i, :n], targets[i, :n] = returns_gae(
            batch['rewards'][i, :n], batch['old_values'][i, :n], gae_lambda=.95)
    return advantages, targets


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()


def measure(operation, device, warmup, repeats):
    for _ in range(warmup):
        operation()
    synchronize(device)
    elapsed = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        operation()
        synchronize(device)
        elapsed.append((time.perf_counter() - started) * 1000)
    return {'median_ms': statistics.median(elapsed), 'min_ms': min(elapsed),
            'max_ms': max(elapsed), 'repeats': repeats, 'warmup': warmup}


def imitation_step(model, optimizer, batch):
    # Preloaded tensors only, no scalar exports or distribution validation sync.
    logits, _ = model(batch['tokens'])
    logits = logits.masked_fill(~batch['masks'], -1e9)
    per_step = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), batch['actions'].flatten(), reduction='none')
    weights = batch['valid'].flatten().to(per_step.dtype)
    loss = (per_step * weights).sum() / weights.sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def memory(device):
    result = {'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'rss_scope': 'cumulative process high-water mark; macOS bytes'}
    if device == 'mps':
        synchronize(device)
        result.update(mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                      mps_recommended_max_bytes=torch.mps.recommended_max_memory())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[256, 1024, 4096])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cpu-contended', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'reports/ppo-v4/gpu-profile.json')
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.warmup < 1 or args.repeats < 1:
        parser.error('positive batch sizes, warmup and repeats required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(720)
    available = torch.backends.mps.is_available()
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Synthetic packing/GAE and GRU64 imitation microbenchmarks; not full PPO or scientific training throughput.',
        'complete_microstep_scope': 'host packing + transfer + GAE + one imitation optimizer update; excludes environment simulation, collection, PPO loss/diagnostics, validation and checkpoints',
        'nn_only_scope': 'preloaded device tensors; GRU64 actor forward, masked cross-entropy backward, Adam step; no critic loss or gradient clipping',
        'cpu_contended': args.cpu_contended,
        'contention_caveat': 'Concurrent CPU training can bias CPU timings and host overhead on both devices; do not choose the production device from these timings alone.' if args.cpu_contended else None,
        'torch_version': torch.__version__, 'python_version': platform.python_version(),
        'platform': platform.platform(), 'cpu_threads': torch.get_num_threads(),
        'mps_built': torch.backends.mps.is_built(), 'mps_available': available,
        'mps_fallback_env': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        'device_support': {}, 'profiles': [],
    }
    for device in ['cpu'] + (['mps'] if available else []):
        # Executing the actual forward/backward below proves operator support.
        for size in args.sizes:
            episodes = synthetic_episodes(size)
            batch = pack_episodes(episodes, device)
            model = ActorCritic(kind='recurrent', hidden=64).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            reference = scalar_advantages(pack_episodes(episodes, 'cpu'))
            actual = whole_episode_advantages(batch, .95)
            max_error = max(float((a.cpu() - b).abs().max()) for a, b in zip(actual, reference))
            if max_error > 1e-6:
                raise AssertionError(f'GAE equivalence failed on {device}: {max_error}')
            operations = {
                'pack_host_and_transfer': lambda: pack_episodes(episodes, device),
                'legacy_pack_host_and_transfer': lambda: legacy_pack(episodes, device),
                'gae_preloaded_tensor': lambda: whole_episode_advantages(batch, .95),
                'gae_preloaded_scalar': lambda: scalar_advantages(batch),
                'nn_only_preloaded_im_update': lambda: imitation_step(model, optimizer, batch),
            }
            timings = {name: measure(fn, device, args.warmup, args.repeats)
                       for name, fn in operations.items()}
            def complete_microstep():
                packed = pack_episodes(episodes, device)
                whole_episode_advantages(packed, .95)
                imitation_step(model, optimizer, packed)
            timings['complete_pack_gae_im_update'] = measure(
                complete_microstep, device, args.warmup, args.repeats)
            report['profiles'].append({'device': device, 'batch_size': size,
                                       'horizon': 5, 'valid_transitions': sum(len(e['actions']) for e in episodes),
                                       'gae_max_abs_error': max_error, 'timings': timings,
                                       'memory': memory(device)})
            report['device_support'][device] = {'packing': True, 'tensor_gae': True,
                                                 'gru64_forward_backward_adam': True,
                                                 'model_parameter_device': str(next(model.parameters()).device)}
            print(f'{device} B={size}: pack={timings["pack_host_and_transfer"]["median_ms"]:.2f}ms '
                  f'GAE={timings["gae_preloaded_tensor"]["median_ms"]:.2f}ms '
                  f'NN={timings["nn_only_preloaded_im_update"]["median_ms"]:.2f}ms '
                  f'complete_microstep={timings["complete_pack_gae_im_update"]["median_ms"]:.2f}ms', flush=True)
            del batch, model, optimizer, actual
            if device == 'mps':
                torch.mps.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
