"""Actual hardware inspection and synchronized complete rollout/update benchmark."""
from pathlib import Path
import importlib.metadata
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
from .envs import PilotConfig, ScientistEnv
from .tasks import finite_family, make_task
from .learning import ActorCritic, collect_episode, episode_batch, ppo_update, save_checkpoint, load_checkpoint


def _command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True)
    except FileNotFoundError:
        return {'unavailable': args[0], 'platform': platform.system()}
    return p.stdout.strip() if p.returncode == 0 else {'error': p.stderr.strip()}


def doctor():
    deps = {m: importlib.metadata.version(m) for m in ('numpy', 'scipy', 'torch', 'PyYAML', 'pytest')}
    return {'python': sys.executable, 'python_version': platform.python_version(),
            'architecture': platform.machine(), 'macos': _command(['/usr/bin/sw_vers']),
            'ram_bytes': _command(['/usr/sbin/sysctl', '-n', 'hw.memsize']),
            'hardware_model': _command(['/usr/sbin/sysctl', '-n', 'hw.model']),
            'chip': _command(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string']),
            'logical_cpus': os.cpu_count(), 'torch_cpu': bool(torch.ones(2).sum().item() == 2),
            'mps_built': torch.backends.mps.is_built(), 'mps_available': torch.backends.mps.is_available(),
            'mps_fallback_environment': os.getenv('PYTORCH_ENABLE_MPS_FALLBACK', 'unset'),
            'dependencies': deps, 'reference_dtype': 'NumPy/SciPy float64 CPU', 'learning_dtype': 'PyTorch float32'}


def synchronize(device):
    if device == 'mps': torch.mps.synchronize()


def peak_memory():
    # macOS ru_maxrss is bytes, Linux KiB. This project records which convention.
    scale = 1 if sys.platform == 'darwin' else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


def benchmark_device(device, episodes=24, repeats=3):
    if device == 'mps' and not torch.backends.mps.is_available():
        return {'status': 'unavailable', 'diagnostic': 'MPS unavailable in this process; check native execution outside sandbox.'}
    if device == 'mps' and os.getenv('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        return {'status': 'rejected', 'diagnostic': 'Disable implicit CPU fallback for an honest MPS benchmark.'}
    torch.set_num_threads(1); torch.manual_seed(120)
    config = PilotConfig(offset_sd=.03)
    family = finite_family()
    model = ActorCritic(kind='recurrent', hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    results = []
    try:
        for repeat in range(repeats+1):
            synchronize(device); start = time.perf_counter()
            rollouts = []
            for i in range(episodes):
                env = ScientistEnv(make_task(family[i % len(family)], 817000+i, config), config)
                rollouts.append(collect_episode(model, env, device))
            synchronize(device); rollout_end = time.perf_counter()
            batch = episode_batch(rollouts, device)  # includes host/device transfers
            metrics = ppo_update(model, opt, batch)
            synchronize(device); update_end = time.perf_counter()
            # Evaluation and save/load are included, not subtracted from timings.
            env = ScientistEnv(make_task(family[0], 9900, config), config)
            collect_episode(model, env, device, deterministic=True)
            with tempfile.TemporaryDirectory(prefix='experimentlab-', dir=tempfile.gettempdir()) as folder:
                path = Path(folder).resolve() / 'checkpoint.pt'
                save_checkpoint(path, model, opt, {'episodes': episodes})
                load_checkpoint(path, model, opt)
            synchronize(device); end = time.perf_counter()
            row = {'total_seconds': end-start, 'rollout_seconds': rollout_end-start,
                   'update_seconds': update_end-rollout_end, 'evaluation_checkpoint_seconds': end-update_end,
                   'transitions': int(batch['valid'].sum().item()), 'episodes': episodes,
                   'transitions_per_second': int(batch['valid'].sum().item())/(end-start),
                   'metrics': metrics}
            if repeat: results.append(row)
        median = float(np.median([r['total_seconds'] for r in results]))
        return {'status': 'ok', 'device': device, 'repeats_excluding_warmup': repeats, 'results': results,
                'median_total_seconds': median, 'peak_process_rss_bytes': peak_memory(),
                'mps_allocated_bytes': torch.mps.current_allocated_memory() if device == 'mps' else 0,
                'forecast_100k_transitions_seconds': 100000/np.median([r['transitions_per_second'] for r in results]),
                'forecast_scope': 'Short end-to-end smoke throughput; excludes planner/teacher generation and long-run thermal effects.'}
    except (RuntimeError, NotImplementedError) as exc:
        return {'status': 'unsupported_workload', 'device': device, 'diagnostic': str(exc)}


def benchmark(episodes=24, repeats=3):
    cpu = benchmark_device('cpu', episodes, repeats)
    mps = benchmark_device('mps', episodes, repeats)
    chosen = 'cpu'
    if cpu['status'] == mps['status'] == 'ok' and mps['median_total_seconds'] < .95*cpu['median_total_seconds']: chosen = 'mps'
    return {'hardware': doctor(), 'cpu': cpu, 'mps': mps, 'selected_device': chosen,
            'selection_rule': 'MPS only if full workload succeeds and is at least 5% faster than CPU.',
            'peak_process_rss_bytes': peak_memory()}
