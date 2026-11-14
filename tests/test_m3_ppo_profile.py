"""Bounded full-loop profiling preserves the existing PPO/scientific contract."""
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def profiler():
    assert importlib.util.find_spec('integrations.m3_ppo_profile') is not None, 'full-loop profiler is missing'
    return importlib.import_module('integrations.m3_ppo_profile')


@pytest.mark.parametrize('kind', ['history', 'recurrent'])
def test_cpu_complete_cycles_include_real_counts_and_checkpoint_resume(tmp_path, kind):
    m = profiler()
    config = m.ProfileConfig(kind=kind, hidden=16, batch=4, cycles=2, seconds_cap=20)
    destination = tmp_path / kind
    result = m.profile(config, destination)
    assert result['status'] == 'completed'
    assert result['stop_reason'] == 'cycles'
    assert result['totals']['training_transitions'] == 4 * 4 * 2
    assert result['totals']['training_policy_decisions'] == 4 * 2
    assert result['totals']['validation_transitions'] == 16 * 4 * 2
    assert result['totals']['all_transitions'] == (4 + 16) * 4 * 2
    assert result['totals']['cycles'] == 2
    assert result['totals']['cycle_seconds'] <= result['totals']['wall_seconds']
    assert result['totals']['measured_stage_seconds'] <= result['totals']['wall_seconds']
    for row in result['rows']:
        assert row['training']['mean_assays'] == 3
        assert row['update']['valid_decisions'] == 16
        assert row['update']['policy_decisions'] == 4
        assert 1 <= row['update']['epochs_taken'] <= 2
        assert row['update']['initial_logp_error'] < result['correctness']['initial_logp_tolerance']
        assert row['checkpoint']['parameters_equal']
        assert row['checkpoint']['optimizer_equal']
        assert row['checkpoint']['continuation_equal']
        assert row['optimizer_devices']['moment_devices'] == ['cpu']
        assert row['stage_seconds']['environment_setup_reset'] > 0
        assert row['stage_seconds']['validation_setup_reset'] > 0
        assert row['stage_seconds']['checkpoint_roundtrip'] > 0
        assert sum(row['stage_seconds'].values()) <= row['cycle_seconds'] + 1e-5
        assert all(math.isfinite(value) for value in row['update'].values())
    assert result['memory']['process_peak_rss_bytes'] > 0
    assert 'current_mps_allocated_bytes' in result['memory']
    assert result['correctness']['source_hashes_unchanged']
    manifest = json.loads((destination / 'manifest.json').read_text())
    assert manifest['source_sha256']['src/experimentlab/ppo_v4.py']
    assert manifest['pools']['train_size'] == 128
    assert manifest['pools']['validation_size'] == 16
    assert manifest['pools']['train_seed'] != manifest['pools']['validation_seed']
    assert manifest['final_split_opened'] is False
    assert json.loads((destination / 'result.json').read_text()) == result
    assert len((destination / 'cycles.jsonl').read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        m.profile(config, destination)


@pytest.mark.parametrize('kwargs', [
    {'batch': 0}, {'batch': True}, {'hidden': 0}, {'threads': 0}, {'cycles': 0},
    {'seconds_cap': 0}, {'seconds_cap': float('nan')}, {'seconds_cap': 601},
    {'backend': 'cuda'}, {'kind': 'latest'}, {'workload': 'forced'},
    {'checkpoint_every': 0}, {'transition_cap': 0}, {'transition_cap': 3_000_001},
    {'initialization': 'v6', 'kind': 'history', 'workload': 'joint'},
    {'initialization': 'v6', 'hidden': 128, 'workload': 'joint'},
    {'initialization': 'v6', 'workload': 'selector'},
])
def test_bad_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        profiler().ProfileConfig(**kwargs)


def test_no_silent_mps_fallback_even_for_cpu_profile(tmp_path, monkeypatch):
    m = profiler()
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    with pytest.raises(RuntimeError, match='FALLBACK'):
        m.profile(m.ProfileConfig(batch=4, cycles=1), tmp_path / 'must-not-exist')
    assert not (tmp_path / 'must-not-exist').exists()


def test_tiny_wall_cap_and_transition_cap_never_start_incomplete_batch(tmp_path):
    m = profiler()
    result = m.profile(m.ProfileConfig(batch=4, cycles=2, seconds_cap=1e-12), tmp_path / 'wall')
    assert result['stop_reason'] == 'seconds_cap'
    assert result['totals']['all_transitions'] == 0
    result = m.profile(m.ProfileConfig(batch=4, cycles=2, transition_cap=1), tmp_path / 'transitions')
    assert result['stop_reason'] == 'transition_cap'
    assert result['totals']['all_transitions'] == 0


def test_joint_profile_counts_observed_native_policy_length(tmp_path):
    m = profiler()
    result = m.profile(m.ProfileConfig(kind='history', hidden=16, batch=4, cycles=1,
                                      workload='joint'), tmp_path / 'joint')
    row = result['rows'][0]
    assert 4 <= row['training']['transitions'] <= 20
    assert row['training']['policy_decisions'] == row['training']['transitions']
    assert row['update']['initial_logp_error'] < result['correctness']['initial_logp_tolerance']


def test_hash_bound_local_v6_warm_start_is_joint_only_and_immutable(tmp_path, monkeypatch):
    m = profiler()
    # The public release excludes pretrained binaries. Exercise the same trusted
    # registry/hash contract with a local synthetic checkpoint, not a private file.
    from experimentlab.learning import ActorCritic
    import hashlib
    model = ActorCritic(kind='recurrent', hidden=64)
    checkpoint = tmp_path / 'registered-fixture.pt'
    torch.save({'architecture': model.architecture, 'model': model.state_dict()}, checkpoint)
    monkeypatch.setattr(m, 'V6_CHECKPOINT', str(checkpoint))
    monkeypatch.setattr(m, 'V6_CHECKPOINT_SHA256', hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    config = m.ProfileConfig(batch=4, cycles=1, workload='joint', initialization='v6')
    result = m.profile(config, tmp_path / 'v6')
    manifest = json.loads(Path(result['manifest']).read_text())
    assert manifest['initial_checkpoint']['sha256'] == m.V6_CHECKPOINT_SHA256
    assert manifest['initial_checkpoint']['optimizer'] == 'new Adam; weights-only warm start'
    assert result['rows'][0]['checkpoint']['continuation_equal']
    monkeypatch.setattr(m, 'V6_CHECKPOINT_SHA256', '0' * 64)
    with pytest.raises(RuntimeError, match='checkpoint hash mismatch'):
        m.profile(config, tmp_path / 'bad-hash')


def test_cli_absolute_entrypoint_from_other_directory(tmp_path):
    command = [sys.executable, str(ROOT / 'scripts/profile_m3_ppo.py'), '--help']
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert '--backend' in result.stdout and '--transition-cap' in result.stdout


def test_prepared_reset_and_cached_workload_match_reference():
    m = profiler()
    from experimentlab.envs import PilotConfig, ScientistEnv
    from experimentlab.private_cache import PrivateDynamicsCache
    from experimentlab.tasks import continuous_family, make_task
    config = PilotConfig()
    mechanism = continuous_family(813661, 1)[0]
    task = make_task(mechanism, 813662, config)
    cached = m.PreparedCachedEnv(task, config, PrivateDynamicsCache([mechanism], config))
    reference = ScientistEnv(task, config)
    cached.prepare_reset()

    def equal(actual, expected):
        if isinstance(expected, np.ndarray):
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True)
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                equal(a, b)
        elif isinstance(expected, float):
            assert actual == pytest.approx(expected, abs=1e-12, rel=0)
        else:
            assert actual == expected

    equal(cached.reset(), reference.reset())
    for action in (*m.SELECTOR_PREFIX, 40):
        equal(cached.step(action), reference.step(action))
    equal(cached.private_result, reference.private_result)
    # A subsequent genuine reset must not return the old prepared observation.
    equal(cached.reset(), reference.reset())


@pytest.mark.parametrize('backend', ['mps', 'split'])
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='native MPS unavailable')
def test_mps_native_cycle_and_split_optimizer_remain_on_device(tmp_path, monkeypatch, backend):
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK', raising=False)
    m = profiler()
    result = m.profile(m.ProfileConfig(hidden=16, batch=4, cycles=1, backend=backend), tmp_path / backend)
    row = result['rows'][0]
    assert row['optimizer_devices']['parameter_devices'] == ['mps:0']
    assert row['optimizer_devices']['moment_devices'] == ['mps:0']
    assert row['checkpoint']['continuation_equal']
    assert row['update']['initial_logp_error'] < result['correctness']['initial_logp_tolerance']
    assert result['memory']['current_mps_allocated_bytes'] > 0
    if backend == 'split':
        assert row['stage_seconds']['mirror_sync_before_rollout'] > 0
        assert row['stage_seconds']['mirror_sync_before_validation'] > 0
