"""M2 training is bounded, provenance-bound, and keeps comparison budgets apart."""
from dataclasses import asdict
import importlib
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def trainer():
    assert importlib.util.find_spec('integrations.ppo_m2_train') is not None, 'bounded M2 trainer is missing'
    return importlib.import_module('integrations.ppo_m2_train')


def tiny(**changes):
    arguments = dict(name='contract-smoke', seed=20, kind='recurrent', hidden=64,
                     batch=4, threads=1, backend='cpu', selector_interactions=20,
                     joint_interactions=24, joint_hard_interactions=24,
                     wall_snapshot_seconds=0., hard_wall_seconds=20.,
                     eval_every_interactions=12, train_pool_size=8,
                     tuning_count=8, smoke=True)
    arguments.update(changes)
    return trainer().TrainConfig(**arguments)


def local_repository(monkeypatch, destination):
    module = trainer()
    hashes = module.source_hashes()
    monkeypatch.setattr(module, 'ROOT', destination)
    monkeypatch.setattr(module, 'source_hashes', lambda: hashes)
    return module


def synthetic_registry(monkeypatch, destination):
    """Portable trusted-input fixture; no unpublished trained weights required."""
    from experimentlab.learning import ActorCritic
    model = ActorCritic(kind='recurrent', hidden=64)
    historical = destination / 'historical.pt'
    torch.save({'architecture': model.architecture, 'model': model.state_dict(),
                'counters': {'transitions': 0}, 'provenance': {'seed': 20, 'scope': 'test_fixture'}}, historical)
    sha = hashlib.sha256(historical.read_bytes()).hexdigest()
    common = {'path': str(historical), 'sha256': sha, 'seed': 20}
    registry = destination / 'registry.json'
    registry.write_text(json.dumps({'registry': {
        'selected_ppo-seed20': {**common, 'calibration': {'duration_penalty': 0., 'dose_penalty': 16., 'stop_bonus': 0.}},
        'imitation_ppo-seed20': {**common, 'calibration': {'duration_penalty': 0., 'dose_penalty': 0., 'stop_bonus': 0.}}}}))
    module = local_repository(monkeypatch, destination)
    monkeypatch.setattr(module, 'REGISTRY_PATH', registry)
    return module, historical


@pytest.mark.parametrize('changes', [
    {'batch': 0}, {'batch': True}, {'hidden': 32}, {'hidden': 1024}, {'threads': 0},
    {'kind': 'latest'}, {'backend': 'cuda'}, {'initialization': 'other'},
    {'seed': -1}, {'name': '../escape'}, {'selector_interactions': -1},
    {'joint_interactions': 4}, {'joint_hard_interactions': 23},
    {'hard_wall_seconds': 0}, {'hard_wall_seconds': float('nan')},
    {'hard_wall_seconds': 3601}, {'wall_snapshot_seconds': -1},
    {'wall_snapshot_seconds': float('inf')}, {'joint_lr': 0},
    {'joint_epochs': 0}, {'entropy_start': float('nan')},
    {'entropy_end': -1}, {'eval_every_interactions': 0},
    {'smoke': False}, {'tuning_count': 0}, {'train_pool_size': 0},
    {'initialization': 'v6'},
])
def test_configuration_rejects_unbounded_or_ambiguous_inputs(changes):
    with pytest.raises(ValueError):
        tiny(**changes)


def test_default_pools_and_initialization_specific_selector_budget():
    m = trainer()
    assert m.TrainConfig().selector_interactions == 1_000_000
    warm = m.TrainConfig(initialization='v6')
    assert warm.selector_interactions == 0
    assert warm.train_pool_size == 4096 and warm.tuning_count == 128


@pytest.mark.parametrize('kind', ['history', 'recurrent'])
def test_complete_cpu_run_accounts_masks_snapshots_and_native_proof(tmp_path, kind):
    m = trainer()
    destination = tmp_path / kind
    result = m.train(tiny(kind=kind), destination)
    assert result['status'] == 'completed'
    assert result['scope'] == 'smoke_only'
    assert result['stop_reason'] == 'requested_snapshots_captured'
    totals = result['totals']
    assert 0 <= 20 - totals['selector_transitions'] < 5
    assert 0 <= 24 - totals['joint_transitions'] < 5
    assert totals['training_transitions'] == totals['selector_transitions'] + totals['joint_transitions']
    assert totals['training_policy_decisions'] < totals['training_transitions']
    assert totals['all_transitions'] == totals['training_transitions'] + totals['validation_transitions']
    assert totals['validation_policy_decisions'] <= totals['validation_transitions']
    assert totals['measured_stage_seconds'] <= totals['wall_seconds'] + 1e-6
    snapshots = result['snapshots']
    assert snapshots['equal_wall'] is None
    assert snapshots['equal_interaction']['phase'] == 'joint'
    assert 0 <= 24 - snapshots['equal_interaction']['joint_transitions'] < 5
    assert result['best_primary']['joint_transitions'] <= 24
    assert result['best_primary']['joint_transitions'] >= 12
    assert result['best_primary']['training_biases_zero']
    assert result['selector_quality']['learnability_established'] is False
    assert result['correctness']['cache_reference']['passed']
    assert result['correctness']['cache_reference']['cases'] == 8
    assert result['correctness']['source_hashes_unchanged']
    assert result['correctness']['max_initial_logp_error'] <= 5e-5
    assert result['resume_supported'] is False
    curves = [json.loads(line) for line in (destination / 'learning_curve.jsonl').read_text().splitlines()]
    updates = [row for row in curves if row['event'] == 'update']
    receipts = [row['checkpoint'] for row in curves if row['event'] == 'checkpoint']
    assert receipts and len({row['path'] for row in receipts}) == len(receipts)
    for record in receipts:
        assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest() == record['sha256']
    assert updates and all(row['old_logp_source'] == 'cpu_rollout' for row in updates)
    for row in updates:
        assert row['rollout']['transitions'] <= 5 * row['rollout']['episodes']
        assert row['update']['valid_decisions'] == row['rollout']['transitions']
        assert row['update']['policy_decisions'] == row['rollout']['policy_decisions']
        assert row['update']['advantage_centering'] == 0
        if row['phase'] == 'selector':
            assert row['rollout']['policy_decisions'] == row['rollout']['episodes']
    manifest = json.loads((destination / 'manifest.json').read_text())
    assert manifest['pools']['train_seed'] == 7401
    assert manifest['pools']['tuning_seed'] == 991311
    assert manifest['pools']['tuning_episode_seed'] == 991312000
    assert manifest['pools']['historical_training_pool_retained']
    assert manifest['confirmation_constructed'] is False and manifest['final_constructed'] is False
    assert set(manifest['pools']['train_hashes']).isdisjoint(manifest['pools']['tuning_hashes'])
    assert manifest['source_sha256']['src/experimentlab/ppo_v4.py']
    assert json.loads((destination / 'result.json').read_text()) == result
    for key in ('selector', 'latest', 'equal_interaction', 'best_primary'):
        state = torch.load(result['checkpoints'][key]['path'], weights_only=False, map_location='cpu')
        assert state['architecture']['kind'] == kind
        assert state['architecture']['horizon'] == 5
        assert 'phase' in state['counters'] and 'phase_episodes' in state['counters']
        assert state['torch_rng'] is not None and state['optimizer']['state']
        assert state['provenance']['calibration'] == m.ZERO_CALIBRATION
    with pytest.raises(FileExistsError):
        m.train(tiny(), destination)


def test_wall_snapshot_can_precede_selector_completion(tmp_path):
    result = trainer().train(tiny(wall_snapshot_seconds=1e-12), tmp_path / 'early-wall')
    wall = result['snapshots']['equal_wall']
    assert wall['phase'] == 'selector'
    assert wall['selector_transitions'] == 0
    assert wall['elapsed_seconds'] >= 1e-12
    assert wall['selection_eligible'] is False
    assert result['snapshots']['equal_interaction']['joint_transitions'] >= 20


def test_post_primary_training_cannot_replace_primary_selection(tmp_path):
    result = trainer().train(tiny(joint_interactions=12, joint_hard_interactions=40,
                                  wall_snapshot_seconds=1000.), tmp_path / 'hard-cap')
    assert result['status'] == 'incomplete_budget'
    assert result['stop_reason'] == 'joint_hard_interactions'
    assert 0 <= 40 - result['totals']['joint_transitions'] < 5
    assert result['totals']['joint_transitions'] > result['snapshots']['equal_interaction']['joint_transitions']
    assert result['best_primary']['joint_transitions'] <= 12
    assert result['snapshots']['equal_wall'] is None
    late = [row for row in result['evaluations'] if row['joint_transitions'] > 12]
    assert all(not row['selection_eligible'] for row in late)


def test_tiny_hard_wall_records_incomplete_without_starting_training(tmp_path):
    result = trainer().train(tiny(hard_wall_seconds=1e-12), tmp_path / 'wall-cap')
    assert result['status'] == 'incomplete_budget'
    assert result['stop_reason'] == 'hard_wall_seconds'
    assert result['totals']['training_transitions'] == 0
    assert result['snapshots']['equal_interaction'] is None


def test_fallback_and_unavailable_backend_are_rejected_before_output(tmp_path, monkeypatch):
    m = trainer()
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    with pytest.raises(RuntimeError, match='FALLBACK'):
        m.train(tiny(), tmp_path / 'fallback')
    assert not (tmp_path / 'fallback').exists()
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK')
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='MPS'):
        m.train(tiny(backend='split'), tmp_path / 'unavailable')


def test_warm_checkpoint_digest_checked_before_deserialization(tmp_path, monkeypatch):
    m = trainer()
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'not a checkpoint')
    registry = tmp_path / 'selection.json'
    registry.write_text(json.dumps({'registry': {'selected_ppo-seed20': {
        'path': str(checkpoint), 'sha256': '0' * 64, 'calibration': m.ZERO_CALIBRATION}}}))
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'REGISTRY_PATH', registry)
    def forbidden(*args, **kwargs):
        raise AssertionError('unverified checkpoint deserialized')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(RuntimeError, match='hash mismatch'):
        m.train(tiny(initialization='v6', selector_interactions=0), tmp_path / 'hash-guard')
    failure = json.loads((tmp_path / 'hash-guard' / 'failure.json').read_text())
    assert failure['error_type'] == 'RuntimeError'


def test_local_initial_record_cannot_escape_repository(tmp_path):
    record = tmp_path / 'initial.json'
    record.write_text(json.dumps({'path': str(tmp_path / 'external.pt'), 'sha256': '0' * 64}))
    with pytest.raises(ValueError, match='repository'):
        trainer().train(tiny(selector_interactions=0, initial_checkpoint_record=str(record)), tmp_path / 'escape')


def test_failed_rollout_retains_completed_physical_calls(tmp_path, monkeypatch):
    m = trainer()
    def fails_after_one_call(model, environments, device, **kwargs):
        environments[0].reset()
        environments[0].step(36)
        raise RuntimeError('injected rollout failure after one completed transition')
    monkeypatch.setattr(m, 'collect_batch', fails_after_one_call)
    with pytest.raises(RuntimeError, match='injected rollout failure'):
        m.train(tiny(), tmp_path / 'failed')
    failure = json.loads((tmp_path / 'failed' / 'failure.json').read_text())
    assert failure['counts']['training_transitions'] == 1
    assert failure['counts']['selector_transitions'] == 1
    assert failure['counts']['training_episodes'] == 1


def test_hidden_512_cpu_microcheck(tmp_path):
    result = trainer().train(tiny(kind='history', hidden=512, selector_interactions=5,
                                  joint_interactions=8, joint_hard_interactions=8), tmp_path / 'large')
    assert result['status'] == 'completed'
    assert result['correctness']['cache_reference']['passed']
    assert result['totals']['wall_seconds'] < 20


def test_local_continuation_retains_episode_offset_and_deployment_policy(tmp_path, monkeypatch):
    m = local_repository(monkeypatch, tmp_path)
    # A temporary repository boundary makes this a real local-record load
    # without writing fixtures into the user's preserved report directories.
    original = m.train(tiny(selector_interactions=0), tmp_path / 'original')
    record = tmp_path / 'initial.json'
    parent = original['checkpoints']['best_primary']
    record.write_text(json.dumps(parent))
    continued = m.train(tiny(selector_interactions=0, initial_checkpoint_record=str(record)), tmp_path / 'continuation')
    rows = [json.loads(line) for line in (tmp_path / 'continuation' / 'learning_curve.jsonl').read_text().splitlines()]
    first = next(row for row in rows if row['event'] == 'update')
    assert first['phase_episode_start'] == parent['joint_episodes']
    assert first['training_biases'] == {'behavior_strength': 0., 'initial_outcome_bias': 0.}
    assert continued['evaluations'][0]['selection_eligible']
    assert continued['initial_checkpoint']['calibration'] == m.ZERO_CALIBRATION
    assert first['namespace'] == m.CONTINUATION_NAMESPACE


@pytest.mark.parametrize('parent_kind', ['selector', 'biased_joint'])
def test_ineligible_local_parent_cannot_bypass_joint_selection_gates(tmp_path, monkeypatch, parent_kind):
    m = local_repository(monkeypatch, tmp_path)
    settings = {} if parent_kind == 'selector' else dict(selector_interactions=0, joint_interactions=5,
                                                        joint_hard_interactions=5, batch=1)
    original = m.train(tiny(**settings), tmp_path / 'original')
    parent = original['checkpoints']['selector' if parent_kind == 'selector' else 'latest']
    state = torch.load(parent['path'], map_location='cpu', weights_only=False)
    if parent_kind == 'selector':
        assert state['counters']['phase'] == 'selector'
    else:
        assert any(state['provenance']['training_biases'].values())
    record = tmp_path / 'initial.json'
    record.write_text(json.dumps(parent))
    continued = m.train(tiny(selector_interactions=0, initial_checkpoint_record=str(record)), tmp_path / 'continuation')
    assert continued['evaluations'][0]['joint_transitions'] == 0
    assert not continued['evaluations'][0]['selection_eligible']
    assert not continued['initial_checkpoint']['incumbent_eligible']
    for evaluation in continued['evaluations']:
        if evaluation['selection_eligible']:
            assert evaluation['joint_transitions'] >= .5 * continued['config']['joint_interactions']
            assert evaluation['training_biases_zero']


@pytest.mark.parametrize('defect', ['role', 'eligibility', 'phase', 'too_early', 'too_late', 'bias', 'source', 'manifest'])
def test_local_parent_exemption_requires_saved_eligibility_evidence(tmp_path, monkeypatch, defect):
    m = local_repository(monkeypatch, tmp_path)
    original = m.train(tiny(selector_interactions=0), tmp_path / 'original')
    parent = original['checkpoints']['best_primary']
    state = torch.load(parent['path'], map_location='cpu', weights_only=False)
    provenance = state['provenance']
    if defect == 'role':
        provenance['checkpoint_role'] = 'selector'
    elif defect == 'eligibility':
        provenance['primary_selection_eligible'] = False
    elif defect == 'phase':
        state['counters']['phase'] = 'selector'
    elif defect == 'too_early':
        state['counters']['joint_transitions'] = 1
    elif defect == 'too_late':
        state['counters']['joint_transitions'] = provenance['config']['joint_interactions'] + 1
    elif defect == 'bias':
        provenance['training_biases']['behavior_strength'] = 1.
    elif defect == 'source':
        provenance['source_sha256']['integrations/ppo_m2_train.py'] = '0' * 64
    elif defect == 'manifest':
        provenance['manifest_hash'] = '0' * 64
    altered = tmp_path / 'altered.pt'
    torch.save(state, altered)
    record = tmp_path / 'initial.json'
    record.write_text(json.dumps({'path': str(altered), 'sha256': hashlib.sha256(altered.read_bytes()).hexdigest()}))
    continued = m.train(tiny(selector_interactions=0, initial_checkpoint_record=str(record)), tmp_path / 'continuation')
    assert not continued['evaluations'][0]['selection_eligible']
    assert not continued['initial_checkpoint']['incumbent_eligible']


def test_historical_zero_step_winner_requires_verified_ancestor_chain(tmp_path, monkeypatch):
    m, historical = synthetic_registry(monkeypatch, tmp_path)
    warm = m.train(tiny(initialization='imitation_ppo', selector_interactions=0), tmp_path / 'warm')
    incumbent = warm['evaluations'][0]['checkpoint']
    record = tmp_path / 'initial.json'
    record.write_text(json.dumps(incumbent))
    continued = m.train(tiny(selector_interactions=0, initial_checkpoint_record=str(record)), tmp_path / 'continued')
    assert continued['evaluations'][0]['joint_transitions'] == 0
    assert continued['evaluations'][0]['selection_eligible']
    assert continued['initial_checkpoint']['incumbent_eligible']
    next_record = tmp_path / 'next.json'
    next_record.write_text(json.dumps(continued['evaluations'][0]['checkpoint']))
    child = m.train(tiny(selector_interactions=0, initial_checkpoint_record=str(next_record)), tmp_path / 'child')
    assert child['evaluations'][0]['selection_eligible']
    historical.write_bytes(b'broken ancestor checkpoint')
    broken = m.train(tiny(selector_interactions=0, initial_checkpoint_record=str(next_record)), tmp_path / 'broken')
    assert not broken['evaluations'][0]['selection_eligible']
    assert not broken['initial_checkpoint']['incumbent_eligible']


def test_v6_incumbent_and_calibration_are_retained(tmp_path, monkeypatch):
    m, _ = synthetic_registry(monkeypatch, tmp_path)
    result = m.train(tiny(initialization='v6', selector_interactions=0,
                          joint_interactions=8, joint_hard_interactions=8,
                          eval_every_interactions=8), tmp_path / 'v6')
    assert result['initial_checkpoint']['calibration']['dose_penalty'] == 16.
    assert result['correctness']['initial_checkpoint_unchanged']
    assert result['evaluations'][0]['selection_eligible']
    assert result['evaluations'][0]['joint_transitions'] == 0
    assert result['best_primary']['joint_transitions'] in (0, *range(4, 9))
    assert result['correctness']['max_initial_logp_error'] <= 5e-5
    state = torch.load(result['checkpoints']['best_primary']['path'], weights_only=False, map_location='cpu')
    assert state['provenance']['calibration']['dose_penalty'] == 16.
    assert result['inherited_pretraining']['included_in_new_training_budget'] is False


def test_historical_imitation_ppo_is_labeled_as_continuation(tmp_path, monkeypatch):
    m, _ = synthetic_registry(monkeypatch, tmp_path)
    result = m.train(tiny(initialization='imitation_ppo', selector_interactions=0,
                                  joint_interactions=5, joint_hard_interactions=5), tmp_path / 'imitation-ppo')
    initial = result['initial_checkpoint']
    assert initial['registry_key'] == 'imitation_ppo-seed20'
    assert 'historical IM->PPO' in initial['initialization_description']
    assert initial['calibration'] == trainer().ZERO_CALIBRATION
    assert result['evaluations'][0]['selection_eligible']
    assert result['correctness']['initial_checkpoint_unchanged']


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='native MPS unavailable')
@pytest.mark.parametrize('backend', ['mps', 'split'])
def test_accelerated_learning_keeps_cpu_probabilities_and_mps_adam(tmp_path, backend):
    result = trainer().train(tiny(backend=backend, selector_interactions=8,
                                  joint_interactions=8, joint_hard_interactions=8), tmp_path / backend)
    assert result['runtime']['rollout_device'] == 'cpu'
    assert result['runtime']['evaluation_device'] == 'cpu'
    assert result['runtime']['learner_device'] == 'mps'
    assert result['optimizer_devices']['parameter_devices'] == ['mps:0']
    assert result['optimizer_devices']['moment_devices'] == ['mps:0']
    assert result['correctness']['max_initial_logp_error'] <= 5e-5
    assert result['correctness']['cache_reference']['passed']


def test_native_and_cached_validation_scores_and_trajectories_match():
    m = trainer()
    from experimentlab.envs import PilotConfig
    from experimentlab.learning import ActorCritic
    from experimentlab.private_cache import PrivateDynamicsCache
    from experimentlab.tasks import continuous_family, make_task
    config = PilotConfig()
    pool = continuous_family(991311, 8)
    tasks = [make_task(mechanism, 991312000 + i, config) for i, mechanism in enumerate(pool)]
    model = ActorCritic(kind='history', hidden=64)
    cached = m.evaluate_policy(model, tasks, config, PrivateDynamicsCache(pool, config))
    native = m.evaluate_policy(model, tasks, config, None)
    for metric in ('mean_return', 'mean_mse', 'aggregate_rmse', 'mean_variable_cost', 'mean_effort', 'mean_assays'):
        assert cached[metric] == pytest.approx(native[metric], rel=0, abs=1e-12)
    assert [row['actions'] for row in cached['rows']] == [row['actions'] for row in native['rows']]


def test_cli_uses_absolute_repository_paths_and_json_configuration(tmp_path):
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(asdict(tiny())))
    destination = tmp_path / 'cli-run'
    process = subprocess.run([sys.executable, str(ROOT / 'scripts/train_ppo_m2.py'),
                              '--config', str(config_path), '--output', str(destination),
                              '--name', 'cli-override'], cwd=tmp_path, capture_output=True,
                             text=True, timeout=20, check=False)
    assert process.returncode == 0, process.stderr
    summary = json.loads(process.stdout)
    assert summary['status'] == 'completed'
    result = json.loads((destination / 'result.json').read_text())
    assert result['name'] == 'cli-override'
    assert result['correctness']['source_hashes_unchanged']
