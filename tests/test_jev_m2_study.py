"""Small offline checks for frozen M2 plans, admission, and replay bindings."""
import copy
from functools import partial
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def study():
    assert importlib.util.find_spec('scripts.jev_m2_study') is not None, 'M2 study runner is missing'
    return importlib.import_module('scripts.jev_m2_study')


def synthetic_registry(destination):
    """Real local input artifacts for admission tests; no historical weights."""
    s = study()
    destination.mkdir(parents=True)
    for name in s.RUNTIME_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s.ROOT / name, target)
    (destination / s.SPEC).write_text('Synthetic specification fixture.\n')
    calibration = {'duration_penalty': 0., 'dose_penalty': 0., 'stop_bonus': 0.}
    registry = {}
    for seed in (20, 21):
        with s.torch.random.fork_rng(devices=[]):
            s.torch.manual_seed(seed)
            model = s.ActorCritic(kind='recurrent', hidden=64)
        checkpoint = destination / f'reports/synthetic/seed{seed}.pt'
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        s.torch.save({'architecture': model.architecture, 'model': model.state_dict(),
                      'provenance': {'seed': seed, 'scope': 'synthetic_test_fixture'}}, checkpoint)
        for index, historical in enumerate(s.NEURAL.values()):
            registry[f'{historical}-seed{seed}'] = {
                'path': str(checkpoint), 'sha256': s.file_hash(checkpoint), 'seed': seed,
                'calibration': {**calibration, 'dose_penalty': float(index)}}
    for name, value in (
        (s.REGISTRY_SOURCE, {'registry': registry, 'scope': 'synthetic_test_fixture'}),
        (s.SCHEDULE_SOURCE, {'fixed': [4, 12, 31], 'fixed_quality': [4, 14, 31],
                             'additive_schedule': [11, 35], 'scope': 'synthetic_test_fixture'}),
    ):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        s.write_new(path, value)
    with s.torch.random.fork_rng(devices=[]):
        s.torch.manual_seed(120)
        model = s.ActorCritic(kind='recurrent', hidden=128)
    selected = destination / 'reports/synthetic/m2-selected.pt'
    s.torch.save({'architecture': model.architecture, 'model': model.state_dict(),
        'synthetic_test_fixture': True,
        'provenance': {'seed': 20, 'trainer_schema': 'm2-ppo-train-v1',
            'scope': 'm2_development_only', 'calibration': calibration,
            'config': {'seed': 20, 'hidden': 128, 'kind': 'recurrent'}}}, selected)
    override = {'seed': 20, 'path': str(selected), 'sha256': s.file_hash(selected),
                'calibration': calibration}
    return s, destination, override


def test_exact_case_counts_and_staged_call_bound():
    s = study()
    for mechanisms, seeds, calls, count in [(1, [20], 79, 23),
                                           (1, list(range(20, 25)), 347, 75),
                                           (8, list(range(20, 25)), 2776, 600)]:
        cases = s.case_plan(mechanisms, seeds)
        assert len(cases) == count
        assert len({c['case_id'] for c in cases}) == count
        assert sum(c['max_calls'] for c in cases) == calls
        pilot = [c for c in cases if c['stage'] == 'pilot']
        assert len(pilot) == 23
        assert {c['task_index'] for c in pilot} == {0}
        assert {c['training_seed'] for c in pilot} == {None, 20}
        assert sum(c['max_calls'] for c in pilot) == 79
        for method in ('jev_only', 'probe'):
            assert sum(c['method'] == method for c in cases) == mechanisms


def test_plan_caps_are_finite_and_fixture_limits_are_explicit():
    s = study()
    assert s.planned_limits(32, .10)['max_input_tokens'] == int(.10 / .042 * 1_000_000)
    limits = s.planned_limits(2800, 1.)
    assert limits['max_request_bytes'] == 65536
    assert limits['max_total_request_bytes'] <= 200 * 1024 * 1024
    for calls, usd in [(2801, 1.), (80, 1.01), (0, .1), (80, float('nan'))]:
        with pytest.raises(ValueError):
            s.planned_limits(calls, usd)
    for seeds in ([21], [20, 21], [20, 20], [19]):
        with pytest.raises(ValueError):
            s.case_plan(1, seeds)
    with pytest.raises(ValueError):
        s.case_plan(9, [20])


def test_new_tasks_and_disjointness_never_construct_confirmation(monkeypatch):
    s = study()
    seen = []
    original = s.continuous_family
    monkeypatch.setattr(s, 'continuous_family', lambda seed, n: (seen.append(seed), original(seed, n))[1])
    tasks = s.development_tasks(8, s.PilotConfig(final_cultures=128))
    hashes = [s.mechanism_hash(t.mechanism) for t in tasks]
    assert seen == [991331]
    assert [t.seed for t in tasks] == list(range(991332000, 991332008))
    s.assert_disjoint(hashes, {'0' * 64})
    with pytest.raises(ValueError, match='duplicate'):
        s.assert_disjoint(hashes + [hashes[0]], set())
    with pytest.raises(ValueError, match='overlap'):
        s.assert_disjoint(hashes, {hashes[0]})


def test_sources_are_runtime_only_and_exclude_original_spec():
    s = study()
    sources = s.source_hashes(s.ROOT)
    assert 'scripts/jev_m2_study.py' in sources
    assert 'integrations/jev_m2_episode.py' in sources
    assert 'autonomous_scientist_build_spec.md' not in sources
    assert not any(p.startswith(('reports/', 'tests/')) for p in sources)


def test_model_override_is_hash_verified_and_cannot_escape_root(tmp_path):
    s = study()
    from experimentlab.learning import ActorCritic
    path = tmp_path / 'reports/m2/selected.pt'
    path.parent.mkdir(parents=True)
    model = ActorCritic(hidden=64, kind='recurrent')
    calibration = {'duration_penalty': 0., 'dose_penalty': 0., 'stop_bonus': 0.}
    state = {'model': model.state_dict(), 'architecture': model.architecture,
             'provenance': {'seed': 20, 'trainer_schema': 'm2-ppo-train-v1',
                'scope': 'm2_development_only', 'calibration': calibration,
                'config': {'seed': 20, 'hidden': 64, 'kind': 'recurrent'}}}
    s.torch.save(state, path)
    record = {'seed': 20, 'path': str(path), 'sha256': s.file_hash(path), 'calibration': calibration}
    assert s.normalize_ppo_records(tmp_path, {'20': record}, [20])['20']['path'] == 'reports/m2/selected.pt'
    without_seed = {key: value for key, value in record.items() if key != 'seed'}
    assert s.normalize_ppo_records(tmp_path, {'20': without_seed}, [20])['20']['seed'] == 20
    with pytest.raises(ValueError, match='hash'):
        s.normalize_ppo_records(tmp_path, {'20': {**record, 'sha256': '0' * 64}}, [20])
    with pytest.raises(ValueError, match='repository'):
        s.normalize_ppo_records(tmp_path, {'20': {**record, 'path': str(tmp_path.parent / 'outside.pt')}}, [20])
    with pytest.raises(ValueError, match='provenance'):
        s.normalize_ppo_records(tmp_path, {'21': {**record, 'seed': 21}}, [21])
    with pytest.raises(ValueError, match='calibration'):
        s.normalize_ppo_records(tmp_path, {'20': {**record, 'calibration': {**calibration, 'stop_bonus': 9.}}}, [20])
    state['architecture']['hidden'] = 128
    s.torch.save(state, path)
    with pytest.raises(ValueError, match='architecture'):
        s.normalize_ppo_records(tmp_path, {'20': {**record, 'sha256': s.file_hash(path)}}, [20])


def test_write_and_existing_output_reject_before_loading_plan(tmp_path):
    s = study()
    path = tmp_path / 'plan.json'
    s.write_new(path, {'first': True})
    with pytest.raises(FileExistsError):
        s.write_new(path, {})
    output = tmp_path / 'existing'
    output.mkdir()
    with pytest.raises(FileExistsError):
        s.run_study(tmp_path / 'missing.json', output, live=True)


def test_live_incomplete_plan_fails_before_credential_lookup(tmp_path, monkeypatch):
    s = study()
    plan = {'worst_case_api_calls': 79, 'planned_api_limits': s.planned_limits(32, .10)}
    monkeypatch.setattr(s, 'read_plan', lambda *args, **kwargs: plan)
    monkeypatch.setattr(s, 'live_key', lambda: pytest.fail('credential lookup before cap rejection'))
    with pytest.raises(ValueError, match='79'):
        s.run_study(tmp_path / 'unused.json', tmp_path / 'new', live=True)
    assert not (tmp_path / 'new').exists()


def test_checkpoint_free_case_replay_rejects_metadata_relabel(tmp_path):
    s = study()
    from experimentlab.inference import PredictionBank
    from experimentlab.tasks import finite_family, make_task
    from integrations.jev_research import Journal
    from integrations.jev_m2_episode import run_candidate
    config = s.PilotConfig(final_cultures=128)
    task = make_task(finite_family()[0], 991332000, config)
    bank = PredictionBank(finite_family(), config)
    case = s.case_plan(1, [20])[8]
    assert case['method'] == 'ppo'
    record = {'seed': 20, 'sha256': 'a' * 64, 'path': 'reports/missing.pt', 'calibration': {}}
    class CommitPolicy:
        metadata = record
        def probabilities(self, obs):
            return [float(i == 36) for i in range(45)]
    manifest = {'model': 'jev-fixture', 'trace_id': 'b' * 32,
                'models': {'ppo-seed20': record}, 'schedules': {'probe': [4, 12, 31]}}
    directory = tmp_path / 'episodes'
    directory.mkdir()
    path = directory / (case['case_id'] + '.jsonl')
    with Journal(path) as journal:
        result = run_candidate(s.ScientistEnv(task, config), 'ppo', bank, journal,
                               policy=CommitPolicy(), schedule=())
    row = {'case_id': case['case_id'], 'method': case['method'], 'journal': str(path.relative_to(tmp_path)),
           'journal_sha256': s.file_hash(path), 'result': result, 'api_receipts': []}
    assert s.verify_case(tmp_path, manifest, case, row, task, config, bank) == result
    changed = copy.deepcopy(manifest)
    changed['models']['ppo-seed20']['sha256'] = 'c' * 64
    with pytest.raises(ValueError, match='binding'):
        s.verify_case(tmp_path, changed, case, row, task, config, bank)
    row['result']['method'] = 'forecast_only'
    with pytest.raises(ValueError, match='result'):
        s.verify_case(tmp_path, manifest, case, row, task, config, bank)


def test_case_replay_rejects_removed_role_label(tmp_path):
    s = study()
    case = s.case_plan(1, [20])[0]
    row = s.not_run_row(case, 'not_run_wall')
    del row['method']
    with pytest.raises(ValueError, match='label'):
        s.verify_case(tmp_path, {}, case, row, None, None, None)


def test_deadline_guard_is_local_and_precedes_transport():
    s = study()
    manifest = {'wall_cap_seconds': 1, 'sources': {}, 'models': {}}
    boundary = s.DispatchBoundary(manifest, 0, clock=lambda: 2)
    boundary.case_id = 'case'
    client = s.BoundaryClient(lambda request: pytest.fail('network after deadline'),
                              boundary=boundary, expected_model='jev-fixture')
    events = []
    client.event_sink = events.append
    with pytest.raises(s.JevError, match='event_sink_error'):
        client({'model': 'jev-fixture', 'state': {}, 'questions': {
            'forecast': {'type': 'noul', 'instructions': 'Test?'}}})
    with pytest.raises(s.JevError, match='event_sink_error'):
        client({'model': 'jev-fixture', 'state': {}, 'questions': {
            'forecast': {'type': 'noul', 'instructions': 'Test?'}}})
    assert client.stats()['attempted_calls'] == 0
    assert boundary.records[0]['origin'] == 'local_dispatch_guard'
    assert boundary.records[0]['reason'] == 'not_run_wall'
    assert len(boundary.records) == 2
    assert [e['kind'] for e in events] == ['api_failure', 'api_failure']
    s.verify_client_stats(client.stats(), client.receipts, s.asdict(s.CallLimits()))


def test_dispatch_guard_rechecks_clock_after_source_checks():
    s = study()
    ticks = iter([0, 2])
    boundary = s.DispatchBoundary({'wall_cap_seconds': 1, 'sources': {}, 'models': {}},
                                  0, clock=lambda: next(ticks))
    assert boundary.check() == 'not_run_wall'


def test_plan_rejects_relabelled_cases_and_unsafe_runtime_cap(monkeypatch):
    s = study()
    monkeypatch.setattr(s, 'exclusions', lambda root: {'hashes': []})
    monkeypatch.setattr(s, 'model_records', lambda root, seeds: {
        f'{method}-seed{seed}': {'path': f'reports/unavailable/{method}-{seed}.pt',
            'seed': seed, 'sha256': 'a' * 64, 'calibration': {}}
        for method in s.NEURAL for seed in seeds})
    plan = s.prepare_plan()
    assert plan['complete_call_allowance'] is False
    changed = copy.deepcopy(plan)
    changed['cases'][0]['method'] = 'ppo'
    with pytest.raises(ValueError, match='relabelled'):
        s.validate_plan(changed)
    changed = copy.deepcopy(plan)
    changed['planned_api_limits']['max_input_tokens'] += 1
    with pytest.raises(ValueError, match='limits'):
        s.validate_plan(changed)


def test_saved_snapshot_verifies_without_any_checkpoint_files(tmp_path, monkeypatch):
    s = study()
    monkeypatch.setattr(s, 'exclusions', lambda root: {'hashes': []})
    monkeypatch.setattr(s, 'model_records', lambda root, seeds: {
        f'{method}-seed{seed}': {'path': f'reports/unavailable/{method}-{seed}.pt',
            'seed': seed, 'sha256': 'a' * 64, 'calibration': {}}
        for method in s.NEURAL for seed in seeds})
    plan = s.prepare_plan(wall_seconds=.01)
    s.write_new(tmp_path / 'plan.json', plan)
    limits = s.fixture_limits(plan['worst_case_api_calls'])
    manifest = {**plan, 'kind': 'study-manifest', 'mode': 'fixture', 'model': 'jev-fixture',
        'response_provider': 'jev-fixture', 'api_limits': limits, 'fixture_caps_only': True,
        'plan_sha256': s.file_hash(tmp_path / 'plan.json'),
        'admission': s.admission_receipt(plan, plan['exclusions'])}
    s.write_new(tmp_path / 'manifest.json', manifest)
    (tmp_path / 'episodes').mkdir()
    rows = [s.not_run_row(case, 'not_run_wall') for case in plan['cases']]
    for row in rows:
        s.write_new(tmp_path / 'episodes' / (row['case_id'] + '.json'), row)
    for name in plan['sources']:
        destination = tmp_path / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s.ROOT / name, destination)
    s.write_new(tmp_path / 'trace-index.json', s.trace_index(s._labelled(rows), 'jev-fixture'))
    stats = s._stats([], limits)
    gate = s.expansion_gate(rows, plan['staged_scope']['pilot_case_ids'], [], stats, limits)
    result = {'manifest_sha256': s.file_hash(tmp_path / 'manifest.json'), 'cases': rows,
        'summary': s.study_summary(rows, 'fixture'), 'client_stats': stats, 'pilot_gate': gate,
        'stop_reason': 'not_run_wall', 'dispatch_guards': [], 'elapsed_seconds': .02,
        'trace_index_sha256': s.file_hash(tmp_path / 'trace-index.json')}
    s.write_new(tmp_path / 'results.json', result)
    assert s.verify_run(tmp_path)['not_run_episodes'] == 23
    command = [sys.executable, str(tmp_path / 'source_snapshot/scripts/jev_m2_study.py'),
               'verify', '--run', str(tmp_path)]
    replay = subprocess.run(command, capture_output=True, text=True, timeout=7)
    assert replay.returncode == 0, replay.stderr
    assert json.loads(replay.stdout)['checkpoint_free_replay'] is True
    result['cases'].pop()
    (tmp_path / 'results.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='cases'):
        s.verify_run(tmp_path)


def test_summary_preserves_not_run_and_missing_forecast_denominators():
    s = study()
    rows = [s.not_run_row({'case_id': 'x', 'method': 'probe'}, 'not_run_wall')]
    summary = s.study_summary(rows, 'fixture')
    assert summary['jev_efficacy_established'] is False
    assert summary['methods']['probe']['planned'] == 1
    assert summary['methods']['probe']['not_run'] == 1
    assert summary['methods']['probe']['attempted'] == 0
    assert summary['methods']['probe']['api']['hosted_calls'] == 0


def test_fixture_reuse_is_disclosed_without_weakening_other_exclusions(tmp_path, monkeypatch):
    s = study()
    task = s.development_tasks(1, s.PilotConfig(final_cultures=128))[0]
    fixture_hash = s.mechanism_hash(task.mechanism)
    historical, hosted, final = ('a' * 64, 'b' * 64, 'c' * 64)
    monkeypatch.setattr(s, 'previous_hashes', lambda root: ({historical}, {}))
    def manifest(name, value):
        path = tmp_path / 'reports/jev-rl-m2' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    manifest('fixture/manifest.json', {'mode': 'fixture', 'tasks': [
        {'mechanism_hash': value} for value in (fixture_hash, historical, hosted)],
        'exclusions': {'hashes': [historical]},
        'prior_fixture_overlap': {'mechanism_hashes': [fixture_hash]}})
    manifest('hosted/manifest.json', {'mode': 'hosted', 'tasks': [{'mechanism_hash': hosted}]})
    manifest('final/freeze.json', {'mechanism_hashes': [final],
        'historical_identifiers': [fixture_hash, historical, hosted]})
    excluded = s.exclusions(tmp_path)
    assert fixture_hash not in excluded['hashes']
    assert {historical, hosted, final} <= set(excluded['hashes'])
    for seed in (7401, 991311):
        assert s.mechanism_hash(s.continuous_family(seed, 1)[0]) in excluded['hashes']
    assert fixture_hash in excluded['prior_fixture_overlap']['mechanism_hashes']
    assert excluded['prior_fixture_overlap']['scope'] == 'disclosed reuse of development cases for fixture-to-hosted engineering validation; not an independent test'
    monkeypatch.setattr(s, 'exclusions', lambda root: excluded)
    monkeypatch.setattr(s, 'model_records', lambda root, seeds: {
        f'{method}-seed{seed}': {'path': f'reports/unavailable/{method}-{seed}.pt',
            'seed': seed, 'sha256': 'd' * 64, 'calibration': {}}
        for method in s.NEURAL for seed in seeds})
    plan = s.prepare_plan()
    assert plan['prior_fixture_overlap']['reused_mechanism_hashes'] == [fixture_hash]
    assert plan['prior_fixture_overlap']['reused_count'] == 1


def test_summary_uses_reset_inclusive_latency_without_changing_raw_online_time():
    s = study()
    common = {'status': 'completed', 'failure': None, 'forecasts': [], 'final_mse': .01,
              'variable_cost': 2., 'total_assay_cost': 5., 'return_': -1., 'effort': .4,
              'assays': 1, 'immediate_stop': False}
    rows = [
        {'case_id': 'm1', 'method': 'incumbent_m1', 'api_receipts': [],
         'result': {**common, 'method': 'jev_rl', 'online_seconds': 1., 'end_to_end_seconds': 1.5}},
        {'case_id': 'm2', 'method': 'forecast_only', 'api_receipts': [],
         'result': {**common, 'method': 'forecast_only', 'online_seconds': 2.,
                    'timing': {'end_to_end_seconds': 2.5, 'reference_seconds': .5}}},
    ]
    original = copy.deepcopy(rows)
    labelled = s._labelled(rows)
    assert labelled[1]['result']['end_to_end_seconds'] == 2.5
    summary = s.study_summary(rows, 'fixture')['methods']
    for method, reset_inclusive, online in [('incumbent_m1', 1.5, 1.), ('forecast_only', 2.5, 2.)]:
        assert summary[method]['mean_end_to_end_seconds'] == reset_inclusive
        assert summary[method]['attempt_metrics']['end_to_end_seconds']['mean'] == reset_inclusive
        assert summary[method]['mean_online_seconds'] == online
    assert rows == original


def test_m2_override_preserves_registered_incumbent_and_standalone_v6(tmp_path):
    s, root, override = synthetic_registry(tmp_path / 'repository')
    registered = s.model_records(root, [20])
    plan = s.prepare_plan(root=root, ppo_records={'20': override})
    s.validate_plan(plan, root=root, files=True)
    new = plan['models']['ppo-seed20']
    assert new['sha256'] == override['sha256'] != registered['ppo-seed20']['sha256']
    assert new['checkpoint_binding']['architecture']['hidden'] == 128
    assert 'ppo_v6-seed20' in plan['models']
    for method in ('incumbent_m1', 'ppo_v6'):
        case = next(c for c in plan['cases'] if c['method'] == method)
        assert s.policy_record(plan, case) == registered['ppo-seed20']
    for method in s.ROLES:
        if method in s.INDEPENDENT:
            continue
        case = next(c for c in plan['cases'] if c['method'] == method)
        assert s.policy_record(plan, case) == new
    local = next(c for c in plan['cases'] if c['method'] == 'ppo_v6')
    assert local['engine_method'] == 'ppo' and local['max_calls'] == 0


def test_current_overlap_rejected_before_credentials_transport_or_output(tmp_path, monkeypatch):
    s, root, _ = synthetic_registry(tmp_path / 'repository')
    plan = s.prepare_plan(root=root, max_calls=80)
    path = tmp_path / 'plan.json'
    s.write_new(path, plan)
    hosted = root / 'reports/jev-rl-m2/prior-hosted/manifest.json'
    hosted.parent.mkdir(parents=True)
    s.write_new(hosted, {'mode': 'hosted', 'tasks': plan['tasks']})
    # Redirect only the repository location; production admission remains active.
    monkeypatch.setattr(s, 'admit_plan', partial(s.admit_plan, root=root))
    monkeypatch.setattr(s, 'live_key', lambda: pytest.fail('credentials before current-overlap admission'))
    monkeypatch.setattr(s, 'HostedTransport', lambda key: pytest.fail('transport before current-overlap admission'))
    with pytest.raises(ValueError, match='overlap'):
        s.run_study(path, tmp_path / 'run', live=True)
    with pytest.raises(ValueError, match='overlap'):
        s.read_plan(path)
    assert not (tmp_path / 'run').exists()


def test_registered_model_relabel_and_preserved_schedule_tamper_are_rejected(tmp_path):
    s, root, _ = synthetic_registry(tmp_path / 'repository')
    plan = s.prepare_plan(root=root, max_calls=80)
    wrong_seed = s.model_records(root, [21])
    assert wrong_seed['ppo-seed21']['sha256'] != plan['models']['ppo-seed20']['sha256']
    for method in s.NEURAL:
        changed = copy.deepcopy(plan)
        changed['models'][f'{method}-seed20'] = {**wrong_seed[f'{method}-seed21'], 'seed': 20}
        with pytest.raises(ValueError, match='registered'):
            s.validate_plan(changed, root=root, files=True)
        changed['registered_inputs']['models'][f'{method}-seed20'] = copy.deepcopy(changed['models'][f'{method}-seed20'])
        if method == 'ppo':
            changed['models']['ppo_v6-seed20'] = copy.deepcopy(changed['models']['ppo-seed20'])
            changed['registered_inputs']['models']['ppo_v6-seed20'] = copy.deepcopy(changed['models']['ppo-seed20'])
        with pytest.raises(ValueError, match='registered'):
            s.validate_plan(changed, root=root, files=True)
    changed = copy.deepcopy(plan)
    changed['schedules']['fixed'] = changed['schedules']['probe'] = [0, 0, 0]
    with pytest.raises(ValueError, match='schedule'):
        s.validate_plan(changed, root=root, files=True)
    changed['registered_inputs']['schedules'] = copy.deepcopy(changed['schedules'])
    with pytest.raises(ValueError, match='schedule'):
        s.validate_plan(changed, root=root, files=True)


def test_real_default_admission_freezes_registered_inputs_without_inference(tmp_path):
    s, root, _ = synthetic_registry(tmp_path / 'repository')
    plan = s.prepare_plan(root=root, max_calls=80)
    assert hasattr(s, 'admit_plan'), 'current admission check is missing'
    receipt = s.admit_plan(plan, root=root)
    assert receipt['phase'] == 'before_credentials_output_and_dispatch'
    assert receipt['registered_inputs_sha256'] == s.digest(plan['registered_inputs'])
    assert receipt['task_hashes'] == [t['mechanism_hash'] for t in plan['tasks']]
    assert receipt['independent_overlap'] == []
    assert receipt['registered_checkpoint_hashes_verified'] is True
    assert receipt['checkpoint_inference_reproduced'] is False
    s.verify_admission(receipt, plan)
    for field, value in [('models_sha256', '0' * 64),
                         ('registered_inputs_sha256', '0' * 64),
                         ('independent_overlap', receipt['task_hashes'])]:
        changed = {**receipt, field: value}
        with pytest.raises(ValueError, match='admission receipt binding'):
            s.verify_admission(changed, plan)
