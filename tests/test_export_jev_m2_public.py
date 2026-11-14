"""Publication derivatives preserve science while making provenance portable."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def exporter():
    assert importlib.util.find_spec('scripts.export_jev_m2_public') is not None, 'publication exporter is missing'
    return importlib.import_module('scripts.export_jev_m2_public')


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    return path


def test_only_known_metadata_paths_change_and_payloads_and_numbers_survive():
    e = exporter()
    original = {'checkpoint': {'path': '/private/project/reports/a.pt'},
                'config': {'initial_checkpoint_record': '/private/project/reports/input.json'},
                'actions': [2, 0, 35], 'measurements': [0.12345678901234567, -1e-22],
                'request': {'question': 'X0 > 0?', 'p': 0.9999999999999999},
                'response': {'probabilities': [0.11, 0.89]}, 'timestamp': '2026-11-14T00:00:00Z'}
    before = copy.deepcopy(original)
    result, changes = e.transform_value(original, Path('/private/project'))
    assert original == before
    assert result['checkpoint']['path'] == 'reports/a.pt'
    assert result['config']['initial_checkpoint_record'] == 'reports/input.json'
    for key in ('actions', 'measurements', 'request', 'response', 'timestamp'):
        assert e.canonical(result[key]) == e.canonical(before[key])
    assert len(changes) == 2


@pytest.mark.parametrize('payload', [
    {'request': {'message': '/private/project/reports/a.json'}},
    {'response': {'url': 'https://service.test/object?access_token=private-value'}},
    {'request': {'authorization': 'Bearer actual-secret-value'}},
])
def test_private_api_payloads_fail_without_redaction(payload):
    e = exporter()
    with pytest.raises(ValueError, match='API payload'):
        e.transform_value(payload, Path('/private/project'))


def test_unrecognized_scientific_text_is_not_silently_redacted():
    e = exporter()
    with pytest.raises(ValueError, match='unrecognized metadata'):
        e.transform_value({'observation': '/private/project/raw'}, Path('/private/project'))


def journal_bytes(e, events):
    previous = '0' * 64
    rows = []
    for seq, event in enumerate(events):
        row = {'seq': seq, 'previous_sha256': previous, 'recorded_at': '2026-11-14T00:00:00Z', 'event': event}
        row['sha256'] = e.digest(row)
        previous = row['sha256']
        rows.append(e.canonical(row))
    return ('\n'.join(rows) + '\n').encode()


def test_hash_chain_rebinding_preserves_actions_payloads_and_timestamps():
    e = exporter()
    events = [{'kind': 'candidate_start', 'policy_metadata': {'path': '/private/project/reports/a.pt'}},
              {'kind': 'api_request', 'request': {'query': 'X0', 'threshold': 0.12345678901234567}},
              {'kind': 'decision', 'action': 35, 'reward': -0.0000000000000007}]
    original = journal_bytes(e, events)
    output, changes = e.transform_bytes('episodes/a.jsonl', original, Path('/private/project'))
    rows = [json.loads(line) for line in output.splitlines()]
    previous = '0' * 64
    for index, row in enumerate(rows):
        sha = row.pop('sha256')
        assert row['previous_sha256'] == previous
        assert row['seq'] == index and row['recorded_at'] == '2026-11-14T00:00:00Z'
        assert e.digest(row) == sha
        previous = sha
    assert rows[0]['event']['policy_metadata']['path'] == 'reports/a.pt'
    assert [r['event'] for r in rows[1:]] == events[1:]
    assert 'rebuild_jsonl_hash_chain' in changes
    tampered = original.replace(b'"action":35', b'"action":34')
    with pytest.raises(ValueError, match='hash chain'):
        e.transform_bytes('episodes/a.jsonl', tampered, Path('/private/project'))


@pytest.mark.parametrize('name', ['model.pt', 'model.pth', 'model.ckpt', 'cache.pyc', '.env', '.env.local',
                                  '.git/config', 'a/__pycache__/b.py', 'bundle.zip'])
def test_forbidden_files_are_never_selected(name):
    assert not exporter().allowed_file(Path(name))


def test_raw_data_and_gzip_are_allowed():
    e = exporter()
    assert e.allowed_file(Path('episodes.jsonl.gz'))
    assert e.allowed_file(Path('result.json'))


@pytest.mark.parametrize('handoff_url', [
    'https://chatgpt.com/c/private-example',
    'https://chatgpt.com/library?file_id=private-example&account_id=private-example',
])
def test_spec_redaction_removes_only_private_handoff_first_line(handoff_url):
    e = exporter()
    raw = ('Private handoff ' + handoff_url + '\n\n# Scientific specification\nalpha = 0.01234567890123456\n').encode()
    result, changes = e.transform_bytes('source_snapshot/autonomous_scientist_build_spec.md', raw, Path('/private/project'))
    assert result == b'\n# Scientific specification\nalpha = 0.01234567890123456\n'
    assert changes == ['remove_private_handoff_first_line']


def test_unrelated_existing_file_blocks_entire_export_before_writes(tmp_path):
    e = exporter()
    source, target = tmp_path / 'source', tmp_path / 'target'
    save(source, 'reports/new.json', {'x': 1})
    save(source, 'reports/conflict.json', {'x': 2})
    conflict = save(target, 'reports/conflict.json', {'keep': True})
    with pytest.raises(FileExistsError, match='unrelated'):
        e.export_paths(source, target, ['reports/new.json', 'reports/conflict.json'], apply=True)
    assert json.loads(conflict.read_text()) == {'keep': True}
    assert not (target / 'reports/new.json').exists()


def test_m1_derivative_hashes_and_explicit_owned_updates(tmp_path):
    e = exporter()
    source, target = tmp_path / 'source', tmp_path / 'target'
    run = 'reports/jev-rl-m1/fixture-04'
    spec = source / run / 'source_snapshot/autonomous_scientist_build_spec.md'
    spec.parent.mkdir(parents=True)
    spec.write_text('Handoff https://chatgpt.com/c/private-example\n# Specification\n')
    old_spec = spec.read_bytes()
    manifest = save(source, run + '/manifest.json', {'sources': {'autonomous_scientist_build_spec.md': e.sha256(old_spec)}})
    save(source, run + '/results.json', {'manifest_sha256': e.sha256(manifest.read_bytes()), 'cases': []})
    paths = sorted(str(p.relative_to(source)) for p in source.rglob('*') if p.is_file())
    receipt = e.export_paths(source, target, paths, apply=True)
    assert spec.read_bytes() == old_spec
    public_manifest = json.loads((target / run / 'manifest.json').read_text())
    public_results = json.loads((target / run / 'results.json').read_text())
    assert public_manifest['sources']['autonomous_scientist_build_spec.md'] == e.sha256((target / run / 'source_snapshot/autonomous_scientist_build_spec.md').read_bytes())
    assert public_results['manifest_sha256'] == e.sha256((target / run / 'manifest.json').read_bytes())
    assert next(r for r in receipt['files'] if r['path'].endswith('manifest.json'))['original_sha256'] != public_results['manifest_sha256']
    spec.write_text('Handoff https://chatgpt.com/c/private-example\n# Updated specification\n')
    manifest = save(source, run + '/manifest.json', {'sources': {'autonomous_scientist_build_spec.md': e.sha256(spec.read_bytes())}})
    save(source, run + '/results.json', {'manifest_sha256': e.sha256(manifest.read_bytes()), 'cases': []})
    with pytest.raises(FileExistsError, match='replace-owned'):
        e.export_paths(source, target, paths, apply=True)
    e.export_paths(source, target, paths, apply=True, replace_owned=True)
    assert 'Updated specification' in (target / run / 'source_snapshot/autonomous_scientist_build_spec.md').read_text()
    (target / run / 'source_snapshot/autonomous_scientist_build_spec.md').write_text('external edit\n')
    with pytest.raises(FileExistsError, match='modified'):
        e.export_paths(source, target, paths, apply=True, replace_owned=True)


def test_candidate_derivative_updates_journal_plan_admission_and_result_bindings(tmp_path):
    e = exporter()
    source, target = tmp_path / 'source', tmp_path / 'target'
    run = 'reports/jev-rl-m2/candidate-fixture-02'
    policy = {'path': str(source / 'reports/a.pt'), 'sha256': 'a' * 64}
    plan = {'kind': 'study-plan', 'models': {'ppo-seed20': policy}, 'registered_inputs': {}, 'sources': {}}
    p = save(source, run + '/plan.json', plan)
    manifest = {**plan, 'kind': 'study-manifest', 'plan_sha256': e.sha256(p.read_bytes()), 'admission': {
        'plan_content_sha256': e.digest(plan), 'models_sha256': e.digest(plan['models']),
        'registered_inputs_sha256': e.digest({}), 'runtime_sources_sha256': e.digest({})}}
    m = save(source, run + '/manifest.json', manifest)
    journal = source / run / 'episodes/test.jsonl'
    journal.parent.mkdir()
    journal.write_bytes(journal_bytes(e, [{'kind': 'candidate_start', 'policy_metadata': policy}, {'kind': 'candidate_end', 'result': {'reward': 0.1111111111111111}}]))
    row = {'journal': 'episodes/test.jsonl', 'journal_sha256': e.sha256(journal.read_bytes()), 'case_id': 'test', 'result': {'actions': [35], 'reward': 0.1111111111111111}}
    save(source, run + '/episodes/test.json', row)
    results = save(source, run + '/results.json', {'manifest_sha256': e.sha256(m.read_bytes()), 'cases': [row]})
    analyzer = source / 'scripts/analyze_jev_m2.py'
    analyzer.parent.mkdir()
    analyzer.write_text('# Fixture analysis source\n')
    analysis_name = 'reports/jev-rl-m2/candidate-analysis-fixture-02/summary.json'
    inputs = {name: e.sha256((source / run / name).read_bytes()) for name in ('manifest.json', 'results.json', 'plan.json', 'episodes/test.jsonl')}
    contract = {'fixture': True}
    save(source, analysis_name, {'schema': 'jev-m2-analysis-v1', 'score': 0.12345678901234567,
        'analysis_contract': contract, 'provenance': {
            'run': str(source / run), 'analysis_source': str(analyzer),
            'analysis_source_sha256': e.sha256(analyzer.read_bytes()), 'analysis_contract_sha256': e.digest(contract),
            'manifest_sha256': e.sha256(m.read_bytes()), 'results_sha256': e.sha256(results.read_bytes()),
            'input_sha256': inputs, 'study_source_sha256': {}, 'checkpoint_metadata_only': plan['models'],
            'verification': {'verified_input_sha256': {k: v for k, v in inputs.items() if '/' not in k}}}})
    paths = sorted(str(p.relative_to(source)) for p in source.rglob('*') if p.is_file())
    e.export_paths(source, target, paths, apply=True)
    public_plan = json.loads((target / run / 'plan.json').read_text())
    public_manifest = json.loads((target / run / 'manifest.json').read_text())
    public_result = json.loads((target / run / 'results.json').read_text())
    assert public_manifest['plan_sha256'] == e.sha256((target / run / 'plan.json').read_bytes())
    assert public_manifest['admission']['plan_content_sha256'] == e.digest(public_plan)
    assert public_manifest['admission']['models_sha256'] == e.digest(public_plan['models'])
    assert public_result['manifest_sha256'] == e.sha256((target / run / 'manifest.json').read_bytes())
    assert public_result['cases'][0] == json.loads((target / run / 'episodes/test.json').read_text())
    assert public_result['cases'][0]['journal_sha256'] == e.sha256((target / run / 'episodes/test.jsonl').read_bytes())
    assert public_result['cases'][0]['result'] == row['result']
    analysis = json.loads((target / analysis_name).read_text())
    assert analysis['score'] == 0.12345678901234567
    assert analysis['provenance']['run'] == run
    assert analysis['provenance']['analysis_source'] == 'scripts/analyze_jev_m2.py'
    assert analysis['provenance']['checkpoint_metadata_only'] == public_plan['models']
    for name, value in analysis['provenance']['input_sha256'].items():
        assert value == e.sha256((target / run / name).read_bytes())
    assert analysis['provenance']['verification']['verified_input_sha256']['manifest.json'] == e.sha256((target / run / 'manifest.json').read_bytes())


def test_confirmation_derivative_reseals_metadata_without_touching_raw_metrics(tmp_path):
    e = exporter()
    source, target = tmp_path / 'source', tmp_path / 'target'
    run = 'reports/jev-rl-m2/confirmation-01'
    records = {'20': {'path': str(source / 'reports/a.pt'), 'sha256': 'a' * 64}}
    freeze = {'selected_ppo': {'cohort': 'scratch_selected'}, 'cohorts': {'scratch_selected': {'records': records}},
              'selected_ppo_records_content_sha256': e.digest(records), 'sources': {}}
    freeze['seal_sha256'] = e.digest(freeze)
    f = save(source, run + '/freeze.json', freeze)
    m = save(source, run + '/run-manifest.json', {'freeze_sha256': e.sha256(f.read_bytes()), 'sources': {}})
    metrics = save(source, run + '/summary.json', {'score': 0.12345678901234567, 'episodes': 5632})
    save(source, run + '/results.json', {'run_manifest_sha256': e.sha256(m.read_bytes()), 'artifacts': {'summary.json': e.sha256(metrics.read_bytes())}})
    paths = sorted(str(p.relative_to(source)) for p in source.rglob('*') if p.is_file())
    e.export_paths(source, target, paths, apply=True)
    public_freeze = json.loads((target / run / 'freeze.json').read_text())
    assert public_freeze['seal_sha256'] == e.digest({k: v for k, v in public_freeze.items() if k != 'seal_sha256'})
    assert public_freeze['selected_ppo_records_content_sha256'] == e.digest(public_freeze['cohorts']['scratch_selected']['records'])
    assert (target / run / 'summary.json').read_bytes() == metrics.read_bytes()
    result = json.loads((target / run / 'results.json').read_text())
    assert result['run_manifest_sha256'] == e.sha256((target / run / 'run-manifest.json').read_bytes())


def test_corrupt_original_manifest_is_not_resigned_as_a_valid_derivative(tmp_path):
    e = exporter()
    source, target = tmp_path / 'source', tmp_path / 'target'
    run = 'reports/jev-rl-m1/fixture-04'
    snapshot = source / run / 'source_snapshot/science.py'
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text('RESULT = 2\n')
    manifest = save(source, run + '/manifest.json', {'sources': {'science.py': '0' * 64}})
    save(source, run + '/results.json', {'manifest_sha256': e.sha256(manifest.read_bytes()), 'cases': []})
    paths = [str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()]
    with pytest.raises(ValueError, match='original.*hash'):
        e.export_paths(source, target, paths, apply=True)
    assert not target.exists()


HOSTED = 'reports/jev-rl-m2/hosted-study-01'
HOSTED_ANALYSIS = 'reports/jev-rl-m2/hosted-analysis-01'
HOSTED_PLAN = 'reports/jev-rl-m2/hosted-plan-study-01.json'
REGISTRATION = 'docs/jev_rl_m2_analysis_registration.json'


def hosted_bundle(source):
    """Small synthetic sealed records; no real outcomes, models, or network."""
    e = exporter()
    script = source / HOSTED / 'source_snapshot/scripts/jev_m2_study.py'
    script.parent.mkdir(parents=True)
    verification = {'verified': True, 'verified_episodes': 2, 'failed_episodes': 1,
                    'not_run_episodes': 1, 'mode': 'hosted', 'hosted_calls': 0}
    script.write_text('import json\nprint(json.dumps(' + repr(verification) + '))\n')
    sources = {'scripts/jev_m2_study.py': e.sha256(script.read_bytes())}
    models = {'ppo-seed20': {'path': str(source / 'reports/model.pt'), 'sha256': 'a' * 64}}
    plan = {'kind': 'study-plan', 'schema_version': 4, 'models': models, 'registered_inputs': {},
            'sources': sources, 'cases': [{'case_id': name} for name in ('complete', 'failed', 'skipped')]}
    staged = save(source, HOSTED_PLAN, plan)
    save(source, HOSTED + '/plan.json', plan)
    manifest = {**plan, 'kind': 'study-manifest', 'mode': 'hosted', 'plan_sha256': e.sha256(staged.read_bytes()),
        'admission': {'plan_content_sha256': e.digest(plan), 'models_sha256': e.digest(models),
                      'registered_inputs_sha256': e.digest({}), 'runtime_sources_sha256': e.digest(sources)}}
    m = save(source, HOSTED + '/manifest.json', manifest)
    rows = []
    for name, status in [('complete', 'completed'), ('failed', 'failed'), ('skipped', 'not_run_gate_failure')]:
        journal = None if name == 'skipped' else 'episodes/' + name + '.jsonl'
        if journal:
            path = source / HOSTED / journal
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(journal_bytes(e, [{'kind': 'start', 'policy_metadata': models['ppo-seed20']},
                                              {'kind': 'api_request', 'request': {'threshold': 0.12345678901234567}}]))
        row = {'case_id': name, 'journal': journal, 'journal_sha256': e.sha256(path.read_bytes()) if journal else None,
               'result': {'status': status, 'score': None if name == 'skipped' else 0.12345678901234567}}
        save(source, HOSTED + '/episodes/' + name + '.json', row)
        rows.append(row)
    trace = save(source, HOSTED + '/trace-index.json', {})
    result = save(source, HOSTED + '/results.json', {'manifest_sha256': e.sha256(m.read_bytes()), 'cases': rows,
        'trace_index_sha256': e.sha256(trace.read_bytes()), 'stop_reason': 'not_run_gate_failure'})
    save(source, HOSTED + '/verification.json', verification)
    analyzer = source / 'scripts/analyze_jev_m2.py'
    analyzer.parent.mkdir()
    analyzer.write_text('# Registered synthetic analyzer\n')
    contract = {'scope': 'synthetic test only'}
    registration = {'schema': 'jev-m2-prospective-analysis-registration-v1',
        'analysis_source': 'scripts/analyze_jev_m2.py', 'analysis_source_sha256': e.sha256(analyzer.read_bytes()),
        'analysis_contract_sha256': e.digest(contract), 'contract': contract,
        'plans': [{'path': HOSTED_PLAN, 'sha256': e.sha256(staged.read_bytes())}]}
    save(source, REGISTRATION, registration)
    inputs = {name: e.sha256((source / HOSTED / name).read_bytes())
              for name in ('manifest.json', 'plan.json', 'results.json', 'trace-index.json', 'episodes/complete.jsonl', 'episodes/failed.jsonl')}
    summary = {'schema': 'jev-m2-analysis-v1', 'mode': 'hosted', 'fixture_mode': False,
        'analysis_contract': contract, 'scientific_benefit': False, 'provenance': {
            'run': str(source / HOSTED), 'analysis_source': str(analyzer),
            'analysis_source_sha256': registration['analysis_source_sha256'], 'analysis_contract_sha256': e.digest(contract),
            'manifest_sha256': e.sha256(m.read_bytes()), 'results_sha256': e.sha256(result.read_bytes()),
            'input_sha256': inputs, 'study_source_sha256': sources, 'checkpoint_metadata_only': models,
            'verification': {**verification, 'verifier_source': str(script), 'verifier_sha256': sources['scripts/jev_m2_study.py'],
                             'verified_input_sha256': {k: v for k, v in inputs.items() if '/' not in k}}}}
    save(source, HOSTED_ANALYSIS + '/summary.json', summary)
    (source / HOSTED_ANALYSIS / 'REPORT.md').write_text('Synthetic only: one completed, one failed, one skipped.\n')
    return sorted(str(p.relative_to(source)) for p in source.rglob('*') if p.is_file())


def test_optional_discovery_never_reads_an_incomplete_live_bundle(tmp_path, monkeypatch):
    e = exporter()
    for name in ('SOURCE_FILES', 'REPORT_FILES', 'REPORT_DIRS'):
        monkeypatch.setattr(e, name, ())
    live = tmp_path / HOSTED / 'episodes/live.jsonl'
    live.parent.mkdir(parents=True)
    live.write_bytes(b'not a complete journal\xff')
    assert e.default_paths(tmp_path) == []
    with pytest.raises(ValueError, match='incomplete hosted'):
        e.export_paths(tmp_path, tmp_path.parent / 'publication', [str(live.relative_to(tmp_path))])


def test_sealed_hosted_binding_survives_public_path_rebinding(tmp_path, monkeypatch):
    e = exporter()
    source, target = tmp_path / 'source', tmp_path / 'public'
    paths = hosted_bundle(source)
    for name in ('SOURCE_FILES', 'REPORT_FILES', 'REPORT_DIRS'):
        monkeypatch.setattr(e, name, ())
    assert set(paths) <= set(e.default_paths(source))
    receipt = e.export_paths(source, target, paths, apply=True)
    original_hash = e.sha256((source / HOSTED_PLAN).read_bytes())
    public_hash = e.sha256((target / HOSTED_PLAN).read_bytes())
    assert public_hash != original_hash
    assert public_hash == e.sha256((target / HOSTED / 'plan.json').read_bytes())
    manifest = json.loads((target / HOSTED / 'manifest.json').read_text())
    plan = json.loads((target / HOSTED / 'plan.json').read_text())
    assert manifest['plan_sha256'] == public_hash
    assert manifest['admission']['plan_content_sha256'] == e.digest(plan)
    assert json.loads((target / REGISTRATION).read_text())['plans'][0]['sha256'] == original_hash
    mapping = next(row for row in receipt['files'] if row['path'] == HOSTED_PLAN)
    assert (mapping['original_sha256'], mapping['public_sha256']) == (original_hash, public_hash)
    summary = json.loads((target / HOSTED_ANALYSIS / 'summary.json').read_text())
    assert summary['provenance']['manifest_sha256'] == e.sha256((target / HOSTED / 'manifest.json').read_bytes())
    assert summary['provenance']['checkpoint_metadata_only'] == plan['models']
    assert not summary['scientific_benefit']
    assert json.loads((source / HOSTED / 'results.json').read_text())['cases'][2]['result'] == json.loads((target / HOSTED / 'results.json').read_text())['cases'][2]['result']


@pytest.mark.parametrize('corruption', ['partial_analysis', 'journal', 'staged_plan', 'registered_contract', 'registered_analyzer', 'unverified_seal'])
def test_corrupted_or_partial_hosted_pair_cannot_be_exported(tmp_path, corruption):
    e = exporter()
    source, target = tmp_path / 'source', tmp_path / 'public'
    paths = hosted_bundle(source)
    if corruption == 'partial_analysis':
        paths.remove(HOSTED_ANALYSIS + '/REPORT.md')
    elif corruption == 'journal':
        (source / HOSTED / 'episodes/failed.jsonl').write_text('{}\n')
    elif corruption == 'staged_plan':
        save(source, HOSTED_PLAN, {'unregistered': True})
    elif corruption == 'unverified_seal':
        save(source, HOSTED + '/verification.json', {'verified': False})
    else:
        registration = json.loads((source / REGISTRATION).read_text())
        registration['analysis_contract_sha256' if corruption == 'registered_contract' else 'analysis_source_sha256'] = 'f' * 64
        save(source, REGISTRATION, registration)
    with pytest.raises(ValueError):
        e.export_paths(source, target, paths, apply=True)
    assert not target.exists()


def minimal_publication(target):
    e = exporter()
    records = []
    for run, script, count in [('reports/jev-rl-m1/fixture-04', 'jev_rl_study.py', 100),
                                ('reports/jev-rl-m1/hosted-20261113-204144', 'jev_rl_study.py', 15),
                                ('reports/jev-rl-m2/candidate-fixture-02', 'jev_m2_study.py', 75),
                                ('reports/jev-rl-m2/confirmation-01', 'ppo_m2_confirm.py', 5632)]:
        name = run + '/source_snapshot/scripts/' + script
        path = target / name
        path.parent.mkdir(parents=True)
        path.write_text('print(' + repr(json.dumps({'verified_episodes': count})) + ')\n')
        records.append({'path': name, 'original_sha256': e.sha256(path.read_bytes()), 'public_sha256': e.sha256(path.read_bytes())})
    save(target, e.RECEIPT, {'schema': e.SCHEMA, 'files': records})
    return records


def test_verify_published_cli_checks_hashes_without_rewriting_or_discovering_unlisted_hosted_data(tmp_path):
    e = exporter()
    records = minimal_publication(tmp_path)
    trap = tmp_path / HOSTED / 'source_snapshot/scripts/jev_m2_study.py'
    trap.parent.mkdir(parents=True)
    trap.write_text('raise AssertionError("unlisted hosted data was executed")\n')
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    command = [sys.executable, '-B', e.__file__, '--verify-published', '--destination', str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    checked = json.loads(result.stdout)
    assert checked['status'] == 'passed' and checked['files_verified'] == len(records)
    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before
    (tmp_path / records[0]['path']).write_text('raise AssertionError("must not execute corrupt source")\n')
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'hash' in result.stdout.lower()


def published_hosted_bundle(tmp_path):
    e = exporter()
    source, target = tmp_path / 'source', tmp_path / 'public'
    receipt = e.export_paths(source, target, hosted_bundle(source), apply=True)
    base_records = minimal_publication(target)
    receipt['files'].extend(base_records)
    save(target, e.RECEIPT, receipt)
    return target


@pytest.mark.parametrize('analysis_mismatch', [False, True])
def test_published_hosted_replay_preserves_failed_skipped_outcomes_and_reproduces_analysis(tmp_path, monkeypatch, analysis_mismatch):
    e = exporter()
    target = published_hosted_bundle(tmp_path)
    before = {str(p.relative_to(target)): p.read_bytes() for p in target.rglob('*') if p.is_file()}
    real_run = subprocess.run
    invocations = []

    def analyze_fixture(argv, **kwargs):
        invocations.append(argv)
        if Path(argv[2]).name != 'analyze_jev_m2.py':
            return real_run(argv, **kwargs)
        # Stub only the registered analysis boundary; snapshot subprocesses run.
        output = Path(argv[argv.index('--output') + 1])
        output.mkdir()
        summary = json.loads((target / HOSTED_ANALYSIS / 'summary.json').read_text())
        if analysis_mismatch:
            summary['scientific_benefit'] = True
        save(output, 'summary.json', summary)
        (output / 'REPORT.md').write_bytes((target / HOSTED_ANALYSIS / 'REPORT.md').read_bytes())
        return subprocess.CompletedProcess(argv, 0, '{}', '')

    monkeypatch.setattr(e.subprocess, 'run', analyze_fixture)
    verified = e.verify_published(target)
    assert verified['hosted_pair_included']
    assert verified['status'] == ('failed' if analysis_mismatch else 'passed')
    hosted = next(run for run in verified['runs'] if run['name'] == 'm2_hosted')
    assert hosted['passed'] and hosted['observed_episodes'] == 2
    assert hosted['scientific_outcomes']['status_counts'] == {'completed': 1, 'failed': 1, 'not_run_gate_failure': 1}
    assert next(run for run in verified['runs'] if run['name'] == 'm2_hosted_analysis')['passed'] is not analysis_mismatch
    assert all('run' not in argv[3:4] for argv in invocations)
    assert {str(p.relative_to(target)): p.read_bytes() for p in target.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('corruption', ['original_plan_hash', 'partial_receipt'])
def test_public_hosted_registration_and_pair_are_checked_before_execution(tmp_path, monkeypatch, corruption):
    e = exporter()
    target = published_hosted_bundle(tmp_path)
    receipt = json.loads((target / e.RECEIPT).read_text())
    if corruption == 'original_plan_hash':
        next(row for row in receipt['files'] if row['path'] == HOSTED_PLAN)['original_sha256'] = 'f' * 64
    else:
        receipt['files'] = [row for row in receipt['files'] if row['path'] != HOSTED_ANALYSIS + '/REPORT.md']
    save(target, e.RECEIPT, receipt)

    def must_not_run(*args, **kwargs):
        pytest.fail('corrupt public binding reached subprocess execution')

    monkeypatch.setattr(e.subprocess, 'run', must_not_run)
    with pytest.raises(ValueError, match='registered plan|incomplete hosted'):
        e.verify_published(target)


@pytest.mark.parametrize('write_option', ['--apply', '--replace-owned', '--verify'])
def test_read_only_cli_rejects_export_options(tmp_path, write_option):
    e = exporter()
    result = subprocess.run([sys.executable, '-B', e.__file__, '--verify-published', '--destination', str(tmp_path), write_option],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'cannot be combined' in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_failed_verifier_diagnostics_do_not_echo_response_content(tmp_path):
    e = exporter()
    records = minimal_publication(tmp_path)
    message = 'PRIVATE_RESPONSE_SENTINEL'
    path = tmp_path / records[0]['path']
    path.write_text('import sys\nprint(' + repr(message) + ')\nprint(' + repr(message) + ', file=sys.stderr)\nraise SystemExit(1)\n')
    records[0]['original_sha256'] = records[0]['public_sha256'] = e.sha256(path.read_bytes())
    save(tmp_path, e.RECEIPT, {'schema': e.SCHEMA, 'files': records})
    result = e.verify_published(tmp_path)
    assert result['status'] == 'failed'
    assert message not in json.dumps(result)
    assert result['runs'][0]['diagnostics']['stdout_sha256'] == e.sha256((message + '\n').encode())
