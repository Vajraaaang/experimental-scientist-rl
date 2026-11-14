"""Publication derivatives preserve science while making provenance portable."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
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
