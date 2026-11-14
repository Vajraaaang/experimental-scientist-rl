"""Synthetic arrays and receipts only; never open study cases or construct pools."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def module():
    assert importlib.util.find_spec('scripts.analyze_jev_m2') is not None, 'M2 analysis missing'
    return importlib.import_module('scripts.analyze_jev_m2')


def study(methods=('ppo', 'arbitration', 'rerank'), count=8, seeds=(20, 21, 22, 23, 24)):
    independent = {'fixed', 'voi', 'jev_only', 'probe'}
    manifest = {'schema_version': 4, 'mode': 'hosted', 'model': 'jev-1.14.0',
                'training_seeds': list(seeds), 'tasks': [{'mechanism_hash': str(i)} for i in range(count)],
                'price_usd_per_million_input_tokens': .042, 'sources': {}, 'cases': [],
                'ppo_source': 'm2_checkpoint_records', 'models': {}}
    rows = []
    for method in methods:
        for seed in (None,) if method in independent else seeds:
            for task in range(count):
                case_id = f'{method}-{seed}-{task}'
                manifest['cases'].append({'case_id': case_id, 'method': method,
                                          'task_index': task, 'training_seed': seed})
                offset = {'arbitration': 1., 'rerank': 2.}.get(method, 0.)
                result = {'method': 'ppo' if method == 'ppo_v6' else method, 'status': 'completed',
                          'failure': None, 'return_': task + (0 if seed is None else seed / 100) + offset,
                          'final_mse': .001, 'effort': .2, 'variable_cost': 2., 'total_assay_cost': 5.,
                          'assays': 3, 'immediate_stop': False, 'actions': [4, 12, 31, 36],
                          'forecasts': [], 'timing': {'end_to_end_seconds': 4.}, 'online_seconds': 99.}
                if method == 'probe':
                    result['forecasts'] = [event(.1 + task / 100, .7, j) for j in range(6)]
                rows.append({'case_id': case_id, 'method': method, 'journal': None,
                             'journal_sha256': None, 'api_receipts': [], 'result': result})
    return manifest, {'cases': rows, 'stop_reason': None, 'dispatch_guards': []}


def event(jev, posterior, index=0):
    probabilities = {'posterior': posterior, 'chance': .5}
    if jev is not None:
        probabilities['jev'] = jev
    return {'action_id': index // 2, 'node': index % 2, 'outcome': 0,
            'probabilities': probabilities, 'brier': {k: p * p for k, p in probabilities.items()}}


def receipt(usage=None, *, value=.2, request='same', span='span', status='success'):
    return {'attempted': True, 'status': status, 'usage': usage,
            'latency_seconds': .1, 'request_bytes': 80, 'request_sha256': request,
            'span_id': span, 'response_sha256': 'response-' + span,
            'response': {'model': 'jev-1.14.0', 'answers': {'q': {'type': 'noul', 'noul': value}}},
            **({'code': 'timeout'} if status == 'failed' else {})}


def test_contract_has_exactly_three_fixed_families():
    m = module()
    contract = m.analysis_contract()
    assert [x['id'] for x in contract['primary_families']] == [
        'arbitration-minus-ppo', 'rerank-minus-arbitration', 'probe-jev-minus-posterior']
    assert contract['bootstrap']['draws'] == 10000
    assert contract['bootstrap']['rng_seed'] == 991499
    assert contract['bootstrap']['families'] == 3
    assert contract['bootstrap']['primary_level'] == 1 - .05 / 3
    contract['bootstrap']['draws'] = 1
    assert m.analysis_contract()['bootstrap']['draws'] == 10000


def test_bootstrap_preserves_pairs_and_independent_seed_mechanism_draws():
    m = module()
    a = np.arange(40.).reshape(5, 8)
    b = a - np.arange(8.)
    actual = m.bootstrap_difference(a, b, draws=256, primary=True)
    rng = np.random.default_rng(991499)
    si = rng.integers(5, size=(256, 5))
    mi = rng.integers(8, size=(256, 8))
    expected = (a - b)[si[:, :, None], mi[:, None, :]].mean(axis=(1, 2))
    assert actual['difference'] == 3.5
    assert actual['ci95'] == pytest.approx(np.quantile(expected, [.025, .975]))
    assert actual['ci98_333333'] == pytest.approx(np.quantile(expected, [.05 / 6, 1 - .05 / 6]))
    with pytest.raises(ValueError):
        m.bootstrap_difference([[1, np.nan]], [[0, 0]])


def test_pilot_never_gets_degenerate_intervals_or_benefit_claim():
    m = module()
    manifest, results = study(methods=('ppo', 'arbitration', 'rerank', 'probe'), count=1, seeds=(20,))
    summary = m.summarize_study(manifest, results)
    for pair in summary['primary_comparisons'].values():
        assert pair['status'] == 'descriptive_only'
        assert pair['reason'] == 'insufficient_mechanisms'
        assert pair['benefit_gate_met'] is None
        for metric in pair['metrics'].values():
            assert metric['ci95'] is None
            assert metric['ci98_333333'] is None
    assert summary['primary_comparisons']['arbitration-minus-ppo']['metrics']['return_']['difference'] == 1.


@pytest.mark.parametrize('failure', ['missing', 'failed', 'not_run_api_limit'])
def test_any_bad_or_missing_planned_seed_blocks_entire_pair(failure):
    m = module()
    manifest, results = study()
    bad = next(r for r in results['cases'] if r['case_id'] == 'ppo-24-7')
    if failure == 'missing':
        results['cases'].remove(bad)
    else:
        bad['result'].update(status=failure, failure='timeout', return_=None, final_mse=None)
    summary = m.summarize_study(manifest, results)
    pair = summary['primary_comparisons']['arbitration-minus-ppo']
    assert pair['status'] == 'blocked'
    assert pair['metrics'] is None
    assert pair['benefit_gate_met'] is None
    assert summary['methods']['ppo']['mean_final_mse'] is None
    assert summary['methods']['ppo']['expected_count'] == 40


def test_fixed_baseline_is_tiled_without_inventing_replicates():
    m = module()
    manifest, results = study(methods=('ppo', 'fixed', 'ppo_v6', 'jev_only', 'arbitration'))
    summary = m.summarize_study(manifest, results)
    pair = summary['secondary_comparisons']['ppo-minus-fixed']
    assert pair['metrics']['return_']['difference'] == pytest.approx(.22)
    assert pair['paired_cells'] == 40
    assert pair['seed_independent_methods_tiled'] == ['fixed']
    assert pair['independent_evaluations'] == {'ppo': 40, 'fixed': 8}
    assert pair['metrics']['return_']['ci98_333333'] is None
    assert summary['methods']['ppo_v6']['expected_count'] == 40
    assert summary['secondary_comparisons']['arbitration-minus-jev_only']['seed_independent_methods_tiled'] == ['jev_only']
    assert summary['methods']['ppo']['mean_end_to_end_seconds'] == 4.
    assert summary['methods']['ppo']['aggregate_rmse'] == pytest.approx(np.sqrt(.001))


def test_m1_and_m2_use_reset_inclusive_latency_without_online_fallback():
    m = module()
    manifest, results = study(methods=('ppo', 'ppo_v6'))
    for row in results['cases']:
        if row['method'] == 'ppo_v6':
            del row['result']['timing']
            row['result']['end_to_end_seconds'] = 7.
    summary = m.summarize_study(manifest, results)
    assert summary['secondary_comparisons']['ppo-minus-ppo_v6']['metrics']['end_to_end_seconds']['difference'] == -3.
    del results['cases'][-1]['result']['end_to_end_seconds']
    summary = m.summarize_study(manifest, results)
    assert summary['methods']['ppo_v6']['mean_end_to_end_seconds'] is None
    assert summary['secondary_comparisons']['ppo-minus-ppo_v6']['metrics']['end_to_end_seconds']['difference'] is None


def test_complete_primary_gain_requires_mse_noninferiority_and_hosted_mode():
    m = module()
    manifest, results = study()
    pair = m.summarize_study(manifest, results)['primary_comparisons']['arbitration-minus-ppo']
    assert pair['benefit_gate_met'] is True
    for row in results['cases']:
        if row['method'] == 'arbitration':
            row['result']['final_mse'] += .0002
    pair = m.summarize_study(manifest, results)['primary_comparisons']['arbitration-minus-ppo']
    assert pair['benefit_gate_met'] is False
    manifest['mode'] = 'fixture'
    pair = m.summarize_study(manifest, results)['primary_comparisons']['arbitration-minus-ppo']
    assert pair['benefit_gate_met'] is None


@pytest.mark.parametrize('seeds', [(20,), (20, 21, 22, 23), (20, 21, 22, 23, 25)])
def test_learned_pairs_require_all_five_registered_seeds_for_intervals(seeds):
    m = module()
    manifest, results = study(methods=('ppo', 'arbitration', 'rerank', 'probe'), seeds=seeds)
    summary = m.summarize_study(manifest, results)
    pair = summary['primary_comparisons']['arbitration-minus-ppo']
    assert pair['status'] == 'descriptive_only'
    assert pair['reason'] == 'requires_all_five_registered_training_seeds'
    assert pair['benefit_gate_met'] is None
    assert pair['metrics']['return_']['difference'] == pytest.approx(1.)
    for metric in [*pair['metrics'].values(), pair['api_cost']['paired_difference']]:
        assert metric['ci95'] is None
        assert metric['ci98_333333'] is None
    secondary = summary['secondary_comparisons']['rerank-minus-ppo']
    assert secondary['metrics']['return_']['ci95'] is None
    # The fixed probe has no learned-policy seed axis.
    probe = summary['primary_comparisons']['probe-jev-minus-posterior']
    assert probe['status'] == 'complete'
    assert probe['benefit_gate_met'] is True
    assert m.analysis_contract()['learned_pair_training_seeds'] == [20, 21, 22, 23, 24]


def test_registered_seed_order_does_not_disable_learned_pair_inference():
    m = module()
    manifest, results = study(seeds=(24, 23, 22, 21, 20))
    pair = m.summarize_study(manifest, results)['primary_comparisons']['arbitration-minus-ppo']
    assert pair['status'] == 'complete'
    assert pair['benefit_gate_met'] is True


def test_favorable_fixture_intervals_cannot_produce_benefit_claims():
    m = module()
    manifest, results = study(methods=('ppo', 'arbitration', 'rerank', 'probe'))
    manifest.update(mode='fixture', model='jev-fixture')
    summary = m.summarize_study(manifest, results)
    assert summary['fixture_mode'] is True
    assert summary['calibration_qualified'] is False
    assert summary['fixture_efficacy_evidence'] is False
    for name, pair in summary['primary_comparisons'].items():
        assert pair['benefit_gate_met'] is None
        assert pair['inference_scope'].startswith('fixture descriptive only')
        if name.startswith('probe-'):
            assert pair['metrics']['brier']['ci98_333333'][1] < 0
        else:
            assert pair['metrics']['return_']['ci98_333333'][0] > 0
            assert pair['metrics']['final_mse']['ci98_333333'][1] <= .0001
    assert 'benefit claims are disabled' in m.report_text(summary)


def test_rerank_minus_hidden_is_descriptive_with_remaining_confounding_noted():
    m = module()
    manifest, results = study(methods=('rerank', 'hidden'))
    summary = m.summarize_study(manifest, results)
    pair = summary['secondary_comparisons']['rerank-minus-hidden']
    assert ['rerank', 'hidden'] in m.analysis_contract()['secondary_comparisons']
    assert pair['metrics']['return_']['difference'] == pytest.approx(2.)
    assert pair['metrics']['return_']['ci95'] == pytest.approx([2., 2.])
    assert pair['metrics']['return_']['ci98_333333'] is None
    assert pair['benefit_gate_met'] is None
    assert 'omit forecast values' in pair['interpretation_note']
    assert 'prompts' in pair['interpretation_note']
    assert 'forecast-call' in pair['interpretation_note']


def test_probe_clusters_six_events_and_uses_no_adaptive_events():
    m = module()
    manifest, results = study(methods=('probe', 'arbitration'))
    for row in results['cases']:
        if row['method'] == 'arbitration':
            row['result']['forecasts'] = [event(1., 0.)] * 30
    summary = m.summarize_study(manifest, results)
    probe = summary['common_probe']
    pair = summary['primary_comparisons']['probe-jev-minus-posterior']
    deltas = np.array([[(.1 + i / 100) ** 2 - .7 ** 2 for i in range(8)]])
    expected = m.bootstrap_difference(deltas, np.zeros_like(deltas), primary=True)
    for key, value in expected.items():
        assert pair['metrics']['brier'][key] == pytest.approx(value)
    assert probe['paired_valid_events'] == 48
    assert probe['expected_events'] == 48
    assert pair['mechanism_clusters'] == 8
    assert pair['benefit_gate_met'] is True


def test_missing_probe_forecasts_remain_missing_and_block_forecast_benefit():
    m = module()
    manifest, results = study(methods=('probe',))
    results['cases'][0]['result']['forecasts'][0] = event(None, .7, 0)
    results['cases'][0]['result']['forecasts'][1] = event(None, .7, 1)
    results['cases'][0]['result']['forecast_failures'] = ['timeout']
    summary = m.summarize_study(manifest, results)
    probe = summary['common_probe']
    observed_sum = sum((.1 + i / 100) ** 2 * (4 if i == 0 else 6) for i in range(8))
    assert probe['paired_valid_events'] == 46
    assert probe['missing_jev_events'] == 2
    assert probe['whole_probe_jev_brier_bounds'] == pytest.approx([observed_sum / 48, (observed_sum + 2) / 48])
    assert probe['whole_probe_delta_brier_bounds'] == pytest.approx([observed_sum / 48 - .49, (observed_sum + 2) / 48 - .49])
    pair = summary['primary_comparisons']['probe-jev-minus-posterior']
    assert pair['status'] == 'blocked'
    assert pair['reason'] == 'incomplete_paired_probe_coverage'
    assert pair['benefit_gate_met'] is None
    assert pair['metrics']['brier']['ci95'] is None


def test_missing_whole_probe_retains_expected_events_and_conservative_bounds():
    m = module()
    manifest, results = study(methods=('probe',), count=1, seeds=(20,))
    results['cases'].clear()
    probe = m.summarize_study(manifest, results)['common_probe']
    assert probe['expected_events'] == 6
    assert probe['paired_valid_events'] == 0
    assert probe['whole_probe_jev_brier_bounds'] == [0., 1.]
    assert probe['whole_probe_delta_brier_bounds'] == [-1., 1.]


def test_probe_bounds_allow_missing_reference_and_report_paired_scores():
    m = module()
    manifest, results = study(methods=('probe',), count=1, seeds=(20,))
    first = results['cases'][0]['result']['forecasts'][0]
    del first['probabilities']['posterior']
    del first['brier']['posterior']
    probe = m.summarize_study(manifest, results)['common_probe']
    assert probe['paired_valid_events'] == 5
    assert probe['paired_mean_jev_brier'] == pytest.approx(.01)
    assert probe['paired_mean_posterior_brier'] == pytest.approx(.49)
    assert probe['whole_probe_delta_brier_bounds'] == pytest.approx([(.06 - 5 * .49 - 1) / 6, (.06 - 5 * .49) / 6])


def test_unknown_usage_preserves_known_cost_and_blocks_dollar_comparison():
    m = module()
    manifest, results = study()
    selected = next(r for r in results['cases'] if r['method'] == 'arbitration')
    selected['api_receipts'] = [receipt({'input_tokens': 100, 'output_tokens': 5}), receipt(status='failed')]
    summary = m.summarize_study(manifest, results)
    api = summary['methods']['arbitration']['api']
    assert api['attempted_calls'] == 2
    assert api['unknown_usage_calls'] == 1
    assert api['known_input_tokens'] == 100
    assert api['input_tokens'] is None
    assert api['estimated_usd'] is None
    assert api['known_estimated_usd'] == pytest.approx(.042 * 100 / 1e6)
    assert summary['primary_comparisons']['arbitration-minus-ppo']['api_cost']['status'] == 'blocked_unknown_usage'
    assert api['resolved_model_counts'] == {'jev-1.14.0': 2}


def test_exact_repeated_request_analysis_counts_answers_not_usage_metadata():
    m = module()
    rows = []
    journals = {}
    for i, value in enumerate((.2, .2, .7)):
        r = receipt({'input_tokens': 100 + i, 'output_tokens': 1}, value=value, span=str(i))
        rows.append({'case_id': str(i), 'method': 'probe', 'api_receipts': [r]})
        journals[str(i)] = [{'kind': 'api_request', 'span_id': str(i), 'request_sha256': 'same',
                            'request': {'model': 'jev-1.14.0', 'questions': {'q': {'type': 'noul'}}}}]
    repeat = m.repeat_variability(rows, journals)
    group = repeat['repeated_requests'][0]
    assert group['actual_attempt_count'] == 3
    assert group['distinct_answer_count'] == 2
    assert group['max_noul_probability_spread'] == pytest.approx(.5)
    assert group['resolved_models'] == ['jev-1.14.0']
    assert repeat['proof_of_weight_drift'] is False


def test_duplicate_cells_and_incomplete_frozen_seed_plan_are_rejected():
    m = module()
    manifest, results = study()
    results['cases'].append(copy.deepcopy(results['cases'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        m.summarize_study(manifest, results)
    results['cases'].pop()
    manifest['cases'] = [c for c in manifest['cases'] if c['case_id'] != 'ppo-24-7']
    with pytest.raises(ValueError, match='planned'):
        m.summarize_study(manifest, results)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_cli_verification_precedes_analysis_and_output_is_exclusive(tmp_path, monkeypatch):
    m = module()
    run, output = tmp_path / 'run', tmp_path / 'analysis'
    manifest, results = study(count=1, seeds=(20,))
    for name, value in [('manifest.json', manifest), ('results.json', results), ('plan.json', {})]:
        write_json(run / name, value)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    verified = []

    def verify(path):
        assert not output.exists()
        verified.append(path)
        return {'verified': True, 'credential_free_replay': True, 'checkpoint_free_replay': True}

    monkeypatch.setattr(m, 'verify_study', verify)
    summary = m.analyze_run(run, output)
    assert verified == [run.resolve()]
    assert summary['provenance']['results_sha256'] == m.file_hash(run / 'results.json')
    assert summary['provenance']['analysis_source_sha256'] == m.file_hash(Path(m.__file__))
    assert summary['provenance']['analysis_contract_sha256'] == m.digest(m.analysis_contract())
    assert {p.name for p in output.iterdir()} == {'summary.json', 'REPORT.md'}
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
    assert 'insufficient_mechanisms' in (output / 'REPORT.md').read_text()
    json.dumps(summary, allow_nan=False)
    with pytest.raises(FileExistsError):
        m.analyze_run(run, output)
    assert len(verified) == 1


def test_failed_verification_writes_no_analysis(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, 'verify_study', lambda run: {'verified': False})
    with pytest.raises(ValueError, match='verif'):
        m.analyze_run(tmp_path, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_analysis_reads_hash_linked_journals_and_reports_api_stages(tmp_path, monkeypatch):
    m = module()
    manifest, results = study(methods=('probe',), count=1, seeds=(20,))
    row = results['cases'][0]
    request = {'model': 'jev-1.14.0', 'questions': {'q': {'type': 'noul'}}}
    response = receipt({'input_tokens': 100, 'output_tokens': 1}, request=m.digest(request))
    events = [{'kind': 'api_request', 'span_id': 'span', 'request': request,
               'request_sha256': m.digest(request)}, {'kind': 'api_response', **response}]
    previous, lines = '0' * 64, []
    for i, event_value in enumerate(events):
        record = {'seq': i, 'previous_sha256': previous, 'recorded_at': 'synthetic', 'event': event_value}
        record['sha256'] = previous = m.digest(record)
        lines.append(m.canonical(record))
    journal = tmp_path / 'run/episodes/probe.jsonl'
    journal.parent.mkdir(parents=True)
    journal.write_text('\n'.join(lines) + '\n')
    row.update(journal='episodes/probe.jsonl', journal_sha256=m.file_hash(journal), api_receipts=[response])
    for name, value in [('manifest.json', manifest), ('results.json', results), ('plan.json', {})]:
        write_json(tmp_path / 'run' / name, value)
    monkeypatch.setattr(m, 'verify_study', lambda path: {'verified': True})
    summary = m.analyze_run(tmp_path / 'run', tmp_path / 'output')
    api = summary['methods']['probe']['api_by_stage']['forecast']
    assert api['attempted_calls'] == 1
    assert summary['provenance']['input_sha256']['episodes/probe.jsonl'] == m.file_hash(journal)
    record = json.loads(lines[0])
    record['event']['request']['model'] = 'changed'
    journal.write_text(m.canonical(record) + '\n' + lines[1] + '\n')
    row['journal_sha256'] = m.file_hash(journal)
    write_json(tmp_path / 'run/results.json', results)
    with pytest.raises(ValueError, match='journal hash'):
        m.analyze_run(tmp_path / 'run', tmp_path / 'tampered-output')
    assert not (tmp_path / 'tampered-output').exists()


def test_verifier_uses_snapshot_in_read_only_verify_mode(tmp_path, monkeypatch):
    m = module()
    script = tmp_path / 'source_snapshot/scripts/jev_m2_study.py'
    script.parent.mkdir(parents=True)
    script.write_text('synthetic verifier source')
    write_json(tmp_path / 'manifest.json', {'sources': {'scripts/jev_m2_study.py': m.file_hash(script)}})
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return type('Result', (), {'returncode': 0, 'stdout': '{"verified": true}', 'stderr': ''})()

    monkeypatch.setattr(m.subprocess, 'run', run)
    assert m.verify_study(tmp_path)['verified'] is True
    assert calls[0][0] == [sys.executable, '-B', str(script), 'verify', '--run', str(tmp_path)]
    script.write_text('modified')
    with pytest.raises(ValueError, match='hash'):
        m.verify_study(tmp_path)
    assert len(calls) == 1


def test_verifier_does_not_execute_unbound_snapshot(tmp_path, monkeypatch):
    m = module()
    script = tmp_path / 'source_snapshot/scripts/jev_m2_study.py'
    script.parent.mkdir(parents=True)
    script.write_text('unbound source')
    write_json(tmp_path / 'manifest.json', {'sources': {}})
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **k: pytest.fail('unbound source executed'))
    with pytest.raises(ValueError, match='bound'):
        m.verify_study(tmp_path)


def test_current_source_verifier_fallback_remains_hash_bound(tmp_path, monkeypatch):
    m = module()
    root, run = tmp_path / 'repo', tmp_path / 'run'
    script = root / 'scripts/jev_m2_study.py'
    script.parent.mkdir(parents=True)
    script.write_text('synthetic current verifier')
    write_json(run / 'manifest.json', {'sources': {'scripts/jev_m2_study.py': m.file_hash(script)}})
    monkeypatch.setattr(m, 'ROOT', root)
    called = []

    def invoke(command, **kwargs):
        called.append(command)
        return type('Result', (), {'returncode': 0, 'stdout': '{"verified":true}', 'stderr': ''})()

    monkeypatch.setattr(m.subprocess, 'run', invoke)
    assert m.verify_study(run)['verifier_source'] == str(script)
    assert called[0][-3:] == ['verify', '--run', str(run)]
