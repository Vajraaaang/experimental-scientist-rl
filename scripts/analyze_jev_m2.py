"""Pre-data M2 paired analysis of a verified, completed or interrupted study.

This command never dispatches Jev, loads weights, reads credentials, or constructs
new evaluation pools. The study's read-only verifier replays its recorded cases
before this module analyzes them. Output must be a new directory.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'jev-m2-analysis-v1'
REGISTERED_TRAINING_SEEDS = (20, 21, 22, 23, 24)
INDEPENDENT = {'prior', 'fixed', 'fixed_quality', 'additive', 'random', 'bo', 'eig', 'voi',
               'jev_only', 'probe'}
METRICS = ('return_', 'final_mse', 'variable_cost', 'total_assay_cost', 'effort',
           'assays', 'immediate_stop', 'end_to_end_seconds')
PRIMARY = (('arbitration', 'ppo'), ('rerank', 'arbitration'))
SELECTED = ('ppo', 'arbitration', 'rerank', 'forecast_only', 'jev_only', 'incumbent_m1')
SECONDARY = (('arbitration', 'hidden'), ('arbitration', 'chance'), ('arbitration', 'public'),
             ('arbitration', 'jev_only'), ('rerank', 'rerank_shadow'), ('rerank', 'hidden'), ('rerank', 'ppo'),
             ('forecast_only', 'ppo'), ('ppo', 'ppo_v6')) + tuple(
                 (method, baseline) for method in SELECTED
                 for baseline in ('imitation', 'calibrated_imitation', 'fixed', 'voi'))
SECONDARY_NOTES = {
    'rerank-minus-hidden': 'Both decision inputs omit forecast values. This compares restricted proposals with the full menu, but prompts and forecast-call order, coverage, and failure rules also differ; the menu effect is not isolated.'}
BOOTSTRAP = {'draws': 10000, 'rng_seed': 991499, 'families': 3,
             'descriptive_level': .95, 'primary_level': 1 - .05 / 3,
             'minimum_mechanisms_for_intervals': 8, 'mse_margin': .0001}
LIMITATIONS = [
    'Development only: intervals and gates are exploratory; no calibration qualification, final-test, or SOTA claim.',
    'Parameter holdout within one simulator family does not establish structural or biological generalization.',
    'At most eight development mechanisms and five training seeds limit precision and generalization; one mechanism has no intervals or inferential success claim.',
    'Learned-method comparisons require all five registered training seeds (20 through 24) for intervals or benefit gates; smaller seed cohorts are descriptive only. The fixed probe resamples mechanisms only.',
    'PPO architecture/checkpoint selection and prior development/fixture reuse can affect results; these comparisons are not an untouched final evaluation.',
    'Seed-independent baselines, Jev-only, and the fixed probe run once per mechanism. Tiling for pairing does not create independent replicates.',
    'Adaptive assay forecasts are selection-affected. The forecast primary uses only the fixed common probe and requires all six paired events per mechanism.',
    'Missing forecasts are never replaced by chance. Conditional paired Brier scores do not establish unconditional whole-probe improvement.',
    'Recorded end-to-end timing includes reset in both engines, reference computations, and synchronous journals; historical native inference speed is not compared.',
    'Estimated API USD is conditional on reported usage and the manifest price; unknown usage is not free. Simulator costs and API dollars are separate units.',
    'Repeated exact requests describe observed reply variability, not proof of model-weight drift; model labels alone do not certify fixed weights.',
    'Prospective registration requires independently recording the analysis source and contract hashes before hosted dispatch; an analysis output alone cannot establish chronology.',
]


def analysis_contract():
    """Return the fixed contract that can be hashed before hosted approval."""
    return {'schema': SCHEMA, 'study_schema_version': 4,
            'primary_families': [
                {'id': f'{left}-minus-{right}', 'left': left, 'right': right,
                 'metrics': ['return_', 'final_mse'],
                 'benefit_rule': 'adjusted return lower > 0 and adjusted MSE upper <= 0.0001'}
                for left, right in PRIMARY] + [
                {'id': 'probe-jev-minus-posterior', 'left': 'jev', 'right': 'posterior',
                 'metrics': ['brier'], 'events': 'six fixed-probe events per mechanism',
                 'benefit_rule': 'complete planned probe and paired coverage; adjusted delta-Brier upper < 0'}],
            'bootstrap': dict(BOOTSTRAP),
            'learned_pair_training_seeds': list(REGISTERED_TRAINING_SEEDS),
            'resampling': 'paired crossed bootstrap of independent seed and mechanism indices; probe resamples whole mechanisms with all six events',
            'missing_outcomes': 'block a comparison unless every planned seed/mechanism cell in both methods completed with finite outcomes',
            'missing_forecasts': 'retain missingness, paired-valid counts, expected counts, and best/worst whole-probe Brier bounds; no chance imputation',
            'secondary_comparisons': [list(pair) for pair in SECONDARY],
            'secondary_interpretation_notes': dict(SECONDARY_NOTES),
            'secondary_scope': 'descriptive paired 95% intervals only; no multiplicity-adjusted secondary claims',
            'fixture_scope': 'descriptive synthetic numbers only; efficacy and calibration benefit claims are disabled even with favorable intervals',
            'cost_scope': 'virtual variable and setup-inclusive assay cost; API cost separately, conditional on known usage',
            'timing_metric': 'reset-inclusive end_to_end_seconds in M1 and M2',
            'seed_independent_methods': sorted(INDEPENDENT),
            'limitations': list(LIMITATIONS)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _metric(row, name):
    result = row['result']
    if name == 'end_to_end_seconds':
        value = result.get('timing', {}).get(name, result.get(name))
    else:
        value = result.get(name)
    if name == 'immediate_stop' and type(value) is bool:
        return float(value)
    return float(value) if _number(value) else None


def bootstrap_difference(a, b, *, draws=10000, seed=991499, primary=False):
    """Pure paired crossed bootstrap; never treat one mechanism as inference.

    Equivalent resampling to ppo_m2_confirm.bootstrap_difference, kept local to
    avoid importing training/scoring code. Independent baseline copies remain
    constant along the seed axis. For probe means the seed axis has length one.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 2 or not a.size or not np.isfinite(a - b).all():
        raise ValueError('paired finite seed by mechanism matrices required')
    if type(draws) is not int or draws < 1:
        raise ValueError('positive bootstrap draw count required')
    delta = a - b
    result = {'difference': float(delta.mean()), 'ci95': None, 'ci98_333333': None}
    if delta.shape[1] < BOOTSTRAP['minimum_mechanisms_for_intervals']:
        return result
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for start in range(0, draws, 256):
        size = min(256, draws - start)
        si = rng.integers(delta.shape[0], size=(size, delta.shape[0]))
        mi = rng.integers(delta.shape[1], size=(size, delta.shape[1]))
        samples[start:start + size] = delta[si[:, :, None], mi[:, None, :]].mean(axis=(1, 2))
    result['ci95'] = np.quantile(samples, [.025, .975]).tolist()
    if primary:
        result['ci98_333333'] = np.quantile(samples, [.05 / 6, 1 - .05 / 6]).tolist()
    return result


def _design(manifest, results):
    if manifest.get('schema_version') != 4:
        raise ValueError('schema-4 study manifest required')
    count, seeds = len(manifest['tasks']), manifest['training_seeds']
    if not count or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('nonempty mechanisms and distinct planned seeds required')
    if len({t['mechanism_hash'] for t in manifest['tasks']}) != count:
        raise ValueError('duplicate mechanism clusters')
    planned, by_id = {}, {}
    for case in manifest['cases']:
        key = (case['method'], case['training_seed'], case['task_index'])
        if key in planned or case['case_id'] in by_id:
            raise ValueError('duplicate planned cell')
        planned[key], by_id[case['case_id']] = case, key
    methods = sorted({key[0] for key in planned})
    for method in methods:
        expected = {(method, seed, i) for seed in ([None] if method in INDEPENDENT else seeds)
                    for i in range(count)}
        if {key for key in planned if key[0] == method} != expected:
            raise ValueError('planned method omits or adds a seed/mechanism cell')
    index = {}
    for row in results['cases']:
        key = by_id.get(row['case_id'])
        if key is None or row['method'] != key[0]:
            raise ValueError('row outside planned cells or label mismatch')
        if key in index:
            raise ValueError('duplicate result cell')
        index[key] = row
    return planned, index, methods


def _api_stages(journals):
    stages = {}
    for events in journals.values():
        for event in events:
            if event.get('kind') == 'api_request':
                types = {q.get('type') for q in event['request']['questions'].values()}
                stages[event['span_id']] = 'forecast' if types == {'noul'} else 'decision' if types == {'choice'} else 'mixed'
    return stages


def api_statistics(receipts, price):
    attempts = [r for r in receipts if r['attempted']]
    known = [r['usage'] for r in attempts if r.get('usage') is not None]
    unknown = len(attempts) - len(known)
    inputs = sum(u['input_tokens'] for u in known)
    outputs = sum(u['output_tokens'] for u in known)
    latency = [r['latency_seconds'] for r in attempts if _number(r.get('latency_seconds'))]
    return {'attempted_calls': len(attempts), 'successful_calls': sum(r['status'] == 'success' for r in attempts),
            'pre_dispatch_refusals': len(receipts) - len(attempts), 'unknown_usage_calls': unknown,
            'usage_complete': unknown == 0, 'known_input_tokens': inputs, 'known_output_tokens': outputs,
            'input_tokens': inputs if not unknown else None, 'output_tokens': outputs if not unknown else None,
            'latency_seconds': sum(latency) if len(latency) == len(attempts) else None,
            'known_latency_seconds': sum(latency), 'latency_known_calls': len(latency),
            'estimated_usd': inputs / 1e6 * price if not unknown else None,
            'known_estimated_usd': inputs / 1e6 * price,
            'price_usd_per_million_input_tokens': price,
            'resolved_model_counts': dict(Counter(r['response']['model'] for r in attempts
                                                  if (r.get('response') or {}).get('model'))),
            'serialized_request_bytes': sum(r.get('request_bytes', 0) for r in attempts),
            'failures': dict(Counter(r.get('code', 'unspecified') for r in receipts if r['status'] == 'failed'))}


def _cohort(method, planned, index, price, stages, guards):
    cells = [key for key in planned if key[0] == method]
    present = [index[key] for key in cells if key in index]
    attempted = [r for r in present if not r['result']['status'].startswith('not_run_')]
    completed = [r for r in present if r['result']['status'] == 'completed']
    valid = [r for r in completed if _metric(r, 'return_') is not None
             and _metric(r, 'final_mse') is not None and _metric(r, 'final_mse') >= 0]
    complete = len(valid) == len(cells)
    receipts = [r for row in present for r in row['api_receipts']]
    row_ids = {row['case_id'] for row in present}
    summary = {'status': 'complete' if complete else 'incomplete', 'expected_count': len(cells),
               'present_count': len(present), 'attempted_count': len(attempted), 'completed_count': len(completed),
               'invalid_completed_outcome_count': len(completed) - len(valid),
               'missing_count': len(cells) - len(present), 'not_run_count': len(present) - len(attempted),
               'failed_count': len(attempted) - len(completed),
               'status_counts': dict(Counter(r['result']['status'] for r in present)),
               'failure_reasons': dict(Counter(r['result'].get('failure') or 'unspecified'
                                              for r in attempted if r['result']['status'] != 'completed')),
               'forecast_failures': dict(Counter(f for r in present for f in r['result'].get('forecast_failures', []))),
               'local_dispatch_guard_failures': dict(Counter(g['reason'] for g in guards if g['case_id'] in row_ids)),
               'completed_immediate_stops': sum(r['result'].get('immediate_stop') is True for r in completed),
               'seed_independent': method in INDEPENDENT,
               'api': api_statistics(receipts, price), 'api_by_stage': {}, 'attempt_metrics': {}}
    for stage in sorted({stages.get(r['span_id'], 'unclassified') for r in receipts}):
        summary['api_by_stage'][stage] = api_statistics([r for r in receipts if stages.get(r['span_id'], 'unclassified') == stage], price)
    for metric in METRICS:
        values = [_metric(r, metric) for r in completed]
        summary['mean_' + metric] = float(np.mean(values)) if complete and all(v is not None for v in values) else None
        if metric in ('variable_cost', 'total_assay_cost', 'assays', 'end_to_end_seconds'):
            measured = [_metric(r, metric) for r in attempted if _metric(r, metric) is not None]
            summary['attempt_metrics'][metric] = {'known_count': len(measured), 'attempted_count': len(attempted),
                                                   'known_total': sum(measured),
                                                   'mean_of_known_attempts': float(np.mean(measured)) if measured else None}
    mse = summary['mean_final_mse']
    summary['aggregate_rmse'] = math.sqrt(mse) if mse is not None else None
    summary['mean_denominator'] = 'all planned cells; outcome means null for an incomplete cohort; attempt costs retain failures separately'
    return summary


def _pair(left, right, manifest, index, methods, *, primary):
    count, seeds = len(manifest['tasks']), manifest['training_seeds']
    result = {'direction': f'{left} minus {right}', 'metrics': None, 'benefit_gate_met': None,
              'inference_scope': 'fixture descriptive only; no benefit claim' if manifest['mode'] == 'fixture' else
                  ('exploratory development only' if primary else 'secondary descriptive only'),
              'api_cost': {'status': 'blocked_incomplete_cohorts', 'paired_difference': None}}
    if f'{left}-minus-{right}' in SECONDARY_NOTES:
        result['interpretation_note'] = SECONDARY_NOTES[f'{left}-minus-{right}']
    if any(method not in methods or methods[method]['status'] != 'complete' for method in (left, right)):
        return {**result, 'status': 'blocked', 'reason': 'requires_all_planned_cells_in_both_methods'}
    used_seeds = [None] if left in INDEPENDENT and right in INDEPENDENT else seeds
    reason = ('insufficient_mechanisms' if count < BOOTSTRAP['minimum_mechanisms_for_intervals'] else
              'requires_all_five_registered_training_seeds' if
              used_seeds != [None] and set(seeds) != set(REGISTERED_TRAINING_SEEDS) else None)
    eligible = reason is None

    def comparison(a, b, *, primary_metric=False):
        if a is None or b is None:
            return {'difference': None, 'ci95': None, 'ci98_333333': None}
        if not eligible:
            return {'difference': float((a - b).mean()), 'ci95': None, 'ci98_333333': None}
        return bootstrap_difference(a, b, primary=primary_metric)

    def matrix(method, metric):
        rows = [[index[(method, None if method in INDEPENDENT else seed, i)] for i in range(count)] for seed in used_seeds]
        if metric == 'estimated_usd':
            values = [[api_statistics(r['api_receipts'], manifest['price_usd_per_million_input_tokens'])['estimated_usd']
                       for r in group] for group in rows]
        else:
            values = [[_metric(r, metric) for r in group] for group in rows]
        return np.asarray(values, float) if all(v is not None for group in values for v in group) else None

    metrics = {}
    for metric in METRICS:
        a, b = matrix(left, metric), matrix(right, metric)
        metrics[metric] = comparison(a, b, primary_metric=primary and metric in ('return_', 'final_mse'))
    result.update(status='complete' if eligible else 'descriptive_only',
                  reason=reason, metrics=metrics,
                  paired_cells=count * len(used_seeds), mechanism_clusters=count,
                  training_seed_count=len(used_seeds),
                  seed_independent_methods_tiled=[m for m in (left, right) if m in INDEPENDENT and len(used_seeds) > 1],
                  independent_evaluations={m: count * (1 if m in INDEPENDENT else len(seeds)) for m in (left, right)})
    if primary and eligible and manifest['mode'] == 'hosted':
        result['benefit_gate_met'] = bool(metrics['return_']['ci98_333333'][0] > 0
                                         and metrics['final_mse']['ci98_333333'][1] <= BOOTSTRAP['mse_margin'])
    a, b = matrix(left, 'estimated_usd'), matrix(right, 'estimated_usd')
    result['api_cost'] = {'status': 'blocked_unknown_usage' if a is None or b is None else 'descriptive_only',
                          'paired_difference': None if a is None or b is None else comparison(a, b),
                          'scope': 'API USD only; separate from return/MSE gate and virtual assay costs; no USD-advantage claim with unknown usage'}
    return result


def _probability(event, name):
    value = event.get('probabilities', {}).get(name)
    if value is None:
        return None
    if not _number(value) or not 0 <= value <= 1:
        raise ValueError('invalid recorded forecast probability')
    return float(value)


def _probe(manifest, index, cohorts):
    count = len(manifest['tasks'])
    planned = 'probe' in cohorts
    per_mechanism, paired_deltas, jev_losses, posterior_losses = [], [], [], []
    paired_jev, paired_posterior = [], []
    delta_lower = delta_upper = 0.
    observed = 0
    for task in range(count) if planned else ():
        row = index.get(('probe', None, task))
        events = [] if row is None else row['result'].get('forecasts', [])
        if len(events) > 6:
            raise ValueError('fixed probe exceeds six expected events per mechanism')
        local_deltas, local_jev, local_posterior = [], [], []
        observed += len(events)
        for event in events:
            y = event.get('outcome')
            if type(y) is not int or y not in (0, 1):
                raise ValueError('binary forecast outcome required')
            probabilities = {name: _probability(event, name) for name in ('jev', 'posterior')}
            losses = {name: None if p is None else (p - y) ** 2 for name, p in probabilities.items()}
            for name, loss in losses.items():
                recorded = event.get('brier', {}).get(name)
                if loss is not None and (not _number(recorded) or not math.isclose(loss, recorded, rel_tol=1e-12, abs_tol=1e-12)):
                    raise ValueError('recorded Brier score differs from probability and outcome')
            j, p = losses['jev'], losses['posterior']
            if j is not None:
                local_jev.append(j)
            if p is not None:
                local_posterior.append(p)
            if j is not None and p is not None:
                local_deltas.append(j - p)
                paired_jev.append(j)
                paired_posterior.append(p)
            delta_lower += (0. if j is None else j) - (1. if p is None else p)
            delta_upper += (1. if j is None else j) - (0. if p is None else p)
        missing_events = 6 - len(events)
        delta_lower -= missing_events
        delta_upper += missing_events
        jev_losses.extend(local_jev)
        posterior_losses.extend(local_posterior)
        paired_deltas.extend(local_deltas)
        per_mechanism.append({'task_index': task, 'expected_events': 6, 'observed_events': len(events),
                              'valid_jev_events': len(local_jev), 'valid_posterior_events': len(local_posterior),
                              'paired_valid_events': len(local_deltas),
                              'paired_mean_delta_brier': float(np.mean(local_deltas)) if local_deltas else None})
    expected = count * 6 if planned else 0
    complete = planned and cohorts['probe']['status'] == 'complete' and len(paired_deltas) == expected
    summary = {'expected_events': expected, 'observed_events': observed,
               'valid_jev_events': len(jev_losses), 'valid_posterior_events': len(posterior_losses),
               'paired_valid_events': len(paired_deltas), 'missing_jev_events': expected - len(jev_losses),
               'paired_coverage': len(paired_deltas) / expected if expected else None,
               'complete_paired_coverage': complete,
               'paired_mean_jev_brier': float(np.mean(paired_jev)) if paired_jev else None,
               'paired_mean_posterior_brier': float(np.mean(paired_posterior)) if paired_posterior else None,
               'conditional_mean_delta_brier': float(np.mean(paired_deltas)) if paired_deltas else None,
               'whole_probe_jev_brier_bounds': [sum(jev_losses) / expected, (sum(jev_losses) + expected - len(jev_losses)) / expected] if expected else None,
               'whole_probe_posterior_brier_bounds': [sum(posterior_losses) / expected, (sum(posterior_losses) + expected - len(posterior_losses)) / expected] if expected else None,
               'whole_probe_delta_brier_bounds': [delta_lower / expected, delta_upper / expected] if expected else None,
               'per_mechanism': per_mechanism,
               'scope': 'fixed common probe only; each mechanism clusters all six events; missingness bounds are not confidence intervals'}
    metric = {'difference': summary['conditional_mean_delta_brier'], 'ci95': None, 'ci98_333333': None}
    pair = {'direction': 'fixed-probe Jev minus posterior Brier', 'status': 'blocked',
            'reason': 'incomplete_paired_probe_coverage', 'metrics': {'brier': metric},
            'mechanism_clusters': count if planned else 0, 'paired_valid_events': len(paired_deltas),
            'expected_events': expected, 'benefit_gate_met': None,
            'inference_scope': 'fixture descriptive only; no benefit claim' if manifest['mode'] == 'fixture' else
                'exploratory development forecast comparison; not calibration qualification'}
    if complete:
        values = np.array([[r['paired_mean_delta_brier'] for r in per_mechanism]])
        pair['metrics']['brier'] = bootstrap_difference(values, np.zeros_like(values), primary=True)
        eligible = count >= BOOTSTRAP['minimum_mechanisms_for_intervals']
        pair.update(status='complete' if eligible else 'descriptive_only', reason=None if eligible else 'insufficient_mechanisms')
        if eligible and manifest['mode'] == 'hosted':
            pair['benefit_gate_met'] = pair['metrics']['brier']['ci98_333333'][1] < 0
    return summary, pair


def repeat_variability(rows, journals):
    requests = {e['span_id']: e for events in journals.values() for e in events if e.get('kind') == 'api_request'}
    groups = defaultdict(list)
    for row in rows:
        for receipt in row['api_receipts']:
            if receipt['attempted'] and receipt.get('request_sha256'):
                groups[receipt['request_sha256']].append((row, receipt))
    repeated = []
    for request_hash, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        answers, noul, requested, resolved = [], defaultdict(list), set(), set()
        for row, receipt in group:
            request = requests.get(receipt['span_id'], {}).get('request', {})
            if request.get('model'):
                requested.add(request['model'])
            response = receipt.get('response') or {}
            if response.get('model'):
                resolved.add(response['model'])
            if isinstance(response.get('answers'), dict):
                answers.append(digest(response['answers']))
                for question, answer in response['answers'].items():
                    if answer.get('type') == 'noul' and _number(answer.get('noul')):
                        noul[question].append(answer['noul'])
        spreads = {q: max(values) - min(values) for q, values in noul.items() if len(values) >= 2}
        repeated.append({'request_sha256': request_hash, 'actual_attempt_count': len(group),
                         'successful_call_count': sum(r['status'] == 'success' for _, r in group),
                         'answers_observed': len(answers), 'distinct_answer_count': len(set(answers)),
                         'distinct_answer_sha256': sorted(set(answers)),
                         'max_noul_probability_spread': max(spreads.values()) if spreads else None,
                         'noul_probability_spread_by_question': spreads,
                         'requested_models': sorted(requested), 'resolved_models': sorted(resolved),
                         'methods': sorted({row['method'] for row, _ in group}),
                         'case_ids': [row['case_id'] for row, _ in group]})
    return {'scope': 'observed repeat variability from exact recorded request hashes; no additional API probes',
            'proof_of_weight_drift': False, 'actual_attempt_count': sum(len(g) for g in groups.values()),
            'distinct_request_count': len(groups), 'repeated_request_count': len(repeated), 'repeated_requests': repeated}


def summarize_study(manifest, results, journals=None):
    """Pure analysis API. CLI verification is deliberately outside this function."""
    journals = {} if journals is None else journals
    planned, index, names = _design(manifest, results)
    price = manifest['price_usd_per_million_input_tokens']
    if not _number(price) or price < 0:
        raise ValueError('finite nonnegative manifest API price required')
    stages = _api_stages(journals)
    cohorts = {name: _cohort(name, planned, index, price, stages, results.get('dispatch_guards', [])) for name in names}
    primary = {f'{a}-minus-{b}': _pair(a, b, manifest, index, cohorts, primary=True) for a, b in PRIMARY}
    probe, primary['probe-jev-minus-posterior'] = _probe(manifest, index, cohorts)
    secondary = {f'{a}-minus-{b}': _pair(a, b, manifest, index, cohorts, primary=False) for a, b in SECONDARY}
    receipts = [r for row in results['cases'] for r in row['api_receipts']]
    return {'schema': SCHEMA, 'mode': manifest['mode'], 'fixture_mode': manifest['mode'] == 'fixture',
            'requested_model': manifest.get('model'),
            'analysis_contract': analysis_contract(),
            'mechanisms': len(manifest['tasks']), 'training_seeds': list(manifest['training_seeds']),
            'ppo_source': manifest.get('ppo_source'), 'methods': cohorts,
            'primary_comparisons': primary, 'secondary_comparisons': secondary, 'common_probe': probe,
            'api': api_statistics(receipts, price), 'repeat_variability': repeat_variability(results['cases'], journals),
            'study_stop_reason': results.get('stop_reason'), 'dispatch_guards': results.get('dispatch_guards', []),
            'failure_origin_note': 'event_sink_error receipts linked to dispatch_guards are local refusals, not provider outages',
            'calibration_qualified': False, 'final_test_evidence': False, 'sota_claim': False,
            'fixture_efficacy_evidence': False, 'limitations': list(LIMITATIONS)}


def _safe_child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('artifact path escapes run/source root')
    return path


def verify_study(run):
    """Invoke only credential/checkpoint-free verification from a bound source."""
    run = Path(run).resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    snapshot = run / 'source_snapshot'
    source_root = snapshot if snapshot.exists() else ROOT
    script = source_root / 'scripts/jev_m2_study.py'
    if 'scripts/jev_m2_study.py' not in manifest['sources']:
        raise ValueError('study verifier must be bound by the manifest source hashes')
    for name, expected in manifest['sources'].items():
        if file_hash(_safe_child(source_root, name)) != expected:
            raise ValueError('verifier source hash mismatch')
    before = {name: file_hash(run / name) for name in ('manifest.json', 'results.json', 'plan.json', 'trace-index.json') if (run / name).is_file()}
    command = [sys.executable, '-B', str(script), 'verify', '--run', str(run)]
    completed = subprocess.run(command, cwd=source_root, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise ValueError('study verification failed: ' + completed.stderr.strip()[-1000:])
    verification = json.loads(completed.stdout)
    if verification.get('verified') is not True:
        raise ValueError('study verification did not succeed')
    if any(file_hash(run / name) != sha for name, sha in before.items()):
        raise ValueError('study input changed during verification')
    if any(file_hash(_safe_child(source_root, name)) != sha for name, sha in manifest['sources'].items()):
        raise ValueError('study source changed during verification')
    return {**verification, 'verifier_source': str(script), 'verifier_sha256': file_hash(script),
            'verified_input_sha256': before}


def _read_journal(path):
    events, previous = [], '0' * 64
    for seq, line in enumerate(path.read_text().splitlines()):
        record = json.loads(line)
        sha = record.pop('sha256')
        if record['seq'] != seq or record['previous_sha256'] != previous or digest(record) != sha:
            raise ValueError('journal hash or order mismatch')
        previous = sha
        events.append(record['event'])
    return events


def _display(value):
    if value is None:
        return 'null'
    if type(value) is float:
        return f'{value:.6g}'
    if isinstance(value, list):
        return '[' + ', '.join(_display(v) for v in value) + ']'
    return str(value)


def report_text(summary):
    lines = ['# M2 paired development analysis', '',
             f"Mode: {summary['mode']}. Mechanisms: {summary['mechanisms']}; planned training seeds: {summary['training_seeds']}.",
             'Fixture mode: descriptive synthetic numbers only; efficacy and calibration benefit claims are disabled.' if summary['fixture_mode'] else
                 'Hosted mode: benefit gates remain exploratory development signals.',
             'All differences are the first method minus its comparator. Null means unavailable or blocked, never zero.',
             'The three prespecified families use 10,000 paired bootstrap draws, RNG 991499, descriptive 95% and Bonferroni 98.333333% intervals. Learned-method intervals require all five registered seeds; the fixed probe requires eight mechanism clusters. Gates are exploratory development signals only.', '',
             '| Primary family | Status / reason | Return or Brier delta (95%; adjusted) | MSE delta (95%; adjusted) | Exploratory gate |',
             '|---|---|---|---|---|']
    for name, pair in summary['primary_comparisons'].items():
        metrics = pair['metrics'] or {}

        def cell(metric):
            value = metrics.get(metric, {})
            return f"{_display(value.get('difference'))} ({_display(value.get('ci95'))}; {_display(value.get('ci98_333333'))})"

        lines.append(f"| {name} | {pair['status']} / {_display(pair.get('reason'))} | {cell('brier' if name.startswith('probe-') else 'return_')} | {cell('final_mse')} | {_display(pair['benefit_gate_met'])} |")
    lines += ['', 'Return benefit requires adjusted return lower > 0 and adjusted MSE upper <= 0.0001. Forecast benefit requires complete paired probe coverage and adjusted delta-Brier upper < 0. Incomplete cohorts are never reduced to successful seeds.', '',
              '| Method | Completed / planned | Return | Raw MSE | sqrt(mean MSE) | Variable / total assay cost | Effort | Assays | Reset-inclusive seconds | Failed / not run / missing | Immediate stops |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for name, row in summary['methods'].items():
        lines.append(f"| {name} | {row['completed_count']}/{row['expected_count']} | {_display(row['mean_return_'])} | {_display(row['mean_final_mse'])} | {_display(row['aggregate_rmse'])} | {_display(row['mean_variable_cost'])} / {_display(row['mean_total_assay_cost'])} | {_display(row['mean_effort'])} | {_display(row['mean_assays'])} | {_display(row['mean_end_to_end_seconds'])} | {row['failed_count']} / {row['not_run_count']} / {row['missing_count']} | {row['completed_immediate_stops']} |")
    probe = summary['common_probe']
    lines += ['', f"Fixed common probe: {probe['paired_valid_events']}/{probe['expected_events']} paired valid events; {probe['missing_jev_events']} missing Jev forecasts. Conditional delta Brier: {_display(probe['conditional_mean_delta_brier'])}. Whole-probe Jev Brier bounds: {_display(probe['whole_probe_jev_brier_bounds'])}; delta-Brier bounds: {_display(probe['whole_probe_delta_brier_bounds'])}.", '',
              '| Role | Attempted calls | Known input / output tokens | Unknown usage calls | API seconds | Estimated USD | Known-usage USD subtotal |',
              '|---|---|---|---|---|---|---|']
    for name, row in summary['methods'].items():
        api = row['api']
        lines.append(f"| {name} | {api['attempted_calls']} | {api['known_input_tokens']} / {api['known_output_tokens']} | {api['unknown_usage_calls']} | {_display(api['latency_seconds'])} | {_display(api['estimated_usd'])} | {_display(api['known_estimated_usd'])} |")
    repeat = summary['repeat_variability']
    lines += ['', f"Repeated exact request hashes: {repeat['repeated_request_count']} groups among {repeat['actual_attempt_count']} actual attempts. Per-group actual counts, model labels, answer hashes and maximum Noul probability spreads are recorded in summary.json; these describe reply variability, not proof of weight drift.",
              'Secondary rerank-minus-hidden: ' + SECONDARY_NOTES['rerank-minus-hidden'],
              f"Study stop reason: {_display(summary['study_stop_reason'])}. Detailed attempted costs, API stages, local guard/failure counts, secondary pairs, and input/source hashes are in summary.json.", '',
              '## Limits', '']
    lines += ['- ' + value for value in summary['limitations']]
    return '\n'.join(lines) + '\n'


def analyze_run(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite analysis output')
    verification = verify_study(run)
    if verification.get('verified') is not True:
        raise ValueError('study verification must succeed before analysis')
    manifest = json.loads((run / 'manifest.json').read_text())
    results = json.loads((run / 'results.json').read_text())
    inputs = {name: file_hash(run / name) for name in ('manifest.json', 'results.json', 'plan.json', 'trace-index.json') if (run / name).is_file()}
    if any(inputs.get(name) != sha for name, sha in verification.get('verified_input_sha256', {}).items()):
        raise ValueError('verified study input changed before analysis')
    journals = {}
    for row in results['cases']:
        if row['journal'] is not None:
            path = _safe_child(run, row['journal'])
            inputs[row['journal']] = file_hash(path)
            if inputs[row['journal']] != row['journal_sha256']:
                raise ValueError('journal hash differs from verified result')
            journals[row['case_id']] = _read_journal(path)
    summary = summarize_study(manifest, results, journals)
    summary['provenance'] = {'run': str(run), 'verification': verification,
                             'manifest_sha256': inputs['manifest.json'], 'results_sha256': inputs['results.json'],
                             'input_sha256': inputs, 'study_source_sha256': copy.deepcopy(manifest['sources']),
                             'analysis_source': str(Path(__file__).resolve()), 'analysis_source_sha256': file_hash(__file__),
                             'analysis_contract_sha256': digest(analysis_contract()),
                             'checkpoint_metadata_only': copy.deepcopy(manifest.get('models', {})),
                             'credential_access': False, 'checkpoint_access': False, 'new_hosted_calls': 0}
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n'
    report = report_text(summary)
    if any(file_hash(_safe_child(run, name)) != sha for name, sha in inputs.items()):
        raise ValueError('study input changed during analysis')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'summary.json').open('x') as stream:
        stream.write(encoded)
    with (output / 'REPORT.md').open('x') as stream:
        stream.write(report)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = analyze_run(args.run, args.output)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({'output': str(args.output.resolve()), 'verified': True, 'mode': summary['mode'],
                      'analysis_contract_sha256': summary['provenance']['analysis_contract_sha256'],
                      'mechanisms': summary['mechanisms']}, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
