"""Bounded development study and credential-free verification. No final-test mode."""
import argparse
from collections import Counter
from dataclasses import asdict
from getpass import getpass, GetPassWarning
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]

import numpy as np
import torch
from experimentlab.cost_policy import PublicCostPolicy
from experimentlab.envs import PilotConfig, ScientistEnv
from experimentlab.learning import ActorCritic, collect_episode
from experimentlab.tasks import continuous_family, make_task, mechanism_hash
from research_stage import UnfilteredBank, quadrature_family
from integrations.jev_client import CallLimits, HostedTransport, JevClient, JevError
from integrations.jev_research import (
    FixtureTransport, FrozenPolicy, Journal, canonical, digest, read_journal,
    run_episode, summarize_results, verify_episode,
)

BASELINES = ('prior','fixed','fixed_quality','additive','random','bo','eig','voi')
NEURAL = {'ppo':'selected_ppo','imitation':'imitation',
          'calibrated_imitation':'calibrated_imitation','imitation_ppo':'imitation_ppo'}
HYBRIDS = ('jev_rl','jev_rl_hidden','jev_rl_chance','jev_rl_public')
HISTORICAL = ('feasibility-v1','research-v2','unfiltered-v3','ppo-v4','ppo-cost-v5','ppo-efficiency-v6')


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_hashes(root, hashes):
    for name, sha in hashes.items():
        path = (Path(root)/name).resolve()
        if not path.is_relative_to(Path(root).resolve()) or not path.is_file() or file_hash(path) != sha:
            raise ValueError('source or artifact hash mismatch: '+name)


def development_tasks(count, config):
    if type(count) is not int or not 1 <= count <= 512:
        raise ValueError('bounded development mechanism count must be 1..512')
    return [make_task(m, 991302000+i, config) for i,m in enumerate(continuous_family(991301,count))]


def previous_hashes(root):
    """Only manifests/freeze metadata, never historical per-episode test outcomes."""
    paths = [root/'manifests/continuous-splits.json']
    for name in HISTORICAL:
        paths.extend((root/'reports'/name).rglob('*manifest*.json'))
        paths.extend((root/'reports'/name).rglob('freeze.json'))
    result = set()
    def visit(value):
        if isinstance(value, dict):
            for item in value.values(): visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
        elif isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value):
            result.add(value)
    for path in paths:
        if path.is_file(): visit(json.loads(path.read_text()))
    return result, {str(p.relative_to(root)):file_hash(p) for p in paths if p.is_file()}


def model_records(root, seeds):
    registry = json.loads((root/'reports/ppo-efficiency-v6/selection.json').read_text())['registry']
    records = {}
    for method, historical in NEURAL.items():
        for seed in seeds:
            old = registry[f'{historical}-seed{seed}']
            # Historical paths may belong to the original development machine.
            relative = 'reports/' + old['path'].split('/reports/',1)[1]
            verify_hashes(root,{relative:old['sha256']})
            records[f'{method}-seed{seed}'] = {k:old[k] for k in ('seed','sha256','calibration')}
            records[f'{method}-seed{seed}']['path'] = relative
    return records


def load_policy(root, record):
    verify_hashes(root,{record['path']:record['sha256']})
    # Only this repository's hash-verified checkpoints are accepted; no arbitrary file argument.
    state = torch.load(root/record['path'],map_location='cpu',weights_only=False)
    model = ActorCritic(**state['architecture'])
    model.load_state_dict(state['model'])
    return PublicCostPolicy(model,**record['calibration']).eval()


def source_hashes(root):
    files = []
    for folder in ('src','integrations','scripts','tests'):
        files.extend((root/folder).rglob('*.py'))
    files += [root/p for p in ('requirements.lock','pyproject.toml','autonomous_scientist_build_spec.md',
                               'docs/jev_rl_protocol.md')]
    return {str(p.relative_to(root)):file_hash(p) for p in sorted(files) if '__pycache__' not in p.parts}


def receipt_stats(receipts):
    attempted = [r for r in receipts if r['attempted']]
    usages = [r['usage'] for r in attempted if r['usage'] is not None]
    input_tokens = sum(u['input_tokens'] for u in usages)
    unknown = sum(r['usage'] is None for r in attempted)
    return {'attempted_calls':len(attempted), 'successful_calls':sum(r['status']=='success' for r in receipts),
            'unknown_usage_calls':unknown, 'input_tokens':input_tokens,
            'output_tokens':sum(u['output_tokens'] for u in usages),
            'serialized_request_bytes':sum(r['request_bytes'] for r in attempted),
            'latency_seconds':sum((r['latency_seconds'] for r in attempted),0.0),
            'estimated_usd':None if unknown else input_tokens/1_000_000*.042,
            'known_estimated_usd':input_tokens/1_000_000*.042,
            'failures':dict(Counter(r['code'] for r in receipts if r['status']=='failed'))}


def validate_api_accounting(receipts, limits):
    """Replay every pre-dispatch budget check, retaining unknown-use reservations."""
    attempted = byte_count = reported_input = reserved_unknown = 0
    blocked = None
    spans=set()
    for index, receipt in enumerate(receipts,1):
        if receipt['call_index']!=index:
            raise ValueError('API call index mismatch')
        if receipt['span_id'] in spans:
            raise ValueError('duplicate API span')
        spans.add(receipt['span_id'])
        size = receipt.get('proposed_request_bytes',receipt['request_bytes'])
        reservation = size+4096
        checks = [(attempted>=limits['max_calls'],'call_limit'),
                  (size>limits['max_request_bytes'],'request_too_large'),
                  (byte_count+size>limits['max_total_request_bytes'],'total_request_limit'),
                  (reported_input+reserved_unknown+reservation>limits['max_input_tokens'],'input_token_limit'),
                  ((reported_input+reserved_unknown+reservation)/1_000_000*.042>limits['max_estimated_usd'],'cost_limit')]
        rejection = blocked or next((code for condition,code in checks if condition),None)
        if receipt['attempted']:
            if rejection is not None:
                raise ValueError('API attempt exceeds frozen cap or blocked client')
            attempted += 1
            byte_count += receipt['request_bytes']
            if receipt['usage'] is None:
                reserved_unknown += reservation
            else:
                reported_input += receipt['usage']['input_tokens']
            if receipt['reservation_overrun']:
                blocked='reservation_overrun'
        elif receipt['code']!=rejection and receipt['code']!='event_sink_error':
            raise ValueError('unjustified API preflight refusal')
        if receipt.get('code') in ('model_drift','event_sink_error'):
            blocked=receipt['code']
    return {'reserved_input_tokens':reserved_unknown,'accounted_input_tokens':reported_input+reserved_unknown,
            'blocked_reason':blocked}


def verify_client_stats(actual, receipts, limits):
    expected = {**receipt_stats(receipts),**validate_api_accounting(receipts,limits)}
    expected.update(usage_complete=expected['unknown_usage_calls']==0,
                    reserved_estimated_usd=expected['reserved_input_tokens']/1_000_000*.042,
                    accounted_estimated_usd=expected['accounted_input_tokens']/1_000_000*.042)
    if canonical(actual)!=canonical(expected):
        raise ValueError('API accounting totals differ from verified receipts')


def trace_index(cases, model):
    calls=[]
    for case in cases:
        for receipt in case['api_receipts']:
            usage=receipt['usage'] or {}
            transport=receipt['transport']
            calls.append({'case_id':case['case_id'],'method':case['result']['method'],
                **{key:receipt[key] for key in ('call_index','trace_id','span_id','started_at_utc','finished_at_utc',
                    'request_sha256','response_sha256','attempted','status','latency_seconds','estimated_usd')},
                'failure':receipt.get('code'),'requested_model':model,
                'resolved_model':receipt.get('response',{}).get('model'),
                'provider_request_id':transport.get('provider_request_id'),'http_status':transport.get('http_status'),
                'retry_after':transport.get('retry_after'),'retry_after_ms':transport.get('retry_after_ms'),
                'input_tokens':usage.get('input_tokens'),'output_tokens':usage.get('output_tokens'),
                'question_order':receipt['question_order'],'candidate_order':receipt['candidate_order']})
    return {'scope':'local trace/span instrumentation plus optional provider request IDs; no provider-side spans',
            'calls':calls}


def study_summary(cases, mode):
    methods = summarize_results([c['result'] for c in cases])
    for method in methods:
        receipts = [r for c in cases if c['result']['method']==method for r in c['api_receipts']]
        methods[method]['api'] = receipt_stats(receipts)
        methods[method]['api']['hosted_calls'] = methods[method]['api']['attempted_calls'] if mode=='hosted' else 0
    return {'scope':'development integration only; no confirmatory inference',
            'mode':mode, 'jev_efficacy_established':False,
            'forecast_comparison_primary':'forecast_probe only; adaptive event sets differ', 'methods':methods}


def run_study(output, *, live=False, count=2, key=None):
    if live and count != 1:
        raise ValueError('hosted M1 pilot allows exactly one development mechanism')
    if not live and count != 2:
        raise ValueError('M1 fixture manifest uses exactly two development mechanisms')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite an existing study directory')
    torch.set_num_threads(1)
    config = PilotConfig(final_cultures=128)
    tasks = development_tasks(count,config)
    hashes = [mechanism_hash(t.mechanism) for t in tasks]
    forbidden, historical_manifests = previous_hashes(ROOT)
    if len(set(hashes)) != count or set(hashes)&forbidden:
        raise ValueError('mechanism leakage or duplicate development task')
    seeds = [20] if live else list(range(20,25))
    models = model_records(ROOT,seeds)
    frozen = json.loads((ROOT/'reports/ppo-efficiency-v6/fresh-locked/freeze.json').read_text())
    schedules = {'fixed':frozen['fixed'],'fixed_quality':frozen['fixed_quality'],
                 'additive':frozen['additive_schedule'],'forecast_probe':frozen['fixed']}
    methods = list(BASELINES)+list(NEURAL)+['jev_only','forecast_probe']+(['jev_rl'] if live else list(HYBRIDS))
    cases = []
    for i in range(count):
        for method in methods:
            for seed in seeds if method in NEURAL or method in HYBRIDS else [None]:
                cases.append({'case_id':f'm{i:03}-{method}-s{seed if seed is not None else "none"}',
                              'task_index':i,'method':method,'training_seed':seed})
    limits = CallLimits() if live else CallLimits(max_calls=2000,max_request_bytes=65536,
                      max_total_request_bytes=64*1024*1024,max_input_tokens=100_000_000,max_estimated_usd=10.)
    model_name = 'jev-1.14.0' if live else 'jev-fixture'
    # A missing/invalid key is rejected before creating any study or making any call.
    transport = HostedTransport(key) if live else FixtureTransport()
    client = JevClient(transport, expected_model=model_name, limits=limits)
    manifest = {'schema_version':2,'scope':'jev-rl-m1-development','mode':'hosted' if live else 'fixture',
                'model':model_name,'config':asdict(config),'generator_seed':991301,
                'trace_id':client.trace_id,
                'tasks':[{'episode_seed':t.seed,'mechanism_hash':h,'target':[float(x) for x in t.target]}
                         for t,h in zip(tasks,hashes)],
                'training_seeds':seeds,'models':models,'cases':cases,'schedules':schedules,
                'api_limits':asdict(limits),'price_usd_per_million_input_tokens':.042,
                'historical_manifest_hashes':historical_manifests,
                'historical_identifiers_checked':len(forbidden),
                'sources':source_hashes(ROOT), 'final_test_opened':False,
                'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
                'device':'cpu','forecast_probe_primary':True,'forced_first_assay':False}
    output.mkdir(parents=True)
    (output/'episodes').mkdir()
    write_new(output/'manifest.json',manifest)
    # Preserve runnable source for verification even after further development.
    for name in manifest['sources']:
        destination = output/'source_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    bank = UnfilteredBank(quadrature_family(2),config)
    loaded = {name:load_policy(ROOT,record) for name,record in models.items()}
    rows, equivalence = [], []
    for case in cases:
        method, seed, index = case['method'],case['training_seed'],case['task_index']
        policy_name = 'ppo' if method in HYBRIDS else method
        model = loaded[f'{policy_name}-seed{seed}'] if method in NEURAL or method in HYBRIDS else None
        path = output/'episodes'/(case['case_id']+'.jsonl')
        before = len(client.receipts)
        with Journal(path) as journal:
            result = run_episode(ScientistEnv(tasks[index],config),method,bank,journal,
                                 schedule=schedules.get(method,()), policy=FrozenPolicy(model,models[f'{policy_name}-seed{seed}']) if model is not None else None,
                                 client=client if method.startswith('jev_') or method=='forecast_probe' else None,
                                 planner_seed=991303000+index)
        row = {'case_id':case['case_id'],'journal':str(path.relative_to(output)),
               'journal_sha256':file_hash(path),'result':result,'api_receipts':client.receipts[before:]}
        write_new(output/'episodes'/(case['case_id']+'.json'),row)
        rows.append(row)
        if method in NEURAL:
            original_env = ScientistEnv(tasks[index],config)
            original = collect_episode(model,original_env,'cpu',deterministic=True)
            matched = result['actions']==original['actions'] and result['final_mse']==original_env.private_result['final_mse']
            equivalence.append({'case_id':case['case_id'],'matches_original_collector':matched})
            if not matched:
                raise ValueError('frozen policy adapter changed behavior')
        print(f"{case['case_id']}: {result['status']}",flush=True)
    verify_hashes(ROOT,manifest['sources'])
    for record in models.values(): verify_hashes(ROOT,{record['path']:record['sha256']})
    result = {'manifest_sha256':file_hash(output/'manifest.json'),'cases':rows,
              'summary':study_summary(rows,manifest['mode']),'client_stats':client.stats(),
              'original_policy_equivalence':equivalence,'historical_models_unchanged':True}
    write_new(output/'trace-index.json',trace_index(rows,model_name))
    result['trace_index_sha256']=file_hash(output/'trace-index.json')
    write_new(output/'results.json',result)
    verification = verify_run(output)
    write_new(output/'verification.json',verification)
    return {'output':str(output),**verification,'api':client.stats()}


def verify_run(output):
    output = Path(output).resolve()
    manifest = json.loads((output/'manifest.json').read_text())
    result = json.loads((output/'results.json').read_text())
    if file_hash(output/'manifest.json') != result['manifest_sha256']:
        raise ValueError('manifest hash mismatch')
    verify_hashes(ROOT,manifest['sources'])
    config = PilotConfig(**manifest['config'])
    if manifest['schema_version']!=2 or manifest['mode'] not in ('fixture','hosted') or asdict(config)!=asdict(PilotConfig(final_cultures=128)):
        raise ValueError('unsupported milestone configuration')
    tasks = development_tasks(len(manifest['tasks']),config)
    if manifest['generator_seed'] != 991301 or manifest['final_test_opened'] is not False:
        raise ValueError('unsupported or nondevelopment manifest')
    for task, record in zip(tasks,manifest['tasks']):
        if task.seed!=record['episode_seed'] or mechanism_hash(task.mechanism)!=record['mechanism_hash'] or list(task.target)!=record['target']:
            raise ValueError('task reconstruction mismatch')
    if [c['case_id'] for c in result['cases']] != [c['case_id'] for c in manifest['cases']]:
        raise ValueError('missing, reordered, or extra cases')
    bank = UnfilteredBank(quadrature_family(2),config)
    for case,row in zip(manifest['cases'],result['cases']):
        verify_hashes(output,{row['journal']:row['journal_sha256']})
        replayed = verify_episode(output/row['journal'],tasks[case['task_index']],config,bank)
        if replayed['method'] != case['method'] or canonical(replayed)!=canonical(row['result']):
            raise ValueError('replayed result mismatch')
        events = read_journal(output/row['journal'])
        start = events[0]
        expected_model = manifest['model'] if case['method'].startswith('jev_') or case['method']=='forecast_probe' else None
        policy_name = 'ppo' if case['method'] in HYBRIDS else case['method']
        expected_policy = manifest['models'].get(f'{policy_name}-seed{case["training_seed"]}')
        if (start['schedule'] != manifest['schedules'].get(case['method'],[])
            or start['planner_seed'] != 991303000+case['task_index']
            or start['model'] != expected_model or start['policy_metadata'] != expected_policy
            or start['trace_id'] != (manifest['trace_id'] if expected_model is not None else None)):
            raise ValueError('condition/seed/model/schedule differs from manifest')
        receipts = [{k:v for k,v in e.items() if k!='kind'} for e in events if e['kind'] in ('api_response','api_failure')]
        if canonical(receipts) != canonical(row['api_receipts']):
            raise ValueError('API receipt mismatch')
    if canonical(study_summary(result['cases'],manifest['mode']))!=canonical(result['summary']):
        raise ValueError('summary mismatch')
    verify_hashes(output,{'trace-index.json':result['trace_index_sha256']})
    if canonical(json.loads((output/'trace-index.json').read_text()))!=canonical(trace_index(result['cases'],manifest['model'])):
        raise ValueError('trace index mismatch')
    aggregate = receipt_stats([r for c in result['cases'] for r in c['api_receipts']])
    verify_client_stats(result['client_stats'],[r for c in result['cases'] for r in c['api_receipts']],manifest['api_limits'])
    return {'verified_episodes':len(result['cases']),
            'failed_episodes':sum(c['result']['status']=='failed' for c in result['cases']),
            'mode':manifest['mode'],'hosted_calls':aggregate['attempted_calls'] if manifest['mode']=='hosted' else 0,
            'credential_free_replay':True,'source_hashes_match':True,
            'scope':'stored receipts and simulator outcomes verified; hosted weights not reproduced'}


def live_key():
    key = os.environ.get('TYPESAFE_API_KEY')
    if not key:
        if not sys.stdin.isatty():
            raise ValueError('Set TYPESAFE_API_KEY in this process or run in an interactive terminal for a hidden prompt.')
        with warnings.catch_warnings():
            warnings.simplefilter('error',GetPassWarning)
            try:
                key = getpass('TypeSafe API key: ')
            except (EOFError,GetPassWarning):
                raise ValueError('Hidden prompt unavailable; set TYPESAFE_API_KEY in the process.') from None
    return key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    run = sub.add_parser('run')
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--live',action='store_true',help='One development mechanism, seed20, at most32 API attempts.')
    verify = sub.add_parser('verify')
    verify.add_argument('--run',type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        result = run_study(args.output,live=args.live,count=1 if args.live else 2,key=live_key() if args.live else None) if args.command=='run' else verify_run(args.run)
    except (ValueError,FileExistsError,JevError) as error:
        print(str(error),file=sys.stderr)
        return 1
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
