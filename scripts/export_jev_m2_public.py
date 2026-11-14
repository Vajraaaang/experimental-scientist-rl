"""Export bounded M1/M2 evidence as an explicit, credential-free publication derivative.

Dry inventory is the default. No source file is edited, no weights are copied,
and verification invokes the copied snapshots without credentials or training.
Execution-provenance hashes remain original; only publication replay bindings
are regenerated. The receipt maps every original file hash to its public hash.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.parse import parse_qsl, urlsplit

RECEIPT = 'reports/jev-rl-m2/publication-export.json'
SCHEMA = 'jev-m2-publication-export-v1'
PATH_FIELDS = {'path', 'initial_checkpoint_record', 'output_dir', 'manifest',
               'selected_checkpoint', 'record_path', 'result_path', 'log_path', 'result', 'python',
               'run', 'analysis_source', 'verifier_source'}
PAYLOAD_FIELDS = {'request', 'response', 'request_payload', 'response_payload'}
SECRET_FIELDS = {'authorization', 'api_key', 'apikey', 'access_token', 'refresh_token',
                 'password', 'client_secret', 'private_key', 'credential', 'credentials'}
TOKEN_QUERY_FIELDS = SECRET_FIELDS | {'key', 'token', 'sig', 'signature', 'x-amz-signature',
                                     'x-amz-credential', 'x-goog-signature', 'file_id', 'account_id'}
SOURCE_FILES = (
    'integrations/jev_demo.py', 'integrations/jev_client.py', 'integrations/jev_research.py',
    'integrations/jev_replay.py', 'integrations/jev_candidates.py', 'integrations/jev_m2_episode.py',
    'integrations/m3_ppo_profile.py', 'integrations/ppo_m2_train.py',
    'scripts/jev_rl_study.py', 'scripts/profile_m3_ppo.py', 'scripts/train_ppo_m2.py',
    'scripts/run_ppo_m2_race.py', 'scripts/jev_m2_study.py', 'scripts/ppo_m2_confirm.py',
    'scripts/analyze_jev_m2.py', 'scripts/export_jev_m2_public.py',
    'tests/test_jev_client.py', 'tests/test_jev_research.py', 'tests/test_jev_study.py',
    'tests/test_m3_ppo_profile.py', 'tests/test_ppo_m2_train.py', 'tests/test_ppo_m2_race.py',
    'tests/test_jev_candidates.py', 'tests/test_jev_m2_episode.py', 'tests/test_jev_m2_study.py',
    'tests/test_ppo_m2_confirm.py', 'tests/test_analyze_jev_m2.py', 'tests/test_export_jev_m2_public.py',
    'docs/jev_capabilities_audit.md', 'docs/jev_rl_requirements.md', 'docs/jev_rl_milestone_plan.md',
    'docs/jev_rl_protocol.md', 'docs/jev_rl_m2_protocol.md', 'docs/jev_rl_m2_hosted_protocol.md',
    'docs/jev_rl_m2_task_derivation.md', 'docs/jev_rl_m2_sources_and_design.md',
    'docs/jev_rl_m2_source_receipts.json', 'docs/jev_rl_m2_analysis_registration.json',
)
REPORT_DIRS = (
    'reports/jev-rl-m1/fixture-04', 'reports/jev-rl-m1/hosted-20261113-204144',
    'reports/jev-rl-m1/failure-examples', 'reports/jev-rl-m1/baseline-checks',
    'reports/jev-rl-m2/race-01', 'reports/jev-rl-m2/hardware-sweep',
    'reports/jev-rl-m2/profile-initial-cpu', 'reports/jev-rl-m2/profile-initial-mps',
    'reports/jev-rl-m2/profile-initial-split', 'reports/jev-rl-m2/profile-capacity512',
    'reports/jev-rl-m2/selected-memory-profile', 'reports/jev-rl-m2/confirmation-freeze-02',
    'reports/jev-rl-m2/confirmation-01', 'reports/jev-rl-m2/candidate-fixture-02',
    'reports/jev-rl-m2/candidate-analysis-fixture-02',
)
REPORT_FILES = tuple('reports/jev-rl-m1/' + name for name in (
    'REPORT.md', 'checkout-audit.json', 'doctor.log', 'failure-checks.log', 'failure_checks.py',
    'fixture-04.log', 'hosted-pilot-status.json', 'preservation-before.json',
    'preservation-after.json', 'snapshot-verification.log', 'tests-final.log', 'validation.json',
)) + tuple('reports/jev-rl-m2/' + name for name in (
    'PROGRESS.md', 'candidate-fixture-02.log', 'confirmation-01.log', 'confirmation-replay.log',
    'fixture-plan-selected-02.json', 'hosted-pilot-review.json', 'launch-receipt.json',
    'memory-profile-config.json', 'memory-profile-initial.json', 'memory-profile-scope.json',
    'memory-profile.log', 'memory-profile.time', 'monitor_resources.py', 'pre-race-tests.log',
    'preserved-m1.json', 'profile-sweep.log', 'profile_sweep.py', 'public-protocol-freeze.json',
    'race-01.log', 'resource-monitor.log', 'resource-samples.jsonl', 'resource-summary.json',
    'study-protocol.json',
))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def digest(value):
    return sha256(canonical(value).encode())


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def allowed_file(path):
    path = Path(path)
    return not (any(p in {'.git', '__pycache__', '.venv'} or p.startswith('.env') for p in path.parts)
                or path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.pyc', '.zip'})


def private_url(text, *, source_fixture=False):
    for raw in re.findall(r'https?://[^\s<>"\)]+', text):
        try:
            url = urlsplit(raw.rstrip("'`.,;"))
        except ValueError:
            return True
        host = (url.hostname or '').lower()
        # Non-routable negative-test fixtures are source, not access links.
        if (host.endswith(('.example', '.invalid')) or (source_fixture and host.endswith('.test'))
                or (source_fixture and host == 'chatgpt.com' and url.path.startswith('/c/private-example'))
                or (source_fixture and host == 'chatgpt.com' and url.path == '/library'
                    and all(value == 'private-example' for _, value in parse_qsl(url.query)))):
            continue
        if (url.username or url.password or
                (host in {'chatgpt.com', 'chat.openai.com'} and url.path.startswith(('/c/', '/share/', '/s/', '/library'))) or
                any(key.lower() in TOKEN_QUERY_FIELDS for key, _ in parse_qsl(url.query))):
            return True
    return False


def secret_text(text):
    return bool(re.search(r'(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)', text))


def private_path(text):
    return bool(re.search(r'/(?:Users|home)/[^/\\\s\'"|]+/', text))


def check_payload(value, source_root):
    text = canonical(value)
    if (str(source_root) in text or private_path(text) or re.search(r'[A-Z]:\\\\Users\\\\', text)
            or private_url(text) or secret_text(text)):
        raise ValueError('private information in API payload; publication must not redact payloads')
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower().replace('-', '_') in SECRET_FIELDS and item not in (None, '', False):
                raise ValueError('credential field in API payload; publication must not redact payloads')
            check_payload(item, source_root)
    elif isinstance(value, list):
        for item in value:
            check_payload(item, source_root)


def transform_value(value, source_root, trail=()):
    """Normalize recognized metadata paths; API payloads are opaque and immutable."""
    changes = []
    root = str(source_root)
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if root in key:
                raise ValueError('unrecognized metadata path in object key')
            if key in PAYLOAD_FIELDS:
                check_payload(item, source_root)
                output[key] = item
            else:
                output[key], child = transform_value(item, source_root, trail + (key,))
                changes.extend(child)
        return output, changes
    if isinstance(value, list):
        output = []
        for item in value:
            transformed, child = transform_value(item, source_root, trail + ('*',))
            output.append(transformed)
            changes.extend(child)
        return output, changes
    if isinstance(value, str):
        if private_url(value) or secret_text(value):
            raise ValueError('private URL or credential in metadata')
        if root in value:
            if not trail or trail[-1] not in PATH_FIELDS or not (value == root or value.startswith(root + '/')):
                raise ValueError('unrecognized metadata contains the project root: ' + '.'.join(trail))
            changes.append('relative_project_path:' + '.'.join(trail))
            return value[len(root) + 1:] if value != root else '.', changes
        if private_path(value):
            raise ValueError('unrecognized metadata contains an external private path: ' + '.'.join(trail))
    return value, changes


def transform_bytes(name, raw, source_root):
    compressed = name.endswith('.gz')
    content = gzip.decompress(raw) if compressed else raw
    text = content.decode('utf-8')
    changes = []
    if name.endswith('autonomous_scientist_build_spec.md'):
        first, separator, rest = text.partition('\n')
        if private_url(first):
            if not separator:
                raise ValueError('private handoff line has no specification body')
            text = rest
            changes.append('remove_private_handoff_first_line')
    jsonl = name.endswith(('.jsonl', '.jsonl.gz'))
    is_json = name.endswith(('.json', '.json.gz'))
    if jsonl:
        records = [json.loads(line) for line in text.splitlines()]
        chained = bool(records and isinstance(records[0], dict) and {'seq', 'previous_sha256', 'sha256', 'event'} <= records[0].keys())
        original_previous = public_previous = '0' * 64
        public_records = []
        for seq, record in enumerate(records):
            if chained:
                signed = {k: v for k, v in record.items() if k != 'sha256'}
                if record['seq'] != seq or record['previous_sha256'] != original_previous or digest(signed) != record['sha256']:
                    raise ValueError('original JSONL hash chain is invalid')
                original_previous = record['sha256']
            updated, child = transform_value(record, source_root)
            changes.extend(child)
            if chained:
                updated['previous_sha256'] = public_previous
                updated['sha256'] = digest({k: v for k, v in updated.items() if k != 'sha256'})
                public_previous = updated['sha256']
            public_records.append(updated)
        if changes:
            text = ''.join(canonical(record) + '\n' for record in public_records)
            if chained:
                changes.append('rebuild_jsonl_hash_chain')
    elif is_json:
        updated, child = transform_value(json.loads(text), source_root)
        changes.extend(child)
        if child:
            text = json_bytes(updated).decode()
    else:
        if private_url(text, source_fixture=name.endswith('.py')) or secret_text(text):
            raise ValueError('private URL or credential in text artifact')
        if str(source_root) in text:
            if name.endswith('.py'):
                raise ValueError('runtime source contains a private project path; source must stay exact')
            text = text.replace(str(source_root), '<PROJECT>')
            changes.append('project_marker_in_text_metadata')
        if private_path(text):
            raise ValueError('external private path remains in text artifact')
    if not changes:
        return raw, []
    content = text.encode()
    return gzip.compress(content, mtime=0) if compressed else content, sorted(set(changes))


def validate_original_bindings(source, original):
    """Never turn a damaged original seal or file binding into a valid derivative."""
    def load(name):
        return json.loads((source / name).read_bytes())

    def require(name, expected):
        if original.get(name, {}).get('sha256') != expected:
            raise ValueError('original artifact hash mismatch: ' + name)

    for base in ('reports/jev-rl-m1/fixture-04', 'reports/jev-rl-m1/hosted-20261113-204144',
                 'reports/jev-rl-m2/candidate-fixture-02'):
        manifest_name = base + '/manifest.json'
        if manifest_name not in original:
            continue
        manifest, result = load(manifest_name), load(base + '/results.json')
        require(manifest_name, result['manifest_sha256'])
        for name, expected in manifest['sources'].items():
            require(base + '/source_snapshot/' + name, expected)
        if 'trace_index_sha256' in result:
            require(base + '/trace-index.json', result['trace_index_sha256'])
        for row in result['cases']:
            if row['journal'] is not None:
                require(base + '/' + row['journal'], row['journal_sha256'])
            if load(base + '/episodes/' + row['case_id'] + '.json') != row:
                raise ValueError('original episode row differs from recorded results')
        if 'plan_sha256' in manifest:
            require(base + '/plan.json', manifest['plan_sha256'])
            plan = load(base + '/plan.json')
            for key, value in plan.items():
                if key != 'kind' and manifest[key] != value:
                    raise ValueError('original plan/manifest binding mismatch')
            for key, value in {'plan_content_sha256': plan, 'registered_inputs_sha256': plan['registered_inputs'],
                               'models_sha256': plan['models'], 'runtime_sources_sha256': plan['sources']}.items():
                if manifest['admission'][key] != digest(value):
                    raise ValueError('original admission hash mismatch')
    for base in ('reports/jev-rl-m2/confirmation-freeze-02', 'reports/jev-rl-m2/confirmation-01'):
        name = base + '/freeze.json'
        if name not in original:
            continue
        freeze = load(name)
        if freeze['seal_sha256'] != digest({k: v for k, v in freeze.items() if k != 'seal_sha256'}):
            raise ValueError('original confirmation seal hash mismatch')
        cohort = freeze['selected_ppo']['cohort']
        if freeze['selected_ppo_records_content_sha256'] != digest(freeze['cohorts'][cohort]['records']):
            raise ValueError('original selected-record hash mismatch')
        if base + '/run-manifest.json' in original:
            manifest, result = load(base + '/run-manifest.json'), load(base + '/results.json')
            require(name, manifest['freeze_sha256'])
            require(base + '/run-manifest.json', result['run_manifest_sha256'])
            for name, expected in manifest['sources'].items():
                require(base + '/source_snapshot/' + name, expected)
            if 'task_manifest_sha256' in manifest:
                require(base + '/task-manifest.json.gz', manifest['task_manifest_sha256'])
            for name, expected in result['artifacts'].items():
                require(base + '/' + name, expected)
    analysis_name = 'reports/jev-rl-m2/candidate-analysis-fixture-02/summary.json'
    if analysis_name in original:
        analysis = load(analysis_name)
        if analysis['schema'] != 'jev-m2-analysis-v1':
            raise ValueError('unsupported analysis publication schema')
        provenance = analysis['provenance']
        run = 'reports/jev-rl-m2/candidate-fixture-02'
        if (source / provenance['run']).resolve() != source / run:
            raise ValueError('original analysis does not bind the selected fixture')
        require('scripts/analyze_jev_m2.py', provenance['analysis_source_sha256'])
        require(run + '/manifest.json', provenance['manifest_sha256'])
        require(run + '/results.json', provenance['results_sha256'])
        for name, expected in provenance['input_sha256'].items():
            require(run + '/' + name, expected)
        for name, expected in provenance['verification']['verified_input_sha256'].items():
            require(run + '/' + name, expected)
        if digest(analysis['analysis_contract']) != provenance['analysis_contract_sha256']:
            raise ValueError('original analysis contract hash mismatch')
        manifest = load(run + '/manifest.json')
        if (provenance['study_source_sha256'] != manifest['sources'] or
                provenance['checkpoint_metadata_only'] != manifest['models']):
            raise ValueError('original analysis source/model metadata mismatch')


def rebind_bundles(files, transformations):
    """Rebind only the schemas that the unmodified snapshot verifiers consume."""
    def load(name):
        return json.loads(files[name])

    def save(name, value, reason):
        if load(name) != value:
            files[name] = json_bytes(value)
            transformations[name].append(reason)

    for base in ('reports/jev-rl-m1/fixture-04', 'reports/jev-rl-m1/hosted-20261113-204144',
                 'reports/jev-rl-m2/candidate-fixture-02'):
        manifest_name, result_name = base + '/manifest.json', base + '/results.json'
        if manifest_name not in files:
            continue
        manifest = load(manifest_name)
        for name in manifest['sources']:
            manifest['sources'][name] = sha256(files[base + '/source_snapshot/' + name])
        plan_name = base + '/plan.json'
        if plan_name in files:
            plan = load(plan_name)
            plan['sources'] = manifest['sources']
            for name in plan.get('prompt_sources', {}):
                plan['prompt_sources'][name] = plan['sources'][name]
            save(plan_name, plan, 'rebind_plan_source_hashes')
            for key, value in plan.items():
                if key != 'kind':
                    manifest[key] = value
            manifest['plan_sha256'] = sha256(files[plan_name])
            for key, value in {'plan_content_sha256': plan, 'registered_inputs_sha256': plan['registered_inputs'],
                               'models_sha256': plan['models'], 'runtime_sources_sha256': plan['sources']}.items():
                manifest['admission'][key] = digest(value)
        save(manifest_name, manifest, 'rebind_public_manifest_and_admission')
        result = load(result_name)
        result['manifest_sha256'] = sha256(files[manifest_name])
        for row in result['cases']:
            if row['journal'] is not None:
                row['journal_sha256'] = sha256(files[base + '/' + row['journal']])
            row_name = base + '/episodes/' + row['case_id'] + '.json'
            save(row_name, row, 'rebind_public_journal_file_hash')
        save(result_name, result, 'rebind_public_result_manifest_and_journals')

    for base in ('reports/jev-rl-m2/confirmation-freeze-02', 'reports/jev-rl-m2/confirmation-01'):
        name = base + '/freeze.json'
        if name not in files:
            continue
        freeze = load(name)
        cohort = freeze['selected_ppo']['cohort']
        freeze['selected_ppo_records_content_sha256'] = digest(freeze['cohorts'][cohort]['records'])
        freeze['seal_sha256'] = digest({k: v for k, v in freeze.items() if k != 'seal_sha256'})
        save(name, freeze, 'reseal_public_freeze_and_selected_records')
        manifest_name, result_name = base + '/run-manifest.json', base + '/results.json'
        if manifest_name in files:
            manifest = load(manifest_name)
            manifest['freeze_sha256'] = sha256(files[name])
            save(manifest_name, manifest, 'rebind_public_freeze_file_hash')
            result = load(result_name)
            result['run_manifest_sha256'] = sha256(files[manifest_name])
            for artifact in result['artifacts']:
                result['artifacts'][artifact] = sha256(files[base + '/' + artifact])
            save(result_name, result, 'rebind_public_run_manifest_and_artifacts')

    analysis_name = 'reports/jev-rl-m2/candidate-analysis-fixture-02/summary.json'
    if analysis_name in files:
        analysis = load(analysis_name)
        provenance = analysis['provenance']
        run = 'reports/jev-rl-m2/candidate-fixture-02'
        for name in provenance['input_sha256']:
            provenance['input_sha256'][name] = sha256(files[run + '/' + name])
        for name in provenance['verification']['verified_input_sha256']:
            provenance['verification']['verified_input_sha256'][name] = sha256(files[run + '/' + name])
        provenance['manifest_sha256'] = sha256(files[run + '/manifest.json'])
        provenance['results_sha256'] = sha256(files[run + '/results.json'])
        manifest = load(run + '/manifest.json')
        provenance['study_source_sha256'] = manifest['sources']
        provenance['checkpoint_metadata_only'] = manifest['models']
        save(analysis_name, analysis, 'rebind_fixture_analysis_to_public_inputs')


def default_paths(source):
    paths = set(SOURCE_FILES + REPORT_FILES)
    for directory in REPORT_DIRS:
        folder = source / directory
        if not folder.is_dir():
            raise FileNotFoundError('required evidence directory missing: ' + directory)
        paths.update(str(p.relative_to(source)) for p in folder.rglob('*') if p.is_file() or p.is_symlink())
    return sorted(paths)


def snapshot_replays(target):
    cases = (
        ('m1_fixture', 'reports/jev-rl-m1/fixture-04', 'jev_rl_study.py', 'verify', '--run', 100),
        ('m1_hosted', 'reports/jev-rl-m1/hosted-20261113-204144', 'jev_rl_study.py', 'verify', '--run', 15),
        ('m2_candidate', 'reports/jev-rl-m2/candidate-fixture-02', 'jev_m2_study.py', 'verify', '--run', 75),
        ('m2_confirmation', 'reports/jev-rl-m2/confirmation-01', 'ppo_m2_confirm.py', 'replay', '--output', 5632),
    )
    environment = {'PATH': os.defpath, 'LANG': 'en_US.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1',
                   'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
    verification = []
    for label, run, script, command, flag, count in cases:
        relative_script = run + '/source_snapshot/scripts/' + script
        argv = [sys.executable, '-B', str(target / relative_script), command, flag, str(target / run)]
        started = time.monotonic()
        completed = subprocess.run(argv, cwd=target, env=environment, capture_output=True, text=True, timeout=900)
        output = completed.stdout.replace(str(target), '<PUBLIC_PROJECT>')
        errors = completed.stderr.replace(str(target), '<PUBLIC_PROJECT>')
        try:
            result = json.loads(output)
        except ValueError:
            result = None
        actual = result.get('verified_episodes', result.get('episodes_replayed')) if isinstance(result, dict) else None
        verification.append({'name': label, 'command': ['python', '-B', relative_script, command, flag, run],
            'exit_code': completed.returncode, 'expected_episodes': count, 'observed_episodes': actual,
            'passed': completed.returncode == 0 and actual == count,
            'wall_seconds': time.monotonic() - started, 'result': result, 'stderr': errors,
            'stdout': output if result is None else None})
        print(label + ': ' + ('passed' if verification[-1]['passed'] else 'FAILED'), flush=True)
    analysis_folder = target / 'reports/jev-rl-m2/candidate-analysis-fixture-02'
    if analysis_folder.exists():
        run = 'reports/jev-rl-m2/candidate-fixture-02'
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='jev-public-analysis-') as temporary:
            generated = Path(temporary) / 'analysis'
            argv = [sys.executable, '-B', str(target / 'scripts/analyze_jev_m2.py'),
                    '--run', str(target / run), '--output', str(generated)]
            completed = subprocess.run(argv, cwd=target, env=environment, capture_output=True, text=True, timeout=900)
            comparisons = {'summary_identical': False, 'report_identical': False}
            if completed.returncode == 0:
                summary, _ = transform_value(json.loads((generated / 'summary.json').read_text()), target)
                comparisons = {'summary_identical': summary == json.loads((analysis_folder / 'summary.json').read_text()),
                               'report_identical': (generated / 'REPORT.md').read_bytes() == (analysis_folder / 'REPORT.md').read_bytes(),
                               'analysis_contract_sha256': summary['provenance']['analysis_contract_sha256'],
                               'mode': summary['mode']}
            errors = completed.stderr.replace(str(target), '<PUBLIC_PROJECT>').replace(temporary, '<ANALYSIS_RECHECK>')
            verification.append({'name': 'm2_fixture_analysis', 'command': ['python', '-B', 'scripts/analyze_jev_m2.py',
                '--run', run, '--output', '<TEMPORARY_ANALYSIS>'], 'exit_code': completed.returncode,
                'passed': completed.returncode == 0 and comparisons['summary_identical'] and comparisons['report_identical'],
                'wall_seconds': time.monotonic() - started, 'result': comparisons, 'stderr': errors})
            print('m2_fixture_analysis: ' + ('passed' if verification[-1]['passed'] else 'FAILED'), flush=True)
    return verification


def export_paths(source, target, paths, *, apply=False, replace_owned=False, verify=False):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('source and destination must be separate trees')
    original, files, transformations, excluded = {}, {}, {}, []
    for name in sorted(set(paths)):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('export path must be repository relative')
        path = source / name
        if path.is_symlink():
            raise ValueError('symlinks are not publication inputs: ' + name)
        if not allowed_file(relative):
            excluded.append({'path': name, 'bytes': path.stat().st_size, 'reason': 'weight_cache_or_private_file'})
            continue
        raw = path.read_bytes()
        original[name] = {'sha256': sha256(raw), 'bytes': len(raw)}
        try:
            files[name], transformations[name] = transform_bytes(name, raw, source)
        except (ValueError, UnicodeError) as error:
            raise ValueError(name + ': ' + str(error)) from None
    validate_original_bindings(source, original)
    rebind_bundles(files, transformations)
    receipt_path = target / RECEIPT
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if previous is not None and previous.get('schema') != SCHEMA:
        raise FileExistsError('unrelated publication receipt already exists')
    if previous is not None and apply and not replace_owned:
        raise FileExistsError('updating an existing export requires --replace-owned')
    owned = {item['path']: item for item in previous['files'] if item['owned']} if previous else {}
    records = []
    for name, public in files.items():
        path = target / name
        is_owned = True
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != target.parent):
            raise ValueError('destination may not contain symlinks: ' + name)
        if path.exists():
            present = sha256(path.read_bytes())
            old = owned.get(name)
            if old and present != old['public_sha256']:
                raise FileExistsError('previously exported file was externally modified: ' + name)
            if present != sha256(public):
                if old is None:
                    raise FileExistsError('refusing to overwrite unrelated existing file: ' + name)
                if not replace_owned:
                    raise FileExistsError('updating an owned export requires --replace-owned: ' + name)
            is_owned = old is not None
        records.append({'path': name, 'original_sha256': original[name]['sha256'],
                        'public_sha256': sha256(public), 'original_bytes': original[name]['bytes'],
                        'public_bytes': len(public), 'owned': is_owned,
                        'transformations': sorted(set(transformations[name]))})
    receipt = {'schema': SCHEMA, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'mode': 'publication_derivative', 'source_root': '<PROJECT>', 'destination_root': '<PUBLIC_PROJECT>',
        'scientific_values_actions_observations_preserved': True,
        'api_request_response_payloads_canonically_unchanged': True,
        'runtime_snapshot_code_unchanged': True, 'credentials_used': False, 'checkpoints_exported': False,
        'hash_binding_semantics': {
            'rebound': ['public source snapshot inventory after private handoff line removal',
                        'public plan/admission/manifest/journal/result bindings',
                        'public confirmation selected-record digest, freeze seal and run manifest',
                        'fixture-only analysis provenance bound to public candidate inputs'],
            'original_execution_provenance': ['checkpoint hashes', 'historical manifest and exclusion hashes',
                'source_sha256 and race_inputs recorded at execution', 'm2_provenance.result_sha256',
                'original_spec_sha256 and original_spec.sha256', 'original registry/schedule input hashes'],
            'meaning': 'Execution provenance continues to name the immutable local originals. Use each file original_sha256/public_sha256 mapping for publication derivatives; historical hashes are not assertions about redacted public bytes.'},
        'limitations': ['No checkpoint inference or full retraining is reproduced.',
                        'Hosted model weights and measured latency are not reproduced.',
                        'Same-machine replay does not establish independent-machine reproduction.'],
        'files': records, 'excluded': excluded,
        'inventory': {'included_files': len(records), 'original_bytes': sum(x['original_bytes'] for x in records),
                      'public_bytes': sum(x['public_bytes'] for x in records),
                      'excluded_files': len(excluded), 'excluded_bytes': sum(x['bytes'] for x in excluded),
                      'largest_public_file_bytes': max((x['public_bytes'] for x in records), default=0),
                      'transformed_files': sum(bool(x['transformations']) for x in records),
                      'reused_existing_files': sum(not x['owned'] for x in records)},
        'privacy_review': {'private_handoff_urls_removed': sum('remove_private_handoff_first_line' in x['transformations'] for x in records),
                           'unresolved_private_paths': [], 'unexpected_private_payloads': [],
                           'scope': 'Selected export inputs only; omitted local files and existing public history are not rewritten.'},
        'verification': {'status': 'not_run', 'runs': []}}
    if not apply:
        return receipt
    # Recheck source hashes and all destinations before the first write.
    for name, item in original.items():
        if sha256((source / name).read_bytes()) != item['sha256']:
            raise ValueError('source changed during export: ' + name)
    for record in records:
        if not record['owned']:
            continue
        path = target / record['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and sha256(path.read_bytes()) == record['public_sha256']:
            continue
        with path.open('wb' if path.exists() else 'xb') as stream:
            stream.write(files[record['path']])
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(json_bytes(receipt))
    if verify:
        runs = snapshot_replays(target)
        receipt['verification'] = {'status': 'passed' if all(x['passed'] for x in runs) else 'failed',
            'credentials_excluded_from_environment': True, 'source': 'copied source_snapshot scripts', 'runs': runs}
        receipt_path.write_bytes(json_bytes(receipt))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Write the reviewed derivative; otherwise inventory only.')
    parser.add_argument('--replace-owned', action='store_true', help='Update only unchanged files owned by an earlier export.')
    parser.add_argument('--verify', action='store_true', help='Run all four copied snapshot verifiers after export.')
    args = parser.parse_args(argv)
    if args.verify and not args.apply:
        parser.error('--verify requires --apply')
    receipt = export_paths(args.source, args.destination, default_paths(args.source.resolve()),
                           apply=args.apply, replace_owned=args.replace_owned, verify=args.verify)
    print(json.dumps({'apply': args.apply, 'inventory': receipt['inventory'], 'privacy_review': receipt['privacy_review'],
                      'verification': receipt['verification'], 'receipt': RECEIPT if args.apply else None}, indent=2))
    return 1 if receipt['verification']['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
