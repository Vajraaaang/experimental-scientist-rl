"""Study manifests must bind independent tasks and the actual measured artifacts."""
import importlib
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def study():
    assert importlib.util.find_spec('scripts.jev_rl_study') is not None, 'study runner is missing'
    return importlib.import_module('scripts.jev_rl_study')


def test_new_development_tasks_are_stable_and_disjoint():
    from experimentlab.envs import PilotConfig
    from experimentlab.tasks import mechanism_hash
    s=study(); tasks=s.development_tasks(2,PilotConfig(final_cultures=128))
    assert [t.seed for t in tasks] == [991302000,991302001]
    assert len(set(mechanism_hash(t.mechanism) for t in tasks)) == 2
    assert [mechanism_hash(t.mechanism) for t in tasks] == [mechanism_hash(t.mechanism) for t in s.development_tasks(2,PilotConfig(final_cultures=128))]
    with pytest.raises(ValueError): s.development_tasks(0,PilotConfig())
    with pytest.raises(ValueError): s.development_tasks(513,PilotConfig())


def test_study_writer_never_overwrites(tmp_path):
    s=study(); path=tmp_path/'receipt.json'
    s.write_new(path,{'first':True})
    with pytest.raises(FileExistsError): s.write_new(path,{'second':True})
    assert 'first' in path.read_text()


def test_sources_fail_closed_when_changed(tmp_path):
    s=study(); source=tmp_path/'source.py'; source.write_text('original')
    hashes={'source.py':s.file_hash(source)}
    s.verify_hashes(tmp_path,hashes)
    source.write_text('changed')
    with pytest.raises(ValueError,match='source'):
        s.verify_hashes(tmp_path,hashes)


def test_accounting_refuses_attempt_beyond_frozen_cap():
    from dataclasses import asdict
    from integrations.jev_client import JevClient,CallLimits
    from integrations.jev_research import FixtureTransport
    s=study()
    assert hasattr(s,'validate_api_accounting'),'cumulative accounting verifier is missing'
    client=JevClient(FixtureTransport(),expected_model='jev-fixture')
    request={'model':'jev-fixture','state':{},'questions':{'test':{'type':'noul','instructions':'Test?'}}}
    client(request);client(request)
    with pytest.raises(ValueError,match='cap'):
        s.validate_api_accounting(client.receipts,asdict(CallLimits(max_calls=1)))


def test_trace_index_connects_case_model_request_and_cost():
    from integrations.jev_client import JevClient
    from integrations.jev_research import FixtureTransport
    s=study()
    assert hasattr(s,'trace_index'),'cross-episode trace index is missing'
    client=JevClient(FixtureTransport(),expected_model='jev-fixture')
    client({'model':'jev-fixture','state':{},'questions':{'test':{'type':'noul','instructions':'Test?'}}})
    cases=[{'case_id':'test-case','result':{'method':'jev_only'},'api_receipts':client.receipts}]
    index=s.trace_index(cases,'jev-fixture')
    assert index['calls'][0]['case_id']=='test-case'
    assert index['calls'][0]['trace_id']==client.trace_id
    assert index['calls'][0]['requested_model']=='jev-fixture'
    assert index['calls'][0]['resolved_model']=='jev-fixture'
    assert index['calls'][0]['provider_request_id'] is None
    assert index['calls'][0]['input_tokens']==0
    assert index['calls'][0]['request_sha256']


def test_client_stats_cannot_contradict_receipts():
    from dataclasses import asdict
    from integrations.jev_client import JevClient,CallLimits
    from integrations.jev_research import FixtureTransport
    s=study()
    assert hasattr(s,'verify_client_stats'),'complete client-stat verification is missing'
    client=JevClient(FixtureTransport(),expected_model='jev-fixture')
    client({'model':'jev-fixture','state':{},'questions':{'test':{'type':'noul','instructions':'Test?'}}})
    stats=client.stats()
    s.verify_client_stats(stats,client.receipts,asdict(CallLimits()))
    for field in ('estimated_usd','known_estimated_usd','reserved_estimated_usd','accounted_estimated_usd','latency_seconds','usage_complete'):
        changed={**stats,field:False if field=='usage_complete' else -123}
        with pytest.raises(ValueError,match='accounting'):
            s.verify_client_stats(changed,client.receipts,asdict(CallLimits()))


def test_zero_attempt_refusal_accounting_remains_verifiable():
    from dataclasses import asdict
    from integrations.jev_client import JevClient,CallLimits,JevError
    from integrations.jev_research import FixtureTransport
    s=study();limits=CallLimits(max_calls=0)
    client=JevClient(FixtureTransport(),expected_model='jev-fixture',limits=limits)
    with pytest.raises(JevError,match='call_limit'):
        client({'model':'jev-fixture','state':{},'questions':{'test':{'type':'noul','instructions':'Test?'}}})
    s.verify_client_stats(client.stats(),client.receipts,asdict(limits))
