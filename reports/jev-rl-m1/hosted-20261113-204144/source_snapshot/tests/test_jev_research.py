"""Prospective ordering, public boundaries, and independent replay contracts."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experimentlab.envs import PilotConfig, ScientistEnv
from experimentlab.inference import PredictionBank
from experimentlab.tasks import finite_family, make_task


def research():
    assert importlib.util.find_spec('integrations.jev_research') is not None, 'research loop is missing'
    return importlib.import_module('integrations.jev_research')


@pytest.fixture
def setup():
    config = PilotConfig(final_cultures=128)
    bank = PredictionBank(finite_family(), config)
    task = make_task(finite_family()[0], 991302000, config)
    return config, bank, task


def test_public_packet_omits_unpurchased_and_private_data(setup):
    r = research(); config, bank, task = setup
    env = ScientistEnv(task, config); obs, _ = env.reset()
    obs, *_ = env.step(4)
    obs['private_sentinel'] = {'W': 'must never be sent'}
    state = r.public_state(obs, env.actions)
    payload = json.dumps(state, allow_nan=False)
    assert 'private_sentinel' not in payload and 'must never' not in payload
    assert set(state['history'][-1]['measurements']) == {'X4', 'X5'}
    assert 'seed' not in state and 'mechanism' not in state
    assert state['remaining_budget'] == obs['budget_ticks'] / 20


def test_reference_forecast_conditions_on_observed_baseline(setup):
    r = research(); config, bank, task = setup
    env = ScientistEnv(task, config); obs, _ = env.reset()
    obs['history'][0]['readouts'][:, 4] = .52
    table = r.reference_forecasts(obs, bank)
    expected = ndtr((.5 - .52) / (.02 / np.sqrt(8)))
    assert table['a0_x4']['threshold'] == pytest.approx(.52)
    assert table['a0_x4']['probabilities']['posterior'] == pytest.approx(expected)
    assert table['a0_x4']['probabilities']['posterior'] < .01
    assert len(table) == 72


def test_candidate_forecasts_follow_affordable_mask_and_keep_commitments(setup):
    r = research(); config, bank, task = setup
    env = ScientistEnv(task, config); obs, _ = env.reset()
    obs['action_mask'][:36] = False
    assert r.reference_forecasts(obs, bank) == {}
    req = r.decision_request(obs, env.actions, 'jev-fixture')
    assert set(req['questions']['next_action']['criteria']) == {f'action_{i}' for i in range(36, 45)}


def test_forecast_is_durable_before_step_and_replay_matches(setup, tmp_path):
    r = research(); config, bank, task = setup
    journal = r.Journal(tmp_path/'trace.jsonl')
    class GuardedEnv(ScientistEnv):
        def step(self, action):
            events = r.read_journal(journal.path)
            assert events[-1]['kind'] == 'action'
            assert any(e['kind'] == 'forecast_frozen' for e in events)
            return super().step(action)
    result = r.run_episode(GuardedEnv(task, config), 'fixed', bank, journal, schedule=[4,12,31])
    journal.close()
    assert result['status'] == 'completed' and result['assays'] == 3
    assert len(result['forecasts']) == 6
    assert result['return_'] == pytest.approx(-100*result['final_mse']-.01*result['variable_cost']-.05*result['effort'])
    assert r.verify_episode(journal.path, task, config, bank) == result


def test_immediate_commit_is_legal_and_has_no_fake_forecast_score(setup, tmp_path):
    r = research(); config, bank, task = setup
    with r.Journal(tmp_path/'trace.jsonl') as journal:
        result = r.run_episode(ScientistEnv(task, config), 'prior', bank, journal)
    assert result['assays'] == 0 and result['forecasts'] == []
    assert result['immediate_stop'] is True


def test_hybrid_and_compute_controls_supply_only_intended_forecasts(setup):
    r = research(); config, bank, task = setup
    env = ScientistEnv(task, config); obs, _ = env.reset()
    table = r.reference_forecasts(obs, bank)
    for row in table.values(): row['probabilities']['jev'] = .9
    policy = np.ones(45)/45
    for mode, expected in [('jev_rl',.9),('jev_rl_chance',.5)]:
        req = r.decision_request(obs, env.actions, 'jev-fixture', r.shown_forecasts(table, mode), policy)
        assert req['state']['forecasts']['a0_x4'] == expected
        assert len(req['state']['rl_action_probabilities']) == 45
        assert 'posterior' not in json.dumps(req)
    hidden = r.decision_request(obs, env.actions, 'jev-fixture', r.shown_forecasts(table,'jev_rl_hidden'), policy)
    assert 'forecasts' not in hidden['state']


def test_journal_detects_tampering_and_refuses_overwrite(setup, tmp_path):
    r = research(); config, bank, task = setup; path = tmp_path/'trace.jsonl'
    with r.Journal(path) as journal:
        r.run_episode(ScientistEnv(task, config), 'fixed', bank, journal, schedule=[4])
    with pytest.raises(FileExistsError): r.Journal(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[-1]['event']['result']['final_mse'] = 0
    path.write_text('\n'.join(json.dumps(record) for record in records)+'\n')
    with pytest.raises(ValueError, match='hash'):
        r.verify_episode(path, task, config, bank)


def test_rehashed_false_result_is_detected_by_simulator(setup, tmp_path):
    r = research(); config, bank, task = setup; path = tmp_path/'trace.jsonl'
    with r.Journal(path) as journal:
        r.run_episode(ScientistEnv(task, config), 'fixed', bank, journal, schedule=[4])
    events = r.read_journal(path)
    events[-1]['result']['final_mse'] = 0
    altered = tmp_path/'altered.jsonl'
    with r.Journal(altered) as journal:
        for event in events: journal.append(event)
    with pytest.raises(ValueError, match='result'):
        r.verify_episode(altered, task, config, bank)


def test_calibration_includes_endpoints_and_failure_denominator():
    r = research()
    rows = [{'probabilities': {'jev': p}, 'outcome': y} for p,y in [(0,0),(1,1),(.5,1)]]
    summary = r.forecast_summary(rows)
    assert summary['jev']['count'] == 3
    assert summary['jev']['brier'] == pytest.approx(1/12)
    assert sum(b['count'] for b in summary['jev']['reliability_bins']) == 3
    methods = r.summarize_results([{'method':'jev_only','status':'failed','failure':'timeout','forecasts':[]}])
    assert methods['jev_only']['failed'] == 1 and methods['jev_only']['completed'] == 0
    assert methods['jev_only']['mean_final_mse'] is None


def test_fixture_hosted_loop_has_both_receipts_and_replays(setup, tmp_path):
    r = research(); config, bank, task = setup
    from integrations.jev_client import JevClient
    client = JevClient(r.FixtureTransport(), expected_model='jev-fixture')
    path = tmp_path/'fixture.jsonl'
    with r.Journal(path) as journal:
        result = r.run_episode(ScientistEnv(task, config), 'jev_only', bank, journal, client=client)
    assert result['status'] == 'completed'
    events = r.read_journal(path)
    assert sum(e['kind'] == 'api_request' for e in events) == client.stats()['attempted_calls']
    assert r.verify_episode(path, task, config, bank) == result


def test_api_failure_retains_episode_and_replay(setup, tmp_path):
    r = research(); config, bank, task = setup
    from integrations.jev_client import JevClient
    def fail(request): raise TimeoutError('private transport text')
    client = JevClient(fail, expected_model='jev-fixture')
    path = tmp_path/'failure.jsonl'
    with r.Journal(path) as journal:
        result = r.run_episode(ScientistEnv(task, config), 'jev_only', bank, journal, client=client)
    assert result['status'] == 'failed' and result['final_mse'] is None
    assert 'private transport' not in path.read_text()
    assert r.verify_episode(path, task, config, bank) == result


def test_full_neural_adapter_matches_original_collector(setup, tmp_path):
    import torch
    from experimentlab.learning import ActorCritic, collect_episode
    from experimentlab.cost_policy import PublicCostPolicy
    r = research(); config, bank, task = setup
    torch.manual_seed(123)
    model = PublicCostPolicy(ActorCritic(kind='recurrent'), dose_penalty=16)
    old_env = ScientistEnv(task, config)
    old = collect_episode(model, old_env, 'cpu', deterministic=True)
    path = tmp_path/'ppo.jsonl'
    with r.Journal(path) as journal:
        result = r.run_episode(ScientistEnv(task,config),'ppo',bank,journal,policy=r.FrozenPolicy(model))
    assert result['actions'] == old['actions']
    assert result['final_mse'] == old_env.private_result['final_mse']
    assert r.verify_episode(path, task, config, bank) == result


def test_hybrid_cannot_silently_omit_rl_policy(setup, tmp_path):
    r = research(); config, bank, task = setup
    from integrations.jev_client import JevClient
    with r.Journal(tmp_path/'missing-policy.jsonl') as journal:
        with pytest.raises(ValueError, match='policy'):
            r.run_episode(ScientistEnv(task,config),'jev_rl',bank,journal,
                          client=JevClient(r.FixtureTransport(),expected_model='jev-fixture'))


def test_replay_rejects_jev_only_relabelled_as_hybrid(setup, tmp_path):
    r = research(); config, bank, task = setup
    from integrations.jev_client import JevClient
    path = tmp_path/'only.jsonl'
    with r.Journal(path) as journal:
        r.run_episode(ScientistEnv(task,config),'jev_only',bank,journal,
                      client=JevClient(r.FixtureTransport(),expected_model='jev-fixture'))
    events=r.read_journal(path)
    events[0]['method']='jev_rl'
    events[-1]['result']['method']='jev_rl'
    altered=tmp_path/'fake-hybrid.jsonl'
    with r.Journal(altered) as journal:
        for event in events: journal.append(event)
    with pytest.raises(ValueError,match='advice'):
        r.verify_episode(altered,task,config,bank)


def test_failed_attempt_costs_and_latency_remain_in_summary():
    r=research()
    complete={'method':'jev_only','status':'completed','failure':None,'forecasts':[],
              'final_mse':.1,'effort':0.,'return_':-10.,'variable_cost':0.,'total_assay_cost':3.,
              'assays':0,'online_seconds':1.,'immediate_stop':True}
    failed={**complete,'status':'failed','failure':'timeout','final_mse':None,'effort':None,
            'return_':None,'variable_cost':7.,'total_assay_cost':10.,'assays':3,'online_seconds':21.}
    result=r.summarize_results([complete,failed])['jev_only']
    assert result['attempt_metrics']['variable_cost'] == {'count':2,'mean':3.5,'total':7.}
    assert result['attempt_metrics']['online_seconds'] == {'count':2,'mean':11.,'total':22.}
    assert result['mean_final_mse'] == .1 and result['completed']==1 and result['failed']==1


def fixture_events(setup, tmp_path):
    r=research(); config,bank,task=setup
    from integrations.jev_client import JevClient
    path=tmp_path/'source.jsonl'
    with r.Journal(path) as journal:
        r.run_episode(ScientistEnv(task,config),'jev_only',bank,journal,
                      client=JevClient(r.FixtureTransport(),expected_model='jev-fixture'))
    return r.read_journal(path)


@pytest.mark.parametrize('mutation',['duplicate_choice','failure_then_retry','false_receipt'])
def test_replay_rejects_extra_calls_and_false_api_receipts(setup,tmp_path,mutation):
    import copy
    r=research(); config,bank,task=setup
    events=fixture_events(setup,tmp_path)
    if mutation=='duplicate_choice':
        i=next(i for i,e in enumerate(events) if e['kind']=='api_request' and 'next_action' in e['request']['questions'])
        events[i:i]=copy.deepcopy(events[i:i+2])
    elif mutation=='failure_then_retry':
        i=next(i for i,e in enumerate(events) if e['kind']=='api_request')
        failure={**copy.deepcopy(events[i+1]),'kind':'api_failure','status':'failed','code':'timeout','response':None}
        events[i:i]=[copy.deepcopy(events[i]),failure]
        events[-1]['result']['forecast_coverage']['jev_requested_events'] += len(events[i]['request']['questions'])
    else:
        i=next(i for i,e in enumerate(events) if e['kind']=='api_response')
        events[i].update(attempted=False,status='failed',usage={'input_tokens':-1,'output_tokens':-1},request_bytes=-1)
    path=tmp_path/'modified.jsonl'
    with r.Journal(path) as journal:
        for event in events: journal.append(event)
    with pytest.raises(ValueError): r.verify_episode(path,task,config,bank)
