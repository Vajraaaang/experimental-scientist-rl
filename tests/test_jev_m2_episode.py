import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experimentlab.envs import PilotConfig,ScientistEnv
from experimentlab.tasks import finite_family,make_task
from experimentlab.inference import PredictionBank
from integrations.jev_client import JevClient,CallLimits
from integrations.jev_research import FixtureTransport,Journal,read_journal


def candidate():
    assert importlib.util.find_spec('integrations.jev_m2_episode') is not None,'candidate episode runner missing'
    return importlib.import_module('integrations.jev_m2_episode')


class OneAssayPolicy:
    metadata={'fixture_policy':'one_assay_then_commit'}
    def probabilities(self,obs):
        p=np.zeros(45);p[4 if obs['experiment_count']==0 else 36]=1.
        return p.tolist()


@pytest.fixture
def setup():
    config=PilotConfig(final_cultures=128)
    return config,PredictionBank(finite_family(),config),make_task(finite_family()[0],991332000,config)


@pytest.mark.parametrize('role',['ppo','forecast_only','rerank','arbitration','hidden','chance','public','jev_only','probe'])
def test_roles_execute_and_replay(role,setup,tmp_path):
    m=candidate();config,bank,task=setup;path=tmp_path/(role+'.jsonl')
    client=None if role=='ppo' else JevClient(FixtureTransport(),expected_model='jev-fixture')
    policy=None if role in ('jev_only','probe') else OneAssayPolicy()
    with Journal(path) as journal:
        result=m.run_candidate(ScientistEnv(task,config),role,bank,journal,policy=policy,client=client)
    assert result['status']=='completed'
    assert m.verify_candidate(path,task,config,bank)==result
    requests=[e['request'] for e in read_journal(path) if e['kind']=='api_request']
    if role=='forecast_only':
        assert result['actions']==[4,36] and len(requests)==1
        assert len(requests[0]['questions'])==2 and result['forecasts'][0]['probabilities']['jev']==.5
    if role=='rerank':
        assert all(len(q['criteria'])<=3 for r in requests for q in r['questions'].values() if q['type']=='choice')


def test_forecast_failure_cannot_block_the_standalone_ppo_policy(setup,tmp_path):
    m=candidate();config,bank,task=setup;path=tmp_path/'missing.jsonl'
    client=JevClient(FixtureTransport(),expected_model='jev-fixture',limits=CallLimits(max_calls=0))
    with Journal(path) as journal:
        result=m.run_candidate(ScientistEnv(task,config),'forecast_only',bank,journal,policy=OneAssayPolicy(),client=client)
    assert result['status']=='completed' and result['actions']==[4,36]
    assert result['forecast_failures']==['call_limit']
    assert all('jev' not in r['probabilities'] for r in result['forecasts'])
    assert m.verify_candidate(path,task,config,bank)==result


def test_frozen_forecast_precedes_the_purchased_observation(setup,tmp_path):
    m=candidate();config,bank,task=setup;path=tmp_path/'ordered.jsonl'
    class CheckedEnv(ScientistEnv):
        def step(self,action):
            events=read_journal(path)
            assert events[-1]['kind']=='candidate_action'
            if self.actions[action].kind=='experiment':
                assert events[-2]['kind']=='candidate_forecast'
            return super().step(action)
    with Journal(path) as journal:
        m.run_candidate(CheckedEnv(task,config),'forecast_only',bank,journal,policy=OneAssayPolicy(),
                        client=JevClient(FixtureTransport(),expected_model='jev-fixture'))


def test_rehashed_scientific_result_change_is_rejected(setup,tmp_path):
    m=candidate();config,bank,task=setup;path=tmp_path/'original.jsonl'
    with Journal(path) as journal:
        m.run_candidate(ScientistEnv(task,config),'forecast_only',bank,journal,policy=OneAssayPolicy(),
                        client=JevClient(FixtureTransport(),expected_model='jev-fixture'))
    events=read_journal(path);events[-1]['result']['final_mse']=0.
    changed=tmp_path/'changed.jsonl'
    with Journal(changed) as journal:
        for event in events:journal.append(event)
    with pytest.raises(ValueError):m.verify_candidate(changed,task,config,bank)


def test_shadow_reranker_pays_for_advice_but_keeps_ppo_choice(setup,tmp_path):
    m=candidate();config,bank,task=setup;path=tmp_path/'shadow.jsonl'
    class StopSuggestion(FixtureTransport):
        def __call__(self,request):
            result=super().__call__(request)
            if 'next_action' in result['answers']:
                answer=result['answers']['next_action'];answer['choice']='action_36'
                answer['probabilities']={k:float(k=='action_36') for k in answer['probabilities']}
            return result
    client=JevClient(StopSuggestion(),expected_model='jev-fixture')
    with Journal(path) as journal:
        result=m.run_candidate(ScientistEnv(task,config),'rerank_shadow',bank,journal,policy=OneAssayPolicy(),client=client)
    assert result['actions']==[4,36] and result['api']['attempted_calls']==3
    assert m.verify_candidate(path,task,config,bank)==result
