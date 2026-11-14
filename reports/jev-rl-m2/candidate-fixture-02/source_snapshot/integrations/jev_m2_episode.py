"""Task-derived Jev roles and replay through the unchanged independent simulator."""
import copy
import time

import numpy as np

from experimentlab.envs import ScientistEnv
from integrations.jev_candidates import (
    candidate_decision_request, proposal_menu, public_family_description,
    rerank_request, selected_forecast_request,
)
from integrations.jev_client import JevError, validate_response
from integrations.jev_demo import describe
from integrations.jev_replay import validate_receipt
from integrations.jev_research import (
    belief_from_public, canonical, digest, episode_result, forecast_request,
    public_state, read_journal, reference_forecasts, score_forecasts,
)
from scripts.jev_rl_study import receipt_stats

ROLES=('ppo','forecast_only','rerank','rerank_shadow','arbitration','hidden','chance','public','jev_only','probe')
ARBITRATION={'arbitration','hidden','chance','public','jev_only'}


def all_forecast_request(obs,actions,model):
    request=forecast_request(obs,actions,model)
    request['state']['public_family']=public_family_description()
    request['state']['candidate_role']='prospective_counterfactual_forecasts'
    for action_id,action in enumerate(actions):
        if action.kind!='experiment' or not obs['action_mask'][action_id]:continue
        for node in action.panel:
            threshold=float(obs['history'][0]['readouts'][:,node].mean())
            request['questions'][f'a{action_id}_x{node}']['instructions']=(
                f'{describe(action)} If this experiment is purchased next on fresh cultures, '
                f'will the mean of its {obs["cohort_size"]} noisy measured X{node} values '
                f'be strictly greater than the observed baseline mean {threshold!r}? '
                'Return this event probability, not final target success.')
    return request


def _shown(table,role):
    if role=='hidden':return None
    name={'chance':'chance','public':'public_heuristic'}.get(role,'jev')
    return {key:value['probabilities'][name] for key,value in table.items()}


def run_candidate(env,role,bank,journal,*,policy=None,client=None,schedule=(4,12,31)):
    if role not in ROLES:raise ValueError('unknown candidate role')
    needs_policy=role not in ('jev_only','probe')
    if needs_policy != (policy is not None):raise ValueError('candidate policy requirement mismatch')
    if role!='ppo' and client is None:raise ValueError('candidate needs an explicit hosted or fixture client')
    started=time.perf_counter();obs,_=env.reset()
    before=0 if client is None else len(client.receipts)
    journal.append({'kind':'candidate_start','role':role,'state':public_state(obs,env.actions),
                    'model':None if client is None else client.expected_model,
                    'trace_id':None if client is None else client.trace_id,
                    'policy_metadata':None if policy is None else policy.metadata,'schedule':list(schedule),
                    'forecast_failure_rule':'forecast-only/probe/post-rerank forecasts cannot veto an already chosen action; failures retained',
                    'decision_failure_rule':'choice or required predecision-forecast failure ends episode; no retry/fallback'})
    if client is not None:client.event_sink=journal.append
    actions=[];scores=[];rewards=[];forecast_failures=[];failure=None
    coverage={'requested_jev_events':0,'validated_jev_events':0,'purchased_events':0,
              'scored_jev_events':0,'scored_reference_events':0,'frozen_candidate_events':0}
    reference_seconds=0.

    def choose(request):
        response=client(request)
        answer=response['answers']['next_action']['choice']
        if answer not in request['questions']['next_action']['criteria']:
            raise JevError('illegal_action')
        return int(answer.split('_')[1]),client.receipts[-1]['span_id']

    def forecast(request,table,required):
        span=None
        if not table:return table,span
        attempts=client.stats()['attempted_calls']
        try:
            response=client(request)
        except JevError as error:
            forecast_failures.append(error.code)
            journal.append({'kind':'candidate_forecast_failure','code':error.code,'required_for_choice':required})
            if required:raise
        else:
            coverage['validated_jev_events']+=len(table)
            span=client.receipts[-1]['span_id']
            for key in table:table[key]['probabilities']['jev']=response['answers'][key]['noul']
        finally:
            if client.stats()['attempted_calls']>attempts:coverage['requested_jev_events']+=len(table)
        return table,span

    try:
        for step in range(env.config.max_experiments+1):
            probabilities=None
            if policy is not None:
                probabilities=policy.probabilities(obs)
                # Validation uses the actual full legal mask, even for standalone PPO.
                proposal_menu(obs,probabilities,env.actions)
                journal.append({'kind':'candidate_policy','step':step,'state_hash':digest(public_state(obs,env.actions)),
                                'probabilities':probabilities,'policy_metadata':policy.metadata})
            tick=time.perf_counter();references=reference_forecasts(obs,bank)
            reference_seconds+=time.perf_counter()-tick
            decision_span=None;forecast_span=None
            if role in ARBITRATION:
                table=copy.deepcopy(references)
                if table:table,forecast_span=forecast(all_forecast_request(obs,env.actions,client.expected_model),table,True)
                journal.append({'kind':'candidate_forecast','step':step,'table':table,'forecast_span_id':forecast_span})
                coverage['frozen_candidate_events']+=len(table)
                request=candidate_decision_request(obs,env.actions,client.expected_model,_shown(table,role),probabilities)
                action,decision_span=choose(request)
            else:
                if role=='probe':
                    index=obs['experiment_count']
                    action=int(schedule[index]) if index<len(schedule) else belief_from_public(obs,bank).commitment()
                elif role in ('rerank','rerank_shadow'):
                    action,decision_span=choose(rerank_request(obs,env.actions,client.expected_model,probabilities,k=3))
                    if role=='rerank_shadow':action=int(np.argmax(probabilities))
                else:action=int(np.argmax(probabilities))
                if not 0<=action<len(env.actions) or not obs['action_mask'][action]:raise JevError('illegal_action')
                journal.append({'kind':'candidate_plan','step':step,'action_id':action,'decision_span_id':decision_span})
                table={key:copy.deepcopy(row) for key,row in references.items() if row['action_id']==action}
                if role!='ppo' and table:
                    request=selected_forecast_request(obs,env.actions,client.expected_model,action)
                    if role=='probe':
                        request['state']['candidate_purpose']='A fixed probe schedule already selected this assay; forecasts cannot change it.'
                    elif role in ('rerank','rerank_shadow'):
                        request['state']['candidate_purpose']='The restricted proposal choice is already frozen; forecasts cannot change it.'
                    table,forecast_span=forecast(request,table,False)
                journal.append({'kind':'candidate_forecast','step':step,'table':table,'forecast_span_id':forecast_span})
                coverage['frozen_candidate_events']+=len(table)
            if type(action) is not int or not 0<=action<len(env.actions) or not obs['action_mask'][action]:
                raise JevError('illegal_action')
            journal.append({'kind':'candidate_action','step':step,'action_id':action,'decision_span_id':decision_span})
            actions.append(action);obs,reward,done,truncated,info=env.step(action);rewards.append(float(reward))
            journal.append({'kind':'candidate_observation','step':step,'state':public_state(obs,env.actions),
                            'reward':float(reward),'done':bool(done)})
            if info or truncated:raise JevError('simulator_error')
            if env.actions[action].kind=='experiment':
                rows=score_forecasts(table,obs['history'][-1]);scores.extend(rows)
                coverage['purchased_events']+=len(env.actions[action].panel)
                coverage['scored_reference_events']+=len(rows)
                coverage['scored_jev_events']+=sum('jev' in row['probabilities'] for row in rows)
                journal.append({'kind':'candidate_scores','step':step,'forecast_span_id':forecast_span,'rows':rows})
            if done:break
        else:raise JevError('decision_limit')
    except JevError as error:
        failure=error.code;journal.append({'kind':'candidate_failure','code':failure})
    result=episode_result(env,role,actions,scores,rewards,'failed' if failure else 'completed',failure)
    result.update(forecast_failures=forecast_failures,forecast_coverage=coverage,
                  api=receipt_stats([] if client is None else client.receipts[before:]),
                  timing={'end_to_end_seconds':time.perf_counter()-started,'reference_seconds':reference_seconds},
                  online_seconds=time.perf_counter()-started)
    journal.append({'kind':'candidate_end','result':result})
    return result


def _without_runtime(event):
    value=copy.deepcopy(event)
    if value.get('kind')=='candidate_end':
        result=value['result'];timing=result.pop('timing')
        if set(timing)!={'end_to_end_seconds','reference_seconds'} or any(type(v) not in (int,float) or not np.isfinite(v) or v<0 for v in timing.values()):
            raise ValueError('invalid candidate timing')
        seconds=result.pop('online_seconds')
        if type(seconds) not in (int,float) or not np.isfinite(seconds) or seconds<0:raise ValueError('invalid loop time')
    return value


class _CheckingJournal:
    def __init__(self,events):self.events=events;self.position=0
    def peek(self):
        if self.position>=len(self.events):raise ValueError('replay exhausted journal')
        return self.events[self.position]
    def append(self,event):
        if canonical(_without_runtime(event))!=canonical(_without_runtime(self.peek())):
            raise ValueError('candidate event/causality/result mismatch')
        self.position+=1


class _RecordedPolicy:
    def __init__(self,journal,metadata,actions):self.journal=journal;self.metadata=metadata;self.actions=actions
    def probabilities(self,obs):
        event=self.journal.peek()
        if event['kind']!='candidate_policy' or event['state_hash']!=digest(public_state(obs,self.actions)):
            raise ValueError('missing or misbound policy advice')
        proposal_menu(obs,event['probabilities'],self.actions)
        return event['probabilities']


class _RecordedClient:
    def __init__(self,journal,start):
        self.journal=journal;self.expected_model=start['model'];self.trace_id=start['trace_id'];self.receipts=[]
        self.event_sink=journal.append
    def stats(self):return receipt_stats(self.receipts)
    def __call__(self,request):
        event=self.journal.peek();request_event=None
        if event['kind']=='api_request':
            request_event=event
            if canonical(event['request'])!=canonical(request) or event['trace_id']!=self.trace_id:
                raise ValueError('candidate API request mismatch')
            self.event_sink(event);event=self.journal.peek()
        if event['kind'] not in ('api_response','api_failure'):raise ValueError('missing candidate API receipt')
        validate_receipt(event,request_event,request)
        if event['trace_id']!=self.trace_id:raise ValueError('candidate receipt trace mismatch')
        if self.receipts and event['call_index']!=self.receipts[-1]['call_index']+1:
            raise ValueError('candidate call index mismatch')
        self.event_sink(event);self.receipts.append({k:v for k,v in event.items() if k!='kind'})
        if event['kind']=='api_failure':raise JevError(event['code'],transport=event['transport'])
        return validate_response(request,event['response'],self.expected_model)


def verify_candidate(path,task,config,bank,expected_start=None):
    events=read_journal(path)
    if events[0]['kind']!='candidate_start' or events[-1]['kind']!='candidate_end':
        raise ValueError('incomplete candidate episode')
    start=events[0]
    if expected_start is not None:
        for key,value in expected_start.items():
            if canonical(start[key])!=canonical(value):raise ValueError('candidate manifest binding mismatch')
    checker=_CheckingJournal(events);env=ScientistEnv(task,config)
    policy=None if start['role'] in ('jev_only','probe') else _RecordedPolicy(checker,start['policy_metadata'],env.actions)
    client=None if start['model'] is None else _RecordedClient(checker,start)
    run_candidate(env,start['role'],bank,checker,policy=policy,client=client,schedule=start['schedule'])
    if checker.position!=len(events):raise ValueError('extra candidate journal events')
    return events[-1]['result']
