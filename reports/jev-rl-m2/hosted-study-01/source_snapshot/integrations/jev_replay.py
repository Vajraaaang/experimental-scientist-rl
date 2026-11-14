"""Credential-free verification against independent, uncached simulator dynamics."""
from datetime import datetime, timezone
import re
import numpy as np

from experimentlab.baselines import DecisionPlanner
from experimentlab.envs import ScientistEnv
from integrations.jev_research import (
    belief_from_public, canonical, decision_request, digest, episode_result,
    forecast_request, local_choice, public_state, read_journal, reference_forecasts,
    score_forecasts, shown_forecasts,
)


def equal(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise ValueError(f'{label} mismatch')


def validate_receipt(event, request_event, proposed_request):
    """Bind accounting to the request and typed response, not self-reported totals."""
    success = event['kind']=='api_response'
    if type(event.get('attempted')) is not bool or event.get('status') != ('success' if success else 'failed'):
        raise ValueError('invalid API receipt status')
    if type(event.get('call_index')) is not int or event['call_index'] < 1:
        raise ValueError('invalid API receipt index')
    if not isinstance(event.get('latency_seconds'),(int,float)) or not np.isfinite(event['latency_seconds']) or event['latency_seconds'] < 0:
        raise ValueError('invalid API receipt latency')
    if success and (not event['attempted'] or 'code' in event):
        raise ValueError('success receipt must represent an attempt')
    request = request_event['request'] if request_event is not None else proposed_request
    size = len(canonical(request).encode())
    for key,length in [('trace_id',32),('span_id',16)]:
        if type(event.get(key)) is not str or not re.fullmatch('[0-9a-f]{'+str(length)+'}',event[key]):
            raise ValueError('invalid local trace identifier')
    for key in ('started_at_utc','finished_at_utc'):
        try:
            stamp=datetime.fromisoformat(event[key])
            if stamp.utcoffset()!=timezone.utc.utcoffset(stamp): raise ValueError
        except (TypeError,ValueError,KeyError):
            raise ValueError('invalid trace timestamp') from None
    if event['request_sha256']!=digest(request) or event['proposed_request_bytes']!=size:
        raise ValueError('trace request hash or size mismatch')
    expected_order=sorted(request['questions'])
    candidate_order={name:sorted(request['questions'][name]['criteria']) for name in expected_order
                     if request['questions'][name]['type']=='choice'}
    equal(event['question_order'],expected_order,'wire question order')
    equal(event['candidate_order'],candidate_order,'wire candidate order')
    if request_event is not None:
        for key in ('trace_id','span_id','started_at_utc','request_sha256','proposed_request_bytes','question_order','candidate_order'):
            equal(event[key],request_event[key],'trace request/receipt '+key)
    from integrations.jev_client import validate_transport_metadata
    equal(event['transport'],validate_transport_metadata(event['transport']),'transport allowlist')
    if event['attempted']:
        if request_event is None or event['call_index']!=request_event['call_index']:
            raise ValueError('API receipt/request index mismatch')
        if event['request_bytes']!=size or event.get('reserved_input_tokens')!=size+4096:
            raise ValueError('API receipt request size/reservation mismatch')
    elif request_event is not None or event['request_bytes']!=0 or event['latency_seconds']!=0:
        raise ValueError('unattempted API receipt claims a request')
    usage = event['usage']
    if usage is not None:
        if not isinstance(usage,dict) or set(usage)!={'input_tokens','output_tokens'} or any(type(v) is not int or not 0<=v<=2**53-1 for v in usage.values()):
            raise ValueError('invalid API receipt usage')
        if not event['attempted'] or event['estimated_usd'] != usage['input_tokens']/1_000_000*.042:
            raise ValueError('API receipt cost mismatch')
    elif event['estimated_usd'] != (None if event['attempted'] else 0.):
        raise ValueError('API receipt unknown cost mismatch')
    expected_overrun = usage is not None and usage['input_tokens']>size+4096
    if event['reservation_overrun'] is not expected_overrun:
        raise ValueError('API reservation overrun mismatch')
    response = event.get('response')
    equal(event['response_sha256'],digest(response) if response is not None else None,'trace response hash')
    if response is not None and response['usage']!=usage:
        raise ValueError('API receipt differs from response usage')
    if success and response is None:
        raise ValueError('API success missing response')
    if success and event['transport'].get('http_status',200)!=200:
        raise ValueError('API success has error HTTP status')
    if not success:
        from integrations.jev_client import JevError
        if event.get('code')!=JevError(event.get('code')).code:
            raise ValueError('unknown API failure code')


def verify_episode(path, task, config, bank):
    from integrations.jev_client import validate_response
    events = read_journal(path)
    start = events[0]
    if start['kind'] != 'episode_start' or events[-1]['kind'] != 'episode_end':
        raise ValueError('incomplete episode journal')
    env = ScientistEnv(task, config)
    obs, _ = env.reset()
    equal(start['state'], public_state(obs, env.actions), 'initial public observation')
    method, schedule = start['method'], start['schedule']
    hosted_method = method.startswith('jev_') or method == 'forecast_probe'
    needs_advice = method.startswith('jev_rl') or method in ('ppo','imitation','calibrated_imitation','imitation_ppo')
    planner = DecisionPlanner(bank, method, 12) if method in ('voi','eig') else None
    if method == 'bo':
        from experimentlab.bo import DiscreteBO
        planner = DiscreteBO(bank, belief_from_public(obs, bank))
    rng = np.random.default_rng(start['planner_seed'])
    step = 0
    table = None
    advice = None
    forecast_response = choice_response = pending_request = None
    forecast_span = decision_span = None
    pending_request_event = None
    decision_call_seen = False
    last_call_index = None
    pending_observation = pending_scores = None
    actions, scores, rewards = [], [], []
    done, failure, api_failure = False, None, None
    coverage = {'frozen_candidate_events':0,'jev_requested_events':0,'jev_validated_events':0,
                'purchased_events':0,'scored_jev_events':0,'scored_reference_events':0,
                'probe_expected_events':sum(len(env.actions[a].panel) for a in schedule) if method=='forecast_probe' else None}
    for index, event in enumerate(events[1:], 1):
        kind = event['kind']
        if pending_observation is not None and kind != 'observation':
            raise ValueError('action must be followed by observation')
        if pending_scores is not None and kind != 'forecast_scores':
            raise ValueError('purchased outcome must be scored')
        if failure and kind != 'episode_end':
            raise ValueError('event after episode failure')
        if api_failure is not None and failure is None and kind!='episode_failure':
            raise ValueError('API failure must terminate episode without retry')
        if done and kind != 'episode_end':
            raise ValueError('event after terminal observation')
        if kind == 'api_request':
            if not hosted_method or pending_request is not None:
                raise ValueError('unexpected API request')
            request = event['request']
            equal(request['model'], start['model'], 'requested model')
            equal(event['trace_id'],start['trace_id'],'episode trace')
            if 'next_action' in request['questions']:
                if decision_call_seen:
                    raise ValueError('duplicate decision call')
                if table is None or method == 'forecast_probe':
                    raise ValueError('choice must follow frozen forecasts')
                if needs_advice != (advice is not None):
                    raise ValueError('condition policy advice missing or unexpected')
                expected = decision_request(obs, env.actions, request['model'], shown_forecasts(table, method), advice)
                decision_call_seen = True
            else:
                if table is not None or forecast_response is not None:
                    raise ValueError('duplicate or late forecast request')
                expected = forecast_request(obs, env.actions, request['model'])
            equal(request, expected, 'public API request')
            pending_request = request
            pending_request_event = event
        elif kind == 'api_response':
            if pending_request is None:
                raise ValueError('response without request')
            validate_receipt(event,pending_request_event,pending_request)
            if last_call_index is not None and event['call_index']!=last_call_index+1:
                raise ValueError('API call indices are not sequential')
            last_call_index = event['call_index']
            response = validate_response(pending_request, event['response'], pending_request['model'])
            if 'next_action' in pending_request['questions']:
                choice_response = response
                decision_span=event['span_id']
            else:
                forecast_response = response
                forecast_span=event['span_id']
                coverage['jev_requested_events'] += len(pending_request['questions'])
                coverage['jev_validated_events'] += len(pending_request['questions'])
            pending_request = None
            pending_request_event = None
        elif kind == 'api_failure':
            proposed = (forecast_request(obs,env.actions,start['model']) if table is None else
                        decision_request(obs,env.actions,start['model'],shown_forecasts(table,method),advice))
            validate_receipt(event,pending_request_event,proposed)
            equal(event['trace_id'],start['trace_id'],'episode failure trace')
            if last_call_index is not None and event['call_index']!=last_call_index+1:
                raise ValueError('API call indices are not sequential')
            last_call_index=event['call_index']
            if pending_request is not None and 'next_action' not in pending_request['questions'] and event['attempted']:
                coverage['jev_requested_events'] += len(pending_request['questions'])
            api_failure = event['code']
            pending_request = None
            pending_request_event = None
        elif kind == 'forecast_frozen':
            if table is not None or pending_request is not None or event['step'] != step:
                raise ValueError('forecast ordering mismatch')
            equal(event['state_hash'], digest(public_state(obs, env.actions)), 'forecast history')
            expected = reference_forecasts(obs, bank)
            if hosted_method and expected:
                if forecast_response is None:
                    raise ValueError('missing prospective forecast response')
                for key in expected:
                    expected[key]['probabilities']['jev'] = forecast_response['answers'][key]['noul']
            equal(event['table'], expected, 'forecast table')
            equal(event['forecast_span_id'],forecast_span,'forecast span')
            table = event['table']
            coverage['frozen_candidate_events'] += len(table)
        elif kind == 'rl_advice':
            if not needs_advice:
                raise ValueError('condition forbids policy advice')
            if table is None or advice is not None or event['step'] != step:
                raise ValueError('policy advice ordering mismatch')
            p = np.asarray(event['probabilities'], dtype=float)
            if p.shape != (45,) or not np.isfinite(p).all() or np.any(p < 0) or abs(p.sum()-1) > 1e-6 or np.any(p[~obs['action_mask']] != 0):
                raise ValueError('invalid masked policy advice')
            advice = event['probabilities']
        elif kind == 'action':
            action = event['action_id']
            if table is None or pending_request is not None or event['step'] != step:
                raise ValueError('action before frozen forecasts')
            if needs_advice != (advice is not None):
                raise ValueError('condition policy advice missing or unexpected')
            if type(action) is not int or not 0 <= action < len(env.actions) or not obs['action_mask'][action]:
                raise ValueError('illegal replay action')
            if method.startswith('jev_'):
                if choice_response is None:
                    raise ValueError('missing decision response')
                equal(f'action_{action}', choice_response['answers']['next_action']['choice'], 'decision action')
                equal(event['decision_span_id'],decision_span,'decision span')
            elif advice is not None:
                equal(event['decision_span_id'],None,'local decision span')
                equal(action, int(np.argmax(advice)), 'learned-policy action')
            else:
                equal(action, local_choice(method, obs, bank, schedule, planner, rng), 'baseline action')
            actions.append(action)
            obs, reward, done_now, truncated, info = env.step(action)
            if truncated or info:
                raise ValueError('invalid simulator replay')
            rewards.append(float(reward))
            pending_observation = {'kind':'observation', 'step':step, 'state':public_state(obs,env.actions),
                                   'reward':float(reward), 'done':bool(done_now)}
        elif kind == 'observation':
            if pending_observation is None:
                raise ValueError('observation without action')
            equal(event, pending_observation, 'public observation or reward')
            pending_observation = None
            done = event['done']
            if not done:
                pending_scores = score_forecasts(table, obs['history'][-1])
            else:
                step += 1
        elif kind == 'forecast_scores':
            if pending_scores is None or event['step'] != step:
                raise ValueError('forecast scoring order mismatch')
            equal(event['rows'], pending_scores, 'forecast scores')
            equal(event['forecast_span_id'],forecast_span,'scored forecast span')
            scores.extend(pending_scores)
            coverage['purchased_events'] += len(env.actions[actions[-1]].panel)
            coverage['scored_reference_events'] += len(pending_scores)
            coverage['scored_jev_events'] += sum('jev' in row['probabilities'] for row in pending_scores)
            pending_scores = None
            table = advice = forecast_response = choice_response = None
            forecast_span = decision_span = None
            decision_call_seen = False
            step += 1
        elif kind == 'episode_failure':
            failure = event['code']
            if api_failure is not None and failure != api_failure:
                raise ValueError('API failure mismatch')
        elif kind == 'episode_end':
            if index != len(events)-1 or pending_request is not None or not (done or failure):
                raise ValueError('unfinished episode')
            result = event['result']
            expected = episode_result(env, method, actions, scores, rewards, 'failed' if failure else 'completed', failure)
            for key, value in expected.items():
                equal(result[key], value, 'result '+key)
            equal(result['forecast_coverage'], coverage, 'forecast coverage')
            for key in ('online_seconds','end_to_end_seconds','decision_seconds','reference_forecast_seconds'):
                values = np.asarray(result[key], dtype=float)
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError('invalid timing receipt')
            return result
        else:
            raise ValueError('unexpected journal event')
    raise ValueError('missing episode result')
