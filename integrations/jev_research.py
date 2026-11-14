"""Prospective public-information controllers; historical simulator stays unchanged."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr
import torch

from experimentlab.baselines import DecisionPlanner, additive_commitment
from experimentlab.inference import Belief
from experimentlab.learning import token, masked_distribution
from integrations.jev_demo import describe


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Journal:
    """Exclusive, flushed pre-observation receipts, with hash-linked local ordering."""
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open('x', encoding='utf-8')
        self.seq = 0
        self.previous = '0' * 64

    def append(self, event):
        record = {'seq': self.seq, 'previous_sha256': self.previous,
                  'recorded_at': datetime.now(timezone.utc).isoformat(), 'event': event}
        record['sha256'] = digest(record)
        self.file.write(canonical(record) + '\n')
        self.file.flush()
        os.fsync(self.file.fileno())
        self.previous = record['sha256']
        self.seq += 1

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_journal(path):
    previous = '0' * 64
    events = []
    for seq, line in enumerate(Path(path).read_text().splitlines()):
        record = json.loads(line)
        sha = record.pop('sha256')
        if record['seq'] != seq or record['previous_sha256'] != previous or digest(record) != sha:
            raise ValueError('journal hash or ordering mismatch')
        events.append(record['event'])
        previous = sha
    if not events:
        raise ValueError('empty journal')
    return events


def public_state(obs, actions):
    """Allowlist serializer: no evaluator object, identifiers, or unbought values."""
    history = []
    for packet in obs['history']:
        action = int(packet['action_id'])
        history.append({'action_id': action,
                        'action': 'baseline' if action == -1 else describe(actions[action]),
                        'cost': int(packet['cost_ticks']) / 20,
                        'measurements': {f'X{i}': [float(v) for v in packet['readouts'][:, i]]
                                         for i in np.flatnonzero(packet['measured_mask'])}})
    return {'task': 'Abstract synthetic regulatory system. Choose experiments then a final intervention.',
            'targets': {f'X{i}': float(v) for i, v in zip(np.flatnonzero(obs['goal_mask']), obs['target'])},
            'remaining_budget': int(obs['budget_ticks']) / 20,
            'experiment_count': int(obs['experiment_count']),
            'cohort_size': int(obs['cohort_size']), 'readout_sd': float(obs['readout_sd']),
            'offset_sd': float(obs['offset_sd']), 'history': history,
            'objective': 'Minimize 100 * final target MSE + 0.01 * assay cost + 0.05 * total intervention dose. Final scoring uses fresh cultures at time 6.',
            'forecast_event': 'Next noisy measured cohort mean strictly exceeds the already observed baseline mean; not final target success.'}


def belief_from_public(obs, bank):
    belief = Belief(bank, obs['target'], admitted=False)
    for packet in obs['history']:
        belief.update(packet)
    return belief


def reference_forecasts(obs, bank):
    if bank.config.offset_sd != 0:
        raise ValueError('M1 forecast reference is qualified only at offset_sd=0')
    belief = belief_from_public(obs, bank)
    baseline = obs['history'][0]['readouts'].mean(axis=0)
    table = {}
    for action_id, action in enumerate(bank.actions):
        if action.kind != 'experiment' or not obs['action_mask'][action_id]:
            continue
        means, cov = bank.observation_model(action_id, list(action.panel))
        se = np.sqrt(np.diagonal(cov, axis1=1, axis2=2) / obs['cohort_size'])
        probabilities = ndtr((means - baseline[list(action.panel)]) / se)
        previous = next((p for p in reversed(obs['history'][1:]) if p['action_id'] == action_id), None)
        for j, node in enumerate(action.panel):
            heuristic = .5
            if previous is not None:
                difference = previous['readouts'][:, node].mean() - baseline[node]
                repeat_se = np.sqrt(2) * obs['readout_sd'] / np.sqrt(obs['cohort_size'])
                heuristic = float(ndtr(difference / repeat_se)) if repeat_se else float(difference > 0)
            table[f'a{action_id}_x{node}'] = {
                'action_id': action_id, 'node': node, 'threshold': float(baseline[node]),
                'probabilities': {'posterior': float(belief.weights @ probabilities[:, j]),
                                  'prior': float(probabilities[:, j].mean()),
                                  'chance': .5, 'public_heuristic': heuristic}}
    return table


def forecast_request(obs, actions, model):
    state = public_state(obs, actions)
    state['candidate_experiments'] = {
        f'action_{i}': describe(a) for i, a in enumerate(actions)
        if a.kind == 'experiment' and obs['action_mask'][i]}
    questions = {}
    for i, action in enumerate(actions):
        if action.kind != 'experiment' or not obs['action_mask'][i]:
            continue
        for node in action.panel:
            threshold = float(obs['history'][0]['readouts'][:, node].mean())
            questions[f'a{i}_x{node}'] = {'type': 'noul', 'instructions': (
                f'If action_{i} is purchased next, will the mean of its {obs["cohort_size"]} '
                f'noisy measured X{node} values be strictly greater than the observed '
                f'baseline mean {threshold!r}? Return the probability of this event.')}
    return {'model': model, 'state': state, 'questions': questions}


def shown_forecasts(table, method):
    if method == 'jev_rl_hidden':
        return None
    source = {'jev_rl_chance': 'chance', 'jev_rl_public': 'public_heuristic'}.get(method, 'jev')
    return {key: row['probabilities'][source] for key, row in table.items()}


def decision_request(obs, actions, model, forecasts=None, rl_probabilities=None):
    state = public_state(obs, actions)
    if forecasts is not None:
        state['forecasts'] = forecasts
    if rl_probabilities is not None:
        state['rl_action_probabilities'] = {f'action_{i}': float(p) for i, p in enumerate(rl_probabilities)
                                            if obs['action_mask'][i]}
        state['rl_advice'] = 'A frozen PPO policy proposes these action probabilities from this same public history; these are not outcome forecasts.'
    return {'model': model, 'state': state, 'questions': {'next_action': {
        'type': 'choice', 'instructions': (
            'Choose a legal next action minimizing the stated final loss and costs. '
            'Use supplied forecasts and policy advice if useful; neither is guaranteed correct. '
            'A sign forecast does not give effect magnitude or value of information. '
            'You may commit immediately or stop whenever another assay is not worth its cost.'),
        'criteria': {f'action_{i}': describe(a) for i, a in enumerate(actions) if obs['action_mask'][i]}}}
    }


class FrozenPolicy:
    def __init__(self, model, metadata=None):
        self.model = model.eval()
        self.history = []
        self.metadata = metadata

    def probabilities(self, obs):
        self.history.append(token(obs))
        with torch.no_grad():
            logits, _ = self.model(torch.as_tensor(np.array(self.history)[None], dtype=torch.float32))
            dist = masked_distribution(logits[:, -1:], torch.as_tensor(obs['action_mask'][None, None]))
        return dist.probs[0, 0].numpy().astype(float).tolist()


def score_forecasts(table, packet):
    rows = []
    for row in table.values():
        if row['action_id'] != packet['action_id']:
            continue
        observed = float(packet['readouts'][:, row['node']].mean())
        outcome = int(observed > row['threshold'])
        rows.append({**row, 'observed_mean': observed, 'outcome': outcome,
                     'brier': {name: float((p-outcome)**2) for name, p in row['probabilities'].items()}})
    return rows


def local_choice(method, obs, bank, schedule, planner, rng):
    belief = belief_from_public(obs, bank)
    if method in ('fixed', 'fixed_quality', 'forecast_probe', 'additive'):
        index = obs['experiment_count']
        if index < len(schedule):
            action = int(schedule[index])
            if not obs['action_mask'][action]:
                raise ValueError('fixed schedule violates budget')
            return action
        return additive_commitment(obs, bank.actions) if method == 'additive' else belief.commitment()
    if method in ('voi', 'eig'):
        return planner.choose(obs, belief, rng)
    if method == 'random':
        legal = np.flatnonzero(obs['action_mask'][:36])
        return int(rng.choice(legal)) if len(legal) else belief.commitment()
    if method == 'bo':
        return planner.choose(obs, belief)
    if method != 'prior':
        raise ValueError('unknown local controller')
    return belief.commitment()


def episode_result(env, method, actions, scores, rewards, status, failure=None):
    result = {'method': method, 'status': status, 'failure': failure, 'actions': actions,
              'assays': int(env.count), 'forecasts': scores, 'immediate_stop': status == 'completed' and env.count == 0,
              'variable_cost': (env.config.budget * 20-env.budget_ticks)/20,
              'setup_cost': 3., 'total_assay_cost': 3.+(env.config.budget*20-env.budget_ticks)/20,
              'final_mse': None, 'effort': None, 'return_': None, 'observed_return': float(sum(rewards))}
    if status == 'completed':
        result.update(env.private_result)
        result['return_'] = float(sum(rewards))
    return result


def run_episode(env, method, bank, journal, *, schedule=(), policy=None, client=None, planner_seed=0):
    from integrations.jev_client import JevError
    episode_start = time.perf_counter()
    if (method.startswith('jev_rl') or method in ('ppo','imitation','calibrated_imitation','imitation_ppo')) and policy is None:
        raise ValueError('condition requires the full learned policy')
    if method in ('jev_only','forecast_probe') and policy is not None:
        raise ValueError('Jev-only/probe condition must not receive a policy')
    obs, _ = env.reset()
    state = public_state(obs, env.actions)
    journal.append({'kind': 'episode_start', 'method': method, 'state': state,
                    'schedule': list(schedule), 'planner_seed': int(planner_seed),
                    'model': client.expected_model if client is not None else None,
                    'trace_id': client.trace_id if client is not None else None,
                    'policy_metadata': policy.metadata if policy is not None else None})
    hosted_method = method.startswith('jev_') or method == 'forecast_probe'
    if hosted_method and client is None:
        raise ValueError('Jev condition needs explicit fixture or hosted client')
    if client is not None:
        client.event_sink = journal.append
    planner = DecisionPlanner(bank, method, 12) if method in ('voi', 'eig') else None
    if method == 'bo':
        from experimentlab.bo import DiscreteBO
        planner = DiscreteBO(bank, belief_from_public(obs, bank))
    rng = np.random.default_rng(planner_seed)
    actions, scores, rewards = [], [], []
    decision_seconds, reference_seconds = [], []
    coverage = {'frozen_candidate_events':0,'jev_requested_events':0,'jev_validated_events':0,
                'purchased_events':0,'scored_jev_events':0,'scored_reference_events':0,
                'probe_expected_events':sum(len(env.actions[a].panel) for a in schedule) if method=='forecast_probe' else None}
    start = time.perf_counter()
    failure = None
    try:
        for step in range(env.config.max_experiments+1):
            tick = time.perf_counter()
            table = reference_forecasts(obs, bank)
            reference_seconds.append(time.perf_counter()-tick)
            tick = time.perf_counter()
            forecast_span = None
            if hosted_method and table:
                before = client.stats()['attempted_calls']
                try:
                    response = client(forecast_request(obs, env.actions, client.expected_model))
                finally:
                    if client.stats()['attempted_calls'] > before:
                        coverage['jev_requested_events'] += len(table)
                coverage['jev_validated_events'] += len(table)
                forecast_span = client.receipts[-1]['span_id']
                for key in table:
                    table[key]['probabilities']['jev'] = response['answers'][key]['noul']
            journal.append({'kind': 'forecast_frozen', 'step': step, 'state_hash': digest(public_state(obs, env.actions)),
                            'table': table, 'forecast_span_id':forecast_span})
            coverage['frozen_candidate_events'] += len(table)
            probabilities = policy.probabilities(obs) if policy is not None else None
            if probabilities is not None:
                journal.append({'kind': 'rl_advice', 'probabilities': probabilities, 'step': step})
            if method.startswith('jev_'):
                request = decision_request(obs, env.actions, client.expected_model, shown_forecasts(table, method), probabilities)
                response = client(request)
                action = int(response['answers']['next_action']['choice'].split('_')[1])
            elif policy is not None:
                action = int(np.argmax(probabilities))
            else:
                action = local_choice(method, obs, bank, schedule, planner, rng)
            if type(action) not in (int, np.int64, np.int32) or not 0 <= action < len(env.actions) or not obs['action_mask'][action]:
                raise JevError('illegal_action')
            decision_seconds.append(time.perf_counter()-tick)
            journal.append({'kind': 'action', 'step': step, 'action_id': int(action),
                            'decision_span_id':client.receipts[-1]['span_id'] if method.startswith('jev_') else None})
            actions.append(int(action))
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(float(reward))
            journal.append({'kind': 'observation', 'step': step, 'state': public_state(obs, env.actions),
                            'reward': float(reward), 'done': bool(done)})
            if truncated or info.get('error'):
                raise JevError('simulator_error')
            if env.actions[action].kind == 'experiment':
                resolved = score_forecasts(table, obs['history'][-1])
                scores.extend(resolved)
                coverage['purchased_events'] += len(env.actions[action].panel)
                coverage['scored_reference_events'] += len(resolved)
                coverage['scored_jev_events'] += sum('jev' in row['probabilities'] for row in resolved)
                journal.append({'kind': 'forecast_scores', 'step': step, 'rows': resolved, 'forecast_span_id':forecast_span})
            if done:
                break
        else:
            raise JevError('decision_limit')
    except JevError as error:
        failure = error.code
        journal.append({'kind': 'episode_failure', 'code': failure})
    result = episode_result(env, method, actions, scores, rewards, 'failed' if failure else 'completed', failure)
    result.update(online_seconds=time.perf_counter()-start, decision_seconds=decision_seconds,
                  reference_forecast_seconds=reference_seconds,
                  end_to_end_seconds=time.perf_counter()-episode_start, forecast_coverage=coverage,
                  timing_scope='instrumented controller loop; includes forecast references and synchronous journal writes; end_to_end also includes reset')
    journal.append({'kind': 'episode_end', 'result': result})
    return result


class FixtureTransport:
    """Deterministic integration fixture. Never describes itself as live Jev."""
    def __call__(self, request):
        if request['model'] != 'jev-fixture':
            raise ValueError('fixture requires explicit fixture model')
        answers = {}
        for key, question in request['questions'].items():
            if question['type'] == 'noul':
                answers[key] = {'type': 'noul', 'noul': .5}
            else:
                options = question['criteria']
                advice = request['state'].get('rl_action_probabilities')
                if advice:
                    choice = max(options, key=lambda k: advice[k])
                else:
                    count = request['state']['experiment_count']
                    proposed = [4,12,31][count] if count < 3 else 36
                    choice = f'action_{proposed}' if f'action_{proposed}' in options else 'action_36'
                answers[key] = {'type': 'choice', 'choice': choice, 'confidence': 1.,
                                'probabilities': {k: float(k == choice) for k in options}}
        return {'model': 'jev-fixture', 'answers': answers, 'usage': {'input_tokens': 0, 'output_tokens': 0}}


def forecast_summary(rows):
    names = sorted({name for row in rows for name in row['probabilities']})
    result = {}
    for name in names:
        pairs = [(row['probabilities'][name], row['outcome']) for row in rows if name in row['probabilities']]
        bins = []
        for index in range(10):
            values = [(p,y) for p,y in pairs if min(int(p*10),9) == index]
            bins.append({'lower': index/10, 'upper': (index+1)/10, 'count': len(values),
                         'mean_probability': float(np.mean([p for p,_ in values])) if values else None,
                         'event_rate': float(np.mean([y for _,y in values])) if values else None})
        result[name] = {'count': len(pairs), 'brier': float(np.mean([(p-y)**2 for p,y in pairs])),
                        'reliability_bins': bins}
    return result


def summarize_results(rows):
    result = {}
    for method in sorted({r['method'] for r in rows}):
        group = [r for r in rows if r['method'] == method]
        complete = [r for r in group if r['status'] == 'completed']
        result[method] = {'attempted': len(group), 'completed': len(complete), 'failed': len(group)-len(complete),
                          'failures': dict(Counter(r['failure'] for r in group if r['status'] != 'completed')),
                          'forecasts': forecast_summary([f for r in group for f in r['forecasts']])}
        result[method]['attempt_metrics'] = {}
        for metric in ('variable_cost','total_assay_cost','assays','online_seconds','end_to_end_seconds'):
            values = [r[metric] for r in group if r.get(metric) is not None]
            result[method]['attempt_metrics'][metric] = {'count':len(values),
                'mean':float(np.mean(values)) if values else None,'total':float(sum(values))}
        coverage_rows = [r['forecast_coverage'] for r in group if 'forecast_coverage' in r]
        result[method]['forecast_coverage'] = {key:sum(r[key] for r in coverage_rows if r[key] is not None)
            for key in coverage_rows[0]} if coverage_rows else {}
        result[method]['mean_denominator'] = 'completed episodes; attempt_metrics separately includes failures'
        for metric in ('final_mse','variable_cost','total_assay_cost','return_','effort','assays','online_seconds','immediate_stop'):
            result[method]['mean_'+metric] = float(np.mean([r[metric] for r in complete])) if complete else None
        result[method]['aggregate_rmse'] = float(np.sqrt(result[method]['mean_final_mse'])) if complete else None
    return result


def verify_episode(path, task, config, bank):
    from integrations.jev_replay import verify_episode as verify
    return verify(path, task, config, bank)
