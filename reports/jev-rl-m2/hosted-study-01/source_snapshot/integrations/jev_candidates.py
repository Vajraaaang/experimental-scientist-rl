"""Pure public-information request builders for the M2 candidate roles.

These functions neither run an episode nor obtain forecasts or reference answers.
The static prior is a declared generator description, never an episode parameter
packet. Incumbent M1 request builders remain unchanged.
"""
import math
from numbers import Real

import numpy as np

from integrations.jev_demo import describe
from integrations.jev_research import decision_request, public_state


def public_family_description():
    """Return a fresh plain-JSON description of the public continuous prior."""
    return {
        'schema': 'experimental-scientist-public-family-v1',
        'scope': ('Abstract synthetic eight-variable benchmark; this generator '
                  'description is not a claim of biological validity.'),
        'variables': {'count': 8, 'controls': [0, 1, 2, 3],
                      'proxies': [4, 5], 'outcomes': [6, 7]},
        'dynamics': {
            'equation': 'dx_i/dt = gamma_i * ((1-u_i) * sigmoid(b_i + sum_j W_ij*x_j) - x_i)',
            'sigmoid': 'sigmoid(z) = 1 / (1 + exp(-z))',
            'bias': 'b_i = -sum_j W_ij / 2',
            'matched_baseline': [.5] * 8,
            'dt': .05,
            'numerical_update': ('Exponential Euler: alpha_i = 1 - exp(-gamma_i*dt); '
                                 'x_i_next = (1-alpha_i)*x_i + alpha_i*(1-u_i)*'
                                 'sigmoid(b_i + sum_j W_ij*x_j).')},
        'prior': {
            'signs': {'count': 5, 'values': [-1, 1],
                      'probabilities': [.5, .5], 'independent': True},
            'control_gain': {'distribution': 'Uniform', 'low': .9, 'high': 1.05},
            'rate_scale': {'distribution': 'Uniform', 'low': .9, 'high': 1.1},
            'independence': 's0, s1, s2, s3, s4, control gain g, and rate scale r are independent.',
            'weight_rules': {
                'W[4,0]': '1.5 * g * s0', 'W[4,1]': '1.5 * g * s1',
                'W[5,2]': '1.5 * g * s2', 'W[5,3]': '1.5 * g * s3',
                'W[6,4]': '1.55', 'W[6,5]': '1.55',
                'W[7,4]': '1.55 * s4', 'W[7,5]': '-1.55 * s4',
                'all_other_entries': '0'},
            'base_rates': [1., 1., 1., 1., 1.2, 1.2, .7, .7],
            'rates': 'gamma_i = base_rates_i * r',
            'finite_diagnostic': 'The 32 sign combinations with g=1 and r=1 form the finite diagnostic family.'},
        'cultures': {
            'fresh_per_assay': True, 'independent_final_cultures': True,
            'initialization': ('Each fresh culture starts at its untreated equilibrium; '
                               'the matched baseline is 0.5 when offset_sd is zero. '
                               'An assay does not carry its physical state into the next assay.'),
            'offsets': ('Independent Normal(0, offset_sd^2) production offsets are '
                        'added to b and held fixed within each culture.'),
            'readouts': ('Independent Normal(0, readout_sd^2) noise is added to each '
                         'purchased variable in each culture. cohort_size, readout_sd, '
                         'and offset_sd are declared in the public state.')},
        'task': {
            'targets': ('The public state supplies the X6 and X7 targets. Each target '
                        'is independently 0.5 plus an equiprobable -0.08 or +0.08; '
                        'the continuous prior is unfiltered by target feasibility.'),
            'budget': 'The public state supplies remaining_budget and experiment_count; only masked legal actions are offered.',
            'assays': ('The pilot permits at most four purchased assays. Each chooses '
                       'no suppression or one control at dose 0.4 or 0.8, duration '
                       '1 or 6, and panel [4,5] or [6,7].'),
            'assay_cost': '1 + 0.2*duration + 0.25*panel_size + 0.25*sum(u); 20 ticks per cost unit.',
            'final_time': 6.,
            'objective': ('Minimize 100 * final latent target MSE + 0.01 * purchased '
                          'assay cost + 0.05 * total final intervention dose. '
                          'Final MSE uses fresh cultures without readout noise.'),
            'forecast_event': ('A future noisy measured cohort mean strictly exceeds '
                               'its already observed baseline mean. This event is '
                               'separate from final target success or policy quality.')}}


def _public_mask(obs, actions):
    mask = np.asarray(obs['action_mask'])
    if len(actions) != 45 or mask.shape != (45,) or mask.dtype.kind != 'b':
        raise ValueError('candidate requests require a 45-action catalogue and boolean legal mask')
    if any(action.kind not in ('experiment', 'commit') for action in actions):
        raise ValueError('unknown action kind')
    if any(mask[i] and action.kind == 'experiment' and action.cost_ticks > obs['budget_ticks']
           for i, action in enumerate(actions)):
        raise ValueError('every legal experiment must be affordable')
    return mask


def _probabilities(probabilities, mask):
    values = np.asarray(probabilities)
    if (values.shape != (45,) or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)):
        raise ValueError('probabilities must be 45 finite numbers in [0, 1]')
    if not math.isclose(math.fsum(values.tolist()), 1., rel_tol=0., abs_tol=1e-6):
        raise ValueError('probabilities must sum to one within 1e-6')
    if np.any(values[~mask] != 0):
        raise ValueError('probabilities must assign zero mass to illegal actions')
    return values.astype(float).tolist()


def _metadata(state, role, action_ids, order, include_prior):
    state.update(candidate_role=role, candidate_action_ids=action_ids,
                 candidate_order=order,
                 candidate_order_scope='candidate_action_ids array only',
                 choice_wire_order=sorted(f'action_{i}' for i in action_ids) if role!='forecast_only' else [])
    if include_prior:
        state['public_family'] = public_family_description()


def selected_forecast_request(obs, actions, model, action_id, include_prior=True):
    """Forecast only a selected affordable assay; commitments have no event."""
    mask = _public_mask(obs, actions)
    if (isinstance(action_id, (bool, np.bool_))
            or not isinstance(action_id, (int, np.integer))
            or not 0 <= action_id < len(actions)):
        raise ValueError('selected action must be a legal experiment ID')
    action_id = int(action_id)
    action = actions[action_id]
    if not mask[action_id] or action.kind != 'experiment':
        raise ValueError('selected action must be a legal experiment; commits have no forecast event')
    if len(action.panel) != 2:
        raise ValueError('selected pilot assay must measure two variables')
    state = public_state(obs, actions)
    description = describe(action)
    state['candidate_experiments'] = {f'action_{action_id}': description}
    _metadata(state, 'forecast_only', [action_id], 'selected_action_only', include_prior)
    state['candidate_purpose'] = ('The PPO policy already selected this assay. Forecasts cannot change '
                                 'the action; score these events separately from policy quality.')
    questions = {}
    for node in action.panel:
        threshold = float(obs['history'][0]['readouts'][:, node].mean())
        questions[f'a{action_id}_x{node}'] = {'type': 'noul', 'instructions': (
            f'{description} If this selected experiment is purchased next on fresh cultures, '
            f'will the mean of its {int(obs["cohort_size"])} noisy measured X{node} values '
            f'be strictly greater than the observed baseline mean {threshold!r}? '
            'Return the probability of this event, not final target success.')}
    return {'model': model, 'state': state, 'questions': questions}


def proposal_menu(obs, probabilities, actions, k=3):
    """Rank up to three legal proposals, retaining the top choice and best commit.

    k=3 starts with the top two legal actions and highest-probability commitment.
    Deduplicate, then fill from the remaining ranking. k=2 retains the top choice
    and best commitment. Ties always prefer the lower catalogue ID, and the final
    menu uses the same descending-probability, ascending-ID order.
    """
    if type(k) is not int or not 2 <= k <= 3:
        raise ValueError('k must be 2 or 3')
    mask = _public_mask(obs, actions)
    values = _probabilities(probabilities, mask)
    ranked = sorted(np.flatnonzero(mask).tolist(), key=lambda i: (-values[i], i))
    commit = next((i for i in ranked if actions[i].kind == 'commit'), None)
    if commit is None:
        raise ValueError('proposal menu requires a legal commitment')
    selected = set(ranked[:k - 1]) | {commit}
    for action_id in ranked:
        if len(selected) >= min(k, len(ranked)):
            break
        selected.add(action_id)
    return [i for i in ranked if i in selected]


def rerank_request(obs, actions, model, probabilities, include_prior=True, k=3):
    """Ask one Choice question over the frozen restricted PPO proposal rule."""
    menu = proposal_menu(obs, probabilities, actions, k=k)
    values = _probabilities(probabilities, _public_mask(obs, actions))
    request = decision_request(obs, actions, model, rl_probabilities=values)
    _metadata(request['state'], 'proposal_reranking', menu,
              'descending_rl_probability_then_ascending_action_id', include_prior)
    request['state']['proposal_limit'] = k
    request['state']['proposal_rule'] = (
        f'Take the top {k - 1} legal actions and highest-probability legal commitment; '
        f'deduplicate and fill to at most {k} from the remaining ranking. '
        'The candidate_action_ids array ranks by descending PPO probability, breaking ties by ascending action ID. '
        'Choice keys are serialized lexicographically in choice_wire_order, which can differ from that rank. '
        'Shown PPO probabilities are the original legal-action probabilities, '
        'not renormalized over this restricted menu.')
    question = request['questions']['next_action']
    question['criteria'] = {f'action_{i}': describe(actions[i]) for i in menu}
    question['instructions'] += (' Choose only from the restricted proposal menu. '
                                 'The restriction is part of this candidate policy, '
                                 'not a change to simulator legality.')
    return request


def candidate_decision_request(obs, actions, model, forecasts=None,
                               rl_probabilities=None, include_prior=True):
    """M1 full arbitration plus declared prior and candidate metadata.

    Forecasts must already be flat probabilities keyed by legal public assay
    events. Reference tables, extra labels and shared mutable input dictionaries
    are not forwarded. The caller remains responsible for forecast provenance.
    """
    mask = _public_mask(obs, actions)
    values = None if rl_probabilities is None else _probabilities(rl_probabilities, mask)
    safe_forecasts = None
    if forecasts is not None:
        events = {f'a{i}_x{node}' for i, action in enumerate(actions)
                  if mask[i] and action.kind == 'experiment' for node in action.panel}
        if (type(forecasts) is not dict or not set(forecasts) <= events
                or any(isinstance(p, (bool, np.bool_)) or not isinstance(p, Real)
                       or not math.isfinite(p) or not 0 <= p <= 1 for p in forecasts.values())):
            raise ValueError('forecasts must be flat probabilities for legal assay events')
        safe_forecasts = {key: float(p) for key, p in forecasts.items()}
    request = decision_request(obs, actions, model, safe_forecasts, values)
    _metadata(request['state'], 'full_arbitration', np.flatnonzero(mask).tolist(),
              'ascending_action_id', include_prior)
    return request
