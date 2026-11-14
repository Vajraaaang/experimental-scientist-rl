"""Public-only request contracts for the narrower M2 candidate roles."""
import copy
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experimentlab.envs import catalogue


def candidates():
    assert importlib.util.find_spec('integrations.jev_candidates') is not None, 'M2 request builders are missing'
    return importlib.import_module('integrations.jev_candidates')


@pytest.fixture
def public_inputs():
    actions = catalogue()
    baseline = np.full((8, 8), .5)
    baseline[:, 4] = .52
    baseline[:, 5] = .47
    obs = {'target': np.array([.42, .58]),
           'goal_mask': np.array([False] * 6 + [True, True]),
           'control_mask': np.array([True] * 4 + [False] * 4),
           'panels': ((4, 5), (6, 7)), 'budget_ticks': 160,
           'experiment_count': 0, 'reset': True,
           'readout_sd': .02, 'offset_sd': 0., 'cohort_size': 8,
           'history': [{'action_id': -1, 'readouts': baseline,
                        'measured_mask': np.ones(8, dtype=bool), 'cost_ticks': 0}],
           'action_mask': np.ones(45, dtype=bool)}
    return obs, actions


def distribution(values):
    probabilities = np.zeros(45)
    for action_id, probability in values.items():
        probabilities[action_id] = probability
    return probabilities


def test_public_prior_is_static_plain_json_and_describes_the_declared_family():
    c = candidates()
    assert not inspect.signature(c.public_family_description).parameters
    family = c.public_family_description()
    assert json.loads(json.dumps(family, allow_nan=False)) == family
    assert family['variables'] == {'count': 8, 'controls': [0, 1, 2, 3],
                                   'proxies': [4, 5], 'outcomes': [6, 7]}
    assert family['dynamics']['matched_baseline'] == [.5] * 8
    assert family['dynamics']['dt'] == .05
    assert family['dynamics']['equation'] == 'dx_i/dt = gamma_i * ((1-u_i) * sigmoid(b_i + sum_j W_ij*x_j) - x_i)'
    assert family['dynamics']['bias'] == 'b_i = -sum_j W_ij / 2'
    prior = family['prior']
    assert prior['signs'] == {'count': 5, 'values': [-1, 1],
                            'probabilities': [.5, .5], 'independent': True}
    assert prior['control_gain'] == {'distribution': 'Uniform', 'low': .9, 'high': 1.05}
    assert prior['rate_scale'] == {'distribution': 'Uniform', 'low': .9, 'high': 1.1}
    assert prior['base_rates'] == [1., 1., 1., 1., 1.2, 1.2, .7, .7]
    assert prior['weight_rules'] == {
        'W[4,0]': '1.5 * g * s0', 'W[4,1]': '1.5 * g * s1',
        'W[5,2]': '1.5 * g * s2', 'W[5,3]': '1.5 * g * s3',
        'W[6,4]': '1.55', 'W[6,5]': '1.55',
        'W[7,4]': '1.55 * s4', 'W[7,5]': '-1.55 * s4',
        'all_other_entries': '0'}
    assert family['cultures']['fresh_per_assay'] is True
    assert family['cultures']['independent_final_cultures'] is True
    assert family['task']['final_time'] == 6.
    assert 'synthetic' in family['scope'] and 'biological' in family['scope']
    family['prior']['control_gain']['low'] = -99
    assert c.public_family_description()['prior']['control_gain']['low'] == .9


def test_selected_assay_has_only_two_explicit_exact_events(public_inputs):
    c = candidates(); obs, actions = public_inputs
    request = c.selected_forecast_request(obs, actions, 'jev-fixture', 6)
    assert set(request) == {'model', 'state', 'questions'}
    assert list(request['questions']) == ['a6_x4', 'a6_x5']
    assert list(request['state']['candidate_experiments']) == ['action_6']
    assert request['state']['candidate_action_ids'] == [6]
    assert request['state']['candidate_order'] == 'selected_action_only'
    assert request['state']['candidate_role'] == 'forecast_only'
    assert request['state']['public_family'] == c.public_family_description()
    for node, threshold in ((4, .52), (5, .47)):
        question = request['questions'][f'a6_x{node}']
        assert question['type'] == 'noul'
        instructions = question['instructions']
        assert request['state']['candidate_experiments']['action_6'] in instructions
        assert 'suppress X0 by dose 0.4' in instructions
        assert 'after 6 time units' in instructions
        assert '8 noisy measured' in instructions
        assert f'X{node}' in instructions
        assert f'baseline mean {threshold!r}' in instructions
        assert 'strictly greater' in instructions
    assert 'policy quality' in request['state']['candidate_purpose']


@pytest.mark.parametrize('action_id', [-1, 45, True, 1.5, 36])
def test_selected_forecast_rejects_non_assay_choices(public_inputs, action_id):
    c = candidates(); obs, actions = public_inputs
    with pytest.raises(ValueError):
        c.selected_forecast_request(obs, actions, 'jev-fixture', action_id)


def test_selected_forecast_requires_a_legal_affordable_assay(public_inputs):
    c = candidates(); obs, actions = public_inputs
    obs['action_mask'][6] = False
    with pytest.raises(ValueError, match='legal'):
        c.selected_forecast_request(obs, actions, 'jev-fixture', 6)
    obs['action_mask'][6] = True
    obs['budget_ticks'] = actions[6].cost_ticks - 1
    with pytest.raises(ValueError, match='affordable'):
        c.selected_forecast_request(obs, actions, 'jev-fixture', 6)


@pytest.mark.parametrize('weights, expected', [
    ({4: .4, 8: .3, 12: .2, 40: .1}, [4, 8, 40]),
    ({4: .2, 8: .1, 36: .5, 37: .2}, [36, 4, 37]),
    ({4: .4, 8: .2, 36: .3, 40: .1}, [4, 36, 8]),
    ({4: .25, 8: .25, 36: .25, 37: .25}, [4, 8, 36]),
    ({4: 1.}, [4, 0, 36]),
])
def test_proposal_menu_preserves_top_choice_commit_and_stable_ranking(public_inputs, weights, expected):
    c = candidates(); obs, actions = public_inputs
    probabilities = distribution(weights)
    menu = c.proposal_menu(obs, probabilities, actions)
    assert menu == expected
    assert all(obs['action_mask'][i] for i in menu)
    assert any(actions[i].kind == 'commit' for i in menu)
    assert len(menu) <= 3 and len(set(menu)) == len(menu)
    np.testing.assert_array_equal(probabilities, distribution(weights))


def test_proposal_menu_obeys_masks_and_small_legal_sets(public_inputs):
    c = candidates(); obs, actions = public_inputs
    obs['action_mask'][:] = False
    obs['action_mask'][[8, 40]] = True
    assert c.proposal_menu(obs, distribution({8: .8, 40: .2}), actions) == [8, 40]
    obs['action_mask'][8] = False
    assert c.proposal_menu(obs, distribution({40: 1.}), actions) == [40]


def test_two_item_menu_retains_best_action_and_best_commit(public_inputs):
    c = candidates(); obs, actions = public_inputs
    assert c.proposal_menu(obs, distribution({4: .5, 8: .4, 40: .1}), actions, k=2) == [4, 40]


@pytest.mark.parametrize('k', [0, 1, 4, 3.0, True])
def test_proposal_menu_rejects_unsupported_size(public_inputs, k):
    c = candidates(); obs, actions = public_inputs
    with pytest.raises(ValueError, match='k'):
        c.proposal_menu(obs, distribution({4: 1.}), actions, k=k)


@pytest.mark.parametrize('probabilities', [
    np.ones(44) / 44,
    np.ones((1, 45)) / 45,
    distribution({4: .5}),
    distribution({4: 1.0001}),
    distribution({4: 1.1, 8: -.1}),
    distribution({4: float('nan')}),
    distribution({4: float('inf')}),
    ['0.022222222222222222'] * 45,
])
def test_proposal_menu_rejects_malformed_probability_vectors(public_inputs, probabilities):
    c = candidates(); obs, actions = public_inputs
    with pytest.raises(ValueError, match='probabilit'):
        c.proposal_menu(obs, probabilities, actions)


def test_proposal_menu_rejects_illegal_mass_or_missing_commit(public_inputs):
    c = candidates(); obs, actions = public_inputs
    obs['action_mask'][4] = False
    with pytest.raises(ValueError, match='illegal'):
        c.proposal_menu(obs, distribution({4: .1, 36: .9}), actions)
    obs['action_mask'][36:] = False
    with pytest.raises(ValueError, match='commit'):
        c.proposal_menu(obs, distribution({8: 1.}), actions)


def test_reranker_offers_only_declared_ranked_proposals(public_inputs):
    c = candidates(); obs, actions = public_inputs
    probabilities = distribution({4: .2, 8: .4, 12: .3, 40: .1})
    request = c.rerank_request(obs, actions, 'jev-fixture', probabilities)
    assert list(request['questions']) == ['next_action']
    question = request['questions']['next_action']
    assert question['type'] == 'choice'
    assert list(question['criteria']) == ['action_8', 'action_12', 'action_40']
    assert request['state']['candidate_action_ids'] == [8, 12, 40]
    assert request['state']['candidate_order'] == 'descending_rl_probability_then_ascending_action_id'
    assert request['state']['candidate_role'] == 'proposal_reranking'
    assert request['state']['proposal_limit'] == 3
    assert request['state']['rl_action_probabilities']['action_4'] == .2
    assert request['state']['public_family'] == c.public_family_description()
    assert 'forecasts' not in request['state']
    assert 'restricted' in question['instructions']


def test_candidate_arbitration_reuses_incumbent_menu_and_clones_forecasts(public_inputs):
    c = candidates(); obs, actions = public_inputs
    from integrations.jev_research import decision_request
    forecasts = {'a6_x4': .23, 'a6_x5': .91}
    probabilities = distribution({4: .2, 8: .4, 12: .3, 40: .1})
    incumbent = decision_request(obs, actions, 'jev-fixture', forecasts, probabilities)
    request = c.candidate_decision_request(obs, actions, 'jev-fixture', forecasts, probabilities)
    assert request['questions'] == incumbent['questions']
    assert request['state']['forecasts'] == forecasts
    assert request['state']['rl_action_probabilities'] == incumbent['state']['rl_action_probabilities']
    assert request['state']['candidate_action_ids'] == list(range(45))
    assert request['state']['candidate_order'] == 'ascending_action_id'
    assert request['state']['candidate_role'] == 'full_arbitration'
    assert request['state']['public_family'] == c.public_family_description()
    request['state']['forecasts']['a6_x4'] = .99
    assert forecasts == {'a6_x4': .23, 'a6_x5': .91}
    assert 'public_family' not in incumbent['state']


@pytest.mark.parametrize('forecasts', [
    {'private_sentinel': .5}, {'posterior': .5}, {'a6_x4': {'posterior': .5}},
    {'a6_x4': float('nan')}, {'a6_x4': -1}, {'a36_x4': .5},
])
def test_candidate_arbitration_rejects_reference_tables_and_non_event_fields(public_inputs, forecasts):
    c = candidates(); obs, actions = public_inputs
    with pytest.raises(ValueError, match='forecast'):
        c.candidate_decision_request(obs, actions, 'jev-fixture', forecasts)


@pytest.mark.parametrize('builder', ['selected', 'rerank', 'arbitration'])
def test_all_builders_ignore_private_and_unpurchased_fields(public_inputs, builder):
    c = candidates(); obs, actions = public_inputs
    readouts = np.full((8, 8), np.nan)
    readouts[:, 4:6] = [.43, .51]
    obs['history'].append({'action_id': 4, 'cost_ticks': actions[4].cost_ticks,
                           'readouts': readouts,
                           'measured_mask': np.array([False] * 4 + [True, True, False, False]),
                           'private_sentinel': 'unbought_secret'})
    obs['experiment_count'] = 1
    obs['budget_ticks'] -= actions[4].cost_ticks
    obs['private_sentinel'] = {'true_weights': 'unbought_secret', 'seed': 'unbought_secret'}
    before = copy.deepcopy(obs)
    probabilities = distribution({4: .7, 36: .3})
    if builder == 'selected':
        request = c.selected_forecast_request(obs, actions, 'jev-fixture', 6, include_prior=False)
    elif builder == 'rerank':
        request = c.rerank_request(obs, actions, 'jev-fixture', probabilities, include_prior=False)
    else:
        request = c.candidate_decision_request(obs, actions, 'jev-fixture',
                                               {'a6_x4': .5}, probabilities, include_prior=False)
    payload = json.dumps(request, allow_nan=False)
    assert 'private_sentinel' not in payload and 'unbought_secret' not in payload
    assert 'true_weights' not in payload and 'posterior' not in payload
    assert 'public_family' not in request['state']
    assert set(request['state']['history'][-1]['measurements']) == {'X4', 'X5'}
    np.testing.assert_array_equal(obs['action_mask'], before['action_mask'])
    np.testing.assert_array_equal(obs['history'][-1]['readouts'], before['history'][-1]['readouts'])
    request['state']['history'][0]['measurements']['X4'][0] = -123
    assert obs['history'][0]['readouts'][0, 4] == .52


def test_candidate_requests_validate_with_existing_client_contract(public_inputs):
    c = candidates(); obs, actions = public_inputs
    from integrations.jev_client import validate_response
    probabilities = distribution({4: .7, 36: .3})
    requests = [c.selected_forecast_request(obs, actions, 'jev-fixture', 6),
                c.rerank_request(obs, actions, 'jev-fixture', probabilities),
                c.candidate_decision_request(obs, actions, 'jev-fixture', {'a6_x4': .5}, probabilities)]
    for request in requests:
        answers = {}
        for name, question in request['questions'].items():
            if question['type'] == 'noul':
                answers[name] = {'type': 'noul', 'noul': .5}
            else:
                options = list(question['criteria'])
                answers[name] = {'type': 'choice', 'choice': options[0], 'confidence': 1.,
                                 'probabilities': {key: float(i == 0) for i, key in enumerate(options)}}
        response = {'model': 'jev-fixture', 'usage': {'input_tokens': 1, 'output_tokens': 1},
                    'answers': answers}
        assert validate_response(request, response, 'jev-fixture') == response
def test_serialized_choice_order_is_explicitly_distinct_from_proposal_rank():
    import json
    from integrations.jev_client import _json_bytes
    from experimentlab.envs import PilotConfig, ScientistEnv
    from experimentlab.tasks import finite_family, make_task
    from integrations.jev_candidates import rerank_request, candidate_decision_request
    config=PilotConfig()
    env=ScientistEnv(make_task(finite_family()[0],981212,config),config)
    obs,_=env.reset()
    probabilities=np.zeros(45);probabilities[8]=.6;probabilities[12]=.3;probabilities[40]=.1
    for request in (rerank_request(obs,env.actions,'jev-fixture',probabilities),
                    candidate_decision_request(obs,env.actions,'jev-fixture',rl_probabilities=probabilities)):
        wire=json.loads(_json_bytes(request))
        assert request['state']['candidate_order_scope']=='candidate_action_ids array only'
        assert request['state']['choice_wire_order']==list(wire['questions']['next_action']['criteria'])
