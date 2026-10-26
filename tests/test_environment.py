import json
import numpy as np
import pytest

from experimentlab.envs import ScientistEnv, PilotConfig, catalogue
from experimentlab.tasks import finite_family, make_task, stream


def make_env(seed=3, **kwargs):
    config = PilotConfig(**kwargs)
    return ScientistEnv(make_task(finite_family()[0], seed, config), config)


def test_action_counts_costs_and_public_masks():
    assert len(catalogue(8, 4, ((4, 5), (6, 7)))) == 45
    assert len(catalogue(4, 2, ((2,), (3,)))) == 25
    env = make_env(budget=0)
    obs, info = env.reset()
    assert obs['action_mask'].sum() == 9
    assert all(a.kind == 'commit' for a, valid in zip(env.actions, obs['action_mask']) if valid)
    assert info == {}


def test_invalid_actions_cannot_loop_and_budget_never_negative():
    env = make_env(budget=0)
    env.reset()
    for i in range(env.config.decision_cap):
        obs, reward, terminated, truncated, info = env.step(-1)
        assert obs['budget_ticks'] >= 0
        assert not truncated
    assert terminated and info['error'] == 'invalid_run'


def test_public_boundary_no_hidden_or_unpurchased_data():
    env = make_env()
    obs, _ = env.reset()
    forbidden = {'W', 'b', 'gamma', 'mechanism', 'family', 'oracle', 'seed', 'evaluation_seed', 'latent', 'offsets'}
    assert forbidden.isdisjoint(obs)
    a = next(i for i, a in enumerate(env.actions) if a.kind == 'experiment')
    obs, r, done, _, info = env.step(a)
    evidence = obs['history'][-1]
    assert np.isnan(evidence['readouts'][:, ~evidence['measured_mask']]).all()
    assert info == {}
    assert r == pytest.approx(-env.config.cost_weight * env.actions[a].cost_ticks / 20)
    assert 'final_mse' not in obs


def test_fresh_replicates_final_independence_and_replay():
    a, b = make_env(), make_env()
    a.reset(); b.reset()
    assay = 0
    a.step(assay)
    commit = 36
    _, ra, *_ = a.step(commit)
    _, rb, *_ = b.step(commit)
    assert a.private_result['final_mse'] == b.private_result['final_mse']
    assert ra == rb
    c = make_env(); c.reset(); c.step(assay)
    assert np.array_equal(a.public_trace[1]['readouts'], c.public_trace[1]['readouts'], equal_nan=True)


def test_config_validation():
    with pytest.raises(ValueError): PilotConfig(budget=-1)
    with pytest.raises(ValueError): PilotConfig(readout_sd=-.1)
    with pytest.raises(ValueError): PilotConfig(max_experiments=9)


def test_every_assay_starts_with_fresh_cultures_not_the_last_endpoint():
    env = make_env(budget=16, readout_sd=0., offset_sd=0.)
    env.reset()
    env.step(11)
    first = env.public_trace[-1]['readouts'].copy()
    env.step(11)
    assert np.array_equal(first, env.public_trace[-1]['readouts'], equal_nan=True)


def test_mask_is_independent_of_hidden_mechanism():
    config = PilotConfig()
    environments = [ScientistEnv(make_task(m, 591, config), config) for m in (finite_family()[0],finite_family()[-1])]
    masks = [e.reset()[0]['action_mask'] for e in environments]
    assert np.array_equal(*masks)
    masks = [e.step(11)[0]['action_mask'] for e in environments]
    assert np.array_equal(*masks)
