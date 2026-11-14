import numpy as np
from experimentlab.envs import ScientistEnv, PilotConfig
from experimentlab.tasks import finite_family, make_task
from experimentlab.inference import PredictionBank, Belief
from experimentlab.baselines import DecisionPlanner, fixed_identification


def test_cached_predictions_posterior_and_exact_likelihood():
    config = PilotConfig()
    bank = PredictionBank(finite_family(), config)
    assert bank.forward_steps == 32 * 9 * 120
    env = ScientistEnv(make_task(finite_family()[0], 10, config), config)
    obs, _ = env.reset()
    belief = Belief(bank, obs['target'])
    prior = belief.weights.copy()
    assay = fixed_identification()[0]
    obs, *_ = env.step(assay)
    belief.update(obs['history'][-1])
    assert np.isclose(belief.weights.sum(), 1.)
    assert belief.weights[0] > prior[0]
    assert np.isfinite(belief.ess)
    record = obs['history'][-1]
    pred = bank.predictions[:, assay, list(bank.actions[assay].panel)]
    y = record['readouts'][:, list(bank.actions[assay].panel)]
    ll = -np.sum((y[None] - pred[:, None])**2, axis=(1, 2)) / (2 * config.readout_sd**2)
    expected = prior * np.exp(ll - ll.max()); expected /= expected.sum()
    assert np.allclose(belief.weights, expected)


def test_planner_cannot_have_negative_expected_costless_information_value():
    config = PilotConfig(cost_weight=0)
    bank = PredictionBank(finite_family(), config)
    belief = Belief(bank, (.42, .42))
    planner = DecisionPlanner(bank, 'voi', draws=64)
    obs, _ = ScientistEnv(make_task(finite_family()[0], 12, config, (.42, .42)), config).reset()
    action, values = planner.choose(obs, belief, np.random.default_rng(2), diagnostics=True)
    assert values['best_expected_value'] >= -belief.risk() - .015
    assert obs['action_mask'][action]


def test_perfectly_uninformative_predictions_make_voi_stop():
    config = PilotConfig(readout_sd=100.)
    bank = PredictionBank(finite_family(), config)
    belief = Belief(bank, (.42, .42))
    obs, _ = ScientistEnv(make_task(finite_family()[0], 12, config, (.42, .42)), config).reset()
    planner = DecisionPlanner(bank, 'voi', draws=128)
    assert planner.choose(obs, belief, np.random.default_rng(1)) >= 36
