from dataclasses import replace
import numpy as np
from experimentlab.envs import PilotConfig, ScientistEnv
from experimentlab.worlds import Mechanism, integrate, equilibrium
from experimentlab.tasks import finite_family, continuous_family, make_task, mechanism_hash
from experimentlab.inference import PredictionBank, Belief
from experimentlab.evaluation import independent_continuous_tasks, run_strategy, paired_interval, tune_fixed


def test_node_permutation_preserves_scientific_dynamics():
    m = finite_family()[3]
    perm = np.array([5,2,0,7,1,6,3,4])
    reordered = Mechanism(m.W[perm][:,perm], m.b[perm], m.gamma[perm])
    u = np.zeros(8); u[1] = .8
    expected = integrate(m, equilibrium(m), u, 6)
    actual = integrate(reordered, equilibrium(reordered), u[perm], 6)
    assert np.allclose(actual, expected[perm], atol=1e-14)


def test_independent_continuous_pools_never_insert_truth_in_bank():
    bank = PredictionBank(continuous_family(6101, 32), PilotConfig())
    train = independent_continuous_tasks(bank, 'train', 8, 551)
    val = independent_continuous_tasks(bank, 'validation', 8, 552)
    h = lambda items: {mechanism_hash(m) for m in items}
    a, b, c = h([t.mechanism for t in train]), h([t.mechanism for t in val]), h(bank.mechanisms)
    assert not(a&b or a&c or b&c)


def test_common_final_selector_and_once_only_experiment_cost():
    config = PilotConfig()
    bank = PredictionBank(finite_family(), config)
    target = (.42,.42)
    idx = int(np.flatnonzero(bank.admitted(target))[0])
    task = make_task(bank.mechanisms[idx], 712, config, target)
    prior, _ = run_strategy(task, bank, 'prior')
    empty_fixed, _ = run_strategy(task, bank, 'fixed', ())
    assert prior['final_mse'] == empty_fixed['final_mse']
    row, _ = run_strategy(task, bank, 'fixed', (11,))
    expected = -100*row['final_mse']-config.effort_weight*row['effort']-config.cost_weight*row['variable_cost']
    assert abs(row['return_']-expected) < 1e-12


def test_cluster_bootstrap_does_not_treat_replicates_as_mechanisms():
    result = paired_interval([1,1,3,3], [0,0,0,0], groups=[0,0,1,1])
    assert result['units'] == 2 and result['mean'] == 2
