import numpy as np
from experimentlab.diagnostics import BranchingEnv
from experimentlab.worlds import fixture
from experimentlab.tasks import make_task
from experimentlab.envs import ScientistEnv, PilotConfig


def test_four_variable_fixture_is_a_runnable_25_action_environment():
    config = PilotConfig(offset_sd=0.)
    env = ScientistEnv(make_task(fixture(1), 481, config, (.41,)), config)
    obs, _ = env.reset()
    assert len(env.actions) == 25 and obs['history'][0]['readouts'].shape == (8,4)
    # A dose.8 short proxy assay then commitment to suppress A.
    obs, reward, done, _, _ = env.step(8)
    assert not done and obs['history'][-1]['measured_mask'].tolist() == [False,False,True,False]
    _, reward, done, _, _ = env.step(22)
    assert done and env.private_result['final_mse'] < 1e-6


def test_branching_step_interface_hides_route_until_purchased():
    for route in (0,1):
        for target in (0,1):
            env = BranchingEnv(route, target, seed=41)
            obs, _ = env.reset()
            assert obs['history'] == [] and 'route' not in obs and 'target' not in obs
            obs, *_ = env.step(0)
            observed_route = obs['history'][-1]['readout']
            obs, *_ = env.step(1+observed_route)
            assert obs['action_mask'].tolist() == [False,False,False,True,True]
            guess = obs['history'][-1]['readout']
            _, reward, done, _, _ = env.step(3+guess)
            assert done and reward == 1.
