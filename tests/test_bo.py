import numpy as np
from experimentlab.bo import objective_observation, DiscreteBO
from experimentlab.envs import PilotConfig, ScientistEnv
from experimentlab.inference import PredictionBank,Belief
from experimentlab.tasks import finite_family,make_task


def test_known_noise_correction_is_retained_when_negative():
    y=np.full((8,2),.42)
    mean,variance=objective_observation(y,np.array([.42,.42]),.02,0.)
    assert mean<0 and np.isfinite(variance) and variance>0


def test_bo_only_buys_fair_final_duration_outcome_assays():
    config=PilotConfig()
    bank=PredictionBank(finite_family(),config)
    obs,_=ScientistEnv(make_task(finite_family()[0],101,config),config).reset()
    belief=Belief(bank,obs['target'])
    selector=DiscreteBO(bank,belief)
    action=selector.choose(obs,belief)
    assert obs['action_mask'][action]
    if action<36:
        assert bank.actions[action].duration==6.
        assert bank.actions[action].panel==(6,7)
