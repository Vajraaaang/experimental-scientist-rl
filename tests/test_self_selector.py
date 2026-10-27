import torch
import numpy as np
from experimentlab.envs import ScientistEnv,PilotConfig
from experimentlab.tasks import continuous_family,make_task


def test_public_self_curriculum_forces_paid_low_dose_assays_but_never_terminal_labels():
    import importlib.util
    assert importlib.util.find_spec('experimentlab.self_selector') is not None,'missing public self-curriculum'
    from experimentlab.self_selector import PublicSelfSelector
    class PreviousPolicy(torch.nn.Module):
        def forward(self,tokens):
            logits=torch.zeros((*tokens.shape[:2],45))
            actions=torch.where(tokens[...,3]<.5,torch.tensor(11),torch.tensor(36))
            logits.scatter_(-1,actions[...,None],10.)
            return logits,torch.zeros(tokens.shape[:2])
    config=PilotConfig();task=make_task(continuous_family(7401,1)[0],771,config,[.42,.58])
    env=PublicSelfSelector(ScientistEnv(task,config),PreviousPolicy(),low_dose=True)
    obs,_=env.reset()
    for _ in range(2):
        assert np.flatnonzero(obs['action_mask']).tolist()==[7]
        obs,_,done,_,_=env.step(7)
        assert not done
    assert not obs['action_mask'][:36].any() and obs['action_mask'][36:].all()
    _,_,done,_,_=env.step(36)
    assert done and env.forced_steps==[True,True,False]
    assert env.private_result['variable_cost']==5.6
