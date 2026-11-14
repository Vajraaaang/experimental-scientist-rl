import torch
import pytest
from experimentlab.learning import ActorCritic
from experimentlab.cost_policy import PublicCostPolicy


def test_zero_cost_calibration_is_identical_to_original_policy():
    torch.manual_seed(4);base=ActorCritic(kind='recurrent',hidden=32)
    x=torch.randn(2,3,75)
    before=base(x);after=PublicCostPolicy(base)(x)
    assert torch.equal(before[0],after[0]) and torch.equal(before[1],after[1])


def test_cost_calibration_uses_only_public_action_and_step_features():
    torch.manual_seed(4);base=ActorCritic(kind='recurrent',hidden=32)
    x=torch.randn(2,2,75);x[:,0,3]=0;x[:,1,3]=.25
    raw=base(x)[0]
    corrected=PublicCostPolicy(base,duration_penalty=2.,dose_penalty=3.,stop_bonus=.5)(x)[0]
    delta=corrected-raw
    assert delta[0,0,7].item()==pytest.approx(-.3,abs=1e-6)
    assert delta[0,1,7].item()==pytest.approx(-2.3,abs=1e-6)
    assert delta[0,1,4].item()==pytest.approx(-.3,abs=1e-6)
    assert delta[0,0,36].item()==pytest.approx(0.,abs=1e-6)
    assert delta[0,1,36].item()==pytest.approx(.5,abs=1e-6)


def test_temporary_duration_exposure_changes_only_public_followup_logits():
    from experimentlab.ppo_v4 import collect_batch
    from experimentlab.envs import PilotConfig,ScientistEnv
    from experimentlab.tasks import finite_family,make_task
    class FixedLogits(torch.nn.Module):
        horizon=5
        def forward(self,x):
            logits=torch.zeros((*x.shape[:2],45),device=x.device);logits[...,7]=2;logits[...,36]=1
            return logits,torch.zeros(x.shape[:2],device=x.device)
    config=PilotConfig();env=ScientistEnv(make_task(finite_family()[0],971,config),config)
    episodes,_=collect_batch(FixedLogits(),[env],'cpu',deterministic=True,followup_duration_bias=6.)
    assert episodes[0]['actions']==[7,36]
    assert episodes[0]['biases'][0][7]==0
    assert episodes[0]['biases'][1][7]==-6
    assert episodes[0]['biases'][1][4]==0
    assert episodes[0]['masks'][1][7]  # Legal environment action was not removed.
