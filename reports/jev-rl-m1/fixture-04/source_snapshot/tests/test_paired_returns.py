import torch
import pytest
from experimentlab.paired_returns import paired_trajectory_advantages


def test_leave_one_out_baseline_preserves_cost_signal_and_excludes_own_return():
    batch={'rewards':torch.tensor([[-.05,-.3],[-.04,-.3],[-.6,0.],[-.6,0.]]),
           'valid':torch.tensor([[1,1],[1,1],[1,0],[1,0]],dtype=torch.bool),
           'old_values':torch.zeros(4,2)}
    advantages,targets=paired_trajectory_advantages(batch,2)
    assert advantages[0,0].item()==pytest.approx(-.01,abs=1e-6)
    assert advantages[1,0].item()==pytest.approx(.01,abs=1e-6)
    assert torch.equal(advantages[2:],torch.zeros(2,2))
    assert targets[0,0].item()==pytest.approx(-.35,abs=1e-6)
    changed={**batch,'rewards':batch['rewards'].clone()};changed['rewards'][0,0]-=.1
    new,_=paired_trajectory_advantages(changed,2)
    assert new[0,0]-advantages[0,0]==pytest.approx(-.1,abs=1e-6)
    assert advantages.sum().abs()<1e-6


def test_paired_return_baseline_requires_complete_groups():
    batch={'rewards':torch.zeros(3,1),'valid':torch.ones(3,1,dtype=torch.bool),'old_values':torch.zeros(3,1)}
    with pytest.raises(ValueError,match='group'):paired_trajectory_advantages(batch,2)


def test_unequal_lengths_keep_exact_leave_one_out_advantages_in_updater():
    from experimentlab.learning import ActorCritic
    from experimentlab.ppo_v4 import behavior_distribution,update_ppo
    torch.manual_seed(42);model=ActorCritic(kind='recurrent',hidden=16)
    batch={'tokens':torch.zeros(2,2,75),'masks':torch.ones(2,2,45,dtype=torch.bool),
           'biases':torch.zeros(2,2,45),'actions':torch.zeros(2,2,dtype=torch.long),
           'rewards':torch.tensor([[0.,0.],[0.,-1.]]),'old_values':torch.zeros(2,2),
           'valid':torch.tensor([[1,0],[1,1]],dtype=torch.bool)}
    batch['policy_valid']=batch['valid'].clone()
    with torch.no_grad():
        logits,_=model(batch['tokens'])
        batch['old_logp']=behavior_distribution(logits,batch['masks'],batch['biases'],batch['valid']).log_prob(batch['actions'])
    a,t=paired_trajectory_advantages(batch,2)
    assert a[0,0]==1 and a[1,0]==-1 and a[1,1]==-1
    result=update_ppo(model,torch.optim.SGD(model.parameters(),lr=0.),batch,epochs=1,
                      prepared_advantages=a,prepared_targets=t,normalize_advantage=False,
                      center_advantage=False,value_coef=0.,entropy_coef=0.)
    assert result['policy_loss']==pytest.approx(1/3,abs=1e-6)
