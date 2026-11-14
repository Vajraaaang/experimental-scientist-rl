import numpy as np
import copy
import pytest
import torch
from experimentlab.learning import ActorCritic
from experimentlab.ppo_v4 import batch_episodes,behavior_distribution,update_ppo,collect_batch
from experimentlab.envs import PilotConfig,ScientistEnv
from experimentlab.tasks import finite_family,make_task


def records():
    rng=np.random.default_rng(5)
    output=[]
    for n in (2,4):
        output.append({'tokens':rng.normal(size=(n,75)).astype(np.float32),'masks':np.ones((n,45),bool),
                       'actions':np.zeros(n,np.int64),'rewards':np.array([-.02]*(n-1)+[-.3]),
                       'old_logp':np.zeros(n),'old_values':np.zeros(n),
                       'biases':np.tile(np.r_[np.zeros(36),np.full(9,-4.)],(n,1)).astype(np.float32),
                       'policy_steps':[False]*(n-1)+[True]})
    return output


def test_public_behavior_bias_recomputed_exactly_and_padding_ignored():
    torch.manual_seed(6);model=ActorCritic(kind='recurrent',hidden=32)
    batch=batch_episodes(records(),'cpu')
    with torch.no_grad():
        logits,values=model(batch['tokens'])
        dist=behavior_distribution(logits,batch['masks'],batch['biases'],batch['valid'])
        batch['old_logp']=dist.log_prob(batch['actions']);batch['old_values']=values
    metrics=update_ppo(model,torch.optim.Adam(model.parameters(),lr=1e-3),batch,epochs=2,entropy_coef=.03)
    assert metrics['initial_logp_error']<1e-6 and metrics['post_kl']>=0
    assert metrics['policy_decisions']==2 and metrics['valid_decisions']==6
    assert metrics['epochs_taken']<=2
    assert all(np.isfinite(v) for v in metrics.values())


def test_policy_loss_masks_cannot_be_empty():
    data=records()
    for r in data:r['policy_steps']=[False]*len(r['actions'])
    with pytest.raises(ValueError,match='policy decision'):batch_episodes(data,'cpu')


def test_raw_advantage_control_preserves_reward_units_in_policy_gradient():
    torch.manual_seed(18);model=ActorCritic(kind='recurrent',hidden=32)
    data=records();data[1]['rewards'][-1]=-.7;batch=batch_episodes(data,'cpu')
    with torch.no_grad():
        logits,_=model(batch['tokens'])
        dist=behavior_distribution(logits,batch['masks'],batch['biases'],batch['valid'])
        batch['old_logp']=dist.log_prob(batch['actions'])
    norms=[]
    for factor in (1.,2.):
        scaled=dict(batch);scaled['rewards']=batch['rewards']*factor
        result=update_ppo(model,torch.optim.SGD(model.parameters(),lr=0.),scaled,epochs=1,
                          value_coef=0.,entropy_coef=0.,normalize_advantage=False)
        norms.append(result['gradient_norm'])
    assert norms[0]>0 and norms[1]==pytest.approx(2*norms[0],rel=1e-5)


def test_batched_deterministic_collection_matches_individual_rollouts():
    from experimentlab.learning import collect_episode
    torch.manual_seed(9);model=ActorCritic(kind='recurrent',hidden=32)
    config=PilotConfig(budget=8)
    tasks=[make_task(m,701+i,config) for i,m in enumerate(finite_family()[:4])]
    envs=[ScientistEnv(t,config) for t in tasks]
    grouped,metrics=collect_batch(model,envs,'cpu',deterministic=True)
    separate=[collect_episode(model,ScientistEnv(t,config),'cpu',deterministic=True) for t in tasks]
    assert [r['actions'] for r in grouped]==[r['actions'] for r in separate]
    assert np.allclose([sum(r['rewards']) for r in grouped],[sum(r['rewards']) for r in separate])
    assert metrics['episodes']==4


def test_precomputed_tensor_advantages_preserve_optimizer_update():
    from experimentlab.tensor_batch import pack_episodes,whole_episode_advantages
    torch.manual_seed(42);first=ActorCritic(kind='recurrent',hidden=32);second=copy.deepcopy(first)
    batch=pack_episodes(records(),'cpu')
    with torch.no_grad():
        logits,values=first(batch['tokens'])
        batch['old_logp']=behavior_distribution(logits,batch['masks'],batch['biases'],batch['valid']).log_prob(batch['actions'])
        batch['old_values']=values
    a,t=whole_episode_advantages(batch,1.)
    original=update_ppo(first,torch.optim.Adam(first.parameters(),lr=.001),batch,epochs=2)
    fast=update_ppo(second,torch.optim.Adam(second.parameters(),lr=.001),batch,epochs=2,
                    prepared_advantages=a,prepared_targets=t)
    assert original['loss']==pytest.approx(fast['loss'],abs=1e-6)
    assert all(torch.allclose(v,second.state_dict()[k],atol=1e-6) for k,v in first.state_dict().items())
