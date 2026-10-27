"""Clipped policy-gradient control with a terminal-return leave-one-out baseline.

Groups contain independent deployments of the same training task. The optimizer
uses observed episode returns only; the policy receives no group identity,
mechanism parameter, unpurchased evidence or counterfactual action label.
This is a declared PPO variance-reduction variant, not unchanged GAE-PPO.
"""
import torch
from .tensor_batch import whole_episode_advantages


def paired_trajectory_advantages(batch,group_size=4):
    B,T=batch['rewards'].shape
    if group_size<2 or B%group_size:raise ValueError('complete groups of at least two rollouts required')
    returns=(batch['rewards']*batch['valid']).sum(-1).reshape(-1,group_size)
    baseline=(returns.sum(-1,keepdim=True)-returns)/(group_size-1)
    centered=(returns-baseline).reshape(B,1)
    advantages=torch.where(batch['valid'],centered.expand(-1,T),0.)
    _,targets=whole_episode_advantages(batch,1.)
    return advantages.detach(),targets.detach()
