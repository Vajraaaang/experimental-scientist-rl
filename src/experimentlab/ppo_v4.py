"""Audited complete-episode PPO controls for the collapse follow-up.

Behavior biases are public training curricula, stored explicitly and included
in both old and new action probabilities. Forced assay positions have no
policy/entropy loss. The scientific reward and evaluation action masks do not
change. This module does not read any hidden task parameters.
"""
import time
import numpy as np
import torch
from torch import nn
from .learning import token,episode_batch,masked_distribution,returns_gae


def behavior_distribution(logits,masks,biases,valid=None):
    if logits.shape!=biases.shape or not torch.isfinite(biases).all():
        raise ValueError('invalid public behavior bias')
    return masked_distribution(logits+biases,masks,valid)


def batch_episodes(episodes,device):
    batch=episode_batch(episodes,device)
    batch['biases']=torch.zeros((*batch['actions'].shape,45),device=device)
    batch['policy_valid']=batch['valid'].clone()
    for i,e in enumerate(episodes):
        n=len(e['actions'])
        if 'biases' in e:batch['biases'][i,:n]=torch.as_tensor(np.asarray(e['biases']),dtype=torch.float32,device=device)
        if 'policy_steps' in e:batch['policy_valid'][i,:n]=torch.as_tensor(e['policy_steps'],dtype=torch.bool,device=device)
    batch['policy_valid'] &= batch['valid']
    if not bool(batch['policy_valid'].any()):raise ValueError('batch has no sampled policy decision')
    return batch


def update_ppo(model,optimizer,batch,epochs=2,clip=.2,entropy_coef=.01,value_coef=.5,
               gae_lambda=1.,target_kl=.02,advantage_std_floor=1e-8,normalize_advantage=True,audit_gradients=False,
               prepared_advantages=None,prepared_targets=None,center_advantage=True):
    valid=batch['valid'];policy=batch['policy_valid']
    if (prepared_advantages is None)!=(prepared_targets is None):
        raise ValueError('prepared advantages and targets must be supplied together')
    if prepared_advantages is None:
        advantages=torch.zeros_like(batch['rewards']);targets=torch.zeros_like(advantages)
        for i in range(len(valid)):
            n=int(valid[i].sum())
            a,ret=returns_gae(batch['rewards'][i,:n],batch['old_values'][i,:n],gae_lambda=gae_lambda)
            advantages[i,:n]=a;targets[i,:n]=ret
    else:
        if prepared_advantages.shape!=batch['rewards'].shape or prepared_targets.shape!=batch['rewards'].shape:
            raise ValueError('prepared return shapes do not match complete episode batch')
        advantages=prepared_advantages;targets=prepared_targets
    advantages=advantages.detach();targets=targets.detach()
    raw_mean=advantages[policy].mean();raw_sd=advantages[policy].std(unbiased=False)
    normalized=advantages-raw_mean if center_advantage else advantages
    if normalize_advantage:normalized=normalized/raw_sd.clamp_min(advantage_std_floor)
    initial_error=0.;epochs_taken=0;grad=torch.tensor(0.);actor_shared=critic_shared=0.
    for epoch in range(epochs):
        logits,values=model(batch['tokens'])
        dist=behavior_distribution(logits,batch['masks'],batch['biases'],valid)
        logp=dist.log_prob(batch['actions'])
        if epoch==0:initial_error=float((logp-batch['old_logp'])[valid].abs().max().detach().cpu())
        ratio=torch.exp(logp-batch['old_logp'])
        surrogate=torch.minimum(ratio*normalized,ratio.clamp(1-clip,1+clip)*normalized)
        value_loss=((values-targets)**2)[valid].mean()
        entropy=dist.entropy()[policy].mean()
        policy_loss=-surrogate[policy].mean()
        loss=policy_loss+value_coef*value_loss-entropy_coef*entropy
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite PPO loss')
        if audit_gradients and epoch==0:
            shared=list(model.encoder.parameters())+(list(model.gru.parameters()) if hasattr(model,'gru') else [])
            def norm_of(objective):
                gradients=torch.autograd.grad(objective,shared,retain_graph=True,allow_unused=True)
                return float(torch.sqrt(sum(g.detach().square().sum() for g in gradients if g is not None)).cpu())
            actor_shared=norm_of(policy_loss-entropy_coef*entropy)
            critic_shared=norm_of(value_coef*value_loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
        optimizer.step();epochs_taken+=1
        with torch.no_grad():
            post_logits,post_values=model(batch['tokens'])
            post=behavior_distribution(post_logits,batch['masks'],batch['biases'],valid)
            delta=post.log_prob(batch['actions'])-batch['old_logp']
            # Accurate expm1 avoids cancellation near zero; clamp only roundoff
            # in this nonnegative diagnostic, never an objective observation.
            post_kl=(torch.expm1(delta)-delta)[policy].clamp_min(0).mean()
            clip_fraction=(torch.abs(torch.exp(delta)-1)>clip)[policy].float().mean()
        if target_kl is not None and float(post_kl.cpu())>target_kl:break
    with torch.no_grad():
        target_variance=targets[valid].var(unbiased=False)
        explained=1-(targets-post_values)[valid].var(unbiased=False)/target_variance if target_variance>1e-12 else torch.tensor(0.,device=targets.device)
    scalar=lambda x:float(x.detach().cpu())
    return {'loss':scalar(loss),'policy_loss':scalar(policy_loss),'value_error':scalar(value_loss),
            'entropy':scalar(entropy),'gradient_norm':scalar(grad),'post_kl':scalar(post_kl),
            'clip_fraction':scalar(clip_fraction),'initial_logp_error':initial_error,'epochs_taken':epochs_taken,
            'advantage_mean':scalar(raw_mean),'advantage_sd':scalar(raw_sd),'explained_variance':scalar(explained),
            'advantage_centering':int(center_advantage),
            'entropy_weight_return_units':entropy_coef*scalar(raw_sd) if normalize_advantage else entropy_coef,
            'gradient_audit':int(audit_gradients),'actor_shared_gradient_norm':actor_shared,'critic_shared_gradient_norm':critic_shared,
            'policy_decisions':int(policy.sum()),'valid_decisions':int(valid.sum())}


def collect_batch(model,envs,device,deterministic=False,public_schedules=None,
                  behavior_strength=0.,minimum_assays=1,maximum_assays=2,terminal_candidates=None,initial_outcome_bias=0.,
                  followup_duration_bias=0.):
    """Synchronous environments, one batched policy forward per decision index.

    Optional schedules are fixed from public action IDs independently of true
    mechanisms. Only the final categorical choice is learned in that control.
    The environment itself keeps its original legal masks and rewards.
    """
    B=len(envs)
    observations=[env.reset()[0] for env in envs]
    records=[{k:[] for k in ('tokens','masks','actions','rewards','old_logp','old_values',
                            'terminated','truncated','biases','policy_steps')} for _ in envs]
    histories=[[] for _ in envs];active=np.ones(B,dtype=bool);step=0
    started=time.perf_counter();returns=np.zeros(B);first_experiment_probability=[]
    while active.any():
        data=np.zeros((B,step+1,75),dtype=np.float32)
        masks=np.zeros((B,45),dtype=bool);biases=np.zeros((B,45),dtype=np.float32)
        sampled=np.ones(B,dtype=bool)
        for i in np.flatnonzero(active):
            obs=observations[i];histories[i].append(token(obs));data[i,:len(histories[i])]=histories[i]
            masks[i]=obs['action_mask'];count=obs['experiment_count']
            if public_schedules is not None:
                schedule=public_schedules[i]
                if count<len(schedule):
                    a=int(schedule[count])
                    if not masks[i,a]:raise ValueError('public curriculum exceeds original resources')
                    masks[i]=False;masks[i,a]=True;sampled[i]=False
                else:
                    masks[i,:36]=False
                    if terminal_candidates is not None:
                        allowed=np.zeros(45,dtype=bool);allowed[list(terminal_candidates[i])]=True
                        masks[i]&=allowed
            elif behavior_strength:
                if count<minimum_assays:biases[i,36:]=-behavior_strength
                elif count>=maximum_assays:biases[i,:36]=-behavior_strength
            if count==0 and initial_outcome_bias and public_schedules is None:
                biases[i,3:36:4]+=initial_outcome_bias
            if count>=1 and followup_duration_bias and public_schedules is None:
                # Catalogue ordering is public: positions2/3 use duration6.
                biases[i,:36]-=followup_duration_bias*np.tile(np.array([0.,0.,1.,1.],dtype=np.float32),9)
        with torch.no_grad():
            logits,values=model(torch.as_tensor(data,device=device))
            live=torch.as_tensor(active[:,None],device=device)
            dist=behavior_distribution(logits[:,-1:],torch.as_tensor(masks[:,None],device=device),
                                       torch.as_tensor(biases[:,None],device=device),live)
            actions=dist.probs.argmax(-1) if deterministic else dist.sample()
            logp=dist.log_prob(actions).flatten().cpu().numpy()
            actions=actions.flatten().cpu().numpy();values=values[:,-1].cpu().numpy()
            if step==0:first_experiment_probability=dist.probs[:,:,:36].sum(-1).flatten().cpu().tolist()
        for i in np.flatnonzero(active):
            record=records[i]
            for key,value in [('tokens',histories[i][-1]),('masks',masks[i].copy()),('actions',int(actions[i])),
                              ('old_logp',float(logp[i])),('old_values',float(values[i])),
                              ('biases',biases[i].copy()),('policy_steps',bool(sampled[i]))]:record[key].append(value)
            obs,reward,terminated,truncated,_=envs[i].step(int(actions[i]))
            observations[i]=obs;returns[i]+=reward
            record['rewards'].append(reward);record['terminated'].append(terminated);record['truncated'].append(truncated)
            if truncated:raise RuntimeError('complete episode required')
            if terminated:active[i]=False
        step+=1
        if step>model.horizon:raise RuntimeError('rollout exceeded model horizon')
    results=[env.private_result for env in envs]
    metrics={'episodes':B,'transitions':sum(len(e['actions']) for e in records),
             'policy_decisions':sum(sum(e['policy_steps']) for e in records),
             'mean_return':float(returns.mean()),'mean_assays':float(np.mean([env.count for env in envs])),
             'mean_variable_cost':float(np.mean([r['variable_cost'] for r in results])),
             'aggregate_rmse':float(np.sqrt(np.mean([r['final_mse'] for r in results]))),
             'initial_experiment_probability':float(np.mean(first_experiment_probability)),
             'seconds':time.perf_counter()-started}
    return records,metrics
