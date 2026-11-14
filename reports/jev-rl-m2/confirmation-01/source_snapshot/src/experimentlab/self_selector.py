"""Training-only public assay curriculum from a frozen previous PPO policy.

Every assay is purchased on its own fresh culture at the original fee. Only the
terminal intervention remains sampled; no terminal label or private mechanism
is supplied by the previous policy. Evaluation uses the unwrapped environment.
"""
import numpy as np
import torch
from .learning import token,masked_distribution


class PublicSelfSelector:
    def __init__(self,environment,previous_policy,low_dose=False):
        self._env=environment;self.previous_policy=previous_policy;self.low_dose=bool(low_dose)
        self.forced_steps=[];self._history=[]

    @property
    def count(self):return self._env.count

    @property
    def private_result(self):return self._env.private_result

    def _public_training_mask(self,observation):
        self._history.append(token(observation))
        device=next(self.previous_policy.parameters(),torch.empty(0)).device
        with torch.no_grad():
            logits,_=self.previous_policy(torch.as_tensor(np.array(self._history)[None],device=device))
            legal=torch.as_tensor(observation['action_mask'][None],device=device)
            chosen=int(masked_distribution(logits[:,-1],legal).probs.argmax().cpu())
        result=dict(observation);mask=observation['action_mask'].copy()
        forced=chosen<36
        if forced:
            if self.low_dose and chosen//4>0 and (chosen//4)%2==0:chosen-=4
            if not mask[chosen]:raise ValueError('projected public curriculum assay is unaffordable')
            mask[:]=False;mask[chosen]=True
        else:mask[:36]=False
        result['action_mask']=mask;self.forced_steps.append(forced)
        return result

    def reset(self):
        self.forced_steps=[];self._history=[]
        observation,info=self._env.reset()
        return self._public_training_mask(observation),info

    def step(self,action):
        observation,reward,terminated,truncated,info=self._env.step(action)
        if not (terminated or truncated):observation=self._public_training_mask(observation)
        return observation,reward,terminated,truncated,info
