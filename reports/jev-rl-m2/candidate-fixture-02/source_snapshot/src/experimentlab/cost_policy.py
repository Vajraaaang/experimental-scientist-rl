"""Transparent public-price calibration; no new learned or private features.

This changes deployment logits and is reported as calibration, not PPO training.
The original environment masks and rewards remain unchanged.
"""
import math
import torch
from torch import nn
from .envs import catalogue


class PublicCostPolicy(nn.Module):
    def __init__(self,base,duration_penalty=0.,dose_penalty=0.,stop_bonus=0.):
        super().__init__();self.base=base;self.horizon=base.horizon
        values=(duration_penalty,dose_penalty,stop_bonus)
        if not all(math.isfinite(v) and v>=0 for v in values):raise ValueError('finite nonnegative calibration required')
        self.parameters_public={'duration_penalty':float(duration_penalty),'dose_penalty':float(dose_penalty),'stop_bonus':float(stop_bonus)}
        actions=catalogue()
        self.register_buffer('long_assay',torch.tensor([a.kind=='experiment' and a.duration==6. for a in actions],dtype=torch.float32))
        self.register_buffer('dose_cost',torch.tensor([.25*sum(a.intervention) if a.kind=='experiment' else 0. for a in actions],dtype=torch.float32))
        self.register_buffer('commitment',torch.tensor([a.kind=='commit' for a in actions],dtype=torch.float32))

    def forward(self,tokens):
        logits,values=self.base(tokens)
        later=(tokens[...,3]>=.25).to(logits.dtype).unsqueeze(-1)
        p=self.parameters_public
        bias=(-p['duration_penalty']*later*self.long_assay-p['dose_penalty']*self.dose_cost+
              p['stop_bonus']*later*self.commitment)
        return logits+bias,values
