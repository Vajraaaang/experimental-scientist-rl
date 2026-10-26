"""Discrete direct-assay Gaussian-process BO with public noisy objective estimates.

The covariance is supplied by the declared hypothetical prior, never truth.
This is a finite GP expected-improvement baseline, not a DAD reproduction.
"""
import numpy as np
from scipy.stats import norm


def objective_observation(readouts,target,readout_sd,effort_penalty):
    """Unbiased readout-variance correction; deliberately do not clip negatives."""
    readouts=np.asarray(readouts)
    P=readouts.shape[1]; K=readouts.shape[0]
    estimate=100*(np.mean((readouts-target)**2)-readout_sd**2)+effort_penalty
    delta=readouts.mean(axis=0)-target
    variance=100**2*np.sum(2*readout_sd**4+4*readout_sd**2*delta**2)/(K*P**2)
    return float(estimate),float(max(variance,1e-12))


class DiscreteBO:
    def __init__(self,bank,initial_belief):
        if bank.config.offset_sd!=0:
            raise ValueError('BO objective-noise model is currently qualified only at zero culture variation')
        self.bank=bank
        losses=initial_belief.losses
        weights=initial_belief.weights
        self.prior_mean=weights@losses
        residual=losses-self.prior_mean
        self.prior_cov=(residual.T*weights)@residual

    def choose(self,obs,belief):
        observed=[]; values=[]; noise=[]
        for packet in obs['history']:
            action_id=packet['action_id']
            if action_id==-1:
                index=0; effort=0.
            else:
                a=self.bank.actions[action_id]
                if a.duration!=6. or a.panel!=(6,7): continue
                index=action_id//4
                effort=self.bank.config.effort_weight*sum(a.intervention)
            value,variance=objective_observation(packet['readouts'][:,6:8],obs['target'],self.bank.config.readout_sd,effort)
            observed.append(index); values.append(value); noise.append(variance)
        mean=self.prior_mean.copy(); cov=self.prior_cov.copy()
        if observed:
            cross=self.prior_cov[:,observed]
            K=self.prior_cov[np.ix_(observed,observed)]+np.diag(noise)+np.eye(len(observed))*1e-10
            mean+=cross@np.linalg.solve(K,np.array(values)-self.prior_mean[observed])
            cov-=cross@np.linalg.solve(K,cross.T)
        sd=np.sqrt(np.maximum(np.diag(cov),1e-16))
        improvement=mean.min()-mean
        z=improvement/sd
        ei=improvement*norm.cdf(z)+sd*norm.pdf(z)
        best,chosen=0.,belief.commitment()
        for index in range(9):
            action=index*4+3
            if not obs['action_mask'][action]: continue
            value=ei[index]-self.bank.config.cost_weight*self.bank.actions[action].cost_ticks/20
            if value>best: best,chosen=value,action
        return int(chosen)
