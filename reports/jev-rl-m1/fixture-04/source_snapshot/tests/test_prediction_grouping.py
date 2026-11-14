import runpy
import numpy as np
from scipy.special import logsumexp
from experimentlab.cli import ROOT
from experimentlab.envs import PilotConfig
from experimentlab.inference import Belief,hypothetical_posteriors


def test_identical_hypothetical_readouts_are_integrated_once_without_value_change():
    api=runpy.run_path(str(ROOT/'scripts/research_stage.py'))
    bank=api['UnfilteredBank'](api['quadrature_family'](2),PilotConfig())
    belief=Belief(bank,(.42,.58))
    a=7;draws=12;seed=92
    means,cov=bank.observation_model(a,list(bank.actions[a].panel));cov=cov/bank.config.cultures
    z=np.random.default_rng(seed).normal(size=(draws,2))
    y=(means[:,None]+np.einsum('hij,qj->hqi',np.linalg.cholesky(cov),z)).reshape(-1,2)
    delta=y[:,None]-means[None]
    ll=-.5*(np.einsum('qhi,hij,qhj->qh',delta,np.linalg.inv(cov),delta)+np.linalg.slogdet(cov)[1])
    logw=ll+np.log(belief.weights)
    old=np.exp(logw-logsumexp(logw,axis=1,keepdims=True));oldmass=np.repeat(belief.weights/draws,draws)
    new,newmass=hypothetical_posteriors(bank,belief,a,draws,np.random.default_rng(seed))
    assert len(new)<len(old)/4
    assert abs(np.sum(oldmass*belief.risk(old))-np.sum(newmass*belief.risk(new)))<1e-12
    assert np.allclose(oldmass@old,newmass@new,atol=1e-12)
