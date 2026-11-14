"""Cached hypothetical-model predictions and checked finite-support beliefs."""
import time
import numpy as np
from scipy.special import expit, logsumexp
from .envs import catalogue
from .worlds import equilibrium


class PredictionBank:
    def __init__(self, mechanisms, config, prediction_cultures=32):
        start = time.perf_counter()
        self.mechanisms, self.config, self.actions = tuple(mechanisms), config, catalogue()
        H, U = len(mechanisms), 9
        R = 1 if config.offset_sd == 0 else prediction_cultures
        W = np.stack([m.W for m in mechanisms])
        b = np.stack([m.b for m in mechanisms])
        gamma = np.stack([m.gamma for m in mechanisms])
        # Fixed hypothetical common random numbers, independent of any task.
        offset = np.random.default_rng(62017).normal(0, config.offset_sd, (H, R, 8))
        x = np.full((H, R, 8), .5)
        for _ in range(2000):
            nxt = expit(b[:, None] + offset + np.einsum('hrj,hij->hri', x, W))
            if np.max(abs(nxt - x)) < 1e-12: break
            x = nxt
        else: raise RuntimeError('bank equilibrium failed')
        self.baseline_mean = nxt.mean(axis=1)
        self.baseline_cov = self._cov(nxt, config.readout_sd)
        x = np.broadcast_to(nxt[:, None], (H, U, R, 8)).copy()
        u = np.array([a.intervention for a in self.actions[36:]])
        steps = int(round(6 / config.dt))
        if abs(steps * config.dt - 6) > 1e-8 or abs(round(1/config.dt)*config.dt - 1) > 1e-8:
            raise ValueError('dt must divide both durations exactly')
        alpha = -np.expm1(-gamma * config.dt)
        endpoints = {}
        for step in range(1, steps + 1):
            production = expit(b[:, None, None] + offset[:, None] + np.einsum('hurj,hij->huri', x, W))
            x = (1 - alpha[:, None, None]) * x + alpha[:, None, None] * (1 - u[None, :, None]) * production
            if step in (int(round(1/config.dt)), steps): endpoints[round(step * config.dt)] = x.copy()
        self.predictions = np.zeros((H, 36, 8))
        self.covariances = np.zeros((H, 36, 8, 8))
        for a, spec in enumerate(self.actions[:36]):
            values = endpoints[round(spec.duration)][:, a // 4]
            self.predictions[:, a] = values.mean(axis=1)
            self.covariances[:, a] = self._cov(values, config.readout_sd)
        self.final_mean = endpoints[6].mean(axis=2)[:, :, 6:8]
        self.final_variance = endpoints[6].var(axis=2)[:, :, 6:8]
        self.nominal_final = np.zeros((H, U, 2))
        if R == 1: self.nominal_final = self.final_mean.copy()
        else:
            from .worlds import integrate
            for h, m in enumerate(mechanisms):
                self.nominal_final[h] = integrate(m, np.broadcast_to(equilibrium(m), (U, 8)), u, 6, config.dt)[:, 6:8]
        self.forward_steps = H * U * R * steps
        self.build_seconds = time.perf_counter() - start
        self.prediction_cultures = R
        self.likelihood = 'exact Gaussian' if R == 1 else 'Gaussian moment approximation to culture-offset mixture'

    @staticmethod
    def _cov(values, noise):
        d = values - values.mean(axis=-2, keepdims=True)
        cov = np.einsum('...ri,...rj->...ij', d, d) / values.shape[-2]
        return cov + np.eye(8) * max(noise**2, 1e-16)

    def admitted(self, target, tolerance=.04):
        losses = np.mean((self.nominal_final - target)**2, axis=-1)
        return (losses.min(axis=1) <= tolerance**2) & (losses.min(axis=1) < losses[:, 0] - .0001)

    def losses(self, target):
        mse = np.mean((self.final_mean - target)**2 + self.final_variance, axis=-1)
        effort = np.array([sum(a.intervention) for a in self.actions[36:]])
        return 100 * mse + self.config.effort_weight * effort

    def observation_model(self, action_id, panel):
        if action_id == -1:
            means, cov = self.baseline_mean, self.baseline_cov
        else:
            means, cov = self.predictions[:, action_id], self.covariances[:, action_id]
        return means[:, panel], cov[:, panel][:, :, panel]


class Belief:
    def __init__(self, bank, target, admitted=False):
        self.bank, self.target = bank, np.array(target)
        self.weights = bank.admitted(self.target).astype(float) if admitted else np.ones(len(bank.mechanisms))
        if self.weights.sum() == 0: raise ValueError('no prior mass after declared admission')
        self.weights /= self.weights.sum()
        self.losses = bank.losses(self.target)

    @property
    def ess(self): return float(1 / np.sum(self.weights**2))

    def risk(self, weights=None):
        return np.min((self.weights if weights is None else weights) @ self.losses, axis=-1)

    def commitment(self): return 36 + int(np.argmin(self.weights @ self.losses))

    def update(self, evidence):
        panel = np.flatnonzero(evidence['measured_mask'])
        means, cov = self.bank.observation_model(evidence['action_id'], panel)
        y = evidence['readouts'][:, panel]
        diff = y[None] - means[:, None]
        inv = np.linalg.inv(cov)
        ll = -.5 * (np.einsum('hri,hij,hrj->h', diff, inv, diff) + len(y) * np.linalg.slogdet(cov)[1])
        logw = np.log(np.maximum(self.weights, 1e-300)) + ll
        # Preserve exact zero mass from admission, rather than allowing revival.
        logw[self.weights == 0] = -np.inf
        self.weights = np.exp(logw - logsumexp(logw))


def hypothetical_posteriors(bank, belief, action_id, draws, rng):
    """Stratified predictive integration over every hypothesis and Gaussian means.

    Exact mean likelihood when offset_sd=0. For culture variation, uses the
    declared Gaussian moment approximation; residual-covariance information is
    omitted in lookahead. Actual updates consume every purchased replicate.
    """
    spec = bank.actions[action_id]
    panel = list(spec.panel)
    means, cov = bank.observation_model(action_id, panel)
    cov = cov / bank.config.cultures
    H, P = means.shape
    # The common noise draws below make observation-equivalent hypotheses
    # generate exactly duplicate hypothetical packets. Integrate each packet
    # once and sum its prior mass; retain every hypothesis in Bayes updates.
    # This is exact duplicate removal, without rounding or particle pruning.
    if not hasattr(bank, '_predictive_groups'): bank._predictive_groups = {}
    key=(action_id,tuple(panel))
    if key not in bank._predictive_groups:
        features=np.concatenate([means,cov.reshape(H,-1)],axis=1)
        _,first,inverse=np.unique(features,axis=0,return_index=True,return_inverse=True)
        bank._predictive_groups[key]=(first,inverse)
    first,inverse=bank._predictive_groups[key]
    group_mass=np.bincount(inverse,weights=belief.weights,minlength=len(first))
    z = rng.normal(size=(draws, P))
    y = means[first, None] + np.einsum('hij,qj->hqi', np.linalg.cholesky(cov[first]), z)
    y = y.reshape(-1, P)
    delta = y[:, None] - means[None]
    inv = np.linalg.inv(cov)
    ll = -.5 * (np.einsum('qhi,hij,qhj->qh', delta, inv, delta) + np.linalg.slogdet(cov)[1])
    logprior = np.full(H, -np.inf)
    np.log(belief.weights, out=logprior, where=belief.weights > 0)
    logw = ll + logprior
    weights = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))
    mass = np.repeat(group_mass / draws, draws)
    return weights, mass
