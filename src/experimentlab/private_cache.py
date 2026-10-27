"""Trusted simulator optimization; never provide this cache to a policy.

Qualification is limited to zero culture offsets, dt=.05, catalogue doses,
and times 0, 1, and 6. Every other request uses the reference simulator.
Cached latent endpoints describe identical fresh zero-offset cultures, not
physical cultures carried forward from a preceding assay. Readout noise is
sampled afresh and the reference culture/final RNG draws are still consumed.
"""
import numpy as np
from scipy.special import expit

from .envs import ScientistEnv, catalogue
from .tasks import mechanism_hash


class PrivateDynamicsCache:
    """CPU float64 endpoints, indexed privately by immutable mechanism hash."""

    def __init__(self, mechanisms, config):
        self._endpoints = {}
        if config.offset_sd != 0. or config.dt != .05:
            return
        unique = {mechanism_hash(m): m for m in mechanisms}
        for n in (4, 8):
            group = [(key, m) for key, m in unique.items() if len(m.b) == n]
            if not group:
                continue
            W = np.stack([m.W for _, m in group])
            b = np.stack([m.b for _, m in group])
            gamma = np.stack([m.gamma for _, m in group])
            # Freeze each mechanism on its own first converged iteration, just
            # as equilibrium() does; other mechanisms may need more iterations.
            x = np.full(b.shape, .5, dtype=np.float64)
            active = np.ones(len(group), dtype=bool)
            for _ in range(2000):
                nxt = expit(b + np.matmul(W, x[..., None])[..., 0])
                converged = np.max(abs(nxt - x), axis=1) < 1e-12
                x[active] = nxt[active]
                active &= ~converged
                if not active.any():
                    break
            else:
                raise RuntimeError('equilibrium iteration did not converge')
            controls = 2 if n == 4 else 4
            panels = ((2,), (3,)) if n == 4 else ((4, 5), (6, 7))
            interventions = np.asarray([a.intervention for a in catalogue(n, controls, panels)
                                        if a.kind == 'commit'], dtype=np.float64)
            endpoints = {0.: np.broadcast_to(x[:, None, :],
                                              (len(group), len(interventions), n)).copy()}
            for duration in (1., 6.):
                latent = endpoints[0.].copy()
                steps = int(np.ceil(duration / config.dt))
                alpha = -np.expm1(-gamma[:, None, :] * (duration / steps))
                for _ in range(steps):
                    latent = ((1 - alpha) * latent + alpha * (1 - interventions) *
                              expit(b[:, None, :] + np.matmul(latent, W.transpose(0, 2, 1))))
                endpoints[duration] = latent
            for index, (key, _) in enumerate(group):
                values = {}
                for duration, latent in endpoints.items():
                    for intervention, endpoint in zip(interventions, latent[index]):
                        endpoint = endpoint.copy()
                        endpoint.flags.writeable = False
                        values[(duration, tuple(intervention))] = endpoint
                self._endpoints[key] = values

    def _lookup(self, mechanism, config, u, duration):
        if config.offset_sd != 0. or config.dt != .05:
            return None
        values = self._endpoints.get(mechanism_hash(mechanism))
        return None if values is None else values.get((duration, tuple(u)))


class CachedScientistEnv(ScientistEnv):
    """Reference public API with an evaluator-owned private dynamics cache."""

    def __init__(self, task, config, cache):
        super().__init__(task, config)
        self._cache = cache

    def _measure(self, u, duration, count, culture_rng, noise_rng):
        endpoint = self._cache._lookup(self._task.mechanism, self.config, u, duration)
        if endpoint is None:
            return super()._measure(u, duration, count, culture_rng, noise_rng)
        shape = (count, self.n)
        culture_rng.normal(0, self.config.offset_sd, shape)
        return endpoint + noise_rng.normal(0, self.config.readout_sd, shape)

    def _final_outcomes(self, u, count, rng, duration=6.):
        endpoint = self._cache._lookup(self._task.mechanism, self.config, u, duration)
        if endpoint is None:
            return super()._final_outcomes(u, count, rng, duration)
        shape = (count, self.n)
        rng.normal(0, self.config.offset_sd, shape)
        return np.broadcast_to(endpoint, shape).copy()
