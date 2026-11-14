"""Private float64 reference dynamics. Never pass these objects to a policy."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit
from scipy.integrate import solve_ivp

SIMULATOR_VERSION = "exp-euler-v1"


@dataclass(frozen=True)
class Mechanism:
    W: np.ndarray
    b: np.ndarray
    gamma: np.ndarray

    def __post_init__(self):
        for name in ('W', 'b', 'gamma'):
            a = np.array(getattr(self, name), dtype=np.float64, copy=True)
            a.flags.writeable = False
            object.__setattr__(self, name, a)
        n = len(self.b)
        if self.W.shape != (n, n) or self.gamma.shape != (n,):
            raise ValueError('inconsistent mechanism shapes')
        if not all(np.isfinite(a).all() for a in (self.W, self.b, self.gamma)):
            raise ValueError('nonfinite mechanism')
        if np.max(np.sum(abs(self.W), axis=1)) > 3.2 + 1e-12 or np.min(self.gamma) <= 0:
            raise ValueError('mechanism violates contraction/rate contract')


@dataclass(frozen=True)
class Cultures:
    b: np.ndarray
    x: np.ndarray


def fixture(h):
    if h not in (-1, 1): raise ValueError('h must be -1 or +1')
    W = np.zeros((4, 4))
    W[2, :2] = [1.5 * h, -1.5 * h]
    W[3, 2] = 3
    return Mechanism(W, -W @ np.full(4, .5), np.array([1, 1, 1.2, .5]))


def equilibrium(m, offsets=None, tolerance=1e-12):
    b = m.b if offsets is None else m.b + offsets
    x = np.full(np.shape(b), .5, dtype=np.float64)
    for _ in range(2000):
        nxt = expit(b + x @ m.W.T)
        if np.max(abs(nxt - x)) < tolerance: return nxt
        x = nxt
    raise RuntimeError('equilibrium iteration did not converge')


def integrate(m, x, u, duration, dt=.05, offsets=None):
    if duration < 0 or dt <= 0: raise ValueError('invalid integration time')
    u = np.asarray(u, dtype=np.float64)
    if np.any(u < 0) or np.any(u > 1): raise ValueError('invalid suppression dose')
    x = np.array(x, dtype=np.float64, copy=True)
    b = m.b if offsets is None else m.b + offsets
    steps = int(np.ceil(duration / dt))
    if not steps: return x
    step = duration / steps
    alpha = -np.expm1(-m.gamma * step)
    for _ in range(steps):
        x = (1 - alpha) * x + alpha * (1 - u) * expit(b + x @ m.W.T)
    return x


def ode_endpoint(m, x, u, duration, offsets=None):
    b = m.b if offsets is None else m.b + offsets
    result = solve_ivp(lambda t, z: m.gamma * ((1 - u) * expit(b + m.W @ z) - z),
                       (0, duration), x, method='DOP853', rtol=1e-11, atol=1e-13)
    if not result.success: raise RuntimeError(result.message)
    return result.y[:, -1]


def fresh_cultures(m, count, offset_sd, rng):
    offsets = rng.normal(0, offset_sd, (count, len(m.b)))
    return Cultures(m.b + offsets, equilibrium(m, offsets))


def measure(m, u, duration, count, offset_sd, readout_sd, culture_rng, noise_rng, dt=.05):
    c = fresh_cultures(m, count, offset_sd, culture_rng)
    latent = integrate(m, c.x, u, duration, dt, c.b - m.b)
    return latent + noise_rng.normal(0, readout_sd, latent.shape)


def final_outcomes(m, u, count, offset_sd, rng, duration=6., dt=.05):
    c = fresh_cultures(m, count, offset_sd, rng)
    return integrate(m, c.x, u, duration, dt, c.b - m.b)


def matched_mechanism(W, rates, baseline=None):
    baseline = np.full(len(W), .5) if baseline is None else np.asarray(baseline)
    return Mechanism(W, logit(baseline) - W @ baseline, rates)
