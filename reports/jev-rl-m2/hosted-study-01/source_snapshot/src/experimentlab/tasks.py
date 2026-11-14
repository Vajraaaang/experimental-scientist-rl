"""Private generation and stable independent random streams."""
from dataclasses import dataclass
from itertools import product
import hashlib
import numpy as np
from .worlds import Mechanism, matched_mechanism, integrate, equilibrium

STREAMS = {'baseline_culture': 1, 'baseline_noise': 2, 'culture': 3, 'noise': 4,
           'final': 5, 'policy': 6, 'mechanism': 7, 'target': 8, 'planner': 9}


def stream(seed, name, index=0):
    return np.random.default_rng(np.random.SeedSequence([int(seed), STREAMS[name], int(index)]))


@dataclass(frozen=True)
class PrivateTask:
    mechanism: Mechanism
    target: tuple
    seed: int


def make_task(mechanism, seed, config, target=None):
    if target is None: target = tuple(.5 + stream(seed, 'target').choice([-.08, .08], 2))
    return PrivateTask(mechanism, tuple(target), int(seed))


def finite_family():
    """Two feed-forward regulatory pathways, four uncertain control signs and route sign.

    32 possible mechanisms; no arbitrary action/reward table. All share baseline .5.
    Full declared finite support is available to the diagnostic belief model.
    """
    result = []
    for signs in product((-1, 1), repeat=5):
        W = np.zeros((8, 8))
        W[4, :2] = np.array(signs[:2]) * 1.5
        W[5, 2:4] = np.array(signs[2:4]) * 1.5
        W[6, 4:6] = [1.55, 1.55]
        W[7, 4:6] = [signs[4] * 1.55, -signs[4] * 1.55]
        result.append(matched_mechanism(W, np.array([1., 1., 1., 1., 1.2, 1.2, .7, .7])))
    return result


def continuous_family(seed, count, parameter_range=(.9, 1.05)):
    """Independently drawn low-dimensional continuous parameter variations."""
    if not 0 < parameter_range[0] <= parameter_range[1] <= 3.2/3:
        raise ValueError('control gain range violates declared row-norm bound')
    rng = stream(seed, 'mechanism')
    finite = finite_family()
    result = []
    for _ in range(count):
        m = finite[int(rng.integers(len(finite)))]
        scale = rng.uniform(*parameter_range)
        W = m.W.copy()
        W[:6] *= scale
        rates = m.gamma * rng.uniform(.9, 1.1)
        result.append(matched_mechanism(W, rates))
    return result


def mechanism_hash(m):
    return hashlib.sha256(m.W.tobytes() + m.b.tobytes() + m.gamma.tobytes()).hexdigest()


def split_manifests():
    """Mechanism-disjoint continuous parameter groups; finite diagnostic is separate."""
    result = {}
    for split, seed in [('train', 7101), ('validation', 8101), ('test', 9101)]:
        mechanisms = continuous_family(seed, 128)
        result[split] = [{'parent_id': f'{split}-{i:04}', 'mechanism_hash': mechanism_hash(m),
                          'seed': seed * 1000 + i} for i, m in enumerate(mechanisms)]
    return result
