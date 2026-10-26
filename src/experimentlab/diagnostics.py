"""Reconstructed supplied audit and exact finite planning controls."""
from itertools import product, combinations_with_replacement
import numpy as np
from scipy.stats import norm
from .worlds import fixture, integrate


class BranchingEnv:
    """Finite public diagnostic interface; not the regulatory simulator."""
    def __init__(self, route, target, seed=0):
        if route not in (0,1) or target not in (0,1): raise ValueError('binary private state required')
        self._route, self._target = route, target
        self._bits = np.random.default_rng(seed).integers(0,2,2)

    def reset(self):
        self.history = []; self.count = 0; self.decisions = 0; self.done = False
        return self._obs(), {}

    def _obs(self):
        return {'history': [dict(p) for p in self.history], 'remaining_assays': 2-self.count,
                'action_mask': np.array([self.count<2]*3+[True]*2)}

    def step(self, action):
        if self.done: raise RuntimeError('reset required')
        self.decisions += 1
        if not isinstance(action,(int,np.integer)) or action<0 or action>=5 or not self._obs()['action_mask'][action]:
            self.done = self.decisions>=6
            return self._obs(), -1. if self.done else 0., self.done, False, {'error':'invalid_action'}
        if action >= 3:
            self.done = True
            return self._obs(), float(action-3 == self._target), True, False, {}
        readout = self._route if action == 0 else (self._target if action-1 == self._route else int(self._bits[action-1]))
        self.count += 1; self.history.append({'action_id': action, 'readout': readout})
        return self._obs(), 0., False, False, {}


def numerical_audit():
    x, u = np.full(4, .5), np.array([.8, 0, 0, 0])
    short = [integrate(fixture(h), x, u, 1.) for h in (1, -1)]
    long = [integrate(fixture(h), x, u, 6.) for h in (1, -1)]
    error = float(norm.cdf(-abs(short[0][2] - .5) / (.02 / np.sqrt(8))))
    correct, wrong = (long[0][3] - .41)**2, (long[1][3] - .41)**2
    expected = float((1 - error) * correct + error * wrong)
    return {'proxy_time1': [float(z[2]) for z in short], 'outcome_time1': [float(z[3]) for z in short],
            'outcome_time6': [float(z[3]) for z in long], 'sign_error': error,
            'informed_mse': expected, 'no_intervention_mse': .0081,
            'negative_return': 100 * expected + .05 * .8 + .01 * 1.65,
            'interpretation': 'One fixed assay solves this diagnostic; no evidence of adaptive RL value.'}


def branching_audit():
    # Enumerate independent route, target, and two irrelevant fair readout bits.
    worlds = list(product((0, 1), repeat=4))
    def read(world, assay):
        route, target, bit0, bit1 = world
        return route if assay == 0 else (target if assay - 1 == route else (bit0, bit1)[assay - 1])
    schedules = {}
    for pair in combinations_with_replacement(range(3), 2):
        bins = {}
        for world in worlds:
            key = tuple(read(world, assay) for assay in pair)
            bins.setdefault(key, [0, 0])[world[1]] += 1
        schedules[str(pair)] = sum(max(counts) for counts in bins.values()) / len(worlds)
    adaptive = np.mean([read(w, 1 + read(w, 0)) == w[1] for w in worlds])
    return {'adaptive_accuracy': float(adaptive), 'best_fixed_accuracy': max(schedules.values()),
            'fixed_schedules': schedules, 'worlds_enumerated': len(worlds),
            'interpretation': 'Routing diagnostic only; no scientific simulator conclusion.'}
