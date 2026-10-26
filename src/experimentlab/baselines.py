"""Public-history experiment selection; one common posterior final selector."""
from itertools import permutations
import numpy as np
from .inference import hypothetical_posteriors


def fixed_identification(controls=(0, 1, 2, 3)):
    # dose .8, duration 6, outcome panel, stable catalogue v1
    return tuple((2 + 2 * c) * 4 + 3 for c in controls)


class DecisionPlanner:
    def __init__(self, bank, method='voi', draws=8, candidates=None):
        if method not in ('voi', 'eig'): raise ValueError('unknown planner')
        self.bank, self.method, self.draws = bank, method, draws
        self.candidates = tuple(range(36)) if candidates is None else tuple(candidates)
        self.hypothetical_outcomes = 0

    def choose(self, obs, belief, rng, diagnostics=False):
        commit = belief.commitment()
        current = -float(belief.risk())
        best_action, best = commit, current
        scores = {}
        # Identical random variates per candidate reduce comparison variance.
        seed = int(rng.integers(2**31))
        prior_entropy = -np.sum(belief.weights * np.log(np.maximum(belief.weights, 1e-300)))
        for a in self.candidates:
            if not obs['action_mask'][a]: continue
            weights, mass = hypothetical_posteriors(self.bank, belief, a, self.draws, np.random.default_rng(seed))
            self.hypothetical_outcomes += len(weights)
            cost = self.bank.config.cost_weight * self.bank.actions[a].cost_ticks / 20
            if self.method == 'voi': score = -float(np.sum(mass * belief.risk(weights))) - cost
            else:
                entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-300)), axis=1)
                score = prior_entropy - float(mass @ entropy) - cost
            scores[str(a)] = score
            threshold = best if self.method == 'voi' else (0 if best_action >= 36 else best)
            if score > threshold + 1e-10: best_action, best = a, score
        diag = {'current_value': current, 'best_expected_value': best, 'candidate_values': scores}
        return (best_action, diag) if diagnostics else best_action


def additive_commitment(obs, actions):
    """Measured single-control effects; missing effects remain zero, not hidden truth."""
    base = obs['history'][0]['readouts'][:, 6:8].mean(axis=0)
    effects = {}
    for packet in obs['history'][1:]:
        a = actions[packet['action_id']]
        nonzero = np.flatnonzero(a.intervention)
        if len(nonzero) == 1 and a.duration == 6 and a.panel == (6, 7):
            c = int(nonzero[0]); dose = a.intervention[c]
            effects[c] = (packet['readouts'][:, 6:8].mean(axis=0) - base) / dose
    predictions = []
    for a in actions[36:]:
        prediction = base.copy()
        for c, effect in effects.items(): prediction += a.intervention[c] * effect
        predictions.append(prediction)
    effort = np.array([sum(a.intervention) for a in actions[36:]])
    score = 100 * np.mean((np.array(predictions) - obs['target'])**2, axis=1) + .05 * effort
    return 36 + int(np.argmin(score))


def fixed_candidates(budget, max_experiments=4):
    """Exhaustive ordered subsets of the four full-dose identification assays.

    Also include every single assay and all repeated-assay schedules up to the
    affordable horizon. This is a declared search space, not global optimality.
    """
    from .envs import catalogue
    actions = catalogue()
    schedules = {()}
    for a in range(36):
        for k in range(1, max_experiments + 1):
            if actions[a].cost_ticks * k <= round(budget * 20): schedules.add((a,) * k)
    ids = fixed_identification()
    for length in range(1, min(4, max_experiments)+1):
        for order in permutations(ids, length):
            if sum(actions[a].cost_ticks for a in order) <= round(budget * 20): schedules.add(order)
    return sorted(schedules)
