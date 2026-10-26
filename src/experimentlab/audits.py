"""Reconstruction of the supplied generous-budget random-network audit.

Original scripts, exact RNG seeds and balancing algorithm were not supplied.
This independently specified reconstruction cannot reproduce their exact rows.
"""
from itertools import product, combinations
import time
import numpy as np
from scipy.special import expit
from .worlds import matched_mechanism, final_outcomes, measure
from .tasks import stream


def interventions():
    result = [np.zeros(8)]
    for i, dose in product(range(4), (.4, .8)):
        u = np.zeros(8); u[i] = dose; result.append(u)
    for (i, j), doses in product(combinations(range(4), 2), product((.4, .8), repeat=2)):
        u = np.zeros(8); u[[i, j]] = doses; result.append(u)
    return np.array(result)


def random_networks(seed, n):
    rng = np.random.default_rng(seed)
    ms = []
    for _ in range(n):
        W = np.zeros((8, 8))
        for i in range(8):
            count = int(rng.integers(2, 4))
            parents = rng.choice(np.delete(np.arange(8), i), count, replace=False)
            # Balance signs within each row; randomize the majority for 3 parents.
            signs = np.array([-1, 1] if count == 2 else [-1, 1, rng.choice([-1, 1])])
            rng.shuffle(signs)
            weights = signs * rng.uniform(.4, 1.5, count)
            weights *= min(1., 3.2/np.sum(abs(weights)))
            W[i, parents] = weights
        ms.append(matched_mechanism(W, rng.uniform(.5, 1.5, 8)))
    targets = .5 + rng.choice([-.08, .08], (n, 2))
    return ms, targets


def batch_nominal(ms, us, duration=6., dt=.05):
    W = np.stack([m.W for m in ms]); b = np.stack([m.b for m in ms]); rates = np.stack([m.gamma for m in ms])
    x = np.full((len(ms), len(us), 8), .5)
    alpha = -np.expm1(-rates * dt)
    for _ in range(round(duration/dt)):
        x = (1-alpha[:, None])*x + alpha[:, None]*(1-us[None])*expit(b[:, None]+np.einsum('huj,hij->hui', x, W))
    return x


def generous_audit(calibration=300, heldout=600, evaluated=128, seed=46193):
    start = time.perf_counter(); us = interventions()
    admitted = []
    counts = []
    for split, n in enumerate((calibration, heldout)):
        ms, goals = random_networks(seed+split, n)
        nominal = batch_nominal(ms, us)[:, :, 6:8]
        mask = np.mean((nominal-goals[:, None])**2, axis=-1).min(axis=1) <= .04**2
        counts.append(int(mask.sum()))
        admitted.append([(ms[i], goals[i], nominal[i], seed*10000+split*1000+i) for i in np.flatnonzero(mask)])
    def predictions(task):
        m, goal, nominal, task_seed = task
        responses = []
        for c in range(4):
            u = np.zeros(8); u[c] = .8
            y = measure(m, u, 6., 8, .03, .02, stream(task_seed, 'culture', c), stream(task_seed, 'noise', c))
            responses.append(y[:, 6:8].mean(axis=0)-.5)
        return np.array(responses)
    calibration_data = [(t, predictions(t)) for t in admitted[0]]
    def additive_choice(task, responses, subset):
        effects = responses.copy()
        effects[[c for c in range(4) if c not in subset]] = 0
        pred = .5 + us[:, :4] @ (effects/.8)
        return int(np.argmin(np.mean((pred-task[1])**2, axis=1)))
    pair_scores = {}
    for subset in combinations(range(4), 2):
        pair_scores[subset] = np.mean([np.mean((t[2][additive_choice(t, r, subset)]-t[1])**2) for t,r in calibration_data])
    best_pair = min(pair_scores, key=pair_scores.get)
    rows = {m: [] for m in ('no_experiment_no_intervention', 'fixed_two_additive', 'fixed_four_additive', 'private_nominal_reference')}
    for task in admitted[1][:evaluated]:
        responses = predictions(task)
        choices = [0, additive_choice(task, responses, best_pair), additive_choice(task, responses, range(4)),
                   int(np.argmin(np.mean((task[2]-task[1])**2, axis=1)))]
        for name, choice in zip(rows, choices):
            y = final_outcomes(task[0], us[choice], 128, .03, stream(task[3], 'final'))[:, 6:8]
            rows[name].append(float(np.mean((y-task[1])**2)))
    costs = [0, 5.8, 11.6, 0]
    summary = {name: {'aggregate_rmse': float(np.sqrt(np.mean(values))),
                      'success_rate': float(np.mean(np.array(values) <= .04**2)),
                      'variable_cost': cost} for (name, values), cost in zip(rows.items(), costs)}
    return {'status': 'independent_reconstruction_not_exact_original_script_reproduction', 'seed': seed,
            'calibration_attempted': calibration, 'calibration_admitted': counts[0],
            'heldout_attempted': heldout, 'heldout_admitted': counts[1],
            'evaluated': min(evaluated, counts[1]), 'tuned_control_pair': list(best_pair),
            'summary': summary, 'seconds': time.perf_counter()-start,
            'original_reported': {'calibration_admitted':119,'heldout_admitted':237,
                                  'rmse':[.08038,.05631,.02755,.02628]},
            'scope': 'Exploratory 33-intervention audit only, not the restricted 45-action pilot or locked evaluation.',
            'limitation': 'Exact original seeds/scripts and per-row sign-balancing convention unavailable; numerical agreement is not asserted.'}
