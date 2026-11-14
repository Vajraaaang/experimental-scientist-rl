"""Development comparisons. No locked-test access in feasibility runs."""
from dataclasses import asdict
from itertools import product
import time
import numpy as np
from scipy.special import logsumexp
from .envs import ScientistEnv, PilotConfig
from .tasks import make_task, stream
from .worlds import measure, final_outcomes
from .inference import Belief
from .baselines import DecisionPlanner, fixed_candidates, fixed_identification, additive_commitment


def development_tasks(bank, n, seed):
    rng = np.random.default_rng(seed)
    tasks, indices, rejected = [], [], 0
    for i in range(n):
        target = .5 + rng.choice([-.08, .08], 2)
        admitted = np.flatnonzero(bank.admitted(target))
        rejected += len(bank.mechanisms) - len(admitted)
        if not len(admitted): raise RuntimeError('target has no admitted support')
        idx = int(rng.choice(admitted)); indices.append(idx)
        tasks.append(make_task(bank.mechanisms[idx], seed * 10000 + i, bank.config, target))
    return tasks, indices


def independent_continuous_tasks(bank, split, n, seed):
    """Actual mechanisms are disjoint from the independently drawn belief bank.

    Targets are chosen before feasibility screening. Fixed split pools and
    rejection rules are public generator assumptions, never policy features.
    """
    from .tasks import continuous_family, mechanism_hash
    from .worlds import integrate, equilibrium
    pool_seed = {'train': 7101, 'validation': 8101, 'test': 9101}[split]
    if split == 'test': raise ValueError('locked test is not available to development runners')
    pool = continuous_family(pool_seed, 128)
    rng = np.random.default_rng(seed)
    tasks = []
    used = set()
    us = np.array([a.intervention for a in bank.actions[36:]])
    for i in range(n):
        target = .5 + rng.choice([-.08, .08], 2)
        candidates = rng.permutation(len(pool))
        for j in candidates:
            if int(j) in used: continue
            m = pool[j]
            y = integrate(m, np.broadcast_to(equilibrium(m), (9,8)), us, 6, bank.config.dt)[:,6:8]
            loss = np.mean((y-target)**2, axis=1)
            if loss.min() <= .04**2 and loss.min() < loss[0]-.0001:
                tasks.append(make_task(m, seed*10000+i, bank.config, target)); used.add(int(j)); break
        else: raise ValueError('not enough distinct admitted mechanisms in declared development pool')
    return tasks


def run_strategy(task, bank, method, schedule=(), planner_draws=12, seed=0):
    env = ScientistEnv(task, bank.config)
    obs, _ = env.reset()
    belief = Belief(bank, obs['target'], admitted=True)
    belief.update(obs['history'][0])
    planner = DecisionPlanner(bank, method, planner_draws) if method in ('voi', 'eig') else None
    rng = np.random.default_rng(seed)
    start = time.perf_counter(); cost = total = 0.
    position = 0; ess = [belief.ess]; choices = []
    while True:
        if method in ('prior', 'reference'): a = belief.commitment()
        elif method in ('fixed', 'additive'):
            a = schedule[position] if position < len(schedule) else (additive_commitment(obs, env.actions) if method == 'additive' else belief.commitment())
            if not obs['action_mask'][a]: a = belief.commitment()
            position += 1
        elif method == 'random':
            possible = np.flatnonzero(obs['action_mask'][:36])
            a = int(rng.choice(possible)) if len(possible) else belief.commitment()
        elif planner is not None: a = planner.choose(obs, belief, rng)
        else: raise ValueError(method)
        choices.append(a)
        obs, reward, done, truncated, _ = env.step(a)
        total += reward
        if done: break
        belief.update(obs['history'][-1]); ess.append(belief.ess)
    result = dict(env.private_result or {})
    result.update(return_=total, assays=env.count, online_seconds=time.perf_counter()-start,
                  ess_min=min(ess), choices=choices,
                  hypothetical_outcomes=0 if planner is None else planner.hypothetical_outcomes)
    return result, env.public_record()


def reference_result(task, bank):
    # Select nominal private mechanism intervention, then score independent fresh final cultures.
    from .worlds import integrate, equilibrium
    actions = bank.actions[36:]
    us = np.array([a.intervention for a in actions])
    y = integrate(task.mechanism, np.broadcast_to(equilibrium(task.mechanism), (9, 8)), us, 6, bank.config.dt)[:, 6:8]
    score = 100*np.mean((y-task.target)**2, axis=1)+bank.config.effort_weight*us.sum(axis=1)
    a = int(np.argmin(score))
    final = final_outcomes(task.mechanism, us[a], bank.config.final_cultures, bank.config.offset_sd,
                           stream(task.seed, 'final'), dt=bank.config.dt)[:, 6:8]
    mse = float(np.mean((final - task.target)**2))
    return {'final_mse': mse, 'effort': float(us[a].sum()), 'variable_cost': 0., 'setup_cost': 3.,
            'return_': -100*mse-bank.config.effort_weight*us[a].sum(), 'assays': 0,
            'online_seconds': 0., 'scope': 'private nominal reference; privileged zero assay cost'}


def tune_fixed(bank, tasks, beam_width=8):
    """All singles and ordered pairs, then beam extension, on calibration only.

    The trusted tuner caches potential measured evidence. Its selector receives
    only each schedule's purchased records. Nothing from this cache reaches a
    policy or a held-out development episode.
    """
    start = time.perf_counter()
    n, H, A, D = len(tasks), len(bank.mechanisms), 36, bank.config.max_experiments
    loglikes = np.zeros((n, D, A, H))
    priors = np.zeros((n, H)); losses = np.zeros((n, H, 9)); truth_loss = np.zeros((n, 9))
    costs = np.array([a.cost_ticks for a in bank.actions[:36]]) / 20
    for t, task in enumerate(tasks):
        obs, _ = ScientistEnv(task, bank.config).reset()
        belief = Belief(bank, task.target, admitted=True); belief.update(obs['history'][0])
        priors[t] = belief.weights; losses[t] = belief.losses
        us = np.array([a.intervention for a in bank.actions[36:]])
        # Paired across final actions in tuning; independent from exploratory draws.
        for j, u in enumerate(us):
            final = final_outcomes(task.mechanism, u, bank.config.final_cultures, bank.config.offset_sd,
                                   stream(task.seed, 'final'), dt=bank.config.dt)[:, 6:8]
            truth_loss[t, j] = 100*np.mean((final-task.target)**2)+bank.config.effort_weight*u.sum()
        for d in range(D):
            for a, spec in enumerate(bank.actions[:36]):
                readouts = measure(task.mechanism, np.array(spec.intervention), spec.duration, bank.config.cultures,
                                   bank.config.offset_sd, bank.config.readout_sd,
                                   stream(task.seed, 'culture', d+1), stream(task.seed, 'noise', d+1), bank.config.dt)
                means, cov = bank.observation_model(a, list(spec.panel))
                delta = readouts[:, list(spec.panel)][None] - means[:, None]
                loglikes[t, d, a] = -.5*(np.einsum('hri,hij,hrj->h', delta, np.linalg.inv(cov), delta) +
                                                       bank.config.cultures*np.linalg.slogdet(cov)[1])
    logprior = np.full_like(priors, -np.inf)
    np.log(priors, out=logprior, where=priors > 0)
    evaluated = {}
    def score(schedule):
        if schedule in evaluated: return evaluated[schedule]
        logw = logprior.copy()
        for d, a in enumerate(schedule): logw += loglikes[:, d, a]
        weights = np.exp(logw-logsumexp(logw, axis=1, keepdims=True))
        chosen = np.argmin(np.einsum('nh,nha->na', weights, losses), axis=1)
        value = -float(np.mean(truth_loss[np.arange(n), chosen])) - bank.config.cost_weight * sum(costs[a] for a in schedule)
        evaluated[schedule] = value
        return value
    candidates = fixed_candidates(bank.config.budget, D)
    if D >= 2:
        candidates.extend((a, b) for a, b in product(range(A), repeat=2) if costs[a]+costs[b] <= bank.config.budget)
    for s in candidates: score(s)
    for length in range(3, D+1):
        previous = sorted((s for s in evaluated if len(s) == length-1), key=score, reverse=True)[:beam_width]
        for s in previous:
            for a in range(A):
                if sum(costs[j] for j in s) + costs[a] <= bank.config.budget: score(s+(a,))
    best = max(evaluated, key=score)
    return best, {'schedule': list(best), 'calibration_return': evaluated[best], 'schedules_evaluated': len(evaluated),
                  'calibration_tasks': n, 'tuning_seconds': time.perf_counter()-start,
                  'search': 'all singles/ordered pairs, repeated assays, identification subsets, beam extensions'}


def paired_interval(a, b, seed=301, samples=2000, groups=None):
    """Mean a-b; bootstrap independent mechanism groups when provided."""
    diff = np.asarray(a)-np.asarray(b)
    if groups is not None:
        labels = np.unique(groups)
        diff = np.array([diff[np.asarray(groups) == g].mean() for g in labels])
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(len(diff), size=(samples, len(diff)))].mean(axis=1)
    return {'mean': float(diff.mean()), 'ci95': [float(v) for v in np.quantile(means, [.025, .975])],
            'unit': 'mechanism' if groups is not None else 'development_episode_conditional_on_finite_support',
            'units': len(diff)}


def summarize(rows):
    return {'aggregate_rmse': float(np.sqrt(np.mean([r['final_mse'] for r in rows]))),
            'success_rate': float(np.mean([r['final_mse'] <= .04**2 for r in rows])),
            'mean_return': float(np.mean([r['return_'] for r in rows])),
            'mean_variable_cost': float(np.mean([r['variable_cost'] for r in rows])),
            'mean_total_assay_cost': float(np.mean([r['variable_cost']+r['setup_cost'] for r in rows])),
            'mean_effort': float(np.mean([r['effort'] for r in rows])),
            'mean_assays': float(np.mean([r['assays'] for r in rows])),
            'mean_online_seconds': float(np.mean([r['online_seconds'] for r in rows]))}


def run_comparison(bank, calibration_count=24, evaluation_count=128, seed=20261026, draws=12, beam=8, bootstrap_samples=2000, gap_threshold=.10):
    calibration, _ = development_tasks(bank, calibration_count, seed)
    tasks, indices = development_tasks(bank, evaluation_count, seed+1)
    fixed, tuning = tune_fixed(bank, calibration, beam)
    identification = fixed_identification()
    while sum(bank.actions[a].cost_ticks for a in identification) > bank.config.budget*20:
        identification = identification[:-1]
    methods = {'prior': (), 'fixed': fixed, 'identification': identification, 'additive': identification,
               'random': (), 'eig': (), 'voi': ()}
    rows = {m: [] for m in methods}; rows['reference'] = []
    example = None
    start = time.perf_counter()
    for i, task in enumerate(tasks):
        for name, schedule in methods.items():
            method = 'fixed' if name == 'identification' else name
            result, public = run_strategy(task, bank, method, schedule, draws, seed=i+100)
            rows[name].append(result)
            if i == 0 and name == 'voi': example = {'public': public, 'post_episode_private_result': result}
        rows['reference'].append(reference_result(task, bank))
    summary = {name: summarize(r) for name, r in rows.items()}
    utilities = lambda m: [r['return_'] for r in rows[m]]
    # Conservative mechanism support grouping avoids treating repeated mechanisms as new mechanisms.
    group_diff = paired_interval(utilities('voi'), utilities('fixed'), groups=indices, samples=bootstrap_samples)
    information = paired_interval(utilities('voi'), utilities('prior'), groups=indices, samples=bootstrap_samples)
    prior_loss = np.mean([r['final_mse'] for r in rows['prior']])
    ref_loss = np.mean([r['final_mse'] for r in rows['reference']])
    fixed_loss = np.mean([r['final_mse'] for r in rows['fixed']])
    adaptive_loss = np.mean([r['final_mse'] for r in rows['voi']])
    recovery = float((fixed_loss-adaptive_loss)/(prior_loss-ref_loss)) if prior_loss > ref_loss else None
    raw_error = paired_interval([r['final_mse'] for r in rows['fixed']], [r['final_mse'] for r in rows['voi']], groups=indices, samples=bootstrap_samples)
    gate_c = bool(group_diff['ci95'][0] > 0 and raw_error['ci95'][0] > 0 and recovery is not None and recovery >= gap_threshold)
    return {'config': asdict(bank.config), 'scope': 'finite_support_development_only',
            'bank': {'hypotheses': len(bank.mechanisms), 'build_seconds': bank.build_seconds,
                     'forward_culture_steps': bank.forward_steps, 'likelihood': bank.likelihood},
            'admission_mass': {str(t): float(bank.admitted(np.array(t)).mean()) for t in product((.42, .58), repeat=2)},
            'tuning': tuning, 'identification_schedule': list(identification), 'summary': summary,
            'information_value': information, 'adaptive_utility_difference': group_diff,
            'adaptive_error_reduction': raw_error, 'gap_recovery': recovery,
            'gate_B': information['ci95'][0] > 0, 'gate_C_screen': gate_c,
            'gate_C_generalization': False, 'rows': rows, 'example': example,
            'evaluation_seconds': time.perf_counter()-start,
            'limitations': ['Finite support shares possible mechanisms; held-out evidence is not mechanism generalization.',
                            'Intervals over sampled finite-support mechanism groups do not establish structural transfer.',
                            'One-step VOI is approximate; absence of benefit is not a proof against all adaptive planners.']}
