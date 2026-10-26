"""Measured inference checks and bounded continuous development extension."""
import time
import numpy as np
from scipy.stats import chi2
from .tasks import finite_family, continuous_family, mechanism_hash, stream
from .worlds import measure
from .envs import PilotConfig
from .inference import PredictionBank, Belief
from .evaluation import independent_continuous_tasks, tune_fixed, run_strategy, reference_result, summarize, paired_interval
from .baselines import DecisionPlanner


def predictive_check(offset_sd, n=256):
    config = PilotConfig(offset_sd=offset_sd)
    bank = PredictionBank(finite_family(), config)
    rng = np.random.default_rng(6027)
    distances = []
    for i in range(n):
        h, a = int(rng.integers(32)), int(rng.integers(36))
        spec = bank.actions[a]
        y = measure(bank.mechanisms[h], np.array(spec.intervention), spec.duration, config.cultures,
                    offset_sd, config.readout_sd, stream(667000+i, 'culture'), stream(667000+i, 'noise'))
        means, cov = bank.observation_model(a, list(spec.panel))
        residual = y[:, list(spec.panel)].mean(axis=0)-means[h]
        distances.append(float(residual @ np.linalg.solve(cov[h]/config.cultures, residual)))
    return {'offset_sd': offset_sd, 'samples': n, 'mean_packet_95pct_coverage': float(np.mean(np.array(distances)<=chi2.ppf(.95,2))),
            'mean_mahalanobis_squared': float(np.mean(distances)), 'expected_under_exact_model': 2.,
            'likelihood': bank.likelihood, 'scope': 'predictive mean coverage; does not alone validate the full posterior'}


def planner_draw_check(n=16):
    config = PilotConfig(readout_sd=.08)
    bank = PredictionBank(finite_family(), config)
    from .evaluation import development_tasks
    from .envs import ScientistEnv
    tasks, _ = development_tasks(bank, n, 3355)
    actions = []; timing = []
    for draws in (12, 64):
        result = []; start = time.perf_counter()
        for i, task in enumerate(tasks):
            obs, _ = ScientistEnv(task, config).reset()
            belief = Belief(bank, obs['target'], admitted=True); belief.update(obs['history'][0])
            result.append(DecisionPlanner(bank, 'voi', draws).choose(obs, belief, np.random.default_rng(i)))
        actions.append(result); timing.append(time.perf_counter()-start)
    return {'draws_per_hypothesis': [12,64], 'first_action_agreement': float(np.mean(np.array(actions[0])==actions[1])),
            'seconds': timing, 'cases': n, 'scope': 'Monte Carlo first-action stability, not proof of planner optimality'}


def continuous_revision(n=64, calibration=24):
    """Development-only parameter variation with a bank independent of truth.

    Keep original low-noise result and stronger fixed identification schedules.
    The predeclared .08 noise slice motivates checking decision uncertainty here.
    No locked outcomes or added private observables are used.
    """
    config = PilotConfig(readout_sd=.08, budget=8)
    bank = PredictionBank(continuous_family(6101, 64), config)
    train = independent_continuous_tasks(bank, 'train', calibration, 6601)
    validation = independent_continuous_tasks(bank, 'validation', n, 6602)
    schedule, tuning = tune_fixed(bank, train)
    rows = {m: [] for m in ('prior','fixed','voi','reference')}
    start = time.perf_counter()
    for i, task in enumerate(validation):
        for method in ('prior','fixed','voi'):
            row, _ = run_strategy(task, bank, method, schedule if method=='fixed' else (), planner_draws=12, seed=i)
            rows[method].append(row)
        rows['reference'].append(reference_result(task, bank))
    hashes = [mechanism_hash(t.mechanism) for t in validation]
    return {'scope': 'development_extension_continuous_parameters_not_locked_test',
            'revision': 'Retain topology, costs, actions and baselines; introduce independent continuous control gain/rate variation at the predeclared .08 noise slice.',
            'uncertain_parameters': {'control_gain_scale': [.9,1.05], 'shared_rate_scale': [.9,1.1]},
            'config': config.__dict__, 'bank_hypotheses': 64, 'truth_inserted_in_bank': False,
            'tuning': tuning, 'summary': {m:summarize(r) for m,r in rows.items()},
            'adaptive_return_difference': paired_interval([r['return_'] for r in rows['voi']], [r['return_'] for r in rows['fixed']], groups=hashes),
            'adaptive_error_reduction': paired_interval([r['final_mse'] for r in rows['fixed']], [r['final_mse'] for r in rows['voi']], groups=hashes),
            'median_minimum_ESS': float(np.median([r['ess_min'] for r in rows['voi']])),
            'validation_mechanism_hashes': hashes,
            'train_mechanism_hashes': [mechanism_hash(t.mechanism) for t in train],
            'bank_mechanism_hashes': [mechanism_hash(m) for m in bank.mechanisms],
            'seconds': time.perf_counter()-start,
            'interpretation': 'Development sensitivity only. Small independent particle bank can collapse; calibrated inference and stronger fixed-design tuning are required before qualifying scientific adaptivity.',
            'rows': rows}
