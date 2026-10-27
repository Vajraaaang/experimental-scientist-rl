"""45-action public experiment/commit interface, independent private evaluator."""
from dataclasses import dataclass, asdict
from itertools import product
import copy
import numpy as np
from .tasks import stream
from .worlds import measure, final_outcomes


@dataclass(frozen=True)
class PilotConfig:
    budget: float = 8.
    max_experiments: int = 4
    decision_cap: int = 10
    cultures: int = 8
    final_cultures: int = 32
    readout_sd: float = .02
    offset_sd: float = 0.
    cost_weight: float = .01
    effort_weight: float = .05
    dt: float = .05

    def __post_init__(self):
        for name in ('budget', 'readout_sd', 'offset_sd', 'cost_weight', 'effort_weight'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'invalid {name}')
        if self.cultures < 1 or self.final_cultures < 1 or not 0 <= self.max_experiments <= 8:
            raise ValueError('invalid counts')
        if self.dt <= 0 or self.decision_cap < 1 or abs(self.budget * 20 - round(self.budget * 20)) > 1e-8:
            raise ValueError('invalid time/budget ticks')


@dataclass(frozen=True)
class Action:
    kind: str
    intervention: tuple
    duration: float = 6.
    panel: tuple = ()
    cost_ticks: int = 0


def catalogue(n=8, controls=4, panels=((4, 5), (6, 7))):
    interventions = [tuple(np.zeros(n))]
    for i, dose in product(range(controls), (.4, .8)):
        u = np.zeros(n); u[i] = dose
        interventions.append(tuple(u))
    experiments = []
    for u, duration, panel in product(interventions, (1., 6.), panels):
        cost = 1 + .2 * duration + .25 * len(panel) + .25 * sum(u)
        experiments.append(Action('experiment', u, duration, tuple(panel), int(round(cost * 20))))
    return tuple(experiments + [Action('commit', u) for u in interventions])


class ScientistEnv:
    """Only reset/step observations and public_trace belong in a policy process.

    Python object introspection is not a security sandbox: private_task and
    private_result belong to the trusted evaluator, never a policy argument.
    """
    def __init__(self, private_task, config=PilotConfig()):
        self._task, self.config = private_task, config
        self.n = len(private_task.mechanism.b)
        if self.n not in (4,8): raise ValueError('only four-variable calibration or eight-variable pilot supported')
        self.controls = 2 if self.n == 4 else 4
        self.panels = ((2,), (3,)) if self.n == 4 else ((4,5), (6,7))
        self.goals = (3,) if self.n == 4 else (6,7)
        if len(private_task.target) != len(self.goals): raise ValueError('target dimension does not match outcomes')
        self.actions = catalogue(self.n, self.controls, self.panels)
        self.private_result = None

    def reset(self):
        self.budget_ticks = int(round(self.config.budget * 20))
        self.count = self.decisions = 0
        self.done = False
        self.private_result = None
        self.public_trace = []
        baseline = self._measure(np.zeros(self.n), 0., self.config.cultures,
                                 stream(self._task.seed, 'baseline_culture'), stream(self._task.seed, 'baseline_noise'))
        self.public_trace.append({'action_id': -1, 'readouts': baseline,
                                  'measured_mask': np.ones(self.n, dtype=bool), 'cost_ticks': 0})
        return self._obs(True), {}

    def _mask(self):
        return np.array([a.kind == 'commit' or (self.count < self.config.max_experiments and
                         a.cost_ticks <= self.budget_ticks) for a in self.actions], dtype=bool)

    def _obs(self, reset=False):
        goal_mask = np.zeros(self.n, dtype=bool); goal_mask[list(self.goals)] = True
        return {'target': np.array(self._task.target), 'goal_mask': goal_mask,
                'control_mask': np.arange(self.n) < self.controls,
                'panels': self.panels, 'budget_ticks': self.budget_ticks,
                'experiment_count': self.count, 'reset': reset,
                'readout_sd': self.config.readout_sd, 'offset_sd': self.config.offset_sd,
                'cohort_size': self.config.cultures,
                'history': copy.deepcopy(self.public_trace), 'action_mask': self._mask()}

    def step(self, action_id):
        if self.done: raise RuntimeError('episode ended; reset required')
        self.decisions += 1
        valid = isinstance(action_id, (int, np.integer)) and 0 <= action_id < len(self.actions)
        valid = valid and self._mask()[action_id]
        if not valid:
            charge = min(2, self.budget_ticks)
            self.budget_ticks -= charge
            self.done = self.decisions >= self.config.decision_cap
            info = {'error': 'invalid_run' if self.done else 'invalid_action'}
            return self._obs(), -10. if self.done else -self.config.cost_weight * charge / 20, self.done, False, info
        a = self.actions[int(action_id)]
        if a.kind == 'commit':
            y = self._final_outcomes(np.array(a.intervention), self.config.final_cultures,
                                     stream(self._task.seed, 'final'))
            mse = float(np.mean((y[:, list(self.goals)] - self._task.target)**2))
            effort = float(sum(a.intervention))
            self.private_result = {'final_mse': mse, 'effort': effort, 'action_id': int(action_id),
                                   'variable_cost': (self.config.budget * 20 - self.budget_ticks) / 20,
                                   'setup_cost': 1+.25*self.n, 'final_cultures': self.config.final_cultures}
            self.done = True
            return self._obs(), -100 * mse - self.config.effort_weight * effort, True, False, {}
        self.budget_ticks -= a.cost_ticks
        self.count += 1
        full = self._measure(np.array(a.intervention), a.duration, self.config.cultures,
                             stream(self._task.seed, 'culture', self.count), stream(self._task.seed, 'noise', self.count))
        mask = np.zeros(self.n, dtype=bool); mask[list(a.panel)] = True
        full[:, ~mask] = np.nan
        self.public_trace.append({'action_id': int(action_id), 'readouts': full, 'measured_mask': mask,
                                  'cost_ticks': a.cost_ticks})
        if self.decisions >= self.config.decision_cap:
            self.done = True
            return self._obs(), -10., True, False, {'error': 'invalid_run'}
        return self._obs(), -self.config.cost_weight * a.cost_ticks / 20, False, False, {}

    def _measure(self, u, duration, count, culture_rng, noise_rng):
        return measure(self._task.mechanism, u, duration, count,
                       self.config.offset_sd, self.config.readout_sd,
                       culture_rng, noise_rng, self.config.dt)

    def _final_outcomes(self, u, count, rng, duration=6.):
        return final_outcomes(self._task.mechanism, u, count,
                              self.config.offset_sd, rng, duration=duration, dt=self.config.dt)

    def public_record(self):
        return {'target': list(self._task.target), 'budget': self.config.budget,
                'trace': copy.deepcopy(self.public_trace), 'actions': [asdict(a) for a in self.actions]}
