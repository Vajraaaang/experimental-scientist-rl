import numpy as np
import pytest

from experimentlab.envs import PilotConfig, ScientistEnv
from experimentlab.tasks import continuous_family, make_task, stream
from experimentlab.worlds import Mechanism, fixture


def assert_equivalent(actual, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equivalent(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_equivalent(a, b)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=0, abs=1e-12)
    else:
        assert actual == expected


def test_all_actions_sequences_match_twenty_continuous_mechanisms():
    from experimentlab.private_cache import PrivateDynamicsCache, CachedScientistEnv

    config = PilotConfig(budget=20.)
    mechanisms = continuous_family(192810, 20)
    cache = PrivateDynamicsCache(mechanisms, config)
    for index, mechanism in enumerate(mechanisms):
        for group in range(9):
            task = make_task(mechanism, 3911 + index * 9 + group, config)
            reference = ScientistEnv(task, config)
            cached = CachedScientistEnv(task, config, cache)
            assert_equivalent(cached.reset(), reference.reset())
            for action in [-1, *range(group * 4, group * 4 + 4), 36 + group]:
                assert_equivalent(cached.step(action), reference.step(action))
            assert_equivalent(cached.public_record(), reference.public_record())
            assert_equivalent(cached.private_result, reference.private_result)


def test_fresh_readout_noise_and_culture_rng_consumption():
    from experimentlab.private_cache import PrivateDynamicsCache, CachedScientistEnv

    config = PilotConfig(budget=20.)
    mechanism = continuous_family(124, 1)[0]
    task = make_task(mechanism, 5923, config)
    cached = CachedScientistEnv(task, config, PrivateDynamicsCache([mechanism], config))
    reference = ScientistEnv(task, config)
    cached.reset()
    cached.step(0)
    first = cached.public_trace[-1]['readouts'].copy()
    cached.step(0)
    second = cached.public_trace[-1]['readouts']
    assert not np.array_equal(first, second, equal_nan=True)
    u = np.asarray(cached.actions[0].intervention)
    c1, c2 = stream(721, 'culture'), stream(721, 'culture')
    n1, n2 = stream(721, 'noise'), stream(721, 'noise')
    assert_equivalent(cached._measure(u, 1., config.cultures, c1, n1),
                      reference._measure(u, 1., config.cultures, c2, n2))
    assert c1.bit_generator.state == c2.bit_generator.state
    assert n1.bit_generator.state == n2.bit_generator.state
    f1, f2 = stream(721, 'final'), stream(721, 'final')
    assert_equivalent(cached._final_outcomes(u, config.final_cultures, f1),
                      reference._final_outcomes(u, config.final_cultures, f2))
    assert f1.bit_generator.state == f2.bit_generator.state
    direct = CachedScientistEnv(task, config, cached._cache)
    direct.reset()
    cached.step(36)
    direct.step(36)
    assert_equivalent(cached.private_result['final_mse'], direct.private_result['final_mse'])


def test_cached_public_boundary_and_fresh_cohort_arrays():
    from experimentlab.private_cache import PrivateDynamicsCache, CachedScientistEnv

    config = PilotConfig(budget=20., readout_sd=0.)
    mechanism = continuous_family(4281, 1)[0]
    cache = PrivateDynamicsCache([mechanism], config)
    cached = CachedScientistEnv(make_task(mechanism, 772, config), config, cache)
    forbidden = {'cache', 'predictions', 'index', 'mechanism_hash', 'W', 'b', 'gamma',
                 'mechanism', 'family', 'oracle', 'seed', 'latent', 'offsets'}

    def check_public(value):
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            assert all(not key.startswith('_') for key in value)
            for child in value.values():
                check_public(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                check_public(child)

    check_public(cached.reset())
    check_public(cached.step(11))
    first = cached.public_trace[-1]['readouts'].copy()
    cached.public_trace[-1]['readouts'][:] = -99.
    check_public(cached.step(11))
    assert_equivalent(cached.public_trace[-1]['readouts'], first)
    check_public(cached.public_record())
    check_public(cached.step(36))


def test_qualified_endpoints_bypass_reference_and_are_hash_keyed(monkeypatch):
    from experimentlab import envs
    from experimentlab.private_cache import PrivateDynamicsCache, CachedScientistEnv

    config = PilotConfig()
    rng = np.random.default_rng(7192)
    W = rng.uniform(-.15, .15, (8, 8))
    mechanisms = [fixture(1), Mechanism(W, rng.normal(0., .4, 8), rng.uniform(.5, 1.5, 8))]
    cache = PrivateDynamicsCache(mechanisms, config)
    expected = []
    for mechanism in mechanisms:
        n = len(mechanism.b)
        target = (.5,) if n == 4 else (.5, .5)
        task = make_task(mechanism, 745, config, target)
        reference = ScientistEnv(task, config)
        records = []
        for action in reference.actions:
            if action.kind == 'commit':
                u = np.asarray(action.intervention)
                for duration in (0., 1., 6.):
                    records.append((u, duration, reference._measure(
                        u, duration, config.cultures, stream(745, 'culture'), stream(745, 'noise'))))
        expected.append(records)

    def reference_forbidden(*args, **kwargs):
        raise AssertionError('qualified request unexpectedly used reference path')

    monkeypatch.setattr(envs, 'measure', reference_forbidden)
    monkeypatch.setattr(envs, 'final_outcomes', reference_forbidden)
    for mechanism, records in zip(mechanisms, expected):
        # A distinct object with the same values must find the same hash entry.
        cloned = Mechanism(mechanism.W, mechanism.b, mechanism.gamma)
        n = len(cloned.b)
        target = (.5,) if n == 4 else (.5, .5)
        cached = CachedScientistEnv(make_task(cloned, 745, config, target), config, cache)
        for u, duration, readout in records:
            assert_equivalent(cached._measure(u, duration, config.cultures,
                              stream(745, 'culture'), stream(745, 'noise')), readout)
        cached.reset()
        cached.step(len(cached.actions) - 1)


@pytest.mark.parametrize('mode', ['offset', 'dt', 'duration', 'intervention', 'missing'])
def test_unsupported_requests_use_reference_path(mode, monkeypatch):
    from experimentlab import envs
    from experimentlab.private_cache import PrivateDynamicsCache, CachedScientistEnv

    config = PilotConfig(offset_sd=.03 if mode == 'offset' else 0.,
                         dt=.1 if mode == 'dt' else .05)
    mechanisms = continuous_family(7441, 2)
    cache = PrivateDynamicsCache(mechanisms[:1], config)
    task = make_task(mechanisms[1 if mode == 'missing' else 0], 783, config)
    cached = CachedScientistEnv(task, config, cache)
    reference = ScientistEnv(task, config)
    duration = 2. if mode == 'duration' else 6.
    intervention = np.zeros(8)
    if mode == 'intervention':
        intervention[0] = .2
    calls = []
    measure, final_outcomes = envs.measure, envs.final_outcomes

    def tracked_measure(*args, **kwargs):
        calls.append('measure')
        return measure(*args, **kwargs)

    def tracked_final(*args, **kwargs):
        calls.append('final')
        return final_outcomes(*args, **kwargs)

    monkeypatch.setattr(envs, 'measure', tracked_measure)
    monkeypatch.setattr(envs, 'final_outcomes', tracked_final)
    actual = cached._measure(intervention, duration, config.cultures,
                             stream(712, 'culture'), stream(712, 'noise'))
    assert calls == ['measure']
    expected = reference._measure(intervention, duration, config.cultures,
                                  stream(712, 'culture'), stream(712, 'noise'))
    assert_equivalent(actual, expected)
    calls.clear()
    actual = cached._final_outcomes(intervention, config.final_cultures,
                                    stream(712, 'final'), duration)
    assert calls == ['final']
    expected = reference._final_outcomes(intervention, config.final_cultures,
                                         stream(712, 'final'), duration)
    assert_equivalent(actual, expected)
