import numpy as np
import pytest

from experimentlab.worlds import Mechanism, fixture, integrate, equilibrium, fresh_cultures, ode_endpoint
from experimentlab.diagnostics import numerical_audit, branching_audit


def test_supplied_fixture_numbers():
    for h, sign in [(1, 1), (-1, -1)]:
        m = fixture(h)
        short = integrate(m, np.full(4, .5), np.array([.8, 0, 0, 0]), 1)
        long = integrate(m, np.full(4, .5), np.array([.8, 0, 0, 0]), 6)
        assert short[2] == pytest.approx(.5 - sign * .042955, abs=6e-7)
        assert short[3] == pytest.approx(.5 - sign * .005140, abs=6e-7)
        assert long[3] == pytest.approx(.5 - sign * .090533, abs=6e-7)


def test_integration_is_bounded_and_refines_toward_independent_solver():
    m = fixture(1)
    x = np.full(4, .5)
    u = np.array([.8, 0, 0, 0])
    exact = ode_endpoint(m, x, u, 6)
    a, b = integrate(m, x, u, 6), integrate(m, x, u, 6, dt=.025)
    assert a.dtype == np.float64
    assert ((a >= 0) & (a <= 1)).all()
    assert np.max(abs(b - exact)) < np.max(abs(a - exact)) < .002
    assert np.allclose(integrate(m, x, np.zeros(4), 6), x)
    assert integrate(m, x, u, .05)[0] > .1  # production suppression, not clamping


def test_equilibria_and_cultures():
    m = fixture(1)
    x = equilibrium(m)
    assert np.max(abs(x - .5)) < 1e-10
    a = fresh_cultures(m, 8, .03, np.random.default_rng(1))
    b = fresh_cultures(m, 8, .03, np.random.default_rng(2))
    assert not np.array_equal(a.b, b.b)
    assert a.x.shape == (8, 4)
    with pytest.raises(ValueError):
        Mechanism(np.ones((4, 4)), np.zeros(4), np.ones(4))


def test_diagnostics_exact_values_and_interpretation():
    audit = numerical_audit()
    assert audit['sign_error'] == pytest.approx(6.2e-10, rel=.03)
    assert audit['informed_mse'] == pytest.approx(2.84e-7, rel=.03)
    assert audit['negative_return'] == pytest.approx(.05653, abs=2e-5)
    b = branching_audit()
    assert b['adaptive_accuracy'] == 1
    assert b['best_fixed_accuracy'] == .75
