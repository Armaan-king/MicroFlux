"""The fit recovers the parameters it was simulated from.

This is the one test the Session 2-3 results rest on. A likelihood with a sign
error, a missing compensator term, or an off-by-one in the recursion will
still optimise to *something* -- it just will not be the truth, and nothing
downstream would notice.

Three truths: the classic single-exponential model with free decay; the
piecewise-baseline multi-scale model with a fixed decay grid; and a
state-dependent model where the same side excites differently depending on
the state it landed in.
"""

import numpy as np
import pytest
from scipy.optimize import check_grad

from microflux.hawkes import (
    Params, _fixed_objective, block_edges, extrapolate, fit_hawkes, fit_poisson,
    ks_exp1, loglik, rescaled_residuals, simulate,
)

SINGLE = Params(
    mu=np.array([[0.5, 0.3]]),
    alpha=np.array([[[0.8, 0.2], [0.3, 0.6]]]),
    beta=np.array([[[2.0, 3.0], [1.5, 2.5]]]),
    edges=np.array([0.0, np.inf]),
)

SCALES = np.array([20.0, 0.5])  # fast and slow decay, known to the fit
MULTI = Params(
    mu=np.array([[0.6, 0.2], [0.2, 0.6]]),  # baseline swaps sides at t = 10000
    alpha=np.array([
        [[4.0, 1.0], [1.0, 4.0]],    # fast scale: branching 0.2 / 0.05
        [[0.1, 0.05], [0.05, 0.1]],  # slow scale: branching 0.2 / 0.1
    ]),
    beta=np.broadcast_to(SCALES[:, None, None], (2, 2, 2)).copy(),
    edges=np.array([0.0, 10_000.0, np.inf]),
)

# Exciting types are (side, state): columns BUY@0, SELL@0, BUY@1, SELL@1.
# In state 1 self-excitation is three times stronger. Cross is unchanged.
STATE = Params(
    mu=np.array([[0.5, 0.5]]),
    alpha=np.array([[[2.0, 0.5, 6.0, 0.5], [0.5, 2.0, 0.5, 6.0]]]),
    beta=np.full((1, 2, 4), 20.0),
    edges=np.array([0.0, np.inf]),
)
STATE_PROB = np.array([0.6, 0.4])


@pytest.fixture(scope="module")
def single():
    ev = simulate(SINGLE, T=20_000.0, seed=1)
    assert 10_000 < len(ev) < 60_000
    return ev


@pytest.fixture(scope="module")
def multi():
    ev = simulate(MULTI, T=20_000.0, seed=2)
    assert 10_000 < len(ev) < 60_000
    return ev


@pytest.fixture(scope="module")
def stated():
    ev = simulate(STATE, T=20_000.0, seed=3, state_prob=STATE_PROB)
    assert 10_000 < len(ev) < 60_000
    assert ev.J == 4
    return ev


def test_single_scale_free_beta_recovers_truth(single):
    got = fit_hawkes(single, T=single.t[-1])
    np.testing.assert_allclose(got.mu, SINGLE.mu, rtol=0.15)
    np.testing.assert_allclose(got.branching, SINGLE.branching, rtol=0.20)
    np.testing.assert_allclose(got.beta, SINGLE.beta, rtol=0.35)


def test_multi_scale_piecewise_baseline_recovers_truth(multi):
    got = fit_hawkes(multi, T=multi.t[-1], block_s=10_000.0, scales=SCALES)
    np.testing.assert_allclose(got.mu, MULTI.mu, rtol=0.20)
    np.testing.assert_allclose(got.branching, MULTI.branching, rtol=0.20)
    # The excitation must land on the right timescale, not just sum right.
    np.testing.assert_allclose(got.branching_by_scale[0], MULTI.branching_by_scale[0], rtol=0.30)


def test_state_dependent_excitation_recovers_truth(stated):
    """The fit must see that a BUY in state 1 triggers three times the
    follow-on of a BUY in state 0 -- and not smear the two together."""
    got = fit_hawkes(stated, T=stated.t[-1], scales=np.array([20.0]))
    np.testing.assert_allclose(got.branching, STATE.branching, rtol=0.25)
    ratio = got.branching[0, 2] / got.branching[0, 0]  # BUY<-BUY@1 over BUY<-BUY@0
    assert 2.2 < ratio < 3.8


def test_state_free_fit_on_state_data_averages_the_states(stated):
    """Collapsing the state should give roughly the frequency-weighted mean of
    the two per-state kernels -- the number the state model must beat."""
    from microflux.hawkes import events

    flat = events(stated.t, stated.m, stated.K)
    got = fit_hawkes(flat, T=flat.t[-1], scales=np.array([20.0]))
    expected = STATE.branching[0, 0] * STATE_PROB[0] + STATE.branching[0, 2] * STATE_PROB[1]
    assert got.branching[0, 0] == pytest.approx(expected, rel=0.25)


def test_hawkes_beats_poisson_on_hawkes_data(single):
    T = single.t[-1]
    assert loglik(fit_hawkes(single, T), single, 0.0, T) > loglik(fit_poisson(single, T), single, 0.0, T)


def test_rescaled_residuals_are_exp1_under_truth(multi):
    """Time-rescaling with the true parameters gives unit exponentials; with
    the Poisson fit it does not. That is what makes it a goodness-of-fit
    measure on real data rather than another likelihood."""
    T = multi.t[-1]
    for r in rescaled_residuals(MULTI, multi, 0.0, T):
        assert ks_exp1(r) < 0.02
    for r in rescaled_residuals(fit_poisson(multi, T), multi, 0.0, T):
        assert ks_exp1(r) > 0.05


def test_heldout_loglik_is_additive_across_a_split(multi):
    """LL[0,T] == LL[0,a] + LL[a,T], with a inside the second baseline block so
    the compensator has to handle a window that starts mid-block."""
    T, a = multi.t[-1], multi.t[-1] * 0.7
    whole = loglik(MULTI, multi, 0.0, T)
    parts = loglik(MULTI, multi, 0.0, a) + loglik(MULTI, multi, a, T)
    assert abs(whole - parts) < 1e-6 * abs(whole)


def test_analytic_gradient_matches_finite_differences(stated):
    """The fixed-beta path ships its own gradient. Check it against finite
    differences at a point away from the optimum, where a wrong sign or a
    dropped term is loud, and check its value agrees with `loglik`. Done on
    the state model so the K != J indexing is exercised."""
    T = stated.t[-1]
    edges = block_edges(T, 10_000.0)
    obj = _fixed_objective(stated, T, edges, STATE.beta)
    theta = np.log(np.concatenate([np.full(4, 0.4), np.full(8, 1.0)]))

    value, _ = obj(theta)
    p = Params(np.full((2, 2), 0.4), np.full((1, 2, 4), 1.0), STATE.beta, edges)
    assert value == pytest.approx(-loglik(p, stated, 0.0, T), rel=1e-9)

    err = check_grad(lambda th: obj(th)[0], lambda th: obj(th)[1], theta, epsilon=1e-6)
    assert err < 1e-3 * np.linalg.norm(obj(theta)[1])


def test_extrapolate_freezes_baseline_at_train_mean(multi):
    p = extrapolate(MULTI, 15_000.0)
    assert p.edges[-2] == 15_000.0
    np.testing.assert_allclose(p.mu[-1], MULTI.mu.mean(0))
    assert loglik(p, multi, 0.0, 15_000.0) == pytest.approx(loglik(MULTI, multi, 0.0, 15_000.0))
