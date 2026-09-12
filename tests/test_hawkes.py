"""The fit recovers the parameters it was simulated from.

This is the one test the Session 2 results rest on. A likelihood with a sign
error, a missing compensator term, or an off-by-one in the recursion will
still optimise to *something* -- it just will not be the truth, and nothing
downstream would notice.

Two truths: the classic single-exponential model with free decay, and the
piecewise-baseline multi-scale model with a fixed decay grid. They exercise
different code paths in `fit_hawkes` and different terms in the compensator.
"""

import numpy as np
import pytest

from microflux.hawkes import (
    Params, extrapolate, fit_hawkes, fit_poisson, ks_exp1, loglik,
    rescaled_residuals, simulate,
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
        [[4.0, 1.0], [1.0, 4.0]],   # fast scale: branching 0.2 / 0.05
        [[0.1, 0.05], [0.05, 0.1]], # slow scale: branching 0.2 / 0.1
    ]),
    beta=np.broadcast_to(SCALES[:, None, None], (2, 2, 2)).copy(),
    edges=np.array([0.0, 10_000.0, np.inf]),
)


@pytest.fixture(scope="module")
def single():
    t, m = simulate(SINGLE, T=20_000.0, seed=1)
    assert 10_000 < len(t) < 60_000
    return t, m


@pytest.fixture(scope="module")
def multi():
    t, m = simulate(MULTI, T=20_000.0, seed=2)
    assert 10_000 < len(t) < 60_000
    return t, m


def test_single_scale_free_beta_recovers_truth(single):
    t, m = single
    got = fit_hawkes(t, m, T=t[-1], K=2)
    np.testing.assert_allclose(got.mu, SINGLE.mu, rtol=0.15)
    np.testing.assert_allclose(got.branching, SINGLE.branching, rtol=0.20)
    np.testing.assert_allclose(got.beta, SINGLE.beta, rtol=0.35)


def test_multi_scale_piecewise_baseline_recovers_truth(multi):
    t, m = multi
    got = fit_hawkes(t, m, T=t[-1], K=2, block_s=10_000.0, scales=SCALES)
    # Two blocks of 10k seconds: the baseline swap must be recovered in both.
    np.testing.assert_allclose(got.mu, MULTI.mu, rtol=0.20)
    np.testing.assert_allclose(got.branching, MULTI.branching, rtol=0.20)
    # And the excitation must land on the right timescale, not just sum right.
    np.testing.assert_allclose(
        got.branching_by_scale[0], MULTI.branching_by_scale[0], rtol=0.30
    )


def test_hawkes_beats_poisson_on_hawkes_data(single):
    t, m = single
    T = t[-1]
    ll_h = loglik(fit_hawkes(t, m, T, 2), t, m, 0.0, T)
    ll_p = loglik(fit_poisson(t, m, T, 2), t, m, 0.0, T)
    assert ll_h > ll_p


def test_rescaled_residuals_are_exp1_under_truth(multi):
    """Time-rescaling with the true parameters gives unit exponentials; with
    the Poisson fit it does not. That is what makes it a goodness-of-fit
    measure on real data rather than another likelihood."""
    t, m = multi
    T = t[-1]
    for r in rescaled_residuals(MULTI, t, m, 0.0, T):
        assert ks_exp1(r) < 0.02
    for r in rescaled_residuals(fit_poisson(t, m, T, 2), t, m, 0.0, T):
        assert ks_exp1(r) > 0.05


def test_heldout_loglik_is_additive_across_a_split(multi):
    """LL[0,T] == LL[0,a] + LL[a,T], with a inside the second baseline block so
    the compensator has to handle a window that starts mid-block."""
    t, m = multi
    T, a = t[-1], t[-1] * 0.7
    whole = loglik(MULTI, t, m, 0.0, T)
    parts = loglik(MULTI, t, m, 0.0, a) + loglik(MULTI, t, m, a, T)
    assert abs(whole - parts) < 1e-6 * abs(whole)


def test_analytic_gradient_matches_finite_differences(multi):
    """The fixed-beta path ships its own gradient. Check it against finite
    differences at a point away from the optimum, where a wrong sign or a
    dropped term is loud, and check its value agrees with `loglik`."""
    from scipy.optimize import check_grad

    from microflux.hawkes import _fixed_objective, block_edges

    t, m = multi
    T = t[-1]
    edges = block_edges(T, 10_000.0)
    beta = MULTI.beta
    obj = _fixed_objective(t, m, T, edges, beta)
    theta = np.log(np.concatenate([[0.4, 0.4, 0.4, 0.4], np.full(8, 1.0)]))

    value, _ = obj(theta)
    p = Params(np.full((2, 2), 0.4), np.full((2, 2, 2), 1.0), beta, edges)
    assert value == pytest.approx(-loglik(p, t, m, 0.0, T), rel=1e-9)

    err = check_grad(lambda th: obj(th)[0], lambda th: obj(th)[1], theta, epsilon=1e-6)
    assert err < 1e-3 * np.linalg.norm(obj(theta)[1])


def test_extrapolate_freezes_baseline_at_train_mean(multi):
    t, m = multi
    p = extrapolate(MULTI, 15_000.0)
    assert p.edges[-2] == 15_000.0
    np.testing.assert_allclose(p.mu[-1], MULTI.mu.mean(0))
    # Inside [0, 15000) nothing changed.
    assert loglik(p, t, m, 0.0, 15_000.0) == pytest.approx(loglik(MULTI, t, m, 0.0, 15_000.0))
