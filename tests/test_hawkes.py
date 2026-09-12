"""The fit recovers the parameters it was simulated from.

This is the one test the whole Session 2 result rests on. A likelihood with a
sign error, a missing compensator term, or an off-by-one in the recursion will
still optimise to *something* -- it just will not be the truth, and nothing
downstream would notice.
"""

import numpy as np
import pytest

from microflux.hawkes import (
    Params, fit_hawkes, fit_poisson, ks_exp1, loglik, rescaled_residuals, simulate,
)

TRUTH = Params(
    mu=np.array([0.5, 0.3]),
    alpha=np.array([[0.8, 0.2], [0.3, 0.6]]),
    beta=np.array([[2.0, 3.0], [1.5, 2.5]]),
)


@pytest.fixture(scope="module")
def sample():
    t, m = simulate(TRUTH, T=20_000.0, seed=1)
    assert 10_000 < len(t) < 60_000, "simulation scale drifted; test would be uninformative"
    return t, m


def test_fit_recovers_truth(sample):
    t, m = sample
    got = fit_hawkes(t, m, T=t[-1], K=2)
    # 20k seconds of a 2-type process pins mu and the branching matrix to
    # within ~15%; beta is the loosest because decay is the hardest to see.
    np.testing.assert_allclose(got.mu, TRUTH.mu, rtol=0.15)
    np.testing.assert_allclose(got.branching, TRUTH.branching, rtol=0.20)
    np.testing.assert_allclose(got.beta, TRUTH.beta, rtol=0.35)


def test_hawkes_beats_poisson_on_hawkes_data(sample):
    t, m = sample
    T = t[-1]
    ll_h = loglik(fit_hawkes(t, m, T, 2), t, m, 0.0, T)
    ll_p = loglik(fit_poisson(t, m, T, 2), t, m, 0.0, T)
    assert ll_h > ll_p


def test_rescaled_residuals_are_exp1_under_truth(sample):
    """Time-rescaling with the true parameters must give unit exponentials;
    with the Poisson fit it must not. This is what makes the test usable as a
    goodness-of-fit measure on real data."""
    t, m = sample
    T = t[-1]
    for r in rescaled_residuals(TRUTH, t, m, 0.0, T):
        assert ks_exp1(r) < 0.02
    for r in rescaled_residuals(fit_poisson(t, m, T, 2), t, m, 0.0, T):
        assert ks_exp1(r) > 0.05


def test_heldout_loglik_uses_history_but_not_parameters(sample):
    """Splitting a window must not change the total: LL[0,T] == LL[0,a] + LL[a,T]."""
    t, m = sample
    T, a = t[-1], t[-1] * 0.7
    whole = loglik(TRUTH, t, m, 0.0, T)
    parts = loglik(TRUTH, t, m, 0.0, a) + loglik(TRUTH, t, m, a, T)
    assert abs(whole - parts) < 1e-6 * abs(whole)
