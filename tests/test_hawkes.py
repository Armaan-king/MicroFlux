"""The fit recovers the parameters it was simulated from.

This is the one test the Session 2-3 results rest on. A likelihood with a sign
error, a missing compensator term, or an off-by-one in the recursion will
still optimise to *something* -- it just will not be the truth, and nothing
downstream would notice.

Three truths: the classic single-exponential model with free decay; the
piecewise-baseline multi-scale model with a fixed decay grid; and the
confound itself -- a persistent state that changes both the exogenous rate
and the self-excitation, which the fit must attribute correctly.
"""

import numpy as np
import pytest
from scipy.optimize import check_grad

from microflux.events import events
from microflux.hawkes import Params, block_edges, loglik, time_in
from microflux.mle import extrapolate, fit_hawkes, fit_poisson, fixed_objective
from microflux.residuals import ks_exp1, rescaled_residuals
from microflux.simulate import random_states, simulate

ONE = np.array([0.0, np.inf])

SINGLE = Params(
    mu=np.array([[[0.5, 0.3]]]),
    alpha=np.array([[[0.8, 0.2], [0.3, 0.6]]]),
    beta=np.array([[[2.0, 3.0], [1.5, 2.5]]]),
    edges=ONE,
)

SCALES = np.array([20.0, 0.5])  # fast and slow decay, known to the fit
MULTI = Params(
    mu=np.array([[[0.6, 0.2]], [[0.2, 0.6]]]),  # baseline swaps sides at t = 10000
    alpha=np.array([
        [[4.0, 1.0], [1.0, 4.0]],    # fast scale: branching 0.2 / 0.05
        [[0.1, 0.05], [0.05, 0.1]],  # slow scale: branching 0.2 / 0.1
    ]),
    beta=np.broadcast_to(SCALES[:, None, None], (2, 2, 2)).copy(),
    edges=np.array([0.0, 10_000.0, np.inf]),
)

# The confound. Two states, persistent (mean dwell 30 s). In state 1 the
# exogenous BUY rate doubles AND BUY<-BUY self-excitation triples. Exciting
# types are (side, state): columns BUY@0, SELL@0, BUY@1, SELL@1.
STATE = Params(
    mu=np.array([[[0.4, 0.4], [0.8, 0.4]]]),
    alpha=np.array([[[2.0, 0.5, 6.0, 0.5], [0.5, 2.0, 0.5, 2.0]]]),
    beta=np.full((1, 2, 4), 20.0),
    edges=ONE,
)

# Marks. No state; two mark classes drawn iid (70% small, 30% big). A big
# order excites three times as much as a small one, on both sides. Exciting
# types are (side, mark): columns BUY@small, SELL@small, BUY@big, SELL@big.
MARK = Params(
    mu=np.array([[[0.5, 0.5]]]),
    alpha=np.array([[[2.0, 0.5, 6.0, 1.5], [0.5, 2.0, 1.5, 6.0]]]),
    beta=np.full((1, 2, 4), 20.0),
    edges=ONE,
)
MARK_PROB = np.array([0.7, 0.3])


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
    st, ss = random_states(T=20_000.0, S=2, mean_dwell=30.0, seed=3)
    ev = simulate(STATE, T=20_000.0, seed=3, step_t=st, step_s=ss)
    assert 10_000 < len(ev) < 60_000
    assert ev.S == 2 and ev.J == 4
    return ev


@pytest.fixture(scope="module")
def marked():
    ev = simulate(MARK, T=20_000.0, seed=4, mark_prob=MARK_PROB)
    assert 10_000 < len(ev) < 60_000
    assert ev.C == 2 and ev.J == 4
    return ev


def test_single_scale_free_beta_recovers_truth(single):
    got = fit_hawkes(single, T=single.t[-1])
    np.testing.assert_allclose(got.mu, SINGLE.mu, rtol=0.15)
    np.testing.assert_allclose(got.branching, SINGLE.branching, rtol=0.20)
    np.testing.assert_allclose(got.beta, SINGLE.beta, rtol=0.35)


def test_multi_scale_piecewise_baseline_recovers_truth(multi):
    got = fit_hawkes(multi, T=multi.t[-1], edges=block_edges(multi.t[-1], 10_000.0), scales=SCALES)
    np.testing.assert_allclose(got.mu, MULTI.mu, rtol=0.20)
    np.testing.assert_allclose(got.branching, MULTI.branching, rtol=0.20)
    # The excitation must land on the right timescale, not just sum right.
    np.testing.assert_allclose(got.branching_by_scale[0], MULTI.branching_by_scale[0], rtol=0.30)


def test_fit_separates_state_rate_from_state_kernel(stated):
    """The confound, resolved: with both mu(s) and alpha(s) free, the fit must
    put the doubled rate in mu and the tripled excitation in alpha."""
    got = fit_hawkes(stated, T=stated.t[-1], scales=np.array([20.0]), state_baseline=True)
    np.testing.assert_allclose(got.mu, STATE.mu, rtol=0.25)
    np.testing.assert_allclose(got.branching, STATE.branching, rtol=0.25)
    ratio = got.branching[0, 2] / got.branching[0, 0]  # BUY<-BUY@1 over BUY<-BUY@0
    assert 2.2 < ratio < 3.8


def test_mark_dependent_excitation_recovers_truth(marked):
    """A big order must be seen to excite three times what a small one does,
    and the mark of the exciting order -- not the excited one -- must be what
    the kernel keys on."""
    got = fit_hawkes(marked, T=marked.t[-1], scales=np.array([20.0]))
    np.testing.assert_allclose(got.branching, MARK.branching, rtol=0.25)
    for i in range(2):  # BUY<-BUY@big / BUY<-BUY@small, then SELL
        ratio = got.branching[i, i + 2] / got.branching[i, i]
        assert 2.2 < ratio < 3.8


def test_state_free_fit_blends_the_states(stated):
    """Dropping the state gives a fit between the two per-state truths -- the
    number a state model has to beat, and a check that S=1 still works."""
    flat = events(stated.t, stated.m, stated.K)
    got = fit_hawkes(flat, T=flat.t[-1], scales=np.array([20.0]))
    assert STATE.branching[0, 0] < got.branching[0, 0] < STATE.branching[0, 2]
    assert STATE.mu[0, 0, 0] < got.mu[0, 0, 0] < STATE.mu[0, 1, 0]


def test_time_in_partitions_the_window(stated):
    T = stated.t[-1]
    tin = time_in(block_edges(T, 5_000.0), stated, 1_000.0, 17_000.0, S=2)
    assert tin.sum() == pytest.approx(16_000.0)
    assert tin.shape == (4, 2)


def test_hawkes_beats_poisson_on_hawkes_data(single):
    T = single.t[-1]
    assert loglik(fit_hawkes(single, T), single, 0.0, T) > loglik(fit_poisson(single, T), single, 0.0, T)


def test_rescaled_residuals_are_exp1_under_truth(stated):
    """Time-rescaling with the true parameters gives unit exponentials; with
    the Poisson fit it does not. That is what makes it a goodness-of-fit
    measure on real data rather than another likelihood. On the state model,
    so the state-dependent baseline integral is exercised."""
    T = stated.t[-1]
    for r in rescaled_residuals(STATE, stated, 0.0, T):
        assert ks_exp1(r) < 0.02
    for r in rescaled_residuals(fit_poisson(stated, T, state_baseline=True), stated, 0.0, T):
        assert ks_exp1(r) > 0.05


def test_heldout_loglik_is_additive_across_a_split(stated):
    """LL[0,T] == LL[0,a] + LL[a,T], with a mid-way so the compensator has to
    handle a window that starts inside a state interval."""
    T, a = stated.t[-1], stated.t[-1] * 0.7
    whole = loglik(STATE, stated, 0.0, T)
    parts = loglik(STATE, stated, 0.0, a) + loglik(STATE, stated, a, T)
    assert abs(whole - parts) < 1e-6 * abs(whole)


def test_torch_objective_matches_numpy_loglik_and_finite_differences(stated):
    """The fixed-beta objective is torch; the reference likelihood is numpy.
    They must agree to 1e-9, and the autograd gradient must match finite
    differences at a point away from the optimum -- a dropped term in the
    objective is loud there."""
    T = stated.t[-1]
    edges = block_edges(T, 10_000.0)
    obj = fixed_objective(stated, T, edges, STATE.beta, S=2)
    theta = np.log(np.concatenate([np.full(8, 0.4), np.full(8, 1.0)]))

    value, _ = obj(theta)
    p = Params(np.full((2, 2, 2), 0.4), np.full((1, 2, 4), 1.0), STATE.beta, edges)
    assert value == pytest.approx(-loglik(p, stated, 0.0, T), rel=1e-9)

    err = check_grad(lambda th: obj(th)[0], lambda th: obj(th)[1], theta, epsilon=1e-6)
    assert err < 1e-3 * np.linalg.norm(obj(theta)[1])


def test_extrapolate_freezes_baseline_per_state(stated):
    p = extrapolate(STATE, stated, 15_000.0)
    assert p.edges[-2] == 15_000.0
    np.testing.assert_allclose(p.mu[-1], STATE.mu[0])  # one block: the mean is itself, per state
    assert loglik(p, stated, 0.0, 15_000.0) == pytest.approx(loglik(STATE, stated, 0.0, 15_000.0))
