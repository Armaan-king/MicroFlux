"""Regressions found in review of the neural pilot (ARCHITECTURE.md D13).

Each test fails on the implementation the pilot ran with and pins the fix.
"""

import numpy as np
import pytest
import torch

from microflux.events import events
from microflux.hawkes import Params
from microflux.mle import fit_poisson
from microflux.experiment import evaluate
from microflux.neural import Attention, Head, TPP, block_loglik, features, loglik

SCALES = np.log(2.0) / np.array([0.005, 0.05, 0.5, 5.0, 50.0])


def _small_model(f, seed=0):
    torch.manual_seed(seed)
    enc = Attention(f.summary.shape[1], K=2, C=2, S=2, d=16, heads=2, layers=1)
    return TPP(enc, Head(16, 2, 2, SCALES))


@pytest.fixture(scope="module")
def tied():
    """A sequence with a same-timestamp pair: BUY at t=10.0 (index 3) then
    SELL at t=10.0 (index 4), in capture order."""
    t = np.array([0.0, 2.0, 5.0, 10.0, 10.0, 10.4, 11.0, 15.0, 20.0, 30.0])
    m = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    c = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 0])
    st, ss = np.array([0.0, 8.0, 16.0]), np.array([0, 1, 0])
    return events(t, m, 2, st, ss, 2, mark=c, C=2)


# --- 1. trailing counts must respect capture order ---------------------------


def test_summaries_exclude_later_events_at_the_same_timestamp(tied):
    f = features(tied, T_end=40.0)
    # index 3 is BUY at t=10.0; index 4 is SELL at the same timestamp, later in capture order.
    # SELL count in the 1 s window at index 3 must be 0 -- the SELL has not happened yet.
    sell_1s = f.summary[:, 1]  # column 1 = SELL, 1 s window (log1p of the count)
    assert sell_1s[3] == pytest.approx(np.log1p(0))
    assert sell_1s[4] == pytest.approx(np.log1p(1))


def test_features_of_a_prefix_do_not_depend_on_later_events(tied):
    """Dropping event 4 (the SELL tied with event 3) must not change anything
    computed for events 0..3 -- inputs are causal in capture order."""
    full = features(tied, T_end=40.0)
    ev_prefix = events(tied.t[:4], tied.m[:4], 2, tied.step_t, tied.step_s, 2, mark=tied.c[:4], C=2)
    prefix = features(ev_prefix, T_end=tied.t[3])
    np.testing.assert_array_equal(full.summary[:4], prefix.summary)
    np.testing.assert_array_equal(full.log_gap[:4], prefix.log_gap)


# --- 2. block log-likelihood must split pieces at block boundaries ------------


def test_each_block_matches_an_independent_likelihood(tied):
    f = features(tied, T_end=40.0)
    model = _small_model(f)
    a, b, block_s = 1.0, 31.0, 7.0  # boundaries at 8, 15, 22, 29 -- inside several pieces
    ll, n = block_loglik(model, f, a, b, N=4, block_s=block_s)
    edges = np.append(np.arange(a, b, block_s), b)
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        assert ll[k] == pytest.approx(loglik(model, f, lo, hi, N=4), rel=1e-6, abs=1e-6), f"block {k}"
        assert n[k] == ((tied.t >= lo) & (tied.t < hi)).sum()


# --- 3. the training objective must be the whole training window -------------


def test_training_objective_equals_window_loglik(tied):
    from microflux.neural import training_objective

    f = features(tied, T_end=40.0)
    model = _small_model(f)
    T_train = 17.0  # falls inside the interval (15, 20]; that clipped tail must be counted
    assert training_objective(model, f, T_train, N=4) == pytest.approx(loglik(model, f, 0.0, T_train, N=4), rel=1e-6)


# --- 4. validation evaluation must never touch the test window ----------------


def test_evaluate_defaults_are_validation_only():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 100, 300))
    ev = events(t, rng.integers(0, 2, 300), 2)
    p = fit_poisson(ev.before(70.0), 70.0)
    # a test window that would raise if evaluated (zero events -> division by zero)
    split = {"train": (0.0, 70.0), "val": (70.0, 85.0), "test": (99.999, 99.999)}
    out = evaluate(p, ev, split)
    assert set(out) == {"train", "val", "ks", "ks_on"}
    assert out["ks_on"] == "val"
