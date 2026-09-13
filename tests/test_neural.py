"""The neural likelihood agrees with the classical one where they must.

The shared head integrates in closed form; a Poisson is the head with
a = b; and the block decomposition is a partition of the total. Any of
these failing would make NLL/event across the ladder incomparable.
"""

import numpy as np
import pytest
import torch

from microflux.events import events
from microflux.hawkes import Params, loglik as hawkes_loglik
from microflux.neural import (
    Attention, Head, SummaryMLP, TPP, block_loglik, features, loglik, rescaled_residuals,
)
from microflux.simulate import random_states, simulate

SCALES = np.log(2.0) / np.array([0.005, 0.05, 0.5, 5.0, 50.0])
# K=2 sides, S=2 states, C=2 marks -> J=8 exciting types, columns (side, state, mark)
TRUTH = Params(
    mu=np.array([[[0.5, 0.4], [0.3, 0.6]]]),
    alpha=np.array([[[3.0, 0.5, 4.0, 0.5, 5.0, 0.5, 6.0, 0.5], [0.5, 3.0, 0.5, 4.0, 0.5, 5.0, 0.5, 6.0]]]),
    beta=np.full((1, 2, 8), 20.0),
    edges=np.array([0.0, np.inf]),
)


@pytest.fixture(scope="module")
def data():
    st, ss = random_states(T=4_000.0, S=2, mean_dwell=20.0, seed=5)
    ev = simulate(TRUTH, T=4_000.0, seed=5, step_t=st, step_s=ss, mark_prob=np.array([[0.6, 0.4], [0.6, 0.4]]))
    return ev, features(ev, T_end=4_000.0)


def test_head_integral_matches_quadrature():
    head = Head(d_in=8, K=2, S=2, scales=SCALES)
    torch.manual_seed(0)
    a, b = torch.rand(3, 2, 5) * 2 + 0.1, torch.rand(3, 2, 5) * 2 + 0.1
    tau0, tau1 = torch.tensor([0.0, 0.2, 1.0]), torch.tensor([0.5, 3.0, 40.0])
    a, b, tau0, tau1 = (x.to(torch.float32) for x in (a, b, tau0, tau1))
    exact = head.integral(a, b, tau0, tau1)
    grid = torch.linspace(0, 1, 20001, dtype=torch.float32)
    for r in range(3):
        taus = tau0[r] + (tau1[r] - tau0[r]) * grid
        lam = head.intensity(a[r:r + 1].expand(len(taus), -1, -1), b[r:r + 1].expand(len(taus), -1, -1), taus)
        num = torch.trapezoid(lam, taus, dim=0)
        assert torch.allclose(exact[r], num, rtol=1e-3)


class ConstHead(Head):
    """a = b = mu / L: a Poisson process, whatever the encoder says."""

    def __init__(self, mu, scales):
        super().__init__(d_in=1, K=len(mu), S=1, scales=scales)
        self.mu = torch.as_tensor(mu, dtype=torch.float32)

    def forward(self, h, state):
        x = (self.mu / self.L)[None, :, None].expand(len(h), -1, self.L)
        return x, x


def test_poisson_is_the_head_with_a_equal_b(data):
    ev, f = data
    mu = np.array([0.7, 0.4])
    enc = SummaryMLP(f.summary.shape[1], K=2, C=2, S=2)
    model = TPP(enc, ConstHead(mu, SCALES))
    a, b = 1_000.0, 3_000.0
    got = loglik(model, f, a, b, N=16)
    want = hawkes_loglik(Params(mu[None, None, :], np.zeros((1, 2, 8)), np.ones((1, 2, 8)), np.array([0.0, np.inf])), ev, a, b)
    assert got == pytest.approx(want, rel=1e-5)  # float32 model, float64 reference
    # and its residuals are exactly mu * inter-event gaps
    r = rescaled_residuals(model, f, a, b, N=16)
    t0 = ev.t[(ev.t >= a) & (ev.t < b) & (ev.m == 0)]
    np.testing.assert_allclose(r[0], mu[0] * np.diff(t0), rtol=1e-4)


def test_blocks_partition_the_likelihood(data):
    ev, f = data
    torch.manual_seed(1)
    model = TPP(Attention(f.summary.shape[1], K=2, C=2, S=2, d=16, heads=2, layers=1), Head(16, 2, 2, SCALES))
    a, b = 500.0, 2_500.0
    ll, n = block_loglik(model, f, a, b, N=16, block_s=100.0)
    assert ll.sum() == pytest.approx(loglik(model, f, a, b, N=16), rel=1e-6)
    assert n.sum() == ((ev.t >= a) & (ev.t < b)).sum()


def test_pieces_cover_every_interval_once(data):
    ev, f = data
    # the pieces of the whole sequence tile [t_0, T_end] exactly
    total = (f.piece_u1 - f.piece_u0).sum()
    assert total == pytest.approx(4_000.0 - ev.t[0], rel=1e-12)
    assert (f.piece_u1 > f.piece_u0).all()
    # every book row strictly inside an interval starts a new piece
    inside = (ev.step_t > ev.t[0]) & (ev.step_t < 4_000.0)
    assert set(np.round(ev.step_t[inside], 9)) <= set(np.round(f.piece_u0, 9))
