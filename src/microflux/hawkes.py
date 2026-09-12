"""Multivariate Hawkes process: the model and its likelihood.

    lambda_i(t) = mu_i(t) + sum_l sum_j sum_{t_k < t, e_k = j} alpha_lij exp(-beta_lij (t - t_k))

mu_i(t) is constant on each of B time blocks; each (excited i, exciting j)
pair has L exponential scales. Both are controls for a way the plain model
can lie: B > 1 stops a slow kernel absorbing non-stationarity, L > 1 stops
one timescale standing in for several. Poisson is B = 1, L = 1, alpha = 0.

The exponential kernel is what keeps everything O(n): one recursion per
(scale, pair). Time is seconds from the window start.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from microflux.events import Events


@dataclass(frozen=True)
class Params:
    mu: np.ndarray     # (B, K)     exogenous rate per time block, per excited type
    alpha: np.ndarray  # (L, K, J)  alpha[l, i, j]: exciting type j excites type i at scale l
    beta: np.ndarray   # (L, K, J)  decay rate of that excitation
    edges: np.ndarray  # (B + 1,)   block boundaries; edges[0] = 0, edges[-1] = inf

    @property
    def K(self) -> int:
        return self.mu.shape[1]

    @property
    def branching(self) -> np.ndarray:
        """(K, J): expected type-i events one exciting-type-j event triggers, all scales."""
        return (self.alpha / self.beta).sum(0)

    @property
    def branching_by_scale(self) -> np.ndarray:
        return self.alpha / self.beta

    @property
    def spectral_radius(self) -> float:
        """Stationarity requires this < 1; near 1 is near-critical. Defined only
        when exciting and excited types coincide (no state)."""
        br = self.branching
        if br.shape[0] != br.shape[1]:
            return float("nan")
        return float(np.abs(np.linalg.eigvals(br)).max())

    @property
    def half_lives(self) -> np.ndarray:
        return np.log(2.0) / self.beta


# --- baseline ---------------------------------------------------------------


def block_edges(T: float, block_s: float | None) -> np.ndarray:
    """Boundaries of the baseline blocks over [0, T]; the last block is open."""
    if block_s is None:
        return np.array([0.0, np.inf])
    return np.append(np.arange(0.0, T, block_s), np.inf)


def block_of(edges: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 2)


def baseline_cum(p: Params, t: np.ndarray) -> np.ndarray:
    """(n, K): integral of mu_i from 0 to each t."""
    lo, hi = p.edges[:-1], p.edges[1:]
    width = np.clip(t[:, None] - lo[None, :], 0.0, (hi - lo)[None, :])
    return width @ p.mu


# --- kernel -----------------------------------------------------------------


@njit(cache=True)
def excitation(t: np.ndarray, e: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """R[k, l, i, j] = sum over earlier events of exciting type j of exp(-beta_lij (t_k - t_l)).

    The recursion R[k] = exp(-beta dt) (R[k-1] + 1[e_{k-1} = j]) is what makes
    the exponential kernel O(n). An event at the same instant as its
    predecessor decays by exp(0) = 1 and is counted -- sequence order is capture
    order, which is the ground-truth ordering (see load.py).
    """
    n = len(t)
    L, K, J = beta.shape
    R = np.zeros((n, L, K, J))
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        prev = e[k - 1]
        for l in range(L):
            for i in range(K):
                for j in range(J):
                    R[k, l, i, j] = np.exp(-beta[l, i, j] * dt) * (
                        R[k - 1, l, i, j] + (1.0 if prev == j else 0.0)
                    )
    return R


# --- likelihood -------------------------------------------------------------


def intensity(p: Params, ev: Events, R: np.ndarray | None = None) -> np.ndarray:
    """lambda_i(t_k) for every event k and excited type i. Shape (n, K)."""
    if R is None:
        R = excitation(ev.t, ev.e, p.beta)
    return p.mu[block_of(p.edges, ev.t)] + np.einsum("lij,nlij->ni", p.alpha, R)


def compensator(p: Params, ev: Events, a: float, b: float) -> np.ndarray:
    """Integral of lambda_i over [a, b], per excited type. Shape (K,).

    Events before `a` contribute their decaying tail over the window; events
    inside it contribute from their own time. Both are the one formula
    (alpha/beta)(exp(-beta max(a - t_k, 0)) - exp(-beta (b - t_k))).
    """
    out = (baseline_cum(p, np.array([b])) - baseline_cum(p, np.array([a])))[0]
    inside = ev.t < b
    tl, el = ev.t[inside], ev.e[inside]
    for j in range(ev.J):
        tj = tl[el == j]
        lead = np.maximum(a - tj, 0.0)[:, None]
        tail = (b - tj)[:, None]
        for l in range(p.beta.shape[0]):
            bl = p.beta[l, :, j][None, :]
            out += (p.alpha[l, :, j] / p.beta[l, :, j]) * (
                np.exp(-bl * lead) - np.exp(-bl * tail)
            ).sum(axis=0)
    return out


def loglik(p: Params, ev: Events, a: float, b: float, R: np.ndarray | None = None) -> float:
    """Log-likelihood of the events in [a, b), given the full history.

    Using history from before `a` is not leakage: a point process conditions on
    what has happened, and what happened is observed. Leakage would be fitting
    the parameters on it, which `mle.fit_hawkes` never does.
    """
    lam = intensity(p, ev, R)
    own = lam[np.arange(len(ev)), ev.m][ev.window(a, b)]
    return float(np.log(own).sum() - compensator(p, ev, a, b).sum())
