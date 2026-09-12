"""Maximum-likelihood estimation for the Hawkes model.

Poisson has a closed form. Hawkes is L-BFGS-B over log-parameters, so
positivity is free and the optimiser walks in the natural scale of rates.

With the decay rates fixed, the recursion is data rather than parameter:
it runs once in numba, and the objective on top of it is a few tensor
operations that PyTorch differentiates. scipy keeps the optimiser -- it is
robust and needs nothing from torch but a function and a gradient. Free
decay is the classic single-exponential model; the recursion is inside its
likelihood, so it keeps numerical gradients.
"""

import numpy as np
import torch
from scipy.optimize import minimize

from microflux.events import Events
from microflux.hawkes import Params, block_of, excitation, loglik, time_in

OPEN = np.array([0.0, np.inf])
torch.set_default_dtype(torch.float64)  # match numpy and scipy exactly


def fit_poisson(ev: Events, T: float, edges: np.ndarray = OPEN, state_baseline: bool = False) -> Params:
    """MLE is the event count over the time spent, per (block, state, type). Closed form."""
    B, S = len(edges) - 1, (ev.S if state_baseline else 1)
    counts = np.zeros((B, S, ev.K))
    np.add.at(counts, (block_of(edges, ev.t), ev.s if S > 1 else 0, ev.m), 1.0)
    tin = time_in(edges, ev, 0.0, T, S)[:, :, None]
    mu = np.divide(counts, tin, out=np.zeros_like(counts), where=tin > 0)
    shape = (1, ev.K, ev.J)
    return Params(mu, np.zeros(shape), np.ones(shape), edges)


def fixed_objective(ev: Events, T: float, edges: np.ndarray, beta: np.ndarray, S: int = 1):
    """Negative log-likelihood and its gradient in log(mu, alpha), beta fixed.

    Everything that does not depend on the parameters is computed once: the
    recursion R, the time in each (block, state) cell, and the compensator
    kernel sum S_lij = sum_{k: e_k = j} (1 - exp(-beta_lij (T - t_k))). What
    remains is

        LL = sum_k log lambda_{m_k}(t_k)  -  sum mu * time_in  -  sum (alpha / beta) * S

    which torch differentiates. Returns (nll, grad) as scipy wants them.
    """
    L, K, J = beta.shape
    B = len(edges) - 1
    Skern = np.zeros((L, K, J))
    for j in range(J):
        tail = (T - ev.t[ev.e == j])[:, None]
        for l in range(L):
            Skern[l, :, j] = (1.0 - np.exp(-beta[l, :, j][None, :] * tail)).sum(0)

    R = torch.from_numpy(excitation(ev.t, ev.e, beta))
    cell = torch.from_numpy(block_of(edges, ev.t) * S + (ev.s if S > 1 else 0))
    tin = torch.from_numpy(time_in(edges, ev, 0.0, T, S).ravel())
    inside = torch.from_numpy(ev.t < T)  # same half-open [0, T) window as `loglik`
    own_idx = (torch.arange(len(ev)), torch.from_numpy(ev.m))
    kern = torch.from_numpy(Skern / beta)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        th = torch.from_numpy(theta).requires_grad_(True)
        x = th.exp()
        mu, alpha = x[:B * S * K].reshape(B * S, K), x[B * S * K:].reshape(L, K, J)
        lam = mu[cell] + torch.einsum("lij,nlij->ni", alpha, R)
        ll = lam[own_idx][inside].log().sum() - (mu * tin[:, None]).sum() - (alpha * kern).sum()
        nll = -ll
        nll.backward()
        return nll.item(), th.grad.numpy().copy()

    return objective


def fit_hawkes(
    ev: Events, T: float, edges: np.ndarray = OPEN, scales: np.ndarray | None = None,
    state_baseline: bool = False,
) -> Params:
    """Maximum likelihood on [0, T].

    Args:
        edges: time-block boundaries for the baseline; `block_edges` builds them.
        scales: fixed decay rates, one kernel scale each, shared across pairs.
            None fits one free beta per pair.
        state_baseline: let mu depend on the state as well as the time block.
            Whether the *kernel* depends on state is fixed by `ev` (its J).
    """
    K, J = ev.K, ev.J
    S = ev.S if state_baseline else 1
    edges = np.asarray(edges, float)
    B = len(edges) - 1
    fixed = scales is not None
    L = len(scales) if fixed else 1
    n_mu, n_alpha = B * S * K, L * K * J
    beta_fixed = (
        np.broadcast_to(np.asarray(scales, float)[:, None, None], (L, K, J)).copy() if fixed else None
    )

    def unpack(theta: np.ndarray) -> Params:
        x = np.exp(theta)
        mu = x[:n_mu].reshape(B, S, K)
        alpha = x[n_mu:n_mu + n_alpha].reshape(L, K, J)
        beta = beta_fixed if fixed else x[n_mu + n_alpha:].reshape(L, K, J)
        return Params(mu, alpha, beta, edges)

    # Start near the model it must beat: half the Poisson rate as baseline and
    # a total branching of 0.3 spread evenly across scales and exciting types.
    rate = np.bincount(ev.m, minlength=K) / T
    beta0 = beta_fixed if fixed else np.full((L, K, J), 5.0)
    parts = [np.tile(rate * 0.5, B * S), (0.3 * beta0 / (L * S)).ravel()]
    if fixed:
        fun, jac = fixed_objective(ev, T, edges, beta_fixed, S), True
    else:
        parts.append(beta0.ravel())
        fun, jac = (lambda theta: -loglik(unpack(theta), ev, 0.0, T)), None
    res = minimize(fun, np.log(np.concatenate(parts)), jac=jac, method="L-BFGS-B",
                   options={"maxiter": 2000})
    if not res.success:
        raise RuntimeError(f"Hawkes MLE did not converge: {res.message}")
    return unpack(res.x)


def extrapolate(p: Params, ev: Events, T: float) -> Params:
    """Freeze the baseline beyond T at its time-weighted train mean, per state.

    A piecewise baseline fit on [0, T] says nothing about what mu does after T.
    Using the last block would reward a lucky final window; the mean is what a
    forecaster without a rate model would assume. The state is still observed
    after T, so the mean is per state. Single-block models pass through
    unchanged in effect.
    """
    w = time_in(p.edges, ev, 0.0, T, p.S)[:, :, None]               # (B, S, 1)
    mean = (p.mu * w).sum(0) / np.maximum(w.sum(0), 1e-12)          # (S, K)
    edges = np.append(p.edges[:-1], [T, np.inf])
    return Params(np.vstack([p.mu, mean[None]]), p.alpha, p.beta, edges)
