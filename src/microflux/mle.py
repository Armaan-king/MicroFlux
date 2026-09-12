"""Maximum-likelihood estimation for the Hawkes model.

Poisson has a closed form. Hawkes is L-BFGS-B over log-parameters, so
positivity is free and the optimiser walks in the natural scale of rates.
With the decay rates fixed the likelihood is concave in (mu, alpha), the
recursion runs once, and the gradient is analytic -- that path is fast and
cannot land in a wrong optimum. Free decay is the classic single-exponential
model and keeps numerical gradients.
"""

import numpy as np
from scipy.optimize import minimize

from microflux.events import Events
from microflux.hawkes import Params, block_edges, block_of, excitation, loglik


def fit_poisson(ev: Events, T: float, block_s: float | None = None) -> Params:
    """MLE is the event rate per type per block. Closed form."""
    edges = block_edges(T, block_s)
    counts = np.zeros((len(edges) - 1, ev.K))
    np.add.at(counts, (block_of(edges, ev.t), ev.m), 1.0)
    widths = np.minimum(edges[1:], T) - edges[:-1]
    shape = (1, ev.K, ev.J)
    return Params(counts / widths[:, None], np.zeros(shape), np.ones(shape), edges)


def fixed_objective(ev: Events, T: float, edges: np.ndarray, beta: np.ndarray):
    """Negative log-likelihood and its gradient in log(mu, alpha), beta fixed.

    With beta fixed the recursion R never changes, and the compensator kernel
    term is (alpha / beta) * S with S precomputed, so an evaluation is one
    einsum. The gradient is closed-form:

        dLL/dmu_bi     = sum_{k: m_k = i, block b} 1 / lambda_i(t_k)  -  |block b|
        dLL/dalpha_lij = sum_{k: m_k = i} R_lij[k] / lambda_i(t_k)   -  S_lij / beta_lij

    which turns 60+ likelihood evaluations per L-BFGS step into one.
    """
    L, K, J = beta.shape
    B = len(edges) - 1
    R = excitation(ev.t, ev.e, beta)
    blk = block_of(edges, ev.t)
    widths = np.minimum(edges[1:], T) - edges[:-1]
    idx = np.arange(len(ev))
    S = np.zeros((L, K, J))
    for j in range(J):
        tail = (T - ev.t[ev.e == j])[:, None]
        for l in range(L):
            S[l, :, j] = (1.0 - np.exp(-beta[l, :, j][None, :] * tail)).sum(0)
    by_type = [ev.m == i for i in range(K)]
    inside = ev.t < T  # same half-open [0, T) window as `loglik`

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.exp(theta)
        mu, alpha = x[:B * K].reshape(B, K), x[B * K:].reshape(L, K, J)
        lam = mu[blk] + np.einsum("lij,nlij->ni", alpha, R)
        own = lam[idx, ev.m]
        ll = np.log(own[inside]).sum() - (mu * widths[:, None]).sum() - (alpha / beta * S).sum()
        inv = np.where(inside, 1.0 / own, 0.0)
        g_mu = np.repeat(-widths[:, None], K, axis=1)
        np.add.at(g_mu, (blk, ev.m), inv)
        g_alpha = -S / beta
        for i, ki in enumerate(by_type):
            g_alpha[:, i, :] += np.einsum("nlj,n->lj", R[ki, :, i, :], inv[ki])
        return -ll, -np.concatenate([g_mu.ravel(), g_alpha.ravel()]) * x

    return objective


def fit_hawkes(
    ev: Events, T: float, block_s: float | None = None, scales: np.ndarray | None = None,
) -> Params:
    """Maximum likelihood on [0, T].

    Args:
        block_s: width of the baseline blocks in seconds; None for one block.
        scales: fixed decay rates, one kernel scale each, shared across pairs.
            None fits one free beta per pair.
    """
    K, J = ev.K, ev.J
    edges = block_edges(T, block_s)
    B = len(edges) - 1
    fixed = scales is not None
    L = len(scales) if fixed else 1
    n_mu, n_alpha = B * K, L * K * J
    beta_fixed = (
        np.broadcast_to(np.asarray(scales, float)[:, None, None], (L, K, J)).copy() if fixed else None
    )

    def unpack(theta: np.ndarray) -> Params:
        x = np.exp(theta)
        mu = x[:n_mu].reshape(B, K)
        alpha = x[n_mu:n_mu + n_alpha].reshape(L, K, J)
        beta = beta_fixed if fixed else x[n_mu + n_alpha:].reshape(L, K, J)
        return Params(mu, alpha, beta, edges)

    # Start near the model it must beat: half the Poisson rate as baseline and
    # a total branching of 0.3 spread evenly across scales and exciting types.
    rate = np.bincount(ev.m, minlength=K) / T
    beta0 = beta_fixed if fixed else np.full((L, K, J), 5.0)
    parts = [np.tile(rate * 0.5, B), (0.3 * beta0 / (L * J / K)).ravel()]
    if fixed:
        fun, jac = fixed_objective(ev, T, edges, beta_fixed), True
    else:
        parts.append(beta0.ravel())
        fun, jac = (lambda theta: -loglik(unpack(theta), ev, 0.0, T)), None
    res = minimize(fun, np.log(np.concatenate(parts)), jac=jac, method="L-BFGS-B",
                   options={"maxiter": 2000})
    if not res.success:
        raise RuntimeError(f"Hawkes MLE did not converge: {res.message}")
    return unpack(res.x)


def extrapolate(p: Params, T: float) -> Params:
    """Freeze the baseline at its train-mean beyond T.

    A piecewise baseline fit on [0, T] says nothing about what mu does after T.
    Using the last block would reward a lucky final window; the mean is what a
    forecaster without a rate model would assume. Single-block models pass
    through unchanged in effect.
    """
    edges = np.append(p.edges[:-1], [T, np.inf])
    mu = np.vstack([p.mu, p.mu.mean(0)])
    return Params(mu, p.alpha, p.beta, edges)
