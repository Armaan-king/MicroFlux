"""Multivariate Hawkes process: piecewise-constant baseline, sum-of-exponential kernels.

    lambda_i(t) = mu_i(t) + sum_l sum_j sum_{t_k < t, m_k = j} alpha_lij exp(-beta_lij (t - t_k))

with mu_i(t) constant on each of B time blocks and L exponential scales per
kernel pair. Two axes, each of which is a control for a way the plain model
can lie:

- B > 1 lets the exogenous rate drift, so a slow kernel cannot absorb
  non-stationarity and report it as excitation.
- L > 1 lets one pair excite at several timescales, which a single exponential
  represents as whichever one the optimiser finds first.

Poisson is B = 1, L = 1, alpha = 0. The single-exponential Hawkes of Bacry,
Mastromatteo & Muzy (2015) s.2 is B = 1, L = 1. Every model shares one
likelihood, so comparisons cannot be confounded by two implementations.

The exponential kernel is what keeps all of this O(n): one recursion per
(scale, pair). Time is seconds from the window start; types are 0..K-1.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.stats import kstest


@dataclass(frozen=True)
class Params:
    mu: np.ndarray     # (B, K)     exogenous rate per time block, per type
    alpha: np.ndarray  # (L, K, K)  alpha[l, i, j]: type-j excites type-i at scale l
    beta: np.ndarray   # (L, K, K)  decay rate of that excitation
    edges: np.ndarray  # (B + 1,)   block boundaries; edges[0] = 0, edges[-1] = inf

    @property
    def K(self) -> int:
        return self.mu.shape[1]

    @property
    def branching(self) -> np.ndarray:
        """(K, K): expected type-i events one type-j event triggers, all scales."""
        return (self.alpha / self.beta).sum(0)

    @property
    def branching_by_scale(self) -> np.ndarray:
        """(L, K, K): the same, split by timescale."""
        return self.alpha / self.beta

    @property
    def spectral_radius(self) -> float:
        """Stationarity requires this < 1. Near 1 means near-critical: most
        events are triggered by other events rather than arriving exogenously."""
        return float(np.abs(np.linalg.eigvals(self.branching)).max())

    @property
    def half_lives(self) -> np.ndarray:
        return np.log(2.0) / self.beta


def block_edges(T: float, block_s: float | None) -> np.ndarray:
    """Boundaries of the baseline blocks over [0, T]; the last block is open."""
    if block_s is None:
        return np.array([0.0, np.inf])
    return np.append(np.arange(0.0, T, block_s), np.inf)


def _block_of(edges: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 2)


def _baseline_cum(p: Params, t: np.ndarray) -> np.ndarray:
    """(n, K): integral of mu_i from 0 to each t."""
    lo, hi = p.edges[:-1], p.edges[1:]
    width = np.clip(t[:, None] - lo[None, :], 0.0, (hi - lo)[None, :])
    return width @ p.mu


@njit(cache=True)
def _excitation(t: np.ndarray, m: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """R[k, l, i, j] = sum over earlier type-j events of exp(-beta_lij (t_k - t_l)).

    The recursion R[k] = exp(-beta dt) (R[k-1] + 1[m_{k-1} = j]) is what makes
    the exponential kernel O(n). An event at the same instant as its
    predecessor decays by exp(0) = 1 and is counted -- sequence order is capture
    order, which is the ground-truth ordering (see load.py).
    """
    n = len(t)
    L, K, _ = beta.shape
    R = np.zeros((n, L, K, K))
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        prev = m[k - 1]
        for l in range(L):
            for i in range(K):
                for j in range(K):
                    R[k, l, i, j] = np.exp(-beta[l, i, j] * dt) * (
                        R[k - 1, l, i, j] + (1.0 if prev == j else 0.0)
                    )
    return R


def intensity(p: Params, t: np.ndarray, m: np.ndarray, R: np.ndarray | None = None) -> np.ndarray:
    """lambda_i(t_k) for every event k and type i. Shape (n, K)."""
    if R is None:
        R = _excitation(t, m, p.beta)
    return p.mu[_block_of(p.edges, t)] + np.einsum("lij,nlij->ni", p.alpha, R)


def compensator(p: Params, t: np.ndarray, m: np.ndarray, a: float, b: float) -> np.ndarray:
    """Integral of lambda_i over [a, b], per type. Shape (K,).

    Events before `a` contribute their decaying tail over the window; events
    inside it contribute from their own time. Both are the one formula
    (alpha/beta)(exp(-beta max(a - t_k, 0)) - exp(-beta (b - t_k))).
    """
    out = (_baseline_cum(p, np.array([b])) - _baseline_cum(p, np.array([a])))[0]
    inside = t < b
    tl, ml = t[inside], m[inside]
    for j in range(p.K):
        tj = tl[ml == j]
        lead = np.maximum(a - tj, 0.0)[:, None]
        tail = (b - tj)[:, None]
        for l in range(p.beta.shape[0]):
            bl = p.beta[l, :, j][None, :]
            out += (p.alpha[l, :, j] / p.beta[l, :, j]) * (
                np.exp(-bl * lead) - np.exp(-bl * tail)
            ).sum(axis=0)
    return out


def loglik(p: Params, t: np.ndarray, m: np.ndarray, a: float, b: float, R: np.ndarray | None = None) -> float:
    """Log-likelihood of the events in [a, b), given the full history.

    Using history from before `a` is not leakage: a point process conditions on
    what has happened, and what happened is observed. Leakage would be fitting
    the parameters on it, which `fit_hawkes` never does.
    """
    lam = intensity(p, t, m, R)
    k = (t >= a) & (t < b)
    own = lam[np.arange(len(t)), m][k]
    return float(np.log(own).sum() - compensator(p, t, m, a, b).sum())


def fit_poisson(t: np.ndarray, m: np.ndarray, T: float, K: int, block_s: float | None = None) -> Params:
    """MLE is the event rate per type per block. Closed form."""
    edges = block_edges(T, block_s)
    counts = np.zeros((len(edges) - 1, K))
    np.add.at(counts, (_block_of(edges, t), m), 1.0)
    widths = np.minimum(edges[1:], T) - edges[:-1]
    return Params(counts / widths[:, None], np.zeros((1, K, K)), np.ones((1, K, K)), edges)


def _fixed_objective(t: np.ndarray, m: np.ndarray, T: float, edges: np.ndarray, beta: np.ndarray):
    """Negative log-likelihood and its gradient in log(mu, alpha), beta fixed.

    With beta fixed the recursion R never changes, and the compensator kernel
    term is (alpha / beta) * S with S precomputed, so an evaluation is one
    einsum. The gradient is closed-form:

        dLL/dmu_bi     = sum_{k: m_k = i, block b} 1 / lambda_i(t_k)  -  |block b|
        dLL/dalpha_lij = sum_{k: m_k = i} R_lij[k] / lambda_i(t_k)   -  S_lij / beta_lij

    which turns 60+ likelihood evaluations per L-BFGS step into one.
    """
    L, K, _ = beta.shape
    B = len(edges) - 1
    R = _excitation(t, m, beta)
    blk = _block_of(edges, t)
    widths = np.minimum(edges[1:], T) - edges[:-1]
    idx = np.arange(len(t))
    S = np.zeros((L, K, K))
    for j in range(K):
        tail = (T - t[m == j])[:, None]
        for l in range(L):
            S[l, :, j] = (1.0 - np.exp(-beta[l, :, j][None, :] * tail)).sum(0)
    by_type = [m == i for i in range(K)]
    inside = t < T  # same half-open [0, T) window as `loglik`

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.exp(theta)
        mu, alpha = x[:B * K].reshape(B, K), x[B * K:].reshape(L, K, K)
        lam = mu[blk] + np.einsum("lij,nlij->ni", alpha, R)
        own = lam[idx, m]
        ll = np.log(own[inside]).sum() - (mu * widths[:, None]).sum() - (alpha / beta * S).sum()
        inv = np.where(inside, 1.0 / own, 0.0)
        g_mu = np.repeat(-widths[:, None], K, axis=1)
        np.add.at(g_mu, (blk, m), inv)
        g_alpha = -S / beta
        for i, ki in enumerate(by_type):
            g_alpha[:, i, :] += np.einsum("nlj,n->lj", R[ki, :, i, :], inv[ki])
        return -ll, -np.concatenate([g_mu.ravel(), g_alpha.ravel()]) * x

    return objective


def fit_hawkes(
    t: np.ndarray, m: np.ndarray, T: float, K: int,
    block_s: float | None = None, scales: np.ndarray | None = None,
) -> Params:
    """Maximum likelihood on [0, T], L-BFGS-B over log-parameters.

    Args:
        block_s: width of the baseline blocks in seconds; None for one block.
        scales: fixed decay rates, one kernel scale each, shared across pairs.
            With beta fixed the likelihood is concave in (mu, alpha), the
            recursion runs once, and the gradient is analytic. None fits one
            free beta per pair -- the classic single-exponential model, with
            numerical gradients.
    """
    edges = block_edges(T, block_s)
    B = len(edges) - 1
    fixed = scales is not None
    L = len(scales) if fixed else 1
    n_mu, n_alpha = B * K, L * K * K
    beta_fixed = (
        np.broadcast_to(np.asarray(scales, float)[:, None, None], (L, K, K)).copy() if fixed else None
    )

    def unpack(theta: np.ndarray) -> Params:
        x = np.exp(theta)
        mu = x[:n_mu].reshape(B, K)
        alpha = x[n_mu:n_mu + n_alpha].reshape(L, K, K)
        beta = beta_fixed if fixed else x[n_mu + n_alpha:].reshape(L, K, K)
        return Params(mu, alpha, beta, edges)

    # Start near the model it must beat: half the Poisson rate as baseline and
    # a total branching of 0.3 spread evenly across scales.
    rate = np.bincount(m, minlength=K) / T
    beta0 = beta_fixed if fixed else np.full((L, K, K), 5.0)
    parts = [np.tile(rate * 0.5, B), (0.3 * beta0 / L).ravel()]
    if fixed:
        fun, jac = _fixed_objective(t, m, T, edges, beta_fixed), True
    else:
        parts.append(beta0.ravel())
        fun, jac = (lambda theta: -loglik(unpack(theta), t, m, 0.0, T)), None
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


def rescaled_residuals(p: Params, t: np.ndarray, m: np.ndarray, a: float, b: float) -> list[np.ndarray]:
    """Time-rescaling residuals per type (Ogata 1981).

    Lambda_i(t) evaluated at the type-i events; consecutive differences are
    iid Exp(1) if the model is correct. This is the goodness-of-fit test, as
    opposed to likelihood, which only ranks models against each other.

    The kernel part is sum_l sum_j (alpha/beta) (N_j(t_k) - R_lij[k]), since
    sum over earlier events of (1 - exp(-beta dt)) is the count minus the
    decayed sum the recursion already holds.
    """
    K = p.K
    R = _excitation(t, m, p.beta)
    onehot = np.eye(K)[m]
    counts = np.cumsum(onehot, axis=0) - onehot  # type-j events strictly before k
    br = p.branching_by_scale
    Lam = _baseline_cum(p, t) + counts @ br.sum(0).T - np.einsum("lij,nlij->ni", br, R)
    inside = (t >= a) & (t < b)
    return [np.diff(Lam[inside & (m == i), i]) for i in range(K)]


def ks_exp1(residuals: np.ndarray) -> float:
    """KS distance from Exp(1). Zero is a perfect fit; compare between models,
    since with 10^5 events every model rejects at p < 0.05."""
    return float(kstest(residuals, "expon").statistic)


def simulate(p: Params, T: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Ogata thinning. Exists so the likelihood can be tested against a
    process with known parameters -- the only proof that the fit is right."""
    rng = np.random.default_rng(seed)
    K = p.K
    R = np.zeros_like(p.alpha)
    mu_max = p.mu.max(0)
    t, times, types = 0.0, [], []
    while True:
        bound = (mu_max + (p.alpha * R).sum((0, 2))).sum()  # intensity only decays until the next event
        dt = rng.exponential(1.0 / bound)
        R = R * np.exp(-p.beta * dt)
        t += dt
        if t > T:
            break
        lam = p.mu[_block_of(p.edges, np.array([t]))[0]] + (p.alpha * R).sum((0, 2))
        if rng.uniform() * bound < lam.sum():
            j = rng.choice(K, p=lam / lam.sum())
            times.append(t)
            types.append(j)
            R[:, :, j] += 1.0
    return np.array(times), np.array(types, dtype=np.int64)
