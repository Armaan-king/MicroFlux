"""Multivariate Hawkes process: piecewise-constant baseline, sum-of-exponential
kernels, and optional state-dependent excitation.

    lambda_i(t) = mu_i(t) + sum_l sum_j sum_{t_k < t, e_k = j} alpha_lij exp(-beta_lij (t - t_k))

Every event has an *excited* type m in 0..K-1 (what it is: BUY or SELL) and an
*exciting* type e in 0..J-1 (how it acts on the future). Without state, e = m
and J = K. With a discrete state s in 0..S-1 observed when the event happens,
e = m + K s and J = K S -- the state-dependent Hawkes process of
Morariu-Patrichi & Pakkanen, where the kernel from event k depends on the
state at k. The recursion, the likelihood and its concavity are unchanged;
only the width of alpha grows.

mu_i(t) is constant on each of B time blocks; each kernel pair has L
exponential scales. Both are controls for a way the plain model can lie:
B > 1 stops a slow kernel absorbing non-stationarity, L > 1 stops one
timescale standing in for several. Poisson is B = 1, L = 1, alpha = 0.

Time is seconds from the window start.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.stats import kstest


@dataclass(frozen=True)
class Events:
    t: np.ndarray  # (n,) seconds, sorted
    m: np.ndarray  # (n,) excited type 0..K-1
    e: np.ndarray  # (n,) exciting type 0..J-1
    K: int
    J: int

    def __len__(self) -> int:
        return len(self.t)

    def window(self, a: float, b: float) -> np.ndarray:
        return (self.t >= a) & (self.t < b)


def events(t: np.ndarray, m: np.ndarray, K: int, state: np.ndarray | None = None, S: int = 1) -> Events:
    """Bundle a sequence. `state` (0..S-1 per event) widens the exciting type."""
    if state is None:
        return Events(t, m, m, K, K)
    return Events(t, m, m + K * state, K, K * S)


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
def _excitation(t: np.ndarray, e: np.ndarray, beta: np.ndarray) -> np.ndarray:
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


def intensity(p: Params, ev: Events, R: np.ndarray | None = None) -> np.ndarray:
    """lambda_i(t_k) for every event k and excited type i. Shape (n, K)."""
    if R is None:
        R = _excitation(ev.t, ev.e, p.beta)
    return p.mu[_block_of(p.edges, ev.t)] + np.einsum("lij,nlij->ni", p.alpha, R)


def compensator(p: Params, ev: Events, a: float, b: float) -> np.ndarray:
    """Integral of lambda_i over [a, b], per excited type. Shape (K,).

    Events before `a` contribute their decaying tail over the window; events
    inside it contribute from their own time. Both are the one formula
    (alpha/beta)(exp(-beta max(a - t_k, 0)) - exp(-beta (b - t_k))).
    """
    out = (_baseline_cum(p, np.array([b])) - _baseline_cum(p, np.array([a])))[0]
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
    the parameters on it, which `fit_hawkes` never does.
    """
    lam = intensity(p, ev, R)
    own = lam[np.arange(len(ev)), ev.m][ev.window(a, b)]
    return float(np.log(own).sum() - compensator(p, ev, a, b).sum())


def fit_poisson(ev: Events, T: float, block_s: float | None = None) -> Params:
    """MLE is the event rate per type per block. Closed form."""
    edges = block_edges(T, block_s)
    counts = np.zeros((len(edges) - 1, ev.K))
    np.add.at(counts, (_block_of(edges, ev.t), ev.m), 1.0)
    widths = np.minimum(edges[1:], T) - edges[:-1]
    shape = (1, ev.K, ev.J)
    return Params(counts / widths[:, None], np.zeros(shape), np.ones(shape), edges)


def _fixed_objective(ev: Events, T: float, edges: np.ndarray, beta: np.ndarray):
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
    R = _excitation(ev.t, ev.e, beta)
    blk = _block_of(edges, ev.t)
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
    """Maximum likelihood on [0, T], L-BFGS-B over log-parameters.

    Args:
        block_s: width of the baseline blocks in seconds; None for one block.
        scales: fixed decay rates, one kernel scale each, shared across pairs.
            With beta fixed the likelihood is concave in (mu, alpha), the
            recursion runs once, and the gradient is analytic. None fits one
            free beta per pair -- the classic single-exponential model, with
            numerical gradients.
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
        fun, jac = _fixed_objective(ev, T, edges, beta_fixed), True
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


def rescaled_residuals(p: Params, ev: Events, a: float, b: float) -> list[np.ndarray]:
    """Time-rescaling residuals per excited type (Ogata 1981).

    Lambda_i(t) evaluated at the type-i events; consecutive differences are
    iid Exp(1) if the model is correct. This is the goodness-of-fit test, as
    opposed to likelihood, which only ranks models against each other.

    The kernel part is sum_l sum_j (alpha/beta) (N_j(t_k) - R_lij[k]), since
    sum over earlier events of (1 - exp(-beta dt)) is the count minus the
    decayed sum the recursion already holds.
    """
    R = _excitation(ev.t, ev.e, p.beta)
    onehot = np.eye(ev.J)[ev.e]
    counts = np.cumsum(onehot, axis=0) - onehot  # exciting-type-j events strictly before k
    br = p.branching_by_scale
    Lam = _baseline_cum(p, ev.t) + counts @ br.sum(0).T - np.einsum("lij,nlij->ni", br, R)
    inside = ev.window(a, b)
    return [np.diff(Lam[inside & (ev.m == i), i]) for i in range(ev.K)]


def ks_exp1(residuals: np.ndarray) -> float:
    """KS distance from Exp(1). Zero is a perfect fit; compare between models,
    since with 10^5 events every model rejects at p < 0.05."""
    return float(kstest(residuals, "expon").statistic)


def simulate(p: Params, T: float, seed: int = 0, state_prob: np.ndarray | None = None) -> Events:
    """Ogata thinning. Exists so the likelihood can be tested against a
    process with known parameters -- the only proof that the fit is right.

    With `state_prob` (S,), each event draws an iid state and excites as
    (type, state): the exogenous-state case of the state-dependent model.
    """
    rng = np.random.default_rng(seed)
    K = p.K
    S = 1 if state_prob is None else len(state_prob)
    R = np.zeros_like(p.alpha)
    mu_max = p.mu.max(0)
    t, times, types, states = 0.0, [], [], []
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
            s = 0 if state_prob is None else rng.choice(S, p=state_prob)
            times.append(t)
            types.append(j)
            states.append(s)
            R[:, :, j + K * s] += 1.0
    m = np.array(types, dtype=np.int64)
    return events(np.array(times), m, K, None if state_prob is None else np.array(states), S)
