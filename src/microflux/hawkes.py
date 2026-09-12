"""Multivariate Hawkes process with exponential kernels, and its Poisson limit.

    lambda_i(t) = mu_i + sum_j sum_{t_l < t, m_l = j} alpha_ij exp(-beta_ij (t - t_l))

The exponential kernel is the baseline every reference starts from (Hawkes
1971; Bacry, Mastromatteo & Muzy 2015 s.2) because it makes the likelihood
O(n) through one recursion per kernel pair, where any other kernel is O(n^2).
Power-law kernels fit financial data better at long lags and are the obvious
next step if the time-rescaling test rejects this one.

A Poisson process is the alpha = 0 special case, so both models share one
likelihood and one evaluation path -- a comparison between them cannot be
confounded by two implementations disagreeing.

Time is in seconds from the start of the window. Types are integers 0..K-1.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.stats import kstest


@dataclass(frozen=True)
class Params:
    mu: np.ndarray     # (K,)
    alpha: np.ndarray  # (K, K)  alpha[i, j]: how much a type-j event excites type i
    beta: np.ndarray   # (K, K)  decay rate of that excitation

    @property
    def branching(self) -> np.ndarray:
        """alpha / beta: expected number of type-i events one type-j event triggers."""
        return self.alpha / self.beta

    @property
    def spectral_radius(self) -> float:
        """Stationarity requires this < 1. Near 1 means near-critical: most
        events are triggered by other events rather than arriving exogenously."""
        return float(np.abs(np.linalg.eigvals(self.branching)).max())


@njit(cache=True)
def _excitation(t: np.ndarray, m: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """R[k, i, j] = sum over earlier type-j events l of exp(-beta_ij (t_k - t_l)).

    The recursion R[k] = exp(-beta dt) (R[k-1] + 1[m_{k-1} = j]) is what makes
    the exponential kernel O(n). An event at the same instant as its
    predecessor decays by exp(0) = 1 and is counted -- sequence order is capture
    order, which is the ground-truth ordering (see load.py).
    """
    n, K = len(t), beta.shape[0]
    R = np.zeros((n, K, K))
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        prev = m[k - 1]
        for i in range(K):
            for j in range(K):
                R[k, i, j] = np.exp(-beta[i, j] * dt) * (R[k - 1, i, j] + (1.0 if prev == j else 0.0))
    return R


def intensity(p: Params, t: np.ndarray, m: np.ndarray, R: np.ndarray | None = None) -> np.ndarray:
    """lambda_i(t_k) for every event k and type i. Shape (n, K)."""
    if R is None:
        R = _excitation(t, m, p.beta)
    return p.mu[None, :] + np.einsum("ij,nij->ni", p.alpha, R)


def compensator(p: Params, t: np.ndarray, m: np.ndarray, a: float, b: float) -> np.ndarray:
    """Integral of lambda_i over [a, b], per type i. Shape (K,).

    Events before `a` still contribute their decaying tail over the window;
    events inside it contribute from their own time. Both are the one formula
    (alpha/beta)(exp(-beta max(a - t_l, 0)) - exp(-beta (b - t_l))).
    """
    K = len(p.mu)
    out = p.mu * (b - a)
    inside = t < b
    tl, ml = t[inside], m[inside]
    for j in range(K):
        tj = tl[ml == j]
        lead = np.maximum(a - tj, 0.0)[:, None]
        tail = (b - tj)[:, None]
        out += (p.alpha[:, j] / p.beta[:, j]) * (
            np.exp(-p.beta[:, j] * lead) - np.exp(-p.beta[:, j] * tail)
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


def fit_poisson(t: np.ndarray, m: np.ndarray, T: float, K: int) -> Params:
    """MLE is the event rate per type. Closed form."""
    mu = np.bincount(m, minlength=K) / T
    return Params(mu, np.zeros((K, K)), np.ones((K, K)))


def fit_hawkes(t: np.ndarray, m: np.ndarray, T: float, K: int) -> Params:
    """Maximum likelihood on [0, T], L-BFGS-B over log-parameters.

    Log-parameterised so positivity is free and the optimiser walks in the
    natural scale of rates. Initialised at half the Poisson rate with modest
    excitation, so the search starts near the model it must beat.
    """
    rate = np.bincount(m, minlength=K) / T

    def unpack(theta: np.ndarray) -> Params:
        mu, alpha, beta = np.split(np.exp(theta), [K, K + K * K])
        return Params(mu, alpha.reshape(K, K), beta.reshape(K, K))

    def nll(theta: np.ndarray) -> float:
        return -loglik(unpack(theta), t, m, 0.0, T)

    theta0 = np.log(np.concatenate([rate * 0.5, np.full(K * K, 0.5), np.full(K * K, 5.0)]))
    res = minimize(nll, theta0, method="L-BFGS-B", options={"maxiter": 500})
    if not res.success:
        raise RuntimeError(f"Hawkes MLE did not converge: {res.message}")
    return unpack(res.x)


def rescaled_residuals(p: Params, t: np.ndarray, m: np.ndarray, a: float, b: float) -> list[np.ndarray]:
    """Time-rescaling residuals per type (Ogata 1981).

    Lambda_i(t) evaluated at the type-i events; consecutive differences are
    iid Exp(1) if the model is correct. This is the goodness-of-fit test, as
    opposed to likelihood, which only ranks models against each other.

    Lambda_i(t_k) = mu_i t_k + sum_j (alpha_ij / beta_ij) (N_j(t_k) - R_ij[k]),
    since sum_l (1 - exp(-beta (t_k - t_l))) is the count minus the decayed sum.
    """
    K = len(p.mu)
    R = _excitation(t, m, p.beta)
    onehot = np.eye(K)[m]
    counts = np.cumsum(onehot, axis=0) - onehot  # type-j events strictly before k
    Lam = (
        p.mu[None, :] * t[:, None]
        + counts @ p.branching.T
        - np.einsum("ij,nij->ni", p.branching, R)
    )
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
    K = len(p.mu)
    R = np.zeros((K, K))
    t, times, types = 0.0, [], []
    while True:
        bound = (p.mu + (p.alpha * R).sum(1)).sum()  # intensity only decays until the next event
        dt = rng.exponential(1.0 / bound)
        R = R * np.exp(-p.beta * dt)
        t += dt
        if t > T:
            break
        lam = p.mu + (p.alpha * R).sum(1)
        if rng.uniform() * bound < lam.sum():
            j = rng.choice(K, p=lam / lam.sum())
            times.append(t)
            types.append(j)
            R[:, j] += 1.0
    return np.array(times), np.array(types, dtype=np.int64)
