"""Goodness of fit by time rescaling (Ogata 1981, 1988).

Likelihood ranks models against each other; this says whether any of them
actually fits. Transform each event time through the model's own integrated
intensity and the gaps are iid Exp(1) if the model is right.

Conditional checks are only valid against information available at the
*start* of each residual interval. The mark of the arriving event, or the
state when it arrives, is measured at the end of the interval and can select
particular waiting times even under a correct model. `interval_covariates`
therefore reports, for each type-i residual, features of the type-i event
that opened the interval -- all F_{t_start}-measurable.

Millisecond timestamps put a floor under every continuous-time check. Two
remedies, used together: `jitter` spreads each event uniformly inside its
tick (the discrete time-rescaling correction of Haslinger, Pipa & Brown
2010), and `diagnose.py` calibrates every statistic against simulations
passed through the same observation process.
"""

import numpy as np
from scipy.stats import kstest

from microflux.events import Events
from microflux.hawkes import Params, baseline_cum, excitation

TICK = 1e-3  # seconds; Binance stamps trades in milliseconds


def rescaled_residuals(p: Params, ev: Events, a: float, b: float) -> list[np.ndarray]:
    """Per excited type: consecutive differences of Lambda_i at the type-i events in [a, b).

    The kernel part is sum_l sum_j (alpha/beta) (N_j(t_k) - R_lij[k]), since
    the sum over earlier events of (1 - exp(-beta dt)) is the count minus the
    decayed sum the recursion already holds.
    """
    Lam = compensator_at_events(p, ev)
    inside = ev.window(a, b)
    return [np.diff(Lam[inside & (ev.m == i), i]) for i in range(ev.K)]


def compensator_at_events(p: Params, ev: Events) -> np.ndarray:
    """(n, K): Lambda_i(t_k) for every event k."""
    R = excitation(ev.t, ev.e, p.beta)
    onehot = np.eye(ev.J)[ev.e]
    counts = np.cumsum(onehot, axis=0) - onehot  # exciting-type-j events strictly before k
    br = p.branching_by_scale
    return baseline_cum(p, ev, ev.t) + counts @ br.sum(0).T - np.einsum("lij,nlij->ni", br, R)


def ks_exp1(residuals: np.ndarray) -> float:
    """KS distance from Exp(1). Zero is a perfect fit; compare between models
    and against the simulated null, since with 10^5 events every model rejects
    at p < 0.05 and millisecond ticks alone put a floor under it."""
    return float(kstest(residuals, "expon").statistic)


def jitter(ev: Events, seed: int = 0, tick: float = TICK) -> Events:
    """Each event moved uniformly within its tick, order preserved.

    Timestamps are floor(t / tick) * tick; the true time is somewhere in the
    tick after. A uniform draw restores a continuous distribution in
    expectation. Events in the same tick keep their capture order by drawing
    sorted offsets within the tick.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, tick, len(ev))
    starts = np.flatnonzero(np.r_[True, np.diff(ev.t) != 0.0])
    for s, e in zip(starts, np.r_[starts[1:], len(ev)]):
        if e - s > 1:  # a run of equal timestamps keeps its capture order
            u[s:e] = np.sort(u[s:e])
    return Events(ev.t + u, ev.m, ev.s, ev.c, ev.e, ev.K, ev.S, ev.C, ev.J, ev.step_t, ev.step_s)


def interval_covariates(ev: Events, a: float, b: float, window: float = 0.1) -> list[dict[str, np.ndarray]]:
    """Per type i, one row per residual in `rescaled_residuals(p, ev, a, b)`:
    features of the type-i event that *opened* the interval.

        mark        its mark class
        state       the state when it happened
        burst       events of any type in the `window` seconds before it
        opp_gap     seconds since the last opposite-side event before it (inf if none)
        hour        its time, in hours from the window start

    Everything here is known at the moment the interval starts.
    """
    t = ev.t
    left = np.searchsorted(t, t - window, side="left")
    burst = np.arange(len(t)) - left
    opp_gap = np.full(len(t), np.inf)
    last = {i: -np.inf for i in range(ev.K)}
    for k in range(len(t)):
        other = max(v for i, v in last.items() if i != ev.m[k]) if ev.K > 1 else -np.inf
        opp_gap[k] = t[k] - other
        last[ev.m[k]] = t[k]
    out = []
    inside = ev.window(a, b)
    for i in range(ev.K):
        k = np.flatnonzero(inside & (ev.m == i))[:-1]  # openers: all but the last type-i event
        out.append({
            "mark": ev.c[k], "state": ev.s[k], "burst": burst[k],
            "opp_gap": opp_gap[k], "hour": (t[k] - a) / 3600.0,
        })
    return out


def conditional_means(residuals: np.ndarray, cov: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(n_bins, 3): mean residual, standard error, count per bin of `cov`.

    Under a correct model every mean is 1 up to its standard error.
    """
    b = np.digitize(cov, edges)
    rows = []
    for k in range(len(edges) + 1):
        r = residuals[b == k]
        rows.append((r.mean() if len(r) else np.nan, r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.nan, len(r)))
    return np.array(rows)


def autocorrelation(residuals: np.ndarray, lags: tuple[int, ...] = (1, 2, 5, 10, 50, 100)) -> np.ndarray:
    """Sample autocorrelation at each lag; iid residuals give ~N(0, 1/n)."""
    x = residuals - residuals.mean()
    v = (x * x).sum()
    return np.array([(x[:-h] * x[h:]).sum() / v for h in lags])


def quantiles(residuals: np.ndarray, probs: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> np.ndarray:
    """(n_probs, 2): empirical vs Exp(1) quantiles. Ratio < 1 at the low end
    means the model does not expect gaps that short."""
    emp = np.quantile(residuals, probs)
    theo = -np.log1p(-np.asarray(probs))
    return np.column_stack([emp, theo])
