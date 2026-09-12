"""Goodness of fit by time rescaling (Ogata 1981).

Likelihood ranks models against each other; this says whether any of them
actually fits. Transform each event time through the model's own integrated
intensity and the gaps are iid Exp(1) if the model is right.
"""

import numpy as np
from scipy.stats import kstest

from microflux.events import Events
from microflux.hawkes import Params, baseline_cum, excitation


def rescaled_residuals(p: Params, ev: Events, a: float, b: float) -> list[np.ndarray]:
    """Per excited type: consecutive differences of Lambda_i at the type-i events in [a, b).

    The kernel part is sum_l sum_j (alpha/beta) (N_j(t_k) - R_lij[k]), since
    the sum over earlier events of (1 - exp(-beta dt)) is the count minus the
    decayed sum the recursion already holds.
    """
    R = excitation(ev.t, ev.e, p.beta)
    onehot = np.eye(ev.J)[ev.e]
    counts = np.cumsum(onehot, axis=0) - onehot  # exciting-type-j events strictly before k
    br = p.branching_by_scale
    Lam = baseline_cum(p, ev, ev.t) + counts @ br.sum(0).T - np.einsum("lij,nlij->ni", br, R)
    inside = ev.window(a, b)
    return [np.diff(Lam[inside & (ev.m == i), i]) for i in range(ev.K)]


def ks_exp1(residuals: np.ndarray) -> float:
    """KS distance from Exp(1). Zero is a perfect fit; compare between models,
    since with 10^5 events every model rejects at p < 0.05."""
    return float(kstest(residuals, "expon").statistic)
