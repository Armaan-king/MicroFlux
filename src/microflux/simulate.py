"""Simulate a Hawkes process with known parameters, by Ogata thinning.

Exists so the likelihood can be tested against a truth -- the only proof
that a fit is right. Not a research tool.
"""

import numpy as np

from microflux.events import Events, events
from microflux.hawkes import Params, block_of


def simulate(p: Params, T: float, seed: int = 0, state_prob: np.ndarray | None = None) -> Events:
    """Events on [0, T]. With `state_prob` (S,), each event draws an iid state
    and excites as (type, state): the exogenous-state case of the
    state-dependent model."""
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
        lam = p.mu[block_of(p.edges, np.array([t]))[0]] + (p.alpha * R).sum((0, 2))
        if rng.uniform() * bound < lam.sum():
            j = rng.choice(K, p=lam / lam.sum())
            s = 0 if state_prob is None else rng.choice(S, p=state_prob)
            times.append(t)
            types.append(j)
            states.append(s)
            R[:, :, j + K * s] += 1.0
    m = np.array(types, dtype=np.int64)
    return events(np.array(times), m, K, None if state_prob is None else np.array(states), S)
