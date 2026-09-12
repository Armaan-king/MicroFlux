"""Simulate a Hawkes process with known parameters, by Ogata thinning.

Exists so the likelihood can be tested against a truth -- the only proof
that a fit is right. Not a research tool.
"""

import numpy as np

from microflux.events import Events, events
from microflux.hawkes import Params, block_of


def random_states(T: float, S: int, mean_dwell: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A state that jumps to a uniformly random value after Exp(mean_dwell) seconds.

    Persistent, like a book imbalance, which is what makes a state-dependent
    rate and a state-dependent kernel hard to tell apart.
    """
    rng = np.random.default_rng(seed)
    n = int(3 * T / mean_dwell) + 10
    step_t = np.concatenate([[0.0], np.cumsum(rng.exponential(mean_dwell, n))])
    step_t = step_t[step_t < T]
    return step_t, rng.integers(0, S, len(step_t))


def simulate(
    p: Params, T: float, seed: int = 0,
    step_t: np.ndarray | None = None, step_s: np.ndarray | None = None,
) -> Events:
    """Events on [0, T] under `p`, with the baseline and the exciting type both
    following the given state function (state 0 throughout if none)."""
    rng = np.random.default_rng(seed)
    K, S = p.K, p.S
    stub = events(np.empty(0), np.empty(0, np.int64), K, step_t, step_s, S)
    R = np.zeros_like(p.alpha)
    mu_max = p.mu.max((0, 1))
    t, times, types = 0.0, [], []
    while True:
        bound = (mu_max + (p.alpha * R).sum((0, 2))).sum()  # intensity only decays until the next event
        dt = rng.exponential(1.0 / bound)
        R = R * np.exp(-p.beta * dt)
        t += dt
        if t > T:
            break
        here = np.array([t])
        s = int(stub.state_at(here)[0])
        lam = p.mu[block_of(p.edges, here)[0], s] + (p.alpha * R).sum((0, 2))
        if rng.uniform() * bound < lam.sum():
            j = rng.choice(K, p=lam / lam.sum())
            times.append(t)
            types.append(j)
            R[:, :, j + K * s] += 1.0
    return events(np.array(times), np.array(types, dtype=np.int64), K, stub.step_t, stub.step_s, S)
