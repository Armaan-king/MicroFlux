"""The event sequence every model consumes.

Each event has an *excited* type m (what it is: BUY or SELL) and an *exciting*
type e (how it acts on the future). Without state, e = m. With a discrete
state s observed when the event happens, e = m + K s -- the state-dependent
Hawkes process of Morariu-Patrichi & Pakkanen, where the kernel from an event
depends on the state it landed in. Bundling both here means no model can
forget to pass the state.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Events:
    t: np.ndarray  # (n,) seconds from window start, sorted
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
