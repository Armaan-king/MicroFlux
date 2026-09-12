"""The event sequence every model consumes, plus the state it happened in.

Each event has an *excited* type m (what it is: BUY or SELL), a state s
observed when it happened, and an *exciting* type e (how it acts on the
future): e = m + K s when the kernel is state-dependent, else e = m. Which
parts of a model use the state is the model's choice; the sequence carries it
either way.

State is a step function of time, not just a label per event: the baseline
mu(t, s) has to be integrated between events too, so the sequence carries
`step_t` / `step_s` -- the times the state changed and what it changed to.
Per-event states are derived from it, so the two can never disagree.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Events:
    t: np.ndarray       # (n,) seconds from window start, sorted
    m: np.ndarray       # (n,) excited type 0..K-1
    s: np.ndarray       # (n,) state 0..S-1
    e: np.ndarray       # (n,) exciting type 0..J-1
    K: int
    S: int
    J: int
    step_t: np.ndarray  # (n_step,) times the state changes; step_t[0] == 0
    step_s: np.ndarray  # (n_step,) state in force from step_t[i]

    def __len__(self) -> int:
        return len(self.t)

    def window(self, a: float, b: float) -> np.ndarray:
        return (self.t >= a) & (self.t < b)

    def state_at(self, t: np.ndarray) -> np.ndarray:
        return self.step_s[np.searchsorted(self.step_t, t, side="right") - 1]

    def before(self, T: float) -> "Events":
        """The events in [0, T); the state function is kept whole."""
        k = self.t < T
        return Events(self.t[k], self.m[k], self.s[k], self.e[k], self.K, self.S, self.J,
                      self.step_t, self.step_s)


def events(
    t: np.ndarray, m: np.ndarray, K: int,
    step_t: np.ndarray | None = None, step_s: np.ndarray | None = None, S: int = 1,
    kernel_state: bool = True,
) -> Events:
    """Bundle a sequence. Without a state function, everything is state 0.

    `kernel_state=False` keeps the state for the baseline but excites by side
    only, so a model can use state in one place and not the other.

    A state function that starts after t = 0 is extended backwards with its
    first value, so every time has a state.
    """
    if step_t is None:
        step_t, step_s, S = np.array([0.0]), np.array([0]), 1
    elif step_t[0] > 0.0:
        step_t, step_s = np.insert(step_t, 0, 0.0), np.insert(step_s, 0, step_s[0])
    step_t, step_s = np.asarray(step_t, float), np.asarray(step_s, np.int64)
    s = step_s[np.searchsorted(step_t, t, side="right") - 1]
    e = m + K * s if kernel_state else m
    return Events(t, m, s, e, K, S, K * S if kernel_state else K, step_t, step_s)
