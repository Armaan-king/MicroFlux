"""The event sequence every model consumes, plus the state it happened in and
the mark it carried.

Each event has an *excited* type m (what it is: BUY or SELL), a state s
observed when it happened, a mark class c it carried (how deep it swept the
book), and an *exciting* type e (how it acts on the future). The exciting
type widens with whatever the kernel is allowed to depend on:

    e = m + K * (s + S * c)        state and mark
    e = m + K * c                  mark only  (kernel_state=False)
    e = m                          neither

State is a step function of time, not just a label per event: the baseline
mu(t, s) has to be integrated between events too, so the sequence carries
`step_t` / `step_s` and derives per-event states from it. A mark is a
property of the event alone, so it is per-event only and never reaches the
baseline.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Events:
    t: np.ndarray       # (n,) seconds from window start, sorted
    m: np.ndarray       # (n,) excited type 0..K-1
    s: np.ndarray       # (n,) state 0..S-1
    c: np.ndarray       # (n,) mark class 0..C-1
    e: np.ndarray       # (n,) exciting type 0..J-1
    K: int
    S: int
    C: int
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
        return Events(self.t[k], self.m[k], self.s[k], self.c[k], self.e[k],
                      self.K, self.S, self.C, self.J, self.step_t, self.step_s)


def events(
    t: np.ndarray, m: np.ndarray, K: int,
    step_t: np.ndarray | None = None, step_s: np.ndarray | None = None, S: int = 1,
    kernel_state: bool = True,
    mark: np.ndarray | None = None, C: int = 1,
) -> Events:
    """Bundle a sequence.

    Without a state function everything is state 0; without `mark` every
    event is mark class 0. `kernel_state=False` keeps the state for the
    baseline but stops it widening the exciting type. A state function that
    starts after t = 0 is extended backwards with its first value.
    """
    if step_t is None:
        step_t, step_s, S = np.array([0.0]), np.array([0]), 1
    elif step_t[0] > 0.0:
        step_t, step_s = np.insert(step_t, 0, 0.0), np.insert(step_s, 0, step_s[0])
    step_t, step_s = np.asarray(step_t, float), np.asarray(step_s, np.int64)
    s = step_s[np.searchsorted(step_t, t, side="right") - 1]
    c = np.zeros(len(t), np.int64) if mark is None else np.asarray(mark, np.int64)
    C = 1 if mark is None else C
    Sk = S if kernel_state else 1
    e = m + K * ((s if kernel_state else 0) + Sk * c)
    return Events(t, m, s, c, e, K, S, C, K * Sk * C, step_t, step_s)
