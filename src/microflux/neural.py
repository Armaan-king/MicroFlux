"""Neural temporal point processes with a shared multiscale intensity head.

Two encoders, one head, one likelihood -- so the comparison between them
isolates what the encoder sees, not how the intensity is parameterised:

    SummaryMLP   causal activity summaries at several timescales + the last
                 event's features             -> h
    Attention    the last N event tokens, self-attention, + the same
                 summaries                    -> h

    head(h, state)  ->  a_il, b_il > 0  for each side i and scale l, and

        lambda_i(tau) = sum_l [ b_il + (a_il - b_il) exp(-beta_l tau) ]

with tau the time since the last event and beta_l the five fixed decay
rates of the classical benchmark. Each component is positive, can rise or
fall, and integrates in closed form while its conditioning is fixed. The
conditioning changes at every event *and at every book update*, so an
inter-event interval is integrated piece by piece across the book rows
inside it, each piece with the state then in force.

The log-likelihood on [a, b) is sum_k log lambda_{m_k}(t_k) over events in
the window minus the compensator over the window, with every piece clipped
to [a, b) -- the same convention as `hawkes.loglik`, on the same events, so
NLL/event is comparable across the whole ladder.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from microflux.events import Events

DTYPE = torch.float32  # the classical likelihood is float64; 1e-7 per event here is far below what is read
EPS = 5e-4  # half a millisecond tick, inside every log
SUMMARY_WINDOWS = (1.0, 10.0, 60.0, 300.0, 1800.0)  # seconds; counts per side ending at the anchor
BIG_WINDOWS = (1.0, 10.0)  # counts of 20+ fill orders per side


# --- features -----------------------------------------------------------------


@dataclass(frozen=True)
class Features:
    """Everything the models need, precomputed once from an Events."""
    t: np.ndarray            # (n,)
    side: np.ndarray         # (n,)
    mark: np.ndarray         # (n,)
    state: np.ndarray        # (n,)   state at the event
    log_gap: np.ndarray      # (n,)   log(t_k - t_{k-1} + eps); 0 for k = 0
    summary: np.ndarray      # (n, F) causal activity summaries at each event
    # compensator pieces over the whole sequence, one row per (interval, book row) piece
    piece_k: np.ndarray      # (P,)  index of the event that CLOSES the interval (anchor is k - 1)
    piece_u0: np.ndarray     # (P,)  absolute start
    piece_u1: np.ndarray     # (P,)  absolute end
    piece_state: np.ndarray  # (P,)  state in force on the piece

    def __len__(self) -> int:
        return len(self.t)


def _trailing_counts(t: np.ndarray, mask: np.ndarray, at: np.ndarray, window: float) -> np.ndarray:
    """Number of masked events in (at - window, at]."""
    tm = t[mask]
    return np.searchsorted(tm, at, side="right") - np.searchsorted(tm, at - window, side="left")


def features(ev: Events, T_end: float) -> Features:
    """Precompute per-event features and the compensator pieces up to T_end."""
    n = len(ev)
    t = ev.t
    log_gap = np.zeros(n)
    log_gap[1:] = np.log(np.diff(t) + EPS)
    cols = []
    for w in SUMMARY_WINDOWS:
        for i in range(ev.K):
            cols.append(_trailing_counts(t, ev.m == i, t, w))
    for w in BIG_WINDOWS:
        for i in range(ev.K):
            cols.append(_trailing_counts(t, (ev.m == i) & (ev.c == ev.C - 1), t, w))
    summary = np.log1p(np.column_stack(cols).astype(float))

    # pieces: interval k is (t_{k-1}, t_k], split at book rows strictly inside;
    # plus the open tail (t_{n-1}, T_end]
    starts = t
    ends = np.append(t[1:], T_end)
    pk, pu0, pu1, ps = [], [], [], []
    step_t, step_s = ev.step_t, ev.step_s
    for k in range(n):
        lo, hi = starts[k], ends[k]
        if hi <= lo:
            continue
        j0 = np.searchsorted(step_t, lo, side="right")   # first book row strictly after lo
        j1 = np.searchsorted(step_t, hi, side="left")    # rows < hi
        cuts = step_t[j0:j1]
        bounds = np.concatenate([[lo], cuts, [hi]])
        st = step_s[np.searchsorted(step_t, bounds[:-1], side="right") - 1]
        for u0, u1, s in zip(bounds[:-1], bounds[1:], st):
            pk.append(k + 1); pu0.append(u0); pu1.append(u1); ps.append(s)
    return Features(t, ev.m, ev.c, ev.s, log_gap, summary,
                    np.array(pk), np.array(pu0), np.array(pu1), np.array(ps, np.int64))


# --- the shared head ----------------------------------------------------------


class Head(nn.Module):
    """h, state -> (a, b) of shape (batch, K, L), both positive."""

    def __init__(self, d_in: int, K: int, S: int, scales: np.ndarray, d_hidden: int = 64):
        super().__init__()
        self.K, self.L = K, len(scales)
        self.register_buffer("beta", torch.as_tensor(np.asarray(scales, float), dtype=DTYPE))
        self.state = nn.Embedding(S, 16)
        self.net = nn.Sequential(nn.Linear(d_in + 16, d_hidden), nn.GELU(), nn.Linear(d_hidden, 2 * K * self.L))

    def forward(self, h: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = nn.functional.softplus(self.net(torch.cat([h, self.state(state)], -1))) + 1e-6
        a, b = out.chunk(2, -1)
        return a.reshape(-1, self.K, self.L), b.reshape(-1, self.K, self.L)

    def intensity(self, a: torch.Tensor, b: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """(batch, K): lambda_i at time-since-anchor tau."""
        decay = torch.exp(-self.beta * tau[:, None, None])
        return (b + (a - b) * decay).sum(-1)

    def integral(self, a: torch.Tensor, b: torch.Tensor, tau0: torch.Tensor, tau1: torch.Tensor) -> torch.Tensor:
        """(batch, K): integral of lambda_i from tau0 to tau1, in closed form."""
        e0, e1 = torch.exp(-self.beta * tau0[:, None, None]), torch.exp(-self.beta * tau1[:, None, None])
        return (b * (tau1 - tau0)[:, None, None] + (a - b) * (e0 - e1) / self.beta).sum(-1)


# --- encoders -----------------------------------------------------------------


class SummaryMLP(nn.Module):
    """The control: summaries and the last event only."""

    def __init__(self, n_summary: int, K: int, C: int, S: int, d: int = 64):
        super().__init__()
        self.K, self.C, self.S = K, C, S
        d_in = n_summary + K + C + S + 1
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.d_out = d

    def forward(self, batch: dict) -> torch.Tensor:
        last = batch["last"]  # (B, 4): side, mark, state, log_gap of the anchor
        x = torch.cat([
            batch["summary"],
            nn.functional.one_hot(last[:, 0].long(), self.K).to(batch["summary"].dtype),
            nn.functional.one_hot(last[:, 1].long(), self.C).to(batch["summary"].dtype),
            nn.functional.one_hot(last[:, 2].long(), self.S).to(batch["summary"].dtype),
            last[:, 3:4],
        ], -1)
        return self.net(x)


class Attention(nn.Module):
    """Self-attention over the last N event tokens, plus the same summaries."""

    def __init__(self, n_summary: int, K: int, C: int, S: int, d: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.side, self.mark, self.state = nn.Embedding(K, d), nn.Embedding(C, d), nn.Embedding(S, d)
        self.time = nn.Linear(2, d)  # log_gap, log_age
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.merge = nn.Sequential(nn.Linear(d + n_summary, d), nn.GELU())
        self.d_out = d

    def forward(self, batch: dict) -> torch.Tensor:
        tok = batch["tokens"]  # (B, N, 5): side, mark, state, log_gap, log_age ; batch["pad"] (B, N) True where padding
        x = (self.side(tok[..., 0].long()) + self.mark(tok[..., 1].long()) + self.state(tok[..., 2].long())
             + self.time(tok[..., 3:5]))
        h = self.encoder(x, src_key_padding_mask=batch["pad"])[:, -1]  # the anchor is the last token
        return self.merge(torch.cat([h, batch["summary"]], -1))


class TPP(nn.Module):
    def __init__(self, encoder: nn.Module, head: Head):
        super().__init__()
        self.encoder, self.head = encoder, head
        self.to(DTYPE)


# --- batching -----------------------------------------------------------------


def _batch(f: Features, anchors: np.ndarray, N: int) -> dict:
    """Model inputs for a set of anchor events (the event whose history is used)."""
    idx = anchors[:, None] - np.arange(N - 1, -1, -1)[None, :]  # (B, N), last column = anchor
    pad = idx < 0
    idx = np.clip(idx, 0, None)
    age = np.log(f.t[anchors][:, None] - f.t[idx] + EPS)
    tokens = np.stack([f.side[idx], f.mark[idx], f.state[idx], f.log_gap[idx], age], -1).astype(float)
    tokens[pad] = 0.0
    last = np.stack([f.side[anchors], f.mark[anchors], f.state[anchors], f.log_gap[anchors]], -1).astype(float)
    return {"tokens": torch.as_tensor(tokens, dtype=DTYPE), "pad": torch.from_numpy(pad),
            "summary": torch.as_tensor(f.summary[anchors], dtype=DTYPE), "last": torch.as_tensor(last, dtype=DTYPE)}


def _pieces_in(f: Features, a: float, b: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compensator pieces clipped to [a, b): (closing event k, tau0, tau1, state)."""
    u0, u1 = np.maximum(f.piece_u0, a), np.minimum(f.piece_u1, b)
    keep = u1 > u0
    k = f.piece_k[keep]
    anchor_t = f.t[k - 1]
    return k, u0[keep] - anchor_t, u1[keep] - anchor_t, f.piece_state[keep]


@dataclass(frozen=True)
class Parts:
    """Everything the likelihood, the block bootstrap and the residuals need, on one window."""
    point: np.ndarray    # (n_ev,)   log lambda_{m_k}(t_k) for events in the window
    point_t: np.ndarray  # (n_ev,)   their times
    comp: np.ndarray     # (P, K)    integral of each piece in the window
    comp_t: np.ndarray   # (P,)      absolute start of each piece
    comp_k: np.ndarray   # (P,)      the event closing each piece's interval


def loglik_parts(model: TPP, f: Features, a: float, b: float, N: int, batch_size: int = 2048) -> Parts:
    """Point log-intensities of the events in [a, b), and the compensator
    integral of every piece clipped to [a, b)."""
    model.eval()
    with torch.no_grad():
        ks = np.flatnonzero((f.t >= a) & (f.t < b) & (np.arange(len(f)) > 0))
        point = np.empty(len(ks))
        for s in range(0, len(ks), batch_size):
            kk = ks[s:s + batch_size]
            h = model.encoder(_batch(f, kk - 1, N))
            a_, b_ = model.head(h, torch.from_numpy(f.state[kk]))
            tau = torch.as_tensor(f.t[kk] - f.t[kk - 1], dtype=DTYPE)
            lam = model.head.intensity(a_, b_, tau)
            point[s:s + batch_size] = torch.log(lam[torch.arange(len(kk)), torch.from_numpy(f.side[kk])]).double().numpy()
        pk, tau0, tau1, pst = _pieces_in(f, a, b)
        comp = np.empty((len(pk), model.head.K))
        for s in range(0, len(pk), batch_size):
            kk = pk[s:s + batch_size]
            h = model.encoder(_batch(f, kk - 1, N))
            a_, b_ = model.head(h, torch.from_numpy(pst[s:s + batch_size]))
            comp[s:s + batch_size] = model.head.integral(
                a_, b_, torch.as_tensor(tau0[s:s + batch_size], dtype=DTYPE),
                torch.as_tensor(tau1[s:s + batch_size], dtype=DTYPE)).double().numpy()
    return Parts(point, f.t[ks], comp, f.t[pk - 1] + tau0, pk)


def loglik(model: TPP, f: Features, a: float, b: float, N: int) -> float:
    q = loglik_parts(model, f, a, b, N)
    return float(q.point.sum() - q.comp.sum())


def block_loglik(model: TPP, f: Features, a: float, b: float, N: int, block_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-block log-likelihood and event count on [a, b), same blocks as `experiment.block_loglik`."""
    q = loglik_parts(model, f, a, b, N)
    edges = np.append(np.arange(a, b, block_s), b)
    bp = np.clip(np.searchsorted(edges, q.point_t, side="right") - 1, 0, len(edges) - 2)
    bc = np.clip(np.searchsorted(edges, q.comp_t, side="right") - 1, 0, len(edges) - 2)
    ll = (np.bincount(bp, weights=q.point, minlength=len(edges) - 1)
          - np.bincount(bc, weights=q.comp.sum(1), minlength=len(edges) - 1))
    return ll, np.bincount(bp, minlength=len(edges) - 1).astype(float)


def rescaled_residuals(model: TPP, f: Features, a: float, b: float, N: int) -> list[np.ndarray]:
    """Time-rescaling residuals per side on [a, b).

    The integral of interval k is the sum of its pieces; Lambda_i(t_k) is the
    cumulative sum of interval integrals up to k. Consecutive differences at
    the side-i events in the window are the residuals, as in
    `residuals.rescaled_residuals`.
    """
    q = loglik_parts(model, f, 0.0, b, N)
    n = len(f)
    Lam = np.zeros((n, model.head.K))
    for i in range(model.head.K):
        Lam[:, i] = np.cumsum(np.bincount(q.comp_k, weights=q.comp[:, i], minlength=n + 1)[:n])
    inside = (f.t >= a) & (f.t < b)
    return [np.diff(Lam[inside & (f.side == i), i]) for i in range(model.head.K)]


# --- training -----------------------------------------------------------------


def train(model: TPP, f: Features, T_train: float, val: tuple[float, float], N: int,
          epochs: int = 30, batch_size: int = 512, lr: float = 1e-3, patience: int = 3, seed: int = 0,
          targets_per_epoch: int | None = None, log=print) -> dict:
    """Adam on the training window, early stopping on validation NLL/event.
    The training objective is the per-interval likelihood: for each event k
    in train, log lambda_{m_k}(t_k) minus the integral over its interval
    (t_{k-1}, t_k], piece by piece across book rows. `targets_per_epoch`
    draws that many training events per epoch without replacement, so the
    validation check comes round more often."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    ks = np.flatnonzero((f.t < T_train) & (np.arange(len(f)) > 0))
    # pieces of the training intervals, grouped by closing event
    pk, tau0, tau1, pst = _pieces_in(f, 0.0, T_train)
    order = np.argsort(pk, kind="stable")
    pk, tau0, tau1, pst = pk[order], tau0[order], tau1[order], pst[order]
    first = np.searchsorted(pk, ks, side="left")
    last = np.searchsorted(pk, ks, side="right")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_val = int(((f.t >= val[0]) & (f.t < val[1])).sum())
    best, best_state, bad, history = np.inf, None, 0, []
    t_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(ks))
        if targets_per_epoch:
            perm = perm[:targets_per_epoch]
        total = 0.0
        for s in range(0, len(perm), batch_size):
            kk = ks[perm[s:s + batch_size]]
            h = model.encoder(_batch(f, kk - 1, N))
            a_, b_ = model.head(h, torch.from_numpy(f.state[kk]))
            lam = model.head.intensity(a_, b_, torch.as_tensor(f.t[kk] - f.t[kk - 1], dtype=DTYPE))
            point = torch.log(lam[torch.arange(len(kk)), torch.from_numpy(f.side[kk])])
            # pieces of these intervals
            sel = np.concatenate([np.arange(first[i], last[i]) for i in perm[s:s + batch_size]])
            owner = np.repeat(np.arange(len(kk)), last[perm[s:s + batch_size]] - first[perm[s:s + batch_size]])
            a_p, b_p = model.head(h[owner], torch.from_numpy(pst[sel]))
            comp = model.head.integral(a_p, b_p, torch.as_tensor(tau0[sel], dtype=DTYPE),
                                       torch.as_tensor(tau1[sel], dtype=DTYPE)).sum(1)
            loss = -(point.sum() - comp.sum()) / len(kk)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(kk)
        v = -loglik(model, f, val[0], val[1], N) / n_val
        history.append((epoch, total / len(perm), v))
        log(f"    epoch {epoch:2d}  train NLL/event {total / len(perm):8.4f}  val {v:8.4f}  ({time.perf_counter() - t_start:.0f}s)")
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: v_.detach().clone() for k, v_ in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"val": best, "epochs": len(history), "seconds": time.perf_counter() - t_start, "history": history}
