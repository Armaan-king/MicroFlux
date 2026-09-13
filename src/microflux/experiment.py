"""What every experiment script shares: the order sequence, the split, the
timescale grid, and one evaluator -- so two scripts cannot disagree on any of
them."""

from pathlib import Path

import numpy as np
import polars as pl

from microflux.book import replay
from microflux.events import Events
from microflux.hawkes import Params, compensator, intensity, loglik
from microflux.residuals import ks_exp1, rescaled_residuals
from microflux.load import collapse_trades, load_stream, partition

TYPES = ("BUY", "SELL")
HALF_LIVES = np.array([0.005, 0.05, 0.5, 5.0, 50.0])  # seconds; one decade apart
SCALES = np.log(2.0) / HALF_LIVES
CACHE = Path("data")


def load_orders(root: str, symbol: str, date: str, minutes: float | None = None, merge_gap_ns: int = 0) -> pl.DataFrame:
    """Aggressive orders in capture order, with `t` in seconds from the first."""
    orders = collapse_trades(load_stream(partition(root, symbol, date), "trades"), merge_gap_ns)
    orders = orders.with_columns(
        ((pl.col("timestamp_ns") - orders["timestamp_ns"][0]) / 1e9).alias("t"),
        (pl.col("aggressor") == "sell").cast(pl.Int64).alias("m"),
    )
    return orders.filter(pl.col("t") < minutes * 60) if minutes else orders


def load_book(root: str, symbol: str, date: str) -> pl.DataFrame:
    """1 Hz book state, replayed once and cached; ~2 min from cold."""
    CACHE.mkdir(exist_ok=True)
    import hashlib
    tag = hashlib.sha1(str(Path(root).resolve()).encode()).hexdigest()[:8]  # two roots can hold the same date
    path = CACHE / f"book-{symbol}-{date}-{tag}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    d = partition(root, symbol, date)
    book = replay(load_stream(d, "snapshots"), load_stream(d, "book_updates"))
    book.write_parquet(path)
    return book


def imbalance_state(orders: pl.DataFrame, book: pl.DataFrame, T_train: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L5-imbalance terciles as a step function of time: (step_t, step_s, cuts).

    Cuts come from train book rows only (1 Hz, so time-weighted). Book rows
    carry emission time; an order matched at T sees the row emitted at or
    before T, which cannot include that order -- leak-free, and up to one
    second stale.
    """
    step_t = (book["timestamp_ns"].to_numpy() - orders["timestamp_ns"][0]) / 1e9
    imb = book["imb5"].to_numpy()
    cuts = np.percentile(imb[step_t < T_train], [100 / 3, 200 / 3])
    return step_t, np.digitize(imb, cuts), cuts


MARK_EDGES = np.array([2, 5, 20])  # fill-count classes: 1 / 2-4 / 5-19 / 20+  (fills = trade rows in the run)
MARK_NAMES = ("1 fill", "2-4", "5-19", "20+")


def mark_class(orders: pl.DataFrame) -> np.ndarray:
    return np.digitize(orders["fills"].to_numpy(), MARK_EDGES)


def splits(T: float) -> dict[str, tuple[float, float]]:
    """70 / 15 / 15 by time. Chronological, never shuffled."""
    return {"train": (0.0, 0.70 * T), "val": (0.70 * T, 0.85 * T), "test": (0.85 * T, T)}


def evaluate(p: Params, ev: Events, split: dict[str, tuple[float, float]],
             segments: tuple[str, ...] = ("train", "val"), ks_on: str = "val") -> dict:
    """Per-event NLL on the named segments; KS(Exp1) per type on `ks_on`.

    Train and validation only by default. Scripts that select or diagnose
    must never see a test statistic; the test segment is scored once, by a
    final-test script that asks for it explicitly, under PROTOCOL.md.
    """
    out = {seg: -loglik(p, ev, *split[seg]) / int(ev.window(*split[seg]).sum()) for seg in segments}
    a, b = split[ks_on]
    out["ks"] = [ks_exp1(r) for r in rescaled_residuals(p, ev, a, b)]
    out["ks_on"] = ks_on
    return out


def matrix(title: str, M: np.ndarray, cols: tuple[str, ...] = TYPES, fmt: str = "{:10.4f}") -> None:
    print(f"\n{title}")
    print(f"{'':>8}" + "".join(f"{'<-' + c:>10}" for c in cols))
    for i, row in enumerate(M):
        print(f"{TYPES[i]:>8}" + "".join(fmt.format(v) for v in row))


# --- uncertainty and controls -----------------------------------------------


def block_loglik(p: Params, ev: Events, a: float, b: float, block_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Log-likelihood and event count per consecutive block of `block_s`
    seconds over [a, b). Blocks are the resampling unit: events within a
    block are dependent, blocks far apart are close to independent."""
    lam = intensity(p, ev)
    own = np.log(lam[np.arange(len(ev)), ev.m])
    edges = np.append(np.arange(a, b, block_s), b)
    ll, n = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        inside = ev.window(lo, hi)
        ll[k] = own[inside].sum() - compensator(p, ev, lo, hi).sum()
        n[k] = inside.sum()
    return ll, n


def gain_ci_from_blocks(d: np.ndarray, n: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Gain per event and its 95% bootstrap interval from per-block
    log-likelihood differences `d` and event counts `n`."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boot = d[idx].sum(1) / n[idx].sum(1)
    return float(d.sum() / n.sum()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def gain_ci(p_ref: Params, p_new: Params, ev_ref: Events, ev_new: Events, a: float, b: float,
            block_s: float = 60.0, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Held-out NLL/event gain of `p_new` over `p_ref` with a 95% block-bootstrap interval.

    Both models are scored on the same time blocks; the per-block difference
    is resampled with replacement. Returns (gain, lo, hi). A gain small
    enough to matter only under one block size is not a gain: see
    `gain_ci_blocks`.
    """
    ll_r, n = block_loglik(p_ref, ev_ref, a, b, block_s)
    ll_n, _ = block_loglik(p_new, ev_new, a, b, block_s)
    return gain_ci_from_blocks(ll_n - ll_r, n, n_boot, seed)


def gain_ci_blocks(p_ref: Params, p_new: Params, ev_ref: Events, ev_new: Events, a: float, b: float,
                   sizes: tuple[float, ...] = (60.0, 300.0, 900.0)) -> list[tuple[float, float, float, float]]:
    """`gain_ci` at several block sizes: (block_s, gain, lo, hi) each. Wider
    blocks respect longer dependence and widen the interval; a tiny gain
    should be read against the widest."""
    return [(s, *gain_ci(p_ref, p_new, ev_ref, ev_new, a, b, block_s=s)) for s in sizes]


def shuffle_within(x: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """`x` permuted separately inside each group. For a mark control the
    group is (split, side): the label loses its timing but keeps its split
    and its side-conditional distribution."""
    rng = np.random.default_rng(seed)
    out = x.copy()
    for g in np.unique(groups):
        k = np.flatnonzero(groups == g)
        out[k] = x[rng.permutation(k)]
    return out


def split_id(t: np.ndarray, split: dict[str, tuple[float, float]]) -> np.ndarray:
    """0 / 1 / 2 for train / val / test."""
    out = np.zeros(len(t), np.int64)
    for k, (_, (a, b)) in enumerate(split.items()):
        out[(t >= a) & (t < b)] = k
    return out
