"""What every experiment script shares: the order sequence, the split, the
timescale grid, and one evaluator -- so two scripts cannot disagree on any of
them."""

from pathlib import Path

import numpy as np
import polars as pl

from microflux.book import replay
from microflux.events import Events
from microflux.hawkes import Params, loglik
from microflux.residuals import ks_exp1, rescaled_residuals
from microflux.load import collapse_trades, load_stream, partition

TYPES = ("BUY", "SELL")
HALF_LIVES = np.array([0.005, 0.05, 0.5, 5.0, 50.0])  # seconds; one decade apart
SCALES = np.log(2.0) / HALF_LIVES
CACHE = Path("data")


def load_orders(root: str, symbol: str, date: str, minutes: float | None = None) -> pl.DataFrame:
    """Aggressive orders in capture order, with `t` in seconds from the first."""
    orders = collapse_trades(load_stream(partition(root, symbol, date), "trades"))
    orders = orders.with_columns(
        ((pl.col("timestamp_ns") - orders["timestamp_ns"][0]) / 1e9).alias("t"),
        (pl.col("aggressor") == "sell").cast(pl.Int64).alias("m"),
    )
    return orders.filter(pl.col("t") < minutes * 60) if minutes else orders


def load_book(root: str, symbol: str, date: str) -> pl.DataFrame:
    """1 Hz book state, replayed once and cached; ~2 min from cold."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"book-{symbol}-{date}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    d = partition(root, symbol, date)
    book = replay(load_stream(d, "snapshots"), load_stream(d, "book_updates"))
    book.write_parquet(path)
    return book


def state_function(book: pl.DataFrame, t0_ns: int, column: str, cuts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The book as a step function of time: (step_t, step_s), with `column`
    bucketed by `cuts`. Book rows carry emission time; an order matched at
    T sees the row emitted at or before T, which cannot include that order
    -- leak-free, and up to one second stale."""
    step_t = (book["timestamp_ns"].to_numpy() - t0_ns) / 1e9
    return step_t, np.digitize(book[column].to_numpy(), cuts)


def splits(T: float) -> dict[str, tuple[float, float]]:
    """70 / 15 / 15 by time. Chronological, never shuffled."""
    return {"train": (0.0, 0.70 * T), "val": (0.70 * T, 0.85 * T), "test": (0.85 * T, T)}


def evaluate(p: Params, ev: Events, split: dict[str, tuple[float, float]]) -> dict:
    """Per-event NLL on every segment; KS(Exp1) per type on test."""
    out = {seg: -loglik(p, ev, a, b) / int(ev.window(a, b).sum()) for seg, (a, b) in split.items()}
    a, b = split["test"]
    out["ks"] = [ks_exp1(r) for r in rescaled_residuals(p, ev, a, b)]
    return out


def matrix(title: str, M: np.ndarray, cols: tuple[str, ...] = TYPES, fmt: str = "{:10.4f}") -> None:
    print(f"\n{title}")
    print(f"{'':>8}" + "".join(f"{'<-' + c:>10}" for c in cols))
    for i, row in enumerate(M):
        print(f"{TYPES[i]:>8}" + "".join(fmt.format(v) for v in row))
