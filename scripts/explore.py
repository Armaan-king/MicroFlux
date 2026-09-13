"""Stage 1: measure the capture, so the event taxonomy is chosen not guessed.

Answers the questions GOAL.md makes a precondition for modelling: what the
event rates are, whether arrivals cluster enough to justify a Hawkes process
over a Poisson one, and at what resolution market state is actually observable.

    python scripts/explore.py [--root C:/tickforge-runs] [--date 2026-09-09]
"""

import argparse

import numpy as np
import polars as pl

from microflux.load import collapse_trades, is_continuous, load_stream, partition


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def describe_gaps(t: np.ndarray, label: str) -> dict:
    """Inter-arrival summary, plus the two statistics that pick the model.

    CV is the coefficient of variation of the gaps and the Fano factor is the
    variance-to-mean ratio of counts per bin. A Poisson process gives 1 for
    both. Above 1 means arrivals bunch -- which is the excitation a Hawkes
    process exists to capture, and the evidence that fitting one is worth it.
    """
    dt = np.diff(t) / 1e9
    span = (t[-1] - t[0]) / 1e9
    counts = np.histogram(t, bins=int(span))[0]  # one-second bins
    cv = dt.std() / dt.mean()
    fano = counts.var() / counts.mean()
    print(
        f"{label:<22} n={len(t):>9,}  rate={len(t) / span:7.2f}/s  "
        f"dt: mean={dt.mean() * 1e3:8.2f}ms p50={np.median(dt) * 1e3:7.2f}ms "
        f"max={dt.max():6.2f}s  zero={100 * (dt == 0).mean():5.2f}%"
    )
    print(f"{'':<22} CV={cv:5.2f}  Fano={fano:7.2f}   (Poisson = 1.0 for both)")
    return {"n": len(t), "rate": len(t) / span, "cv": cv, "fano": fano}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/tickforge-runs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2026-09-09")
    args = ap.parse_args()

    directory = partition(args.root, args.symbol, args.date)
    book = load_stream(directory, "book_updates")
    trades = load_stream(directory, "trades")

    rule("CAPTURE")
    span = (book["timestamp_ns"].max() - book["timestamp_ns"].min()) / 1e9
    print(f"{args.symbol} {args.date}   span={span / 3600:.2f}h   files/sessions={book['session'].n_unique()}")
    print(f"one unbroken Binance sequence: {is_continuous(book)}"
          "   (first_seq == prev last_seq + 1 at every boundary)")

    rule("BOOK UPDATES -- is this a point process or a clock?")
    describe_gaps(book["timestamp_ns"].to_numpy(), "book_updates")
    levels = book.select(
        (pl.col("bids").list.len() + pl.col("asks").list.len()).alias("n")
    )["n"].to_numpy()
    print(f"{'':<22} levels/update: p50={np.median(levels):.0f} "
          f"p90={np.percentile(levels, 90):.0f} max={levels.max()}")
    print("\n-> a fixed 1 Hz grid. Depth arrival times carry no market information,")
    print("   so depth cannot be an event type in a temporal point process.")

    rule("TRADES -- raw fills vs aggressive orders")
    describe_gaps(trades["timestamp_ns"].to_numpy(), "raw fills")
    orders = collapse_trades(trades)
    describe_gaps(orders["timestamp_ns"].to_numpy(), "aggressive orders")
    print(f"\ncollapse: {trades.height:,} fills -> {orders.height:,} orders "
          f"({orders.height / trades.height:.1%}, {trades.height / orders.height:.1f} fills each)")
    fills = orders["fills"].to_numpy()
    print(f"fills per order: p50={np.median(fills):.0f} p90={np.percentile(fills, 90):.0f} "
          f"max={fills.max()}  (levels swept by one order)")

    rule("PER-SIDE -- the event types a Hawkes process would carry")
    for side in ("buy", "sell"):
        describe_gaps(
            orders.filter(pl.col("aggressor") == side)["timestamp_ns"].to_numpy(),
            f"{side.upper()}_ORDER",
        )
    ties = orders.group_by("timestamp_ns").len().filter(pl.col("len") > 1).height
    print(f"\nremaining simultaneous orders (opposite sides, same ms): {ties:,} "
          f"({100 * ties / orders.height:.2f}%)")

    rule("CHRONOLOGICAL SPLIT -- 70/15/15 by time, no shuffling")
    t = orders["timestamp_ns"].to_numpy()
    lo, hi = t[0], t[-1]
    for name, a, b in [("train", 0.0, 0.70), ("val", 0.70, 0.85), ("test", 0.85, 1.0)]:
        s, e = lo + (hi - lo) * a, lo + (hi - lo) * b
        n = int(((t >= s) & (t < e)).sum())
        print(f"{name:<6} {(e - s) / 1e9 / 3600:5.2f}h  orders={n:>8,}  start_ns={int(s)}")


if __name__ == "__main__":
    main()
