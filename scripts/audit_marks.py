"""Audit of the fill-count mark: what it measures, and whether the D9 result
survives a different grouping rule.

`fills` is the number of trade rows in a same-millisecond same-side run.
That is resting orders matched, not price levels, and it cannot tell one
taker order from two in the same millisecond. `levels` (distinct prices in
the run) is the nearer proxy for depth consumed. This script reports how the
two relate and refits the marked model under an alternative rule that also
joins same-side runs 1 ms apart -- the millisecond-boundary split -- to see
whether the mark effect is an artifact of where a run is cut.

Evaluated on the validation segment. The 2026-09-09 test segment has been
inspected repeatedly and is exploratory from here on.

    python scripts/audit_marks.py [--root C:/tickforge-runs] [--date 2026-09-09]
"""

import argparse

import numpy as np
import polars as pl

from microflux.events import events
from microflux.experiment import (
    HALF_LIVES, MARK_EDGES, MARK_NAMES, SCALES, evaluate, load_orders, splits,
)
from microflux.hawkes import block_edges
from microflux.mle import extrapolate, fit_hawkes


def marked_gain(orders: pl.DataFrame, mark: np.ndarray, C: int, block_s: float) -> tuple[float, np.ndarray]:
    """Validation NLL gain of the marked over the unmarked kernel, and the
    BUY<-BUY branching-by-scale table of the marked fit."""
    t, m = orders["t"].to_numpy(), orders["m"].to_numpy()
    split = splits(t[-1])
    T_train = split["train"][1]
    blocks = block_edges(T_train, block_s)
    flat, marked = events(t, m, 2), events(t, m, 2, mark=mark, C=C)
    p0 = fit_hawkes(flat.before(T_train), T_train, blocks, SCALES)
    p1 = fit_hawkes(marked.before(T_train), T_train, blocks, SCALES)
    v0 = evaluate(extrapolate(p0, flat, T_train), flat, split)["val"]
    v1 = evaluate(extrapolate(p1, marked, T_train), marked, split)["val"]
    table = np.array([[p1.branching_by_scale[l, 0, 2 * c] for c in range(C)] for l in range(len(HALF_LIVES))])
    return v0 - v1, table


def show(table: np.ndarray, names: tuple[str, ...]) -> None:
    print(f"{'half-life':>10}" + "".join(f"{n:>10}" for n in names))
    for l, h in enumerate(HALF_LIVES):
        print(f"{h:>9.3f}s" + "".join(f"{v:10.4f}" for v in table[l]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/tickforge-runs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2026-09-09")
    ap.add_argument("--block-minutes", type=float, default=15.0)
    args = ap.parse_args()
    block = args.block_minutes * 60

    print("=" * 72)
    print("1. what does `fills` measure?")
    print("=" * 72)
    o = load_orders(args.root, args.symbol, args.date)
    f, lv = o["fills"].to_numpy(), o["levels"].to_numpy()
    print(f"orders: {len(o):,}   fills p50={np.median(f):.0f} p90={np.percentile(f, 90):.0f} max={f.max()}   "
          f"levels p50={np.median(lv):.0f} p90={np.percentile(lv, 90):.0f} max={lv.max()}")
    print(f"corr(log fills, log levels) = {np.corrcoef(np.log(f), np.log(lv))[0, 1]:.3f}   "
          f"fills/levels p50={np.median(f / lv):.2f} p90={np.percentile(f / lv, 90):.2f}")
    print("\nfills class  ->  distribution of levels")
    lv_edges = np.array([2, 3, 6])
    lv_names = ("1 lvl", "2", "3-5", "6+")
    fc, lc = np.digitize(f, MARK_EDGES), np.digitize(lv, lv_edges)
    print(f"{'':<10}" + "".join(f"{n:>9}" for n in lv_names))
    for c, name in enumerate(MARK_NAMES):
        row = [100 * np.mean(lc[fc == c] == k) for k in range(4)]
        print(f"{name:<10}" + "".join(f"{v:8.1f}%" for v in row))

    print("\n" + "=" * 72)
    print("2. sensitivity to the grouping rule  (validation NLL gain of marks)")
    print("=" * 72)
    rules = [("same ms, same side  (default)", 0), ("same side, gap <= 1 ms", 1_000_000)]
    for name, gap in rules:
        oo = load_orders(args.root, args.symbol, args.date, merge_gap_ns=gap)
        ff = oo["fills"].to_numpy()
        gain, table = marked_gain(oo, np.digitize(ff, MARK_EDGES), 4, block)
        print(f"\n[{name}]   orders={len(oo):,}   val gain of marks = {gain:+.4f}")
        show(table, MARK_NAMES)

    print("\n" + "=" * 72)
    print("3. levels as the mark instead of fills  (validation NLL gain)")
    print("=" * 72)
    gain, table = marked_gain(o, lc, 4, block)
    print(f"val gain of levels-mark = {gain:+.4f}")
    show(table, lv_names)


if __name__ == "__main__":
    main()
