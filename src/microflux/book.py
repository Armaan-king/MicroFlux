"""Replay a TickForge capture into a 1 Hz book-state series.

TickForge serves the snapshot and the diffs and deliberately not the
reconstructed book, so that features are recomputed from events and stay
reproducible. This is that recomputation. It applies Binance depth-diff
semantics -- a level with quantity zero is removed -- and nothing else: no
inference of what caused a change, which the data cannot support.

One row per book update, at the update's emission time. That is the state
resolution the capture has (see ARCHITECTURE.md D1).
"""

import heapq

import numpy as np
import polars as pl

DEPTH = 10  # levels summed for depth; imbalance uses 1 and 5


def _levels(col: pl.Series) -> list[list[tuple[float, float]]]:
    """List(Struct(price, quantity)) -> [[(price, qty), ...], ...] as floats.

    Cast in polars, not per element: Decimal(38, 18) -> Float64 is exact for
    the two-decimal prices Binance sends, and the same decimal always maps to
    the same float, so floats are safe dict keys here.
    """
    frame = col.to_frame("l").select(
        pl.col("l").list.eval(
            pl.struct(pl.element().struct.field("price").cast(pl.Float64).alias("p"),
                      pl.element().struct.field("quantity").cast(pl.Float64).alias("q"))
        )
    )
    return [[(d["p"], d["q"]) for d in row] for row in frame["l"].to_list()]


def replay(snapshot: pl.DataFrame, updates: pl.DataFrame) -> pl.DataFrame:
    """Book features at every update. Columns: timestamp_ns, capture_seq,
    best_bid, best_ask, spread, mid, imb1, imb5, bid_depth, ask_depth.

    Requires the updates to be gap-free from the snapshot onward, which
    `load.is_continuous` establishes. Updates already covered by the snapshot
    are skipped, per the Binance procedure.
    """
    assert snapshot.height == 1, "replay expects exactly one seed snapshot"
    seq = int(snapshot["last_seq"][0])
    bids = {p: q for p, q in _levels(snapshot["bids"])[0] if q > 0}
    asks = {p: q for p, q in _levels(snapshot["asks"])[0] if q > 0}

    updates = updates.sort("session", "capture_seq")
    ub, ua = _levels(updates["bids"]), _levels(updates["asks"])
    last_seq = updates["last_seq"].to_numpy()
    ts = updates["timestamp_ns"].to_numpy()
    cs = updates["capture_seq"].to_numpy()

    rows = []
    for k in range(updates.height):
        if last_seq[k] <= seq:
            continue
        for p, q in ub[k]:
            if q > 0:
                bids[p] = q
            else:
                bids.pop(p, None)
        for p, q in ua[k]:
            if q > 0:
                asks[p] = q
            else:
                asks.pop(p, None)
        seq = last_seq[k]

        top_b = heapq.nlargest(DEPTH, bids.items())
        top_a = heapq.nsmallest(DEPTH, asks.items())
        bb, ba = top_b[0][0], top_a[0][0]
        assert bb < ba, f"crossed book at capture_seq={cs[k]}: bid {bb} >= ask {ba}"
        bq1, aq1 = top_b[0][1], top_a[0][1]
        bq5 = sum(q for _, q in top_b[:5])
        aq5 = sum(q for _, q in top_a[:5])
        rows.append((
            ts[k], cs[k], bb, ba, ba - bb, (bb + ba) / 2,
            (bq1 - aq1) / (bq1 + aq1), (bq5 - aq5) / (bq5 + aq5),
            sum(q for _, q in top_b), sum(q for _, q in top_a),
        ))

    return pl.DataFrame(
        rows,
        schema=["timestamp_ns", "capture_seq", "best_bid", "best_ask", "spread", "mid",
                "imb1", "imb5", "bid_depth", "ask_depth"],
        orient="row",
    )


def align(order_ts: np.ndarray, book_ts: np.ndarray) -> np.ndarray:
    """Index of the last book row emitted at or before each order; -1 if none.

    Book rows carry emission time and orders carry match time, both Binance
    server clocks. A row emitted at E <= T cannot include the effect of a
    trade matched at T, so this is leak-free -- and up to one second stale,
    which is the resolution the capture has.
    """
    return np.searchsorted(book_ts, order_ts, side="right") - 1
