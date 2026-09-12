"""Load a TickForge capture into research frames.

Reads the stored Parquet partitions directly rather than through the research
HTTP API. Same bytes, same ordering rule -- the API exists so a consumer need
not rebuild TickForge's Binance, sequencing and recovery logic, and reading a
partition it already wrote rebuilds none of that. Swap in the HTTP path when
the capture lives on another machine.

Ordering is ``(session, capture_seq)``, never a timestamp: both timestamp
columns have ties and they disagree with each other. See TickForge
`research.read_range`.
"""

import re
from pathlib import Path

import polars as pl

SESSION_FILE = re.compile(r"^(?P<stream>[a-z_]+)-(?P<session>\d+)\.parquet$")
STREAMS = ("snapshots", "book_updates", "trades")


def partition(root: str | Path, symbol: str, date: str, exchange: str = "binance") -> Path:
    return Path(root) / exchange / symbol.upper() / date


def session_files(directory: Path) -> dict[int, dict[str, Path]]:
    """``{session: {stream: path}}`` for one partition.

    Mirrors TickForge's `replay.session_files`, including its exclusion of
    `features` -- derived output that a consumer recomputes.
    """
    sessions: dict[int, dict[str, Path]] = {}
    for path in directory.glob("*.parquet"):
        match = SESSION_FILE.match(path.name)
        if match is None or match["stream"] == "features":
            continue
        sessions.setdefault(int(match["session"]), {})[match["stream"]] = path
    return sessions


def load_stream(directory: Path, stream: str) -> pl.DataFrame:
    """One stream for a partition, session column added, in replay order.

    The stored files hold the session in their filename only, so a merged frame
    needs the column added before it can be sorted -- exactly what TickForge's
    `/export` does for the same reason.
    """
    files = session_files(directory)
    frames = [
        pl.read_parquet(paths[stream]).with_columns(pl.lit(session, pl.Int64).alias("session"))
        for session, paths in sorted(files.items())
        if stream in paths
    ]
    if not frames:
        raise FileNotFoundError(f"no {stream} in {directory}")
    return pl.concat(frames).sort("session", "capture_seq")


def is_continuous(book_updates: pl.DataFrame) -> bool:
    """Whether the book updates form one unbroken Binance sequence.

    A capture rolled into several files is still one synchronised stream; a
    genuine reconnect is not, and joining across one silently fabricates
    continuity the market never had. Binance's own `first_seq`/`last_seq` settle
    it -- contiguous means no update was missed, whatever the filenames suggest.

    Checked rather than assumed, because ARCHITECTURE.md's rule is that
    separate sessions must not be *presumed* continuous. This is the evidence.
    """
    edges = book_updates.select(
        (pl.col("first_seq") - pl.col("last_seq").shift(1)).alias("gap")
    ).drop_nulls()
    return bool((edges["gap"] == 1).all())


def collapse_trades(trades: pl.DataFrame) -> pl.DataFrame:
    """Raw fills -> aggressive orders.

    Binance's ``@trade`` stream reports one row per fill, so a single marketable
    order sweeping the book arrives as up to 231 rows sharing a millisecond
    timestamp. Left raw, ~91% of consecutive trades are separated by dt=0 and
    every point-process likelihood degenerates.

    Consecutive fills sharing `(timestamp_ns, aggressor)` are one aggressor's
    decision, so they collapse to one event carrying the summed quantity and the
    number of fills -- the depth of book that order consumed. This reconstructs
    what Binance's ``@aggTrade`` stream would have delivered.

    Grouped on a run index, not on the timestamp alone: two genuinely separate
    orders can land in the same millisecond on opposite sides, and merging those
    would invent a trade that never happened.
    """
    run = (
        (pl.col("timestamp_ns") != pl.col("timestamp_ns").shift(1))
        | (pl.col("aggressor") != pl.col("aggressor").shift(1))
    ).cum_sum().alias("run")

    return (
        trades.with_columns(run)
        .group_by("run", maintain_order=True)
        .agg(
            pl.col("session").first(),
            pl.col("capture_seq").first(),
            pl.col("timestamp_ns").first(),
            pl.col("received_ns").first(),
            pl.col("aggressor").first(),
            pl.col("quantity").cast(pl.Float64).sum().alias("quantity"),
            pl.col("price").cast(pl.Float64).first().alias("price_first"),
            pl.col("price").cast(pl.Float64).last().alias("price_last"),
            pl.len().alias("fills"),
            pl.col("trade_id").first().alias("trade_id"),
        )
        .drop("run")
    )


def load_capture(root: str | Path, symbol: str, date: str) -> dict[str, pl.DataFrame]:
    """Every stream for one partition, plus the collapsed trade process."""
    directory = partition(root, symbol, date)
    out = {s: load_stream(directory, s) for s in STREAMS if _has(directory, s)}
    out["orders"] = collapse_trades(out["trades"])
    return out


def _has(directory: Path, stream: str) -> bool:
    return any(stream in paths for paths in session_files(directory).values())
