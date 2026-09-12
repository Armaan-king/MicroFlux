"""Stage 2: Poisson vs Hawkes on aggressive orders, chronological split.

Fits on train only. Reports per-event NLL on train/val/test and the
time-rescaling KS distance on test -- likelihood ranks the models, KS says
whether either actually fits.

    python fit.py [--root C:/tickforge-runs] [--date 2026-09-09] [--minutes N]
"""

import argparse
import time

import numpy as np
import polars as pl

from microflux.hawkes import (
    Params, fit_hawkes, fit_poisson, ks_exp1, loglik, rescaled_residuals,
)
from microflux.load import collapse_trades, load_stream, partition

TYPES = ("BUY", "SELL")


def load_orders(root: str, symbol: str, date: str, minutes: float | None) -> tuple[np.ndarray, np.ndarray]:
    orders = collapse_trades(load_stream(partition(root, symbol, date), "trades"))
    t = (orders["timestamp_ns"].to_numpy() - orders["timestamp_ns"][0]) / 1e9
    m = (orders["aggressor"] == "sell").cast(pl.Int64).to_numpy()
    if minutes:
        keep = t < minutes * 60
        t, m = t[keep], m[keep]
    return t, m


def report(name: str, p: Params, t: np.ndarray, m: np.ndarray, splits: dict[str, tuple[float, float]]) -> None:
    print(f"\n--- {name} ---")
    for seg, (a, b) in splits.items():
        n = int(((t >= a) & (t < b)).sum())
        nll = -loglik(p, t, m, a, b) / n
        line = f"{seg:<6} n={n:>8,}  NLL/event={nll:8.4f}"
        if seg == "test":
            ks = [ks_exp1(r) for r in rescaled_residuals(p, t, m, a, b)]
            line += "   KS(Exp1): " + "  ".join(f"{k}={v:.4f}" for k, v in zip(TYPES, ks))
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/tickforge-runs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2026-09-09")
    ap.add_argument("--minutes", type=float, default=None, help="use only the first N minutes")
    args = ap.parse_args()

    t, m = load_orders(args.root, args.symbol, args.date, args.minutes)
    T = t[-1]
    splits = {"train": (0.0, 0.70 * T), "val": (0.70 * T, 0.85 * T), "test": (0.85 * T, T)}
    T_train = splits["train"][1]
    tr = t < T_train
    print(f"{len(t):,} orders over {T / 3600:.2f}h   train={tr.sum():,}  "
          f"types: BUY={int((m == 0).sum()):,} SELL={int((m == 1).sum()):,}")

    poisson = fit_poisson(t[tr], m[tr], T_train, 2)
    report("Poisson", poisson, t, m, splits)

    t0 = time.perf_counter()
    hawkes = fit_hawkes(t[tr], m[tr], T_train, 2)
    print(f"\n(Hawkes MLE took {time.perf_counter() - t0:.1f}s)")
    report("Hawkes (exp kernel)", hawkes, t, m, splits)

    print("\n--- Hawkes parameters (fit on train) ---")
    print(f"mu (exogenous rate, /s):    BUY={hawkes.mu[0]:.4f}  SELL={hawkes.mu[1]:.4f}")
    print(f"Poisson rate for reference: BUY={poisson.mu[0]:.4f}  SELL={poisson.mu[1]:.4f}")
    print("\nbranching alpha/beta  (row = excited type, col = exciting type)")
    print(f"{'':>8}{'<-BUY':>10}{'<-SELL':>10}")
    for i, row in enumerate(hawkes.branching):
        print(f"{TYPES[i]:>8}{row[0]:10.4f}{row[1]:10.4f}")
    print(f"spectral radius = {hawkes.spectral_radius:.4f}   (fraction of events that are endogenous)")
    print("\nhalf-life of excitation ln2/beta (seconds)")
    print(f"{'':>8}{'<-BUY':>10}{'<-SELL':>10}")
    for i, row in enumerate(np.log(2) / hawkes.beta):
        print(f"{TYPES[i]:>8}{row[0]:10.3f}{row[1]:10.3f}")


if __name__ == "__main__":
    main()
