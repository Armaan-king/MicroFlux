"""Stage 2: the Hawkes ladder on aggressive orders, chronological split.

Six models, each adding one thing, so every improvement has one cause:

    Poisson                       constant rate
    Poisson + mu(t)               rate drifts -- how much of the story is drift alone?
    Hawkes                        one exponential per pair, free decay        (D4)
    Hawkes + mu(t)                same, but drift cannot hide in slow kernels  (A)
    Hawkes x5                     five fixed timescales per pair
    Hawkes x5 + mu(t)             both controls                                 (B)

Fits on train only. Reports per-event NLL on train/val/test and the
time-rescaling KS distance on test -- likelihood ranks the models, KS says
whether any of them actually fits.

    python fit.py [--root C:/tickforge-runs] [--date 2026-09-09] [--minutes N] [--block-minutes 15]
"""

import argparse
import time

import numpy as np
import polars as pl

from microflux.hawkes import (
    Params, extrapolate, fit_hawkes, fit_poisson, ks_exp1, loglik, rescaled_residuals,
)
from microflux.load import collapse_trades, load_stream, partition

TYPES = ("BUY", "SELL")
HALF_LIVES = np.array([0.005, 0.05, 0.5, 5.0, 50.0])  # seconds; one decade apart
SCALES = np.log(2.0) / HALF_LIVES


def load_orders(root: str, symbol: str, date: str, minutes: float | None) -> tuple[np.ndarray, np.ndarray]:
    orders = collapse_trades(load_stream(partition(root, symbol, date), "trades"))
    t = (orders["timestamp_ns"].to_numpy() - orders["timestamp_ns"][0]) / 1e9
    m = (orders["aggressor"] == "sell").cast(pl.Int64).to_numpy()
    if minutes:
        keep = t < minutes * 60
        t, m = t[keep], m[keep]
    return t, m


def evaluate(p: Params, t: np.ndarray, m: np.ndarray, splits: dict[str, tuple[float, float]]) -> dict:
    out = {}
    for seg, (a, b) in splits.items():
        n = int(((t >= a) & (t < b)).sum())
        out[seg] = -loglik(p, t, m, a, b) / n
    a, b = splits["test"]
    out["ks"] = [ks_exp1(r) for r in rescaled_residuals(p, t, m, a, b)]
    return out


def matrix(title: str, M: np.ndarray, fmt: str = "{:10.4f}") -> None:
    print(f"\n{title}")
    print(f"{'':>8}{'<-BUY':>10}{'<-SELL':>10}")
    for i, row in enumerate(M):
        print(f"{TYPES[i]:>8}" + "".join(fmt.format(v) for v in row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/tickforge-runs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2026-09-09")
    ap.add_argument("--minutes", type=float, default=None, help="use only the first N minutes")
    ap.add_argument("--block-minutes", type=float, default=15.0, help="baseline block width")
    args = ap.parse_args()

    t, m = load_orders(args.root, args.symbol, args.date, args.minutes)
    T = t[-1]
    splits = {"train": (0.0, 0.70 * T), "val": (0.70 * T, 0.85 * T), "test": (0.85 * T, T)}
    T_train = splits["train"][1]
    tr = t < T_train
    tt, mt = t[tr], m[tr]
    block = args.block_minutes * 60
    print(f"{len(t):,} orders over {T / 3600:.2f}h   train={tr.sum():,}  "
          f"BUY={int((m == 0).sum()):,} SELL={int((m == 1).sum()):,}   "
          f"baseline blocks of {args.block_minutes:g} min")

    ladder = [
        ("Poisson",           lambda: fit_poisson(tt, mt, T_train, 2)),
        ("Poisson + mu(t)",   lambda: fit_poisson(tt, mt, T_train, 2, block_s=block)),
        ("Hawkes",            lambda: fit_hawkes(tt, mt, T_train, 2)),
        ("Hawkes + mu(t)",    lambda: fit_hawkes(tt, mt, T_train, 2, block_s=block)),
        ("Hawkes x5",         lambda: fit_hawkes(tt, mt, T_train, 2, scales=SCALES)),
        ("Hawkes x5 + mu(t)", lambda: fit_hawkes(tt, mt, T_train, 2, block_s=block, scales=SCALES)),
    ]

    fits: dict[str, Params] = {}
    rows = []
    for name, fit in ladder:
        t0 = time.perf_counter()
        p = fit()
        secs = time.perf_counter() - t0
        fits[name] = p
        ev = evaluate(extrapolate(p, T_train), t, m, splits)
        rows.append((name, ev, p.spectral_radius, secs))
        print(f"  fitted {name:<20} {secs:6.1f}s")

    print(f"\n{'model':<20}{'train':>9}{'val':>9}{'test':>9}   {'KS BUY':>7} {'KS SELL':>8}   {'rho':>6}")
    print(f"{'':<20}{'NLL/event':>27}   {'test, Exp(1)':>16}")
    for name, ev, rho, _ in rows:
        print(f"{name:<20}{ev['train']:9.4f}{ev['val']:9.4f}{ev['test']:9.4f}   "
              f"{ev['ks'][0]:7.4f} {ev['ks'][1]:8.4f}   {rho:6.3f}")

    print("\n" + "=" * 70)
    print("A: does cross-excitation survive a drifting baseline?")
    print("=" * 70)
    for name in ("Hawkes", "Hawkes + mu(t)"):
        p = fits[name]
        matrix(f"[{name}]  branching alpha/beta", p.branching)
        matrix(f"[{name}]  half-life (s)", p.half_lives[0], "{:10.3f}")
    print(f"\nbaseline mu(t) across train blocks, BUY: "
          f"{fits['Hawkes + mu(t)'].mu[:, 0].min():.3f} .. {fits['Hawkes + mu(t)'].mu[:, 0].max():.3f} /s")

    print("\n" + "=" * 70)
    print("B: which timescales carry the excitation?  (branching per scale)")
    print("=" * 70)
    p = fits["Hawkes x5 + mu(t)"]
    print(f"{'half-life':>10}" + "".join(f"{f'{TYPES[i]}<-{TYPES[j]}':>12}" for i in range(2) for j in range(2)))
    for l, h in enumerate(HALF_LIVES):
        print(f"{h:>9.3f}s" + "".join(f"{p.branching_by_scale[l, i, j]:12.4f}" for i in range(2) for j in range(2)))
    print(f"{'total':>10}" + "".join(f"{p.branching[i, j]:12.4f}" for i in range(2) for j in range(2)))


if __name__ == "__main__":
    main()
