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

from microflux.experiment import HALF_LIVES, SCALES, TYPES, evaluate, load_orders, matrix, splits
from microflux.events import events
from microflux.hawkes import Params, block_edges
from microflux.mle import extrapolate, fit_hawkes, fit_poisson


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/tickforge-runs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2026-09-09")
    ap.add_argument("--minutes", type=float, default=None, help="use only the first N minutes")
    ap.add_argument("--block-minutes", type=float, default=15.0, help="baseline block width")
    args = ap.parse_args()

    orders = load_orders(args.root, args.symbol, args.date, args.minutes)
    ev = events(orders["t"].to_numpy(), orders["m"].to_numpy(), K=2)
    T = ev.t[-1]
    split = splits(T)
    T_train = split["train"][1]
    tr = ev.t < T_train
    train = ev.before(T_train)
    blocks = block_edges(T_train, args.block_minutes * 60)
    print(f"{len(ev):,} orders over {T / 3600:.2f}h   train={tr.sum():,}  "
          f"BUY={int((ev.m == 0).sum()):,} SELL={int((ev.m == 1).sum()):,}   "
          f"baseline blocks of {args.block_minutes:g} min")

    ladder = [
        ("Poisson",           lambda: fit_poisson(train, T_train)),
        ("Poisson + mu(t)",   lambda: fit_poisson(train, T_train, blocks)),
        ("Hawkes",            lambda: fit_hawkes(train, T_train)),
        ("Hawkes + mu(t)",    lambda: fit_hawkes(train, T_train, blocks)),
        ("Hawkes x5",         lambda: fit_hawkes(train, T_train, scales=SCALES)),
        ("Hawkes x5 + mu(t)", lambda: fit_hawkes(train, T_train, blocks, SCALES)),
    ]

    fits: dict[str, Params] = {}
    rows = []
    for name, fit in ladder:
        t0 = time.perf_counter()
        p = fit()
        secs = time.perf_counter() - t0
        fits[name] = p
        rows.append((name, evaluate(extrapolate(p, ev, T_train), ev, split), p.spectral_radius, secs))
        print(f"  fitted {name:<20} {secs:6.1f}s")

    print(f"\n{'model':<20}{'train':>9}{'val':>9}{'test':>9}   {'KS BUY':>7} {'KS SELL':>8}   {'rho':>6}")
    print(f"{'':<20}{'NLL/event':>27}   {'test, Exp(1)':>16}")
    for name, ev_, rho, _ in rows:
        print(f"{name:<20}{ev_['train']:9.4f}{ev_['val']:9.4f}{ev_['test']:9.4f}   "
              f"{ev_['ks'][0]:7.4f} {ev_['ks'][1]:8.4f}   {rho:6.3f}")

    print("\n" + "=" * 70)
    print("A: does cross-excitation survive a drifting baseline?")
    print("=" * 70)
    for name in ("Hawkes", "Hawkes + mu(t)"):
        p = fits[name]
        matrix(f"[{name}]  branching alpha/beta", p.branching)
        matrix(f"[{name}]  half-life (s)", p.half_lives[0], fmt="{:10.3f}")
    p = fits["Hawkes + mu(t)"]
    print(f"\nbaseline mu(t) across train blocks, BUY: {p.mu[:, 0, 0].min():.3f} .. {p.mu[:, 0, 0].max():.3f} /s")

    print("\n" + "=" * 70)
    print("B: which timescales carry the excitation?  (branching per scale)")
    print("=" * 70)
    p = fits["Hawkes x5 + mu(t)"]
    pairs = [(i, j) for i in range(2) for j in range(2)]
    print(f"{'half-life':>10}" + "".join(f"{TYPES[i] + '<-' + TYPES[j]:>12}" for i, j in pairs))
    for l, h in enumerate(HALF_LIVES):
        print(f"{h:>9.3f}s" + "".join(f"{p.branching_by_scale[l, i, j]:12.4f}" for i, j in pairs))
    print(f"{'total':>10}" + "".join(f"{p.branching[i, j]:12.4f}" for i, j in pairs))


if __name__ == "__main__":
    main()
