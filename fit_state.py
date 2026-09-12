"""Stage 3: does the book state an order lands in change what it triggers?

Three fits of the same Hawkes x5 + mu(t) model, differing only in how the
exciting type is defined:

    no state          e = side
    state             e = (side, L5-imbalance tercile at the order)
    shuffled state    e = (side, a random permutation of those terciles)

The shuffled fit has exactly the parameters of the state fit and none of the
information. Whatever held-out gain it shows is what extra parameters buy on
their own; the state must beat it to have said anything.

Tercile cuts are computed on train orders only. Spread is not a state
variable here: it is one tick 99.9% of the time on this capture.

    python fit_state.py [--root C:/tickforge-runs] [--date 2026-09-09] [--block-minutes 15]
"""

import argparse
import time

import numpy as np

from microflux.experiment import (
    HALF_LIVES, SCALES, TYPES, evaluate, load_book, load_orders, splits, with_book,
)
from microflux.events import events
from microflux.mle import extrapolate, fit_hawkes

STATES = ("ask-heavy", "balanced", "bid-heavy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/tickforge-runs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2026-09-09")
    ap.add_argument("--block-minutes", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    orders = with_book(load_orders(args.root, args.symbol, args.date), load_book(args.root, args.symbol, args.date))
    t, m, imb = orders["t"].to_numpy(), orders["m"].to_numpy(), orders["imb5"].to_numpy()
    T = t[-1]
    split = splits(T)
    T_train = split["train"][1]
    tr = t < T_train
    block = args.block_minutes * 60

    cuts = np.percentile(imb[tr], [100 / 3, 200 / 3])
    state = np.digitize(imb, cuts)
    shuffled = np.random.default_rng(args.seed).permutation(state)
    print(f"{len(t):,} orders with book state   train={tr.sum():,}   "
          f"imb5 tercile cuts (train) = {cuts[0]:+.3f}, {cuts[1]:+.3f}")
    for s, name in enumerate(STATES):
        n = int((state[tr] == s).sum())
        print(f"   {name:<10} n={n:>7,}   BUY share={100 * (m[tr][state[tr] == s] == 0).mean():5.1f}%")

    variants = {
        "no state":       events(t, m, 2),
        "state":          events(t, m, 2, state, 3),
        "shuffled state": events(t, m, 2, shuffled, 3),
    }
    fits, rows = {}, []
    for name, ev in variants.items():
        train = events(ev.t[tr], ev.m[tr], 2, None if ev.J == 2 else (ev.e[tr] // 2), ev.J // 2)
        t0 = time.perf_counter()
        p = fit_hawkes(train, T_train, block_s=block, scales=SCALES)
        secs = time.perf_counter() - t0
        fits[name] = p
        rows.append((name, evaluate(extrapolate(p, T_train), ev, split), secs))
        print(f"  fitted {name:<16} {secs:6.1f}s   params={p.mu.size + p.alpha.size}")

    print(f"\n{'model':<16}{'train':>9}{'val':>9}{'test':>9}   {'KS BUY':>7} {'KS SELL':>8}")
    print(f"{'':<16}{'NLL/event':>27}   {'test, Exp(1)':>16}")
    for name, ev_, _ in rows:
        print(f"{name:<16}{ev_['train']:9.4f}{ev_['val']:9.4f}{ev_['test']:9.4f}   "
              f"{ev_['ks'][0]:7.4f} {ev_['ks'][1]:8.4f}")
    base, real, shuf = (r[1]["test"] for r in rows)
    print(f"\nheld-out gain over no-state, test NLL/event:  state {base - real:+.4f}   shuffled {base - shuf:+.4f}")

    p = fits["state"]
    print("\n" + "=" * 70)
    print("branching by state of the EXCITING order   (row = excited, col = exciting)")
    print("=" * 70)
    for s, name in enumerate(STATES):
        print(f"\n[{name}]")
        print(f"{'':>8}{'<-BUY':>10}{'<-SELL':>10}")
        for i in range(2):
            print(f"{TYPES[i]:>8}" + "".join(f"{p.branching[i, j + 2 * s]:10.4f}" for j in range(2)))

    print("\n" + "=" * 70)
    print("where does the state act?  self-excitation branching per scale")
    print("=" * 70)
    for pair, (i, j) in (("BUY<-BUY", (0, 0)), ("SELL<-SELL", (1, 1))):
        print(f"\n{pair:<12}" + "".join(f"{s:>12}" for s in STATES))
        for l, h in enumerate(HALF_LIVES):
            print(f"{h:>9.3f}s   " + "".join(f"{p.branching_by_scale[l, i, j + 2 * s]:12.4f}" for s in range(3)))
        print(f"{'total':>10}   " + "".join(f"{p.branching[i, j + 2 * s]:12.4f}" for s in range(3)))


if __name__ == "__main__":
    main()
