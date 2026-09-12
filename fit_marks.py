"""Stage 3b: does an order that swept deeper excite more?

Mark = fill count of the aggressive order, in classes 1 / 2-4 / 5-19 / 20+.
The mark class of the *exciting* order widens its exciting type, exactly as
the book state does, so the shape of the effect is read off rather than
assumed. Five fits:

    Hawkes x5 + mu(t)                        D5: unmarked
    Hawkes x5 + mu(t) + k(c)                 marked kernel
    ... shuffled marks                       control: same parameters, no information
    Hawkes x5 + mu(t,s) + k(s)               D7: state, unmarked
    Hawkes x5 + mu(t,s) + k(s, c)            state and mark together

The likelihood is for times and sides given the marks. A full marked
likelihood would add a mark density; with marks iid given side it does not
depend on the kernel parameters and cannot change the fit.

    python fit_marks.py [--root C:/tickforge-runs] [--date 2026-09-09] [--block-minutes 15]
"""

import argparse
import time

import numpy as np

from microflux.events import events
from microflux.experiment import (
    HALF_LIVES, MARK_NAMES, SCALES, TYPES, evaluate, imbalance_state, load_book, load_orders,
    mark_class, splits,
)
from microflux.hawkes import block_edges
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

    orders = load_orders(args.root, args.symbol, args.date)
    book = load_book(args.root, args.symbol, args.date)
    t, m = orders["t"].to_numpy(), orders["m"].to_numpy()
    T = t[-1]
    split = splits(T)
    T_train = split["train"][1]
    blocks = block_edges(T_train, args.block_minutes * 60)
    step_t, step_s, _ = imbalance_state(orders, book, T_train)
    mark = mark_class(orders)
    shuffled = np.random.default_rng(args.seed).permutation(mark)
    C = len(MARK_NAMES)

    tr = t < T_train
    print(f"{len(t):,} orders over {T / 3600:.2f}h   train={tr.sum():,}")
    for c, name in enumerate(MARK_NAMES):
        k = tr & (mark == c)
        print(f"   {name:<8} n={k.sum():>7,}   share={100 * k.sum() / tr.sum():5.1f}%   "
              f"BUY share={100 * (m[k] == 0).mean():5.1f}%")

    variants = {
        "Hawkes x5 + mu(t)":             (events(t, m, 2), {}),
        "Hawkes x5 + mu(t) + k(c)":      (events(t, m, 2, mark=mark, C=C), {}),
        "  ... shuffled marks":          (events(t, m, 2, mark=shuffled, C=C), {}),
        "Hawkes x5 + mu(t,s) + k(s)":    (events(t, m, 2, step_t, step_s, 3), {"state_baseline": True}),
        "Hawkes x5 + mu(t,s) + k(s,c)":  (events(t, m, 2, step_t, step_s, 3, mark=mark, C=C), {"state_baseline": True}),
    }
    fits, rows = {}, []
    for name, (ev, kw) in variants.items():
        t_start = time.perf_counter()
        p = fit_hawkes(ev.before(T_train), T_train, blocks, SCALES, **kw)
        secs = time.perf_counter() - t_start
        fits[name] = (p, ev)
        rows.append((name, evaluate(extrapolate(p, ev, T_train), ev, split)))
        print(f"  fitted {name:<30} {secs:6.1f}s   params={p.mu.size + p.alpha.size}   J={ev.J}")

    print(f"\n{'model':<30}{'train':>9}{'val':>9}{'test':>9}   {'KS BUY':>7} {'KS SELL':>8}")
    print(f"{'':<30}{'NLL/event':>27}   {'test, Exp(1)':>16}")
    for name, r in rows:
        print(f"{name:<30}{r['train']:9.4f}{r['val']:9.4f}{r['test']:9.4f}   {r['ks'][0]:7.4f} {r['ks'][1]:8.4f}")
    base = rows[0][1]["test"]
    print(f"\nheld-out gain over Hawkes x5 + mu(t), test NLL/event:")
    for name, r in rows[1:]:
        print(f"   {name:<30} {base - r['test']:+.4f}")

    p, ev = fits["Hawkes x5 + mu(t) + k(c)"]
    print("\n" + "=" * 72)
    print("A: self-excitation by mark of the exciting order  (branching, all scales)")
    print("=" * 72)
    print(f"{'':<12}" + "".join(f"{n:>10}" for n in MARK_NAMES))
    for pair, (i, j) in (("BUY<-BUY", (0, 0)), ("SELL<-SELL", (1, 1)), ("SELL<-BUY", (1, 0)), ("BUY<-SELL", (0, 1))):
        print(f"{pair:<12}" + "".join(f"{p.branching[i, j + 2 * c]:10.4f}" for c in range(C)))

    print("\n" + "=" * 72)
    print("B: where the mark acts  --  BUY<-BUY branching per scale")
    print("=" * 72)
    print(f"{'half-life':>10}" + "".join(f"{n:>10}" for n in MARK_NAMES))
    for l, h in enumerate(HALF_LIVES):
        print(f"{h:>9.3f}s" + "".join(f"{p.branching_by_scale[l, 0, 2 * c]:10.4f}" for c in range(C)))

    p, ev = fits["Hawkes x5 + mu(t,s) + k(s,c)"]
    print("\n" + "=" * 72)
    print("C: state x mark  --  BUY<-BUY branching at 5 ms (row = state, col = mark)")
    print("=" * 72)
    print(f"{'':<12}" + "".join(f"{n:>10}" for n in MARK_NAMES))
    for s, sname in enumerate(STATES):
        print(f"{sname:<12}" + "".join(f"{p.branching_by_scale[0, 0, 2 * (s + 3 * c)]:10.4f}" for c in range(C)))
    print("\n   ... and at 5 s")
    print(f"{'':<12}" + "".join(f"{n:>10}" for n in MARK_NAMES))
    for s, sname in enumerate(STATES):
        print(f"{sname:<12}" + "".join(f"{p.branching_by_scale[3, 0, 2 * (s + 3 * c)]:10.4f}" for c in range(C)))


if __name__ == "__main__":
    main()
