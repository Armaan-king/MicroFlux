"""Stage 3b: does the fill count of an order predict what it triggers?

Mark = fill count of the aggressive order, in classes 1 / 2-4 / 5-19 / 20+.
The mark class of the *exciting* order widens its exciting type, exactly as
the book state does, so the shape of the effect is read off rather than
assumed. Fills are trade rows in a same-millisecond same-side run -- a proxy
for liquidity consumed (see audit_marks.py).

Gains are reported on validation with 95% block-bootstrap intervals (60 s
blocks). The control shuffles marks within (split, side) so the label keeps
its split and its side-conditional distribution and loses only its timing;
several seeds give its spread. The test segment is exploratory.

    python fit_marks.py [--root C:/tickforge-runs] [--date 2026-09-09] [--seeds 5]
"""

import argparse
import time

import numpy as np

from microflux.events import events
from microflux.experiment import (
    HALF_LIVES, MARK_NAMES, SCALES, TYPES, evaluate, gain_ci, imbalance_state, load_book,
    load_orders, mark_class, shuffle_within, split_id, splits,
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
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    orders = load_orders(args.root, args.symbol, args.date)
    book = load_book(args.root, args.symbol, args.date)
    t, m = orders["t"].to_numpy(), orders["m"].to_numpy()
    split = splits(t[-1])
    T_train = split["train"][1]
    a_val, b_val = split["val"]
    blocks = block_edges(T_train, args.block_minutes * 60)
    step_t, step_s, _ = imbalance_state(orders, book, T_train)
    mark = mark_class(orders)
    C = len(MARK_NAMES)
    groups = split_id(t, split) * 2 + m  # (split, side)

    tr = t < T_train
    print(f"{len(t):,} orders over {t[-1] / 3600:.2f}h   train={tr.sum():,}   val=[{a_val / 3600:.2f}h, {b_val / 3600:.2f}h)")
    for c, name in enumerate(MARK_NAMES):
        k = tr & (mark == c)
        print(f"   {name:<8} n={k.sum():>7,}   share={100 * k.sum() / tr.sum():5.1f}%   "
              f"BUY share={100 * (m[k] == 0).mean():5.1f}%")

    def fit(ev, **kw):
        t0 = time.perf_counter()
        p = extrapolate(fit_hawkes(ev.before(T_train), T_train, blocks, SCALES, **kw), ev, T_train)
        return p, time.perf_counter() - t0

    ev_flat = events(t, m, 2)
    ev_mark = events(t, m, 2, mark=mark, C=C)
    ev_state = events(t, m, 2, step_t, step_s, 3)
    ev_both = events(t, m, 2, step_t, step_s, 3, mark=mark, C=C)
    p_flat, s0 = fit(ev_flat)
    p_mark, s1 = fit(ev_mark)
    p_state, s2 = fit(ev_state, state_baseline=True)
    p_both, s3 = fit(ev_both, state_baseline=True)
    print(f"\n  fitted: flat {s0:.0f}s   marks {s1:.0f}s   state {s2:.0f}s   both {s3:.0f}s")

    print(f"\n{'model':<30}{'val NLL/event':>14}   {'gain over flat':>15}   {'95% CI':>18}   {'KS BUY':>7} {'KS SELL':>8}")
    rows = [("Hawkes x5 + mu(t)", p_flat, ev_flat), ("Hawkes x5 + mu(t) + k(c)", p_mark, ev_mark),
            ("Hawkes x5 + mu(t,s) + k(s)", p_state, ev_state), ("Hawkes x5 + mu(t,s) + k(s,c)", p_both, ev_both)]
    for name, p, ev in rows:
        r = evaluate(p, ev, split)
        g, lo, hi = gain_ci(p_flat, p, ev_flat, ev, a_val, b_val)
        print(f"{name:<30}{r['val']:14.4f}   {g:+15.4f}   [{lo:+.4f}, {hi:+.4f}]   {r['ks'][0]:7.4f} {r['ks'][1]:8.4f}")

    print(f"\ncontrol: marks shuffled within (split, side), {args.seeds} seeds")
    ctrl = []
    for seed in range(args.seeds):
        ev_sh = events(t, m, 2, mark=shuffle_within(mark, groups, seed), C=C)
        p_sh, _ = fit(ev_sh)
        g, lo, hi = gain_ci(p_flat, p_sh, ev_flat, ev_sh, a_val, b_val)
        ctrl.append(g)
        print(f"   seed {seed}: gain {g:+.4f}   [{lo:+.4f}, {hi:+.4f}]")
    print(f"   mean {np.mean(ctrl):+.4f}   sd {np.std(ctrl, ddof=1):.4f}")

    p = p_mark
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

    p = p_both
    print("\n" + "=" * 72)
    print("C: state x mark  --  BUY<-BUY branching at 5 ms (row = state, col = mark)")
    print("=" * 72)
    print(f"{'':<12}" + "".join(f"{n:>10}" for n in MARK_NAMES))
    for s, sname in enumerate(STATES):
        print(f"{sname:<12}" + "".join(f"{p.branching_by_scale[0, 0, 2 * (s + 3 * c)]:10.4f}" for c in range(C)))


if __name__ == "__main__":
    main()
