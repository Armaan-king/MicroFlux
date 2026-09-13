"""Stage 3: does the book state an order lands in change what it triggers --
or does it only change how many orders arrive?

State = L5-imbalance tercile of the last book row at or before the order, as
a step function of time. Seven fits, differing only in where the state enters:

    Poisson + mu(t)                      no state
    Poisson + mu(t,s)                    state changes the rate, nothing propagates
    Hawkes x5 + mu(t)                    D5: no state
    Hawkes x5 + mu(t,s)                  state in the baseline only
    Hawkes x5 + mu(t)   + kernel(s)      D6: state in the kernel only -- the confounded model
    Hawkes x5 + mu(t,s) + kernel(s)      both free: the fit decides what the state does
    ... same, state series shifted 2 h   control: same parameters, no information

Tercile cuts come from train book rows only. Spread is not a state variable
here: it is one tick 99.9% of the time on this capture.

    python scripts/fit_state.py [--root C:/tickforge-runs] [--date 2026-09-09] [--block-minutes 15]
"""

import argparse
import time

import numpy as np

from microflux.events import events
from microflux.experiment import (
    HALF_LIVES, SCALES, TYPES, evaluate, imbalance_state, load_book, load_orders, splits,
)
from microflux.hawkes import block_edges
from microflux.mle import extrapolate, fit_hawkes, fit_poisson

STATES = ("ask-heavy", "balanced", "bid-heavy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/tickforge-runs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2026-09-09")
    ap.add_argument("--block-minutes", type=float, default=15.0)
    ap.add_argument("--shift-hours", type=float, default=2.0)
    args = ap.parse_args()

    orders = load_orders(args.root, args.symbol, args.date)
    book = load_book(args.root, args.symbol, args.date)
    t, m = orders["t"].to_numpy(), orders["m"].to_numpy()
    T = t[-1]
    split = splits(T)
    T_train = split["train"][1]
    blocks = block_edges(T_train, args.block_minutes * 60)

    step_t, step_s, cuts = imbalance_state(orders, book, T_train)
    shifted = np.roll(step_s, int(args.shift_hours * 3600))  # book rows are 1 Hz

    def seq(kernel_state: bool, states=step_s):
        return events(t, m, 2, step_t, states, 3, kernel_state=kernel_state)

    ev_flat, ev_state, ev_shift = seq(False), seq(True), seq(True, shifted)
    tr = ev_state.before(T_train)
    print(f"{len(t):,} orders over {T / 3600:.2f}h   train={len(tr):,}   "
          f"imb5 tercile cuts (train book rows) = {cuts[0]:+.3f}, {cuts[1]:+.3f}")
    for s, name in enumerate(STATES):
        n = int((tr.s == s).sum())
        print(f"   {name:<10} n={n:>7,}   BUY share={100 * (tr.m[tr.s == s] == 0).mean():5.1f}%")

    ladder = [
        ("Poisson + mu(t)",            ev_flat,  lambda ev: fit_poisson(ev, T_train, blocks)),
        ("Poisson + mu(t,s)",          ev_flat,  lambda ev: fit_poisson(ev, T_train, blocks, state_baseline=True)),
        ("Hawkes x5 + mu(t)",          ev_flat,  lambda ev: fit_hawkes(ev, T_train, blocks, SCALES)),
        ("Hawkes x5 + mu(t,s)",        ev_flat,  lambda ev: fit_hawkes(ev, T_train, blocks, SCALES, state_baseline=True)),
        ("Hawkes x5 + mu(t) + k(s)",   ev_state, lambda ev: fit_hawkes(ev, T_train, blocks, SCALES)),
        ("Hawkes x5 + mu(t,s) + k(s)", ev_state, lambda ev: fit_hawkes(ev, T_train, blocks, SCALES, state_baseline=True)),
        ("  ... shifted state",        ev_shift, lambda ev: fit_hawkes(ev, T_train, blocks, SCALES, state_baseline=True)),
    ]
    fits, rows = {}, []
    for name, ev, fit in ladder:
        t_start = time.perf_counter()
        p = fit(ev.before(T_train))
        secs = time.perf_counter() - t_start
        fits[name] = p
        rows.append((name, evaluate(extrapolate(p, ev, T_train), ev, split, ks_on="val")))
        print(f"  fitted {name:<28} {secs:6.1f}s   params={p.mu.size + p.alpha.size}")

    print(f"\n{'model':<28}{'train':>9}{'val':>9}{'test':>9}   {'val KS BUY':>11} {'SELL':>7}")
    print(f"{'':<28}{'NLL/event':>27}   {'test, Exp(1)':>16}")
    for name, r in rows:
        print(f"{name:<28}{r['train']:9.4f}{r['val']:9.4f}{r['test']:9.4f}   {r['ks'][0]:11.4f} {r['ks'][1]:7.4f}")
    base = rows[2][1]["test"]
    print(f"\nheld-out gain over Hawkes x5 + mu(t), test NLL/event:")
    for name, r in rows[3:]:
        print(f"   {name:<28} {base - r['test']:+.4f}")

    print("\n" + "=" * 72)
    print("A: self-excitation by state of the exciting order  --  kernel-only vs both free")
    print("=" * 72)
    print(f"{'':<12}{'ask-heavy':>12}{'balanced':>12}{'bid-heavy':>12}    {'range':>8}")
    for name in ("Hawkes x5 + mu(t) + k(s)", "Hawkes x5 + mu(t,s) + k(s)"):
        p = fits[name]
        print(f"[{name}]")
        for pair, (i, j) in (("BUY<-BUY", (0, 0)), ("SELL<-SELL", (1, 1))):
            v = [p.branching[i, j + 2 * s] for s in range(3)]
            print(f"{pair:<12}" + "".join(f"{x:12.4f}" for x in v) + f"    {max(v) - min(v):8.4f}")

    p = fits["Hawkes x5 + mu(t,s) + k(s)"]
    print("\n" + "=" * 72)
    print("B: exogenous rate by state  --  train-mean of mu(t,s), orders/s")
    print("=" * 72)
    print(f"{'':<12}{'ask-heavy':>12}{'balanced':>12}{'bid-heavy':>12}")
    for i in range(2):
        print(f"{TYPES[i]:<12}" + "".join(f"{p.mu[:-1, s, i].mean() if p.mu.shape[0] > 1 else p.mu[0, s, i]:12.4f}" for s in range(3)))

    print("\n" + "=" * 72)
    print("C: where the state acts  --  self-excitation branching per scale, both free")
    print("=" * 72)
    for pair, (i, j) in (("BUY<-BUY", (0, 0)), ("SELL<-SELL", (1, 1))):
        print(f"\n{pair:<12}" + "".join(f"{s:>12}" for s in STATES))
        for l, h in enumerate(HALF_LIVES):
            print(f"{h:>9.3f}s   " + "".join(f"{p.branching_by_scale[l, i, j + 2 * s]:12.4f}" for s in range(3)))
        print(f"{'total':>10}   " + "".join(f"{p.branching[i, j + 2 * s]:12.4f}" for s in range(3)))


if __name__ == "__main__":
    main()
