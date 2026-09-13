"""The two classical extensions the diagnostics support, against the benchmark.

diagnose.py found, on validation: residual mean ~1.10 and rising through
the window (the frozen train-mean baseline over-predicts a quieter period
-- a rate-tracking failure), and residual autocorrelation 0.08-0.10 at
lags 1-2 dying by lag 50 (unmodelled clustering at the 0.2-2 s scale). It
found no saturation and no opposite-side inhibition beyond the null.

    E1  slow scales     + 500 s and 5000 s half-lives: recent activity level
                        predicts the near future, so the baseline is tracked
                        out of sample instead of frozen
    E2  dense grid      ten half-decade scales 5 ms .. 160 s: room for
                        structure between the decade grid points

Both stay concave. Pre-registered here before running. Validation only.

    python fit_extend.py [--root C:/tickforge-runs] [--date 2026-09-09]
"""

import argparse
import time

import numpy as np

from microflux.events import events
from microflux.experiment import (
    HALF_LIVES, MARK_NAMES, TYPES, evaluate, gain_ci, imbalance_state, load_book, load_orders,
    mark_class, splits,
)
from microflux.hawkes import block_edges
from microflux.mle import extrapolate, fit_hawkes
from microflux.residuals import conditional_means, interval_covariates, jitter, rescaled_residuals

GRIDS = {
    "benchmark  x5":     HALF_LIVES,
    "E1  x7 slow":       np.append(HALF_LIVES, [500.0, 5000.0]),
    "E2  x10 dense":     np.array([0.005, 0.016, 0.05, 0.16, 0.5, 1.6, 5.0, 16.0, 50.0, 160.0]),
    "E1+E2  x12":        np.array([0.005, 0.016, 0.05, 0.16, 0.5, 1.6, 5.0, 16.0, 50.0, 160.0, 500.0, 5000.0]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/tickforge-runs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2026-09-09")
    ap.add_argument("--block-minutes", type=float, default=15.0)
    args = ap.parse_args()

    orders = load_orders(args.root, args.symbol, args.date)
    book = load_book(args.root, args.symbol, args.date)
    t, m = orders["t"].to_numpy(), orders["m"].to_numpy()
    split = splits(t[-1])
    T_train = split["train"][1]
    a_val, b_val = split["val"]
    blocks = block_edges(T_train, args.block_minutes * 60)
    step_t, step_s, _ = imbalance_state(orders, book, T_train)
    ev = events(t, m, 2, step_t, step_s, 3, mark=mark_class(orders), C=len(MARK_NAMES))
    evj = jitter(ev, 0)

    fits = {}
    print(f"{'model':<18}{'val NLL/event':>14}   {'gain over benchmark':>20}   {'95% CI':>18}   {'KS BUY':>7} {'KS SELL':>8}   "
          f"{'resid mean by hour (BUY)':>26}")
    for name, hl in GRIDS.items():
        t0 = time.perf_counter()
        p = extrapolate(fit_hawkes(ev.before(T_train), T_train, blocks, np.log(2.0) / hl, state_baseline=True), ev, T_train)
        secs = time.perf_counter() - t0
        fits[name] = p
        r = evaluate(p, ev, split)
        ref = fits["benchmark  x5"]
        g, lo, hi = gain_ci(ref, p, ev, ev, a_val, b_val)
        res = rescaled_residuals(p, evj, a_val, b_val)
        cov = interval_covariates(evj, a_val, b_val)
        by_hour = conditional_means(res[0], cov[0]["hour"], np.array([0.4, 0.8]))[:, 0]
        print(f"{name:<18}{r['val']:14.4f}   {g:+20.4f}   [{lo:+.4f}, {hi:+.4f}]   {r['ks'][0]:7.4f} {r['ks'][1]:8.4f}   "
              f"{by_hour[0]:8.3f} {by_hour[1]:8.3f} {by_hour[2]:8.3f}   ({secs:.0f}s)")

    p = fits["E1  x7 slow"]
    print("\nE1: branching at the slow scales  (row = excited, col = exciting side, summed over state and mark)")
    for l, h in ((5, 500.0), (6, 5000.0)):
        br = p.branching_by_scale[l].reshape(2, -1, 2).sum(1)  # (K, K): sum over (state, mark) of the exciting side
        print(f"   {h:>6.0f}s  " + "  ".join(f"{TYPES[i]}<-{TYPES[j]} {br[i, j]:.4f}" for i in range(2) for j in range(2)))


if __name__ == "__main__":
    main()
