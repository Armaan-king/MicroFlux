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
    E1' slow, no blocks E1 scales with a single time block per state, so the
                        level is carried by excitation rather than frozen
                        (pre-registered 2026-09-13 after D11)

All stay concave. Validation only. The decomposition splits each model's
predicted validation count into baseline and excitation, against the
observed count: over-prediction is a floor problem only if the observed
count is below the baseline alone.

    python scripts/fit_extend.py [--root C:/tickforge-runs] [--date 2026-09-09]
"""

import argparse
import time

import numpy as np

from microflux.events import events
from microflux.experiment import (
    HALF_LIVES, MARK_NAMES, TYPES, evaluate, gain_ci_blocks, imbalance_state, load_book,
    load_orders, mark_class, splits,
)
from microflux.hawkes import baseline_cum, block_edges, compensator
from microflux.mle import OPEN, extrapolate, fit_hawkes
from microflux.residuals import conditional_means, interval_covariates, jitter, rescaled_residuals

SLOW = np.append(HALF_LIVES, [500.0, 5000.0])
DENSE = np.array([0.005, 0.016, 0.05, 0.16, 0.5, 1.6, 5.0, 16.0, 50.0, 160.0])
GRIDS = {  # name: (half-lives, blocks or None for a single block)
    "benchmark  x5":     (HALF_LIVES, True),
    "E1  x7 slow":       (SLOW, True),
    "E2  x10 dense":     (DENSE, True),
    "E1+E2  x12":        (np.append(DENSE, [500.0, 5000.0]), True),
    "E1' x7 no blocks":  (SLOW, False),
}


def decompose(p, ev, a: float, b: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed, predicted-baseline and predicted-excitation counts per side on [a, b)."""
    observed = np.bincount(ev.m[ev.window(a, b)], minlength=ev.K).astype(float)
    base = (baseline_cum(p, ev, np.array([b])) - baseline_cum(p, ev, np.array([a])))[0]
    total = compensator(p, ev, a, b)
    return observed, base, total - base


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
    print(f"{'model':<18}{'val NLL/event':>14}   {'gain over benchmark':>20}   {'95% CI, 60 s':>18}   {'300 s':>18}   {'900 s':>18}   "
          f"{'val KS BUY':>11} {'SELL':>7}   {'resid mean by hour (BUY)':>26}")
    for name, (hl, blocked) in GRIDS.items():
        t0 = time.perf_counter()
        edges = blocks if blocked else OPEN
        p = extrapolate(fit_hawkes(ev.before(T_train), T_train, edges, np.log(2.0) / hl, state_baseline=True), ev, T_train)
        secs = time.perf_counter() - t0
        fits[name] = p
        r = evaluate(p, ev, split, ks_on="val")
        ref = fits["benchmark  x5"]
        cis = gain_ci_blocks(ref, p, ev, ev, a_val, b_val)
        g, lo, hi = cis[0][1:]
        res = rescaled_residuals(p, evj, a_val, b_val)
        cov = interval_covariates(evj, a_val, b_val)
        by_hour = conditional_means(res[0], cov[0]["hour"], np.array([0.4, 0.8]))[:, 0]
        print(f"{name:<18}{r['val']:14.4f}   {g:+20.4f}   " + "   ".join(f"[{lo:+.4f}, {hi:+.4f}]" for _, _, lo, hi in cis)
              + f"   {r['ks'][0]:11.4f} {r['ks'][1]:7.4f}   {by_hour[0]:8.3f} {by_hour[1]:8.3f} {by_hour[2]:8.3f}   ({secs:.0f}s)")

    print("\ndecomposition of the predicted validation count   (observed | baseline + excitation = predicted)")
    print(f"{'model':<18}" + "".join(f"{s + ' obs':>10}{'base':>9}{'excit':>9}{'pred':>9}   " for s in TYPES))
    for name, p in fits.items():
        obs, base, exc = decompose(p, ev, a_val, b_val)
        print(f"{name:<18}" + "".join(f"{obs[i]:10.0f}{base[i]:9.0f}{exc[i]:9.0f}{base[i] + exc[i]:9.0f}   " for i in range(2)))

    for name in ("E1  x7 slow", "E1' x7 no blocks"):
        p = fits[name]
        print(f"\n[{name}] branching at the slow scales  (row = excited, col = exciting side, summed over state and mark)")
        for l, h in ((5, 500.0), (6, 5000.0)):
            br = p.branching_by_scale[l].reshape(2, -1, 2).sum(1)  # (K, K): sum over (state, mark) of the exciting side
            print(f"   {h:>6.0f}s  " + "  ".join(f"{TYPES[i]}<-{TYPES[j]} {br[i, j]:.4f}" for i in range(2) for j in range(2)))
        print(f"   baseline mu (train-mean per state, BUY): " + "  ".join(f"{v:.3f}" for v in p.mu[-1, :, 0]) + " /s")


if __name__ == "__main__":
    main()
