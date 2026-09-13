"""Residual diagnostics of the working model, calibrated against simulation.

What structure is left once the marked, state-dependent, multi-scale Hawkes
has done its work? The time-rescaling residuals answer that -- but only
against a null that went through the same observation process: millisecond
ticks, same-tick merging, a model fitted rather than known. So every
statistic is reported twice: on the validation segment of the real capture,
and on synthetic data simulated from the fitted model, observed the same
way, refitted the same way.

Conditioning variables are features of the event that *opened* each
residual interval (see residuals.interval_covariates); nothing measured at
the arriving event is used.

Validation only. The test segment is exploratory from here on.

    python scripts/diagnose.py [--root C:/tickforge-runs] [--date 2026-09-09] [--replicates 2]
"""

import argparse
import time

import numpy as np

from microflux.events import Events, events
from microflux.experiment import (
    MARK_NAMES, SCALES, TYPES, imbalance_state, load_book, load_orders, mark_class, splits,
)
from microflux.hawkes import block_edges
from microflux.mle import extrapolate, fit_hawkes
from microflux.residuals import (
    autocorrelation, conditional_means, interval_covariates, jitter, ks_exp1, quantiles,
    rescaled_residuals,
)
from microflux.simulate import observe, simulate

STATES = ("ask-heavy", "balanced", "bid-heavy")
LAGS = (1, 2, 5, 10, 50, 100)
PROBS = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
COND = {
    "mark":    (np.array([1, 2, 3]),                 MARK_NAMES),
    "state":   (np.array([1, 2]),                    STATES),
    "burst":   (np.array([1, 2, 5, 10]),             ("0", "1", "2-4", "5-9", "10+")),
    "opp_gap": (np.array([0.01, 0.1, 1.0, 10.0]),    ("<10ms", "10-100ms", "0.1-1s", "1-10s", ">10s")),
    "hour":    (np.array([0.4, 0.8]),                ("first", "middle", "last")),
}


def diagnostics(p, ev: Events, a: float, b: float, seed: int) -> dict:
    """Every statistic, on jittered times, for the window [a, b)."""
    evj = jitter(ev, seed)
    res = rescaled_residuals(p, evj, a, b)
    cov = interval_covariates(evj, a, b)
    out = {"ks_raw": [ks_exp1(r) for r in rescaled_residuals(p, ev, a, b)],
           "ks": [ks_exp1(r) for r in res],
           "q": [quantiles(r, PROBS) for r in res],
           "acf": [autocorrelation(r, LAGS) for r in res],
           "n": [len(r) for r in res]}
    for name, (edges, _) in COND.items():
        out[name] = [conditional_means(r, c[name], edges) for r, c in zip(res, cov)]
    return out


def synthetic(p, ev: Events, T_train: float, T_end: float, mark_prob: np.ndarray, blocks, seed: int) -> dict:
    """Simulate from the fitted model, observe, refit, diagnose -- the null."""
    sim = observe(simulate(p, T_end, seed, ev.step_t, ev.step_s, mark_prob))
    q = fit_hawkes(sim.before(T_train), T_train, blocks, SCALES, state_baseline=True)
    return diagnostics(extrapolate(q, sim, T_train), sim, T_train, T_end, seed) | {"events": len(sim)}


def show_row(label: str, real: np.ndarray, null: np.ndarray, fmt: str = "{:8.3f}") -> None:
    print(f"{label:<14}" + "".join(fmt.format(v) for v in real) + "   |" + "".join(fmt.format(v) for v in null))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/tickforge-runs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2026-09-09")
    ap.add_argument("--block-minutes", type=float, default=15.0)
    ap.add_argument("--replicates", type=int, default=2)
    args = ap.parse_args()

    orders = load_orders(args.root, args.symbol, args.date)
    book = load_book(args.root, args.symbol, args.date)
    t, m = orders["t"].to_numpy(), orders["m"].to_numpy()
    split = splits(t[-1])
    T_train, T_val = split["train"][1], split["val"][1]
    blocks = block_edges(T_train, args.block_minutes * 60)
    step_t, step_s, _ = imbalance_state(orders, book, T_train)
    mark = mark_class(orders)
    ev = events(t, m, 2, step_t, step_s, 3, mark=mark, C=4)

    t0 = time.perf_counter()
    p = extrapolate(fit_hawkes(ev.before(T_train), T_train, blocks, SCALES, state_baseline=True), ev, T_train)
    print(f"working model fitted in {time.perf_counter() - t0:.0f}s   val window = [{T_train / 3600:.2f}h, {T_val / 3600:.2f}h)")
    real = diagnostics(p, ev, T_train, T_val, seed=0)

    tr = ev.before(T_train)
    mark_prob = np.array([np.bincount(tr.c[tr.m == i], minlength=4) / (tr.m == i).sum() for i in range(2)])
    nulls = []
    for r in range(args.replicates):
        t0 = time.perf_counter()
        nulls.append(synthetic(p, ev, T_train, T_val, mark_prob, blocks, seed=100 + r))
        print(f"null replicate {r}: {nulls[-1]['events']:,} events, {time.perf_counter() - t0:.0f}s")
    null = {k: np.mean([n[k] for n in nulls], axis=0) for k in real}

    print(f"\n{'':<14}{'REAL (validation)':^32}   |{'NULL (simulated, mean of ' + str(args.replicates) + ')':^32}")
    print(f"{'':<14}" + "".join(f"{s:>16}" for s in TYPES) + "   |" + "".join(f"{s:>16}" for s in TYPES))
    print("=" * 84)
    print("KS distance from Exp(1)")
    show_row("  raw ticks", real["ks_raw"], null["ks_raw"], "{:16.4f}")
    show_row("  jittered", real["ks"], null["ks"], "{:16.4f}")
    show_row("  n", real["n"], null["n"], "{:16,.0f}")

    for i, side in enumerate(TYPES):
        print(f"\n--- {side} ---   residual quantile / Exp(1) quantile")
        show_row("  prob", np.array(PROBS), np.array(PROBS), "{:8.2f}")
        show_row("  ratio", real["q"][i][:, 0] / real["q"][i][:, 1], null["q"][i][:, 0] / null["q"][i][:, 1])
        print(f"      autocorrelation at lag   (iid: ~N(0, 1/n), 2 s.e. = {2 / np.sqrt(real['n'][i]):.4f})")
        show_row("  lag", np.array(LAGS), np.array(LAGS), "{:8.0f}")
        show_row("  acf", real["acf"][i], null["acf"][i], "{:8.4f}")
        for name, (_, labels) in COND.items():
            print(f"      mean residual by {name} of the interval opener   (correct model: 1.00 +/- s.e.)")
            print(f"{'':<14}" + "".join(f"{l:>8}" for l in labels) + "   |" + "".join(f"{l:>8}" for l in labels))
            show_row("  mean", real[name][i][:, 0], null[name][i][:, 0])
            show_row("  s.e.", real[name][i][:, 1], null[name][i][:, 1])
            show_row("  n", real[name][i][:, 2], null[name][i][:, 2], "{:8.0f}")


if __name__ == "__main__":
    main()
