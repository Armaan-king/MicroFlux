"""Stage 4 pilot: does detailed event history add predictive value beyond
compact activity summaries?  Matched inputs (side, mark class, state).

    classical    M4 and E1' refit here; the better on validation is the reference
    MLP          summaries at 1 s .. 30 min + the last event, shared head
    attention    the last N event tokens + the same summaries, shared head

Same head (five fixed decay scales, exact piecewise integration across
book rows), same likelihood convention and windows as the classical
models, so NLL/event is comparable. Each neural model is trained from
several seeds; gains are reported against the classical reference with
block-bootstrap intervals at 60 / 300 / 900 s. Validation only; the test
segment of this session is exploratory.

Sizes are pilot defaults (PROTOCOL.md s.7): width 64, 2 layers, 4 heads,
N in {64, 128}; 256 and 512 are a follow-up. Each epoch draws 40k of the
121k training events so the validation check comes round often; float32
for the neural models. Nothing here identifies a mechanism; these are
predictive gains.

    python fit_neural.py [--seeds 3] [--horizons 64,128] [--epochs 40] [--targets-per-epoch 40000]
"""

import argparse
import time

import numpy as np
import torch

from microflux import neural
from microflux.events import events
from microflux.experiment import (
    HALF_LIVES, MARK_NAMES, SCALES, TYPES, block_loglik, gain_ci_from_blocks, imbalance_state,
    load_book, load_orders, mark_class, splits,
)
from microflux.hawkes import block_edges, loglik as hawkes_loglik
from microflux.mle import OPEN, extrapolate, fit_hawkes
from microflux.residuals import jitter, ks_exp1, rescaled_residuals

BLOCKS = (60.0, 300.0, 900.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/tickforge-runs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2026-09-09")
    ap.add_argument("--block-minutes", type=float, default=15.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--horizons", default="64,128")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--targets-per-epoch", type=int, default=40_000, help="training events drawn per epoch")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads() - 2))

    orders = load_orders(args.root, args.symbol, args.date)
    book = load_book(args.root, args.symbol, args.date)
    t, m = orders["t"].to_numpy(), orders["m"].to_numpy()
    T = t[-1]
    split = splits(T)
    T_train = split["train"][1]
    a_val, b_val = split["val"]
    n_val = int(((t >= a_val) & (t < b_val)).sum())
    step_t, step_s, _ = imbalance_state(orders, book, T_train)
    ev = events(t, m, 2, step_t, step_s, 3, mark=mark_class(orders), C=len(MARK_NAMES))
    evj = jitter(ev, 0)
    print(f"{len(ev):,} orders   train < {T_train / 3600:.2f}h   val [{a_val / 3600:.2f}h, {b_val / 3600:.2f}h) n={n_val:,}   "
          f"threads={torch.get_num_threads()}")

    # --- classical reference ---------------------------------------------------
    blocks = block_edges(T_train, args.block_minutes * 60)
    slow = np.log(2.0) / np.append(HALF_LIVES, [500.0, 5000.0])
    classical = {}
    for name, edges, scales in (("M4", blocks, SCALES), ("E1'", OPEN, slow)):
        t0 = time.perf_counter()
        p = extrapolate(fit_hawkes(ev.before(T_train), T_train, edges, scales, state_baseline=True), ev, T_train)
        v = -hawkes_loglik(p, ev, a_val, b_val) / n_val
        ks = [ks_exp1(r) for r in rescaled_residuals(p, evj, a_val, b_val)]
        classical[name] = (p, v, ks)
        print(f"  {name:<10} val NLL/event {v:8.4f}   val KS {ks[0]:.4f} / {ks[1]:.4f}   ({time.perf_counter() - t0:.0f}s)")
    ref_name = min(classical, key=lambda k: classical[k][1])
    p_ref, v_ref, _ = classical[ref_name]
    ref_blocks = {s: block_loglik(p_ref, ev, a_val, b_val, s) for s in BLOCKS}
    print(f"  reference = {ref_name}")

    # --- neural -----------------------------------------------------------------
    f = neural.features(ev, T_end=T)
    fj = neural.features(evj, T_end=T)
    n_sum = f.summary.shape[1]
    horizons = [int(x) for x in args.horizons.split(",")]
    runs = [("MLP", None)] + [("attention", N) for N in horizons]

    print(f"\n{'model':<16}{'seed':>5}{'val NLL':>9}{'gain vs ' + ref_name:>14}   {'CI 60 s':>18}   {'300 s':>18}   {'900 s':>18}   "
          f"{'val KS':>13}   {'epochs':>6} {'time':>7}")
    results = {}
    for kind, N in runs:
        label = kind if N is None else f"{kind} N={N}"
        gains = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            if kind == "MLP":
                enc, Nn = neural.SummaryMLP(n_sum, 2, len(MARK_NAMES), 3, d=args.width), 1
            else:
                enc, Nn = neural.Attention(n_sum, 2, len(MARK_NAMES), 3, d=args.width, heads=args.heads, layers=args.layers), N
            model = neural.TPP(enc, neural.Head(enc.d_out, 2, 3, SCALES, d_hidden=args.width))
            fit = neural.train(model, f, T_train, (a_val, b_val), Nn, epochs=args.epochs, seed=seed,
                               patience=args.patience, targets_per_epoch=args.targets_per_epoch,
                               log=(lambda s: None) if seed else print)
            cis = []
            for s in BLOCKS:
                ll_n, n = neural.block_loglik(model, f, a_val, b_val, Nn, s)
                cis.append(gain_ci_from_blocks(ll_n - ref_blocks[s][0], n))
            ks = [ks_exp1(r) for r in neural.rescaled_residuals(model, fj, a_val, b_val, Nn)]
            gains.append(cis[0][0])
            print(f"{label:<16}{seed:>5}{fit['val']:9.4f}{cis[0][0]:+14.4f}   "
                  + "   ".join(f"[{lo:+.4f}, {hi:+.4f}]" for _, lo, hi in cis)
                  + f"   {ks[0]:.4f}/{ks[1]:.4f}   {fit['epochs']:>6} {fit['seconds']:6.0f}s")
        results[label] = gains
        print(f"{label:<16}{'mean':>5}{'':>9}{np.mean(gains):+14.4f}   sd {np.std(gains, ddof=1) if len(gains) > 1 else 0:.4f}")

    print("\nattention minus MLP, mean over seeds:")
    for label, g in results.items():
        if label != "MLP":
            print(f"   {label:<16} {np.mean(g) - np.mean(results['MLP']):+.4f}")


if __name__ == "__main__":
    main()
