"""Stage 4: the matched-input neural comparison, with artifacts.

    classical    M4 and E1' fit here; the better on validation is the reference
    MLP          activity summaries at 1 s .. 30 min + the last event, shared head
    attention    the last N = 64 event tokens + the same summaries, shared head

Same head (five fixed decay scales, exact piecewise integration across
book rows), same likelihood convention and windows as the classical models.
Five predetermined seeds (0..4); patience 8, minimum 10 and maximum 40
epochs, fixed before starting. Every run writes config, learning curve,
resumable checkpoint, result and per-block log-likelihood contributions to
runs/<session>/<model>_seed<k>/.

Reported, each with paired block-bootstrap intervals at 60 / 300 / 900 s:
MLP vs reference, attention vs reference, attention vs MLP (same seed,
same blocks). Seed-to-seed variability is reported separately as the s.d.
of validation NLL. Validation only; the test segment is scored by a
separate final-test stage under PROTOCOL.md, never here.

    python fit_neural.py [--date 2026-09-09] [--root C:/tickforge-runs] [--seeds 0,1,2,3,4]
"""

import argparse
import json
import time
from pathlib import Path

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
from microflux.runs import Run, code_revision, data_identity

BLOCKS = (60.0, 300.0, 900.0)
N_TOKENS = 64


def ci_str(g):
    return f"{g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/tickforge-runs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2026-09-09")
    ap.add_argument("--session", default=None, help="run-directory name; default <symbol>-<date>")
    ap.add_argument("--block-minutes", type=float, default=15.0)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--targets-per-epoch", type=int, default=40_000)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads() - 2))
    session = args.session or f"{args.symbol}-{args.date}"
    out = Path(args.runs) / session
    seeds = [int(x) for x in args.seeds.split(",")]
    rev = code_revision(Path(__file__).parent)

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
    identity = data_identity(args.root, args.symbol, args.date, orders["timestamp_ns"].to_numpy(), split)
    print(f"{session}: {len(ev):,} orders over {T / 3600:.2f}h   train < {T_train / 3600:.2f}h   "
          f"val [{a_val / 3600:.2f}h, {b_val / 3600:.2f}h) n={n_val:,}   code {rev}")

    # --- classical reference ---------------------------------------------------
    blocks = block_edges(T_train, args.block_minutes * 60)
    slow = np.log(2.0) / np.append(HALF_LIVES, [500.0, 5000.0])
    classical = {}
    for name, edges, scales in (("M4", blocks, SCALES), ("E1'", OPEN, slow)):
        t0 = time.perf_counter()
        p = extrapolate(fit_hawkes(ev.before(T_train), T_train, edges, scales, state_baseline=True), ev, T_train)
        v = -hawkes_loglik(p, ev, a_val, b_val) / n_val
        ks = [ks_exp1(r) for r in rescaled_residuals(p, evj, a_val, b_val)]
        secs = time.perf_counter() - t0
        classical[name] = {"p": p, "val": v, "ks": ks, "seconds": secs}
        print(f"  {name:<10} val NLL/event {v:8.4f}   val KS {ks[0]:.4f} / {ks[1]:.4f}   ({secs:.0f}s)")
    ref_name = min(classical, key=lambda k: classical[k]["val"])
    ref = classical[ref_name]
    ref_blocks = {s: block_loglik(ref["p"], ev, a_val, b_val, s) for s in BLOCKS}
    run = Run(out / f"classical_{ref_name.replace(chr(39), 'prime')}")
    run.config({"model": ref_name, "data": identity, "code": rev, "block_minutes": args.block_minutes})
    run.result({"val_nll": ref["val"], "val_ks": ref["ks"], "seconds": ref["seconds"],
                "all_classical": {k: {"val_nll": v["val"], "val_ks": v["ks"]} for k, v in classical.items()}})
    for s in BLOCKS:
        run.blocks(s, *ref_blocks[s], np.append(np.arange(a_val, b_val, s), b_val))
    print(f"  reference = {ref_name}")

    # --- neural -----------------------------------------------------------------
    f = neural.features(ev, T_end=T)
    fj = neural.features(evj, T_end=T)
    n_sum = f.summary.shape[1]
    results = {"MLP": {}, "attention": {}}
    print(f"\n{'model':<11}{'seed':>5}{'val NLL':>9}   {'vs ' + ref_name + ' (60 s)':>26}   {'300 s':>26}   {'900 s':>26}   "
          f"{'val KS':>13}   {'ep':>3} {'stop':>5} {'time':>6}")
    for kind in ("MLP", "attention"):
        for seed in seeds:
            torch.manual_seed(seed)
            if kind == "MLP":
                enc, N = neural.SummaryMLP(n_sum, 2, len(MARK_NAMES), 3, d=args.width), 1
            else:
                enc, N = neural.Attention(n_sum, 2, len(MARK_NAMES), 3, d=args.width, heads=args.heads, layers=args.layers), N_TOKENS
            model = neural.TPP(enc, neural.Head(enc.d_out, 2, 3, SCALES, d_hidden=args.width))
            run = Run(out / f"{kind}_seed{seed}")
            run.config({"model": kind, "seed": seed, "N": N, "width": args.width, "layers": args.layers, "heads": args.heads,
                        "max_epochs": args.max_epochs, "min_epochs": args.min_epochs, "patience": args.patience,
                        "targets_per_epoch": args.targets_per_epoch, "scales_half_life_s": HALF_LIVES.tolist(),
                        "summary_windows_s": list(neural.SUMMARY_WINDOWS), "big_windows_s": list(neural.BIG_WINDOWS),
                        "reference": ref_name, "data": identity, "code": rev})
            fit = neural.train(model, f, T_train, (a_val, b_val), N, max_epochs=args.max_epochs, min_epochs=args.min_epochs,
                               patience=args.patience, seed=seed, targets_per_epoch=args.targets_per_epoch, run=run,
                               log=print if seed == seeds[0] else (lambda s: None))
            blocks_n = {s: neural.block_loglik(model, f, a_val, b_val, N, s) for s in BLOCKS}
            for s in BLOCKS:
                run.blocks(s, *blocks_n[s], np.append(np.arange(a_val, b_val, s), b_val))
            gains = {s: gain_ci_from_blocks(blocks_n[s][0] - ref_blocks[s][0], blocks_n[s][1]) for s in BLOCKS}
            ks = [ks_exp1(r) for r in neural.rescaled_residuals(model, fj, a_val, b_val, N)]
            res = {"val_nll": fit["val"], "epochs": fit["epochs"], "stopped_early": fit["stopped_early"],
                   "seconds": fit["seconds"], "val_ks": ks,
                   "gain_vs_reference": {str(int(s)): list(g) for s, g in gains.items()}, "history": fit["history"]}
            run.result(res)
            results[kind][seed] = {"blocks": blocks_n, **res}
            print(f"{kind:<11}{seed:>5}{fit['val']:9.4f}   " + "   ".join(f"{ci_str(gains[s]):>26}" for s in BLOCKS)
                  + f"   {ks[0]:.4f}/{ks[1]:.4f}   {fit['epochs']:>3} {'early' if fit['stopped_early'] else 'max':>5} {fit['seconds']:5.0f}s")
        vals = [results[kind][sd]["val_nll"] for sd in seeds]
        print(f"{kind:<11}{'':>5}{np.mean(vals):9.4f}   seed s.d. of val NLL {np.std(vals, ddof=1):.4f}   "
              f"mean gain vs {ref_name} " + "  ".join(f"{np.mean([results[kind][sd]['gain_vs_reference'][str(int(s))][0] for sd in seeds]):+.4f}@{int(s)}s" for s in BLOCKS))

    print(f"\nattention vs MLP, paired on blocks within each seed:")
    print(f"{'seed':>5}   {'60 s':>26}   {'300 s':>26}   {'900 s':>26}")
    diffs = []
    for seed in seeds:
        g = {s: gain_ci_from_blocks(results["attention"][seed]["blocks"][s][0] - results["MLP"][seed]["blocks"][s][0],
                                    results["attention"][seed]["blocks"][s][1]) for s in BLOCKS}
        diffs.append(g[60.0][0])
        print(f"{seed:>5}   " + "   ".join(f"{ci_str(g[s]):>26}" for s in BLOCKS))
    print(f"{'mean':>5}   {np.mean(diffs):+.4f}   seed s.d. {np.std(diffs, ddof=1):.4f}")

    summary = {"session": session, "reference": ref_name, "code": rev, "seeds": seeds,
               "classical": {k: {"val_nll": v["val"], "val_ks": v["ks"]} for k, v in classical.items()},
               "MLP": {sd: {k: v for k, v in r.items() if k != "blocks"} for sd, r in results["MLP"].items()},
               "attention": {sd: {k: v for k, v in r.items() if k != "blocks"} for sd, r in results["attention"].items()},
               "attention_minus_MLP_60s": diffs}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"\nartifacts: {out}")


if __name__ == "__main__":
    main()
