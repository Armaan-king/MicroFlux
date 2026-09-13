"""Final test stage: score the selected models on the test segment, once.

Run only after validation selection is finished and recorded in
runs/<session>/summary.json (neural) and replication.json (classical).
Refuses sessions listed in EXPLORATORY. Writes runs/<session>/final_test.json
and appends a dated line to PROTOCOL.md's log so the touch is on record.

    python scripts/final_test.py --root ... --date ... --session ...
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
    HALF_LIVES, MARK_NAMES, REPO, SCALES, block_loglik, gain_ci_from_blocks, imbalance_state, load_book,
    load_orders, mark_class, splits,
)
from microflux.hawkes import block_edges
from microflux.mle import OPEN, extrapolate, fit_hawkes
from microflux.runs import code_revision

EXPLORATORY = {"BTCUSDT-2026-09-09"}  # test segment inspected before the protocol froze
BLOCKS = (60.0, 300.0, 900.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--block-minutes", type=float, default=15.0)
    ap.add_argument("--runs", default=str(REPO / "runs"))
    args = ap.parse_args()
    if args.session in EXPLORATORY:
        raise SystemExit(f"{args.session} is exploratory; its test segment is not scored.")
    out = Path(args.runs) / args.session
    if (out / "final_test.json").exists():
        raise SystemExit(f"{out / 'final_test.json'} exists; the test segment is scored once.")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    ref_name = summary["reference"]

    orders = load_orders(args.root, args.symbol, args.date)
    book = load_book(args.root, args.symbol, args.date)
    t, m = orders["t"].to_numpy(), orders["m"].to_numpy()
    split = splits(t[-1])
    T_train = split["train"][1]
    a_te, b_te = split["test"]
    n_te = int(((t >= a_te) & (t < b_te)).sum())
    step_t, step_s, _ = imbalance_state(orders, book, T_train)
    ev = events(t, m, 2, step_t, step_s, 3, mark=mark_class(orders), C=len(MARK_NAMES))
    blocks = block_edges(T_train, args.block_minutes * 60)

    # classical reference, refit deterministically on train
    edges, scales = (OPEN, np.log(2.0) / np.append(HALF_LIVES, [500.0, 5000.0])) if ref_name == "E1'" else (blocks, SCALES)
    p_ref = extrapolate(fit_hawkes(ev.before(T_train), T_train, edges, scales, state_baseline=True), ev, T_train)
    ref_blocks = {s: block_loglik(p_ref, ev, a_te, b_te, s) for s in BLOCKS}
    result = {"session": args.session, "code": code_revision(Path(__file__).parent), "reference": ref_name,
              "test_window": [a_te, b_te], "n_test": n_te,
              "reference_test_nll": float(-sum(ref_blocks[60.0][0]) / n_te), "models": {}}

    f = neural.features(ev, T_end=t[-1])
    n_sum = f.summary.shape[1]
    for kind in ("MLP", "attention"):
        for seed in summary["seeds"]:
            cfg = json.loads((out / f"{kind}_seed{seed}" / "config.json").read_text(encoding="utf-8"))
            if kind == "MLP":
                enc, N = neural.SummaryMLP(n_sum, 2, len(MARK_NAMES), 3, d=cfg["width"]), 1
            else:
                enc, N = neural.Attention(n_sum, 2, len(MARK_NAMES), 3, d=cfg["width"], heads=cfg["heads"], layers=cfg["layers"]), cfg["N"]
            model = neural.TPP(enc, neural.Head(enc.d_out, 2, 3, SCALES, d_hidden=cfg["width"]))
            model.load_state_dict(torch.load(out / f"{kind}_seed{seed}" / "best.pt", weights_only=True))
            bl = {s: neural.block_loglik(model, f, a_te, b_te, N, s) for s in BLOCKS}
            gains = {str(int(s)): list(gain_ci_from_blocks(bl[s][0] - ref_blocks[s][0], bl[s][1])) for s in BLOCKS}
            result["models"][f"{kind}_seed{seed}"] = {"test_nll": float(-bl[60.0][0].sum() / n_te), "gain_vs_reference": gains}
            print(f"{kind:<11} seed {seed}  test NLL {result['models'][f'{kind}_seed{seed}']['test_nll']:8.4f}   "
                  + "  ".join(f"{g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}]@{s}s" for s, g in gains.items()))
    (out / "final_test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(REPO / "PROTOCOL.md", "a", encoding="utf-8") as h:
        h.write(f"- **{time.strftime('%Y-%m-%d')}** — final test scored once on `{args.session}` (runs/{args.session}/final_test.json).\n")
    print(f"\nwritten {out / 'final_test.json'}; PROTOCOL.md log appended")


if __name__ == "__main__":
    main()
