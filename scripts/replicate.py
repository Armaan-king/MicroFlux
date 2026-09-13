"""Replication runner: the registered classical comparison on one session.

Fits the PROTOCOL.md ladder on train, scores validation with paired
block-bootstrap intervals, runs the registered controls and the calibrated
diagnostics, and evaluates hypotheses H1-H9 with their registered
statistics. Writes runs/<session>/replication.json and replication.md.
Validation only; the test segment belongs to final_test.py.

    python scripts/replicate.py --root C:/tickforge-runs/archive-2h --date 2026-09-09 --session BTCUSDT-2026-09-09-early
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from microflux.events import events
from microflux.experiment import (
    HALF_LIVES, MARK_NAMES, REPO, SCALES, TYPES, gain_ci_blocks, imbalance_state, load_book, load_orders,
    mark_class, shuffle_within, split_id, splits,
)
from microflux.hawkes import block_edges
from microflux.mle import OPEN, extrapolate, fit_hawkes, fit_poisson
from microflux.residuals import (
    conditional_means, interval_covariates, jitter, ks_exp1, rescaled_residuals,
)
from microflux.runs import code_revision, data_identity
from microflux.simulate import observe, simulate

STATES = ("ask-heavy", "balanced", "bid-heavy")
SLOW = np.log(2.0) / np.append(HALF_LIVES, [500.0, 5000.0])
DENSE = np.log(2.0) / np.array([0.005, 0.016, 0.05, 0.16, 0.5, 1.6, 5.0, 16.0, 50.0, 160.0])
CTRL_SEEDS = (0, 1, 2, 3, 4)
COND = {"burst": np.array([1, 2, 5, 10]), "opp_gap": np.array([0.01, 0.1, 1.0, 10.0]), "hour": np.array([0.4, 0.8])}


def fit(ev, T_train, edges, scales, **kw):
    return extrapolate(fit_hawkes(ev.before(T_train), T_train, edges, scales, **kw), ev, T_train)


def diagnostics(p, ev, a, b, seed):
    evj = jitter(ev, seed)
    res = rescaled_residuals(p, evj, a, b)
    cov = interval_covariates(evj, a, b)
    out = {"ks": [ks_exp1(r) for r in res]}
    for name, edges in COND.items():
        out[name] = [conditional_means(r, c[name], edges) for r, c in zip(res, cov)]
    return out


def null_diagnostics(p, ev, T_train, b, edges, scales, replicates):
    tr = ev.before(T_train)
    mark_prob = np.array([np.bincount(tr.c[tr.m == i], minlength=ev.C) / (tr.m == i).sum() for i in range(ev.K)])
    outs = []
    for r in range(replicates):
        sim = observe(simulate(p, b, 100 + r, ev.step_t, ev.step_s, mark_prob))
        q = fit(sim, T_train, edges, scales, state_baseline=True)
        outs.append(diagnostics(q, sim, T_train, b, 100 + r))
    return {k: np.mean([o[k] for o in outs], axis=0) for k in outs[0]}


def within_two_se(real, null):
    """real and null are (bins, 3) tables: mean, s.e., n. True if every bin's
    real mean is within 2 real s.e. of the null mean."""
    ok = True
    for r, n in zip(real, null):
        if np.isnan(r[0]) or np.isnan(n[0]) or r[2] < 30:
            continue
        ok &= abs(r[0] - n[0]) <= 2 * r[1]
    return bool(ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--block-minutes", type=float, default=15.0)
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--runs", default=str(REPO / "runs"))
    args = ap.parse_args()
    out = Path(args.runs) / args.session
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    orders = load_orders(args.root, args.symbol, args.date)
    book = load_book(args.root, args.symbol, args.date)
    t, m = orders["t"].to_numpy(), orders["m"].to_numpy()
    T = t[-1]
    split = splits(T)
    T_train = split["train"][1]
    a_val, b_val = split["val"]
    blocks = block_edges(T_train, args.block_minutes * 60)
    step_t, step_s, cuts = imbalance_state(orders, book, T_train)
    mark = mark_class(orders)
    C = len(MARK_NAMES)
    identity = data_identity(args.root, args.symbol, args.date, orders["timestamp_ns"].to_numpy(), split)
    R = {"session": args.session, "code": code_revision(Path(__file__).parent), "data": identity,
         "imbalance_cuts": cuts.tolist(), "hypotheses": {}}
    print(f"{args.session}: {len(t):,} orders over {T / 3600:.2f}h   train < {T_train / 3600:.2f}h   "
          f"val [{a_val / 3600:.2f}h, {b_val / 3600:.2f}h) n={int(((t >= a_val) & (t < b_val)).sum()):,}")

    # --- sequences ---------------------------------------------------------------
    ev_flat = events(t, m, 2)
    ev_mark = events(t, m, 2, mark=mark, C=C)
    ev_state = events(t, m, 2, step_t, step_s, 3)
    ev_state_nok = events(t, m, 2, step_t, step_s, 3, kernel_state=False)
    ev_both = events(t, m, 2, step_t, step_s, 3, mark=mark, C=C)

    # --- ladder ------------------------------------------------------------------
    P = {}
    P["M0"] = extrapolate(fit_poisson(ev_flat.before(T_train), T_train, blocks), ev_flat, T_train)
    P["M1"] = fit(ev_flat, T_train, blocks, SCALES)
    P["M2"] = fit(ev_mark, T_train, blocks, SCALES)
    P["M3"] = fit(ev_state, T_train, blocks, SCALES, state_baseline=True)
    P["M4"] = fit(ev_both, T_train, blocks, SCALES, state_baseline=True)
    P["M4nok"] = fit(events(t, m, 2, step_t, step_s, 3, kernel_state=False, mark=mark, C=C), T_train, blocks, SCALES, state_baseline=True)
    P["E1"] = fit(ev_both, T_train, blocks, SLOW, state_baseline=True)
    P["E2"] = fit(ev_both, T_train, blocks, DENSE, state_baseline=True)
    P["E1'"] = fit(ev_both, T_train, OPEN, SLOW, state_baseline=True)
    EV = {"M0": ev_flat, "M1": ev_flat, "M2": ev_mark, "M3": ev_state, "M4": ev_both, "E1": ev_both, "E2": ev_both, "E1'": ev_both,
          "M4nok": events(t, m, 2, step_t, step_s, 3, kernel_state=False, mark=mark, C=C)}
    print(f"  ladder fitted ({time.perf_counter() - t_all:.0f}s)")

    def gains(ref, new):
        return [(s, g, lo, hi) for s, g, lo, hi in gain_ci_blocks(P[ref], P[new], EV[ref], EV[new], a_val, b_val)]

    G = {}
    for ref, new in (("M1", "M2"), ("M4nok", "M4"), ("M4", "E1"), ("M4", "E2"), ("M4", "E1'"), ("M0", "M1"), ("M1", "M4")):
        G[f"{new}-{ref}"] = gains(ref, new)
        print(f"  {new} - {ref}: " + "  ".join(f"{g:+.4f} [{lo:+.4f}, {hi:+.4f}] @{int(s)}s" for s, g, lo, hi in G[f"{new}-{ref}"]))
    R["gains"] = {k: [list(x) for x in v] for k, v in G.items()}

    # --- controls ----------------------------------------------------------------
    groups = split_id(t, split) * 2 + m
    ctrl_mark = []
    for seed in CTRL_SEEDS:
        ev_sh = events(t, m, 2, mark=shuffle_within(mark, groups, seed), C=C)
        p_sh = fit(ev_sh, T_train, blocks, SCALES)
        ctrl_mark.append(gain_ci_blocks(P["M1"], p_sh, ev_flat, ev_sh, a_val, b_val, sizes=(60.0,))[0][1])
    seg = split_id(t, split)
    ctrl_state = []
    for k, frac in enumerate((1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6)):
        shifted = step_s.copy()
        for sid, (a, b) in enumerate(split.values()):
            rows = np.flatnonzero((step_t >= a) & (step_t < b))
            shifted[rows] = np.roll(step_s[rows], int(frac * len(rows)))
        ev_sh = events(t, m, 2, step_t, shifted, 3, mark=mark, C=C)
        ev_sh_nok = events(t, m, 2, step_t, shifted, 3, kernel_state=False, mark=mark, C=C)
        p_sh, p_sh_nok = fit(ev_sh, T_train, blocks, SCALES, state_baseline=True), fit(ev_sh_nok, T_train, blocks, SCALES, state_baseline=True)
        ctrl_state.append(gain_ci_blocks(p_sh_nok, p_sh, ev_sh_nok, ev_sh, a_val, b_val, sizes=(60.0,))[0][1])
    R["controls"] = {"mark_shuffle": ctrl_mark, "state_shift": ctrl_state}
    print(f"  controls: marks {np.mean(ctrl_mark):+.4f} +/- {np.std(ctrl_mark, ddof=1):.4f}   "
          f"state {np.mean(ctrl_state):+.4f} +/- {np.std(ctrl_state, ddof=1):.4f}   ({time.perf_counter() - t_all:.0f}s)")

    # --- diagnostics with nulls ----------------------------------------------------
    D = {k: diagnostics(P[k], EV[k], a_val, b_val, 0) for k in ("M4", "E1", "E1'")}
    NULL = {"M4": null_diagnostics(P["M4"], ev_both, T_train, b_val, blocks, SCALES, args.replicates),
            "E1": null_diagnostics(P["E1"], ev_both, T_train, b_val, blocks, SLOW, args.replicates),
            "E1'": null_diagnostics(P["E1'"], ev_both, T_train, b_val, OPEN, SLOW, args.replicates)}
    R["diagnostics"] = {k: {"ks": D[k]["ks"], "null_ks": NULL[k]["ks"].tolist(),
                            "hour": [x.tolist() for x in D[k]["hour"]], "null_hour": [x.tolist() for x in NULL[k]["hour"]]} for k in D}
    print(f"  diagnostics done ({time.perf_counter() - t_all:.0f}s)")

    # --- hypotheses ----------------------------------------------------------------
    H = R["hypotheses"]
    g = {k: v[0][1] for k, v in G.items()}
    lo = {k: v[0][2] for k, v in G.items()}
    H["H1"] = {"gain": g["M2-M1"], "control_mean": float(np.mean(ctrl_mark)), "control_sd": float(np.std(ctrl_mark, ddof=1)),
               "confirmed": bool(g["M2-M1"] > 0.05 and lo["M2-M1"] > 0 and g["M2-M1"] > np.mean(ctrl_mark) + 3 * np.std(ctrl_mark, ddof=1))}
    fast = P["M2"].branching_by_scale[:2].sum(0)  # scales <= 50 ms
    share = {c: [float(fast[i, i + 2 * c] / P["M2"].branching[i, i + 2 * c]) for i in range(2)] for c in range(C)}
    H["H2"] = {"fast_share_by_mark": share,
               "confirmed": bool(all(share[c][i] <= share[c + 1][i] + 1e-9 for c in range(C - 1) for i in range(2))
                                 and all(share[C - 1][i] > 0.7 and share[0][i] < 0.3 for i in range(2)))}
    H["H3"] = {"gain": g["M4-M4nok"], "control_mean": float(np.mean(ctrl_state)), "control_sd": float(np.std(ctrl_state, ddof=1)),
               "confirmed": bool(g["M4-M4nok"] > 0.005 and lo["M4-M4nok"] > 0 and g["M4-M4nok"] > np.mean(ctrl_state) + 3 * np.std(ctrl_state, ddof=1))}
    b4 = P["M4"].branching_by_scale
    def col(i, s, c): return i + 2 * (s + 3 * c)
    h4 = {"BUY_5ms": [float(b4[0, 0, col(0, s, 0)]) for s in range(3)], "BUY_5s": [float(b4[3, 0, col(0, s, 0)]) for s in range(3)],
          "SELL_5ms": [float(b4[0, 1, col(1, s, 0)]) for s in range(3)], "SELL_5s": [float(b4[3, 1, col(1, s, 0)]) for s in range(3)]}
    inc = lambda v: v[0] < v[1] < v[2]
    dec = lambda v: v[0] > v[1] > v[2]
    H["H4"] = {**h4, "confirmed": bool(inc(h4["BUY_5ms"]) and dec(h4["BUY_5s"]) and dec(h4["SELL_5ms"]) and inc(h4["SELL_5s"]))}
    mu = P["M4"].mu[-1]  # the extrapolated row is the time-weighted train mean per state
    H["H5"] = {"mu_BUY": mu[:, 0].tolist(), "mu_SELL": mu[:, 1].tolist(), "confirmed": bool(inc(mu[:, 0]) and dec(mu[:, 1]))}
    H["H6"] = {"confirmed": bool(all(within_two_se(D["M4"][k][i], NULL["M4"][k][i]) for k in ("burst", "opp_gap") for i in range(2)))}
    H["H7"] = {"M4_hour": [float(x[0]) for x in D["M4"]["hour"][0]], "E1_hour": [float(x[0]) for x in D["E1"]["hour"][0]],
               "confirmed": bool(all(within_two_se(D["E1"]["hour"][i], NULL["E1"]["hour"][i]) for i in range(2)))}
    H["H8"] = {"gain": g["E2-M4"], "confirmed": bool(lo["E2-M4"] > 0)}
    H["H9"] = {"gain": g["E1'-M4"], "E1p_hour": [float(x[0]) for x in D["E1'"]["hour"][0]],
               "confirmed": bool(lo["E1'-M4"] > 0 and all(within_two_se(D["E1'"]["hour"][i], NULL["E1'"]["hour"][i]) for i in range(2)))}
    R["seconds"] = time.perf_counter() - t_all
    (out / "replication.json").write_text(json.dumps(R, indent=2, default=float), encoding="utf-8")

    lines = [f"# Replication — {args.session}", "", f"code `{R['code']}` · {identity['orders']:,} orders · fingerprint `{identity['fingerprint']}` · {R['seconds']:.0f}s", "",
             "| | verdict | statistic |", "|---|---|---|"]
    for k, v in H.items():
        stat = {kk: vv for kk, vv in v.items() if kk != "confirmed"}
        lines.append(f"| {k} | {'confirmed' if v['confirmed'] else 'not confirmed'} | `{json.dumps(stat, default=lambda x: round(float(x), 4))[:160]}` |")
    lines += ["", "## Gains (validation, 60 / 300 / 900 s blocks)", ""]
    for k, v in G.items():
        lines.append(f"- **{k}**: " + " · ".join(f"{gg:+.4f} [{l:+.4f}, {h:+.4f}] @{int(s)}s" for s, gg, l, h in v))
    lines += ["", f"controls: marks shuffled within (split, side) {np.mean(ctrl_mark):+.4f} ± {np.std(ctrl_mark, ddof=1):.4f}; "
              f"state shifted within split {np.mean(ctrl_state):+.4f} ± {np.std(ctrl_state, ddof=1):.4f}", "",
              "## KS (jittered) vs simulated null", ""]
    for k in D:
        lines.append(f"- {k}: real {D[k]['ks'][0]:.4f} / {D[k]['ks'][1]:.4f} · null {NULL[k]['ks'][0]:.4f} / {NULL[k]['ks'][1]:.4f}")
    (out / "replication.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
