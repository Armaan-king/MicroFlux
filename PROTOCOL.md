# MicroFlux — Comparison Protocol

**Status: frozen 2026-09-13.** Changes to this document require a dated entry in its log and apply only to sessions not yet evaluated under it.

Everything learned so far comes from one 8-hour capture (BTCUSDT, 2026-09-09) whose test segment has been inspected repeatedly. Those results are **exploratory**. This protocol fixes, before any fresh session is opened, what will be computed on it, what counts as a control, and what each hypothesis predicts — so that a confirmation is a confirmation.

---

## 1. Data construction (fixed)

| step | rule | code |
|---|---|---|
| source | TickForge partition, all files of one UTC date, ordered on `(session, capture_seq)` | `load.load_stream` |
| continuity | files join only if Binance `first_seq == prev.last_seq + 1` at every boundary; otherwise the date is split into runs and each run is a separate session | `load.is_continuous` |
| events | consecutive trades sharing `(timestamp_ns, aggressor)` collapse to one aggressive order; `merge_gap_ns = 0` | `load.collapse_trades` |
| mark | fill count of the order, classes `1 / 2–4 / 5–19 / 20+` (edges fixed at 2, 5, 20) | `experiment.mark_class` |
| book | replayed from the session's snapshot and diffs; state row at each 1 Hz update | `book.replay` |
| state | L5 imbalance `(bid − ask) / (bid + ask)` over the top 5 levels, terciles; **cuts from train book rows of the same session** | `experiment.imbalance_state` |
| time | seconds from the session's first order; ms ticks kept as recorded | `experiment.load_orders` |

## 1b. Session eligibility and selection (fixed 2026-09-13)

A TickForge partition is one **session** if `load.is_continuous` holds across all its files; otherwise each continuous run is its own session. A session is **eligible** for replication if it has ≥ 1 snapshot, span ≥ 1 h, ≥ 10,000 collapsed orders, and **no time overlap** with any session already used for exploration or a prior replication. Sessions are taken in chronological order of capture as they become eligible; none is skipped or chosen on its results. Every session runs the same configuration: `scripts/replicate.py` for the classical ladder, controls, diagnostics and H1–H9, then `scripts/fit_neural.py`, then `scripts/final_test.py` once — all three via `scripts/run_session.py`.

Inventory at freeze: `BTCUSDT-2026-09-09` (8.00 h, 11:32–19:32 UTC) — **exploratory**; `BTCUSDT-2026-09-09-early` (2.00 h, 05:04–07:04 UTC, `C:/tickforge-runs/archive-2h`) — **eligible, fresh session 1**; 2026-09-07 and 2026-09-08 partitions (1.2 min and 0.6 min) — ineligible. A second fresh session requires a new capture.

## 2. Split (fixed)

70 / 15 / 15 **by time**, per session. Never shuffled.

- **train** — all parameter fitting, all preprocessing statistics (tercile cuts, mark frequencies).
- **validation** — model selection, diagnostics, ablations, hypothesis checks.
- **test** — touched **once per model** under this protocol, after everything else is frozen. For 2026-09-09 the test segment is already exploratory and is reported as such.

Held-out evaluation uses the model's full observed history (events before the window are conditioned on, not fitted on) and the baseline frozen at its time-weighted train mean per state (`mle.extrapolate`).

## 3. Models (fixed ladder)

All with five fixed exponential scales (half-lives 5 ms, 50 ms, 0.5 s, 5 s, 50 s) unless stated, baseline blocks of 15 minutes, MLE by L-BFGS-B on train.

```
M0   Poisson + mu(t)
M1   Hawkes x5 + mu(t)
M2   Hawkes x5 + mu(t) + k(c)                      marks in the kernel
M3   Hawkes x5 + mu(t,s) + k(s)                    state in baseline and kernel
M4   Hawkes x5 + mu(t,s) + k(s,c)                  BENCHMARK
E1   M4 with + 500 s, 5000 s scales                pre-registered extension (rate tracking)
E2   M4 with ten half-decade scales 5 ms .. 160 s  pre-registered extension (intermediate scales)
E1'  M4 scales + 500 s, 5000 s, single baseline    pre-registered 2026-09-13 after D11: the level carried by excitation, so intensity can track a quiet window
N*   neural models per docs/ml-design-brief.md      matched inputs first, extended inputs second
```

## 4. Metrics (fixed)

- **Primary:** held-out negative log-likelihood per event, reported as the **gain** over the relevant reference with a **95% block-bootstrap interval** (60 s blocks, 2000 resamples; `experiment.gain_ci`), with 300 s and 900 s blocks as sensitivity (`experiment.gain_ci_blocks`). Both models scored on identical blocks. Validation scripts compute KS on validation only (`evaluate(ks_on="val")`).
- **Goodness of fit:** time-rescaling residuals on **jittered** times (`residuals.jitter`), KS distance from Exp(1) per side, and conditional residual means by features of the **interval opener** (mark, state, burst count in the prior 100 ms, time since last opposite-side event, hour). Every statistic is reported next to a **simulated null**: the fitted model simulated on the same state function with train mark frequencies, passed through `simulate.observe` (ms ticks, same-tick merge), refitted, and diagnosed identically; ≥ 2 replicates.
- **Secondary:** next-event type accuracy and log-time error, for neural models and the benchmark alike.

## 5. Controls (fixed)

- **Marks:** shuffled **within (split, side)** — the label keeps its split and its side-conditional distribution and loses only its timing. Five seeds; report mean and s.d. of the gain.
- **State:** the state series circularly shifted **within each split** by five offsets at 1/6 … 5/6 of the segment's length (30 … 150 min on an 8 h session; scaled on shorter ones). Report mean and s.d.
- A gain is attributable to the information only if it exceeds the control mean by more than three control s.d. and its own interval excludes zero.

## 6. Pre-registered hypotheses

Each stated with the statistic that tests it and the direction predicted from the 2026-09-09 exploratory session. A fresh session **confirms** a hypothesis when the predicted direction holds with the interval excluding zero (or, for shape hypotheses, the ordering holds in every side).

| | hypothesis | statistic | predicted |
|---|---|---|---|
| H1 | Marks in the kernel carry held-out information | gain M2 − M1, vs control | gain > 0.05; control ≈ 0 |
| H2 | Excitation from 5+ fill orders concentrates at ≤ 50 ms; from 1-fill orders at ≥ 0.5 s | share of `BUY←BUY` and `SELL←SELL` branching at scales ≤ 50 ms, by mark class | monotone increasing in mark class; > 0.7 for 20+, < 0.3 for 1 fill |
| H3 | State in the kernel carries information beyond state in the baseline | gain M4 − (M4 with `kernel_state=False`), vs control | gain > 0.005; control ≈ 0 |
| H4 | Fast same-side excitation from small orders rises with imbalance on the aggressor's side; slow falls | `BUY←BUY@1-fill` at 5 ms and at 5 s across ask-heavy → bid-heavy; mirrored for SELL | 5 ms increasing, 5 s decreasing, in both sides |
| H5 | Exogenous rate follows imbalance | time-weighted train mean of μ by state | BUY increasing, SELL decreasing in bid-heaviness |
| H6 | No saturation, no opposite-side inhibition beyond the benchmark | conditional residual means by burst count and by opposite-side gap, vs null | within 2 s.e. of the null in every bin |
| H7 | Rate tracking is the dominant residual; slow scales fix it | residual mean by hour for M4 vs E1 | M4 drifts from 1; E1 within 2 s.e. of the null |
| H8 | Structure exists between the decade scales | gain E2 − M4 | interval excludes zero |
| H9 | An excitation-carried baseline tracks a quiet held-out window where a frozen one cannot | residual mean by hour, E1′ vs M4; gain E1′ − M4 | E1′ within 2 s.e. of the null; gain interval excludes zero |

## 7. Neural comparison (decision rule)

A neural model is judged on validation gains over **the best classical model on the ladder** (M4, E1, E2, E1′ or a retained combination, whichever wins on validation of the session under test), with matched inputs first. Gains are read at 60 s, 300 s and 900 s block sizes; a gain that holds only at the narrowest is not reported as one. Neural fits use ≥ 3 training seeds and report the spread and the runtime. It earns a place in the analysis if its gain interval excludes zero on validation **and** the direction replicates on ≥ 2 fresh sessions under this protocol. Passing residual diagnostics does not make it right; failing them does make it wrong.

Extended-input runs report the gain over the matched-input run of the same architecture, so the value of new information and the value of the architecture are never added together.

**Executable defaults (fixed 2026-09-13, `scripts/fit_neural.py`):** MLP over activity summaries and attention over the last **N = 64** event tokens, both on the shared five-scale head; width 64, 2 layers, 4 heads; seeds **0, 1, 2, 3, 4**; patience **8**; **minimum 10, maximum 40** epochs; 40,000 training units per epoch; Adam 1e-3; float32. Reported: MLP vs reference, attention vs reference, attention vs MLP paired within seed, each at 60 / 300 / 900 s; seed s.d. of validation NLL separately. On a session shorter than ~4 h the 300 s and 900 s intervals rest on too few blocks and are reported but not read. N = 64 is a practical pilot choice, not a finding.

---

## Log

- **2026-09-13** — frozen after the diagnostics pass on 2026-09-09 (ARCHITECTURE.md D9–D10).
- **2026-09-13** — E1 and E2 run on the exploratory session (D11): H7 refuted for E1, H8 not confirmed. E1′ and H9 added; apply to fresh sessions only.
- **2026-09-13** — E1′ run on the exploratory session (D12): gain +0.0039 with all three block sizes excluding zero; residual drift reduced, not removed. H9 partially met; stands for fresh sessions.
- **2026-09-13** — corrected comparison on the exploratory session (D14): attention +0.0257 vs E1′, seed s.d. 0.0029, 5/5 clear; MLP −0.0040. Fresh session 1 `BTCUSDT-2026-09-09-early`: H1 confirmed; H2 direction only; H3–H9 not confirmed as registered; neural validation gains not clear of zero (18 blocks); final test scored once, MLP −0.0140, attention −0.0057. **Neural advantage not replicated on fresh session 1.** Classical model retained as reference. Collection batch 1 recorded prospectively (`docs/collection-batch-1.md`).
- **2026-09-13** — four implementation defects found in review of the pilot (D13) fixed with regression tests; pilot numbers superseded. Session eligibility (§1b), scaled state-shift offsets, and executable neural defaults fixed. `final_test.py` is the only path that scores a test segment.
- **2026-09-13** — neural pilot on the exploratory session (D13, superseded): summaries-MLP −0.0020 ± 0.0034 vs E1′; attention N = 64 +0.0164 ± 0.0084, 3/3 seeds clear zero; N = 128 no different. Fresh-session neural runs use N = 64, patience 8, an epoch floor, 5 seeds.
- **2026-09-13** — test isolation fixed: `evaluate` scored KS on test in every validation script; now validation by default. §7 comparator list includes E1′ and retained combinations; block-size sensitivity and training seeds added.
- **2026-09-13** — final test scored once on `BTCUSDT-2026-09-09-early` (runs/BTCUSDT-2026-09-09-early/final_test.json).
- **2026-09-14** — final test scored once on `BTCUSDT-2026-09-14` (runs/BTCUSDT-2026-09-14/final_test.json).
- **2026-09-15** — batch session 1 `BTCUSDT-2026-09-14` (D15): H1, H2, H5, H8 confirmed; attention +0.0278 on validation and **+0.0339 on the final test**, 5/5 seeds clear at 77 / 16 / 6 blocks; MLP +0.0076 / +0.0037. Batch conclusion pending session 2.
- **2026-09-15** — reporting audit, with no model changes or test rescoring: H8's confirmation uses the primary 60 s interval; its 300 s sensitivity interval includes zero. H3 fails both its >0.005 threshold and exclusion of zero. The batch's 300 s / 900 s intervals on 76-minute validation/test windows are insufficient for inference. The planned second capture date (2026-09-15) was missed; next feasible target 2026-09-16, with the same eligibility and evaluation rules (`docs/handoff-2026-09-15.md`).
