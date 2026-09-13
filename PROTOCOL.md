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

- **Primary:** held-out negative log-likelihood per event, reported as the **gain** over the relevant reference with a **95% block-bootstrap interval** (60 s blocks, 2000 resamples; `experiment.gain_ci`). Both models scored on identical blocks.
- **Goodness of fit:** time-rescaling residuals on **jittered** times (`residuals.jitter`), KS distance from Exp(1) per side, and conditional residual means by features of the **interval opener** (mark, state, burst count in the prior 100 ms, time since last opposite-side event, hour). Every statistic is reported next to a **simulated null**: the fitted model simulated on the same state function with train mark frequencies, passed through `simulate.observe` (ms ticks, same-tick merge), refitted, and diagnosed identically; ≥ 2 replicates.
- **Secondary:** next-event type accuracy and log-time error, for neural models and the benchmark alike.

## 5. Controls (fixed)

- **Marks:** shuffled **within (split, side)** — the label keeps its split and its side-conditional distribution and loses only its timing. Five seeds; report mean and s.d. of the gain.
- **State:** the state series circularly shifted **within each split** by five offsets (30, 60, 90, 120, 150 min). Report mean and s.d.
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

A neural model is judged on validation gains over **the best classical model on the ladder** (M4, E1 or E2, whichever wins on validation), with matched inputs first. It earns a place in the analysis if its gain interval excludes zero on validation **and** the direction replicates on ≥ 2 fresh sessions under this protocol. Passing residual diagnostics does not make it right; failing them does make it wrong.

Extended-input runs report the gain over the matched-input run of the same architecture, so the value of new information and the value of the architecture are never added together.

---

## Log

- **2026-09-13** — frozen after the diagnostics pass on 2026-09-09 (ARCHITECTURE.md D9–D10).
- **2026-09-13** — E1 and E2 run on the exploratory session (D11): H7 refuted for E1, H8 not confirmed. E1′ and H9 added; apply to fresh sessions only.
