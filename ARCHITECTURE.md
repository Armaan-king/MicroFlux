# MicroFlux — Architecture

## Status

This document describes the **current working architecture** of MicroFlux.

MicroFlux is a research project, so its architecture is expected to evolve as we explore the data, papers, and modelling approaches.

This document should capture the best current understanding of the system without prematurely fixing implementation details.

---

# High-Level Architecture

MicroFlux sits downstream of TickForge.

```text
Binance
   ↓
TickForge
   ↓
Research API
   ↓
MicroFlux
```

TickForge handles market-data infrastructure and correctness.

MicroFlux handles:

```text
research data construction
        ↓
event / market-state representation
        ↓
temporal modelling
        ↓
training and experiments
        ↓
evaluation and analysis
        ↓
optional generative modelling
```

---

# System Boundary

## TickForge

TickForge remains responsible for:

* Binance market-data ingestion
* normalized market events
* L2 order-book reconstruction
* sequence validation and recovery
* deterministic capture ordering
* historical storage
* replay
* generic microstructure features

MicroFlux should consume TickForge through its research API rather than reproduce these capabilities.

## MicroFlux

MicroFlux is responsible for:

* retrieving historical TickForge data
* local research caching
* research-specific event construction
* dataset generation
* feature selection
* temporal representations
* statistical and neural models
* training
* evaluation
* experiment management
* research analysis
* generative modelling if pursued

---

# Current Data Flow

```text
TickForge Research API
          ↓
Historical Parquet Export
          ↓
MicroFlux Local Cache
          ↓
Research Data Processing
          ↓
Event + Market-State Representation
          ↓
Model-Ready Dataset
          ↓
Research Models
          ↓
Training + Evaluation
          ↓
Analysis
```

The representation between these stages is intentionally not fixed yet.

---

# Repository Structure

Flat library modules, one job each; experiment scripts under `scripts/`, run from the repository root. Files appear when they have a purpose; the six planned subdirectories (`tickforge/ data/ models/ training/ evaluation/ analysis/`) are unused until a second module of the same kind exists.

```text
src/microflux/
├── load.py         TickForge partitions → frames; continuity check; fill → order collapse
├── book.py         snapshot + diffs → 1 Hz book state
├── events.py       Events: (t, excited type, state, exciting type) + state as a step function of time
├── hawkes.py       Params, kernel recursion, intensity, compensator, log-likelihood
├── mle.py          fit_poisson, fit_hawkes (torch autograd objective when β fixed, scipy L-BFGS-B), extrapolate
├── residuals.py    time-rescaling residuals, KS distance
├── simulate.py     Ogata thinning — synthetic truth for the tests
└── experiment.py   shared loader, split, timescale grid, evaluator, cache

scripts/
├── explore.py      Stage 1 measurements
├── fit.py          Stage 2 six-model ladder
├── fit_state.py    Stage 3 state-dependent excitation, with time-shift control
├── fit_marks.py    Stage 3b marked excitation by fill count, with CIs and within-split shuffle control
├── audit_marks.py  what the mark measures; grouping-rule sensitivity
├── diagnose.py     residual diagnostics on validation, calibrated against a simulated null
├── fit_extend.py   pre-registered classical extensions E1 / E2 / E1′, with the baseline–excitation decomposition
├── fit_neural.py   Stage 4: summary MLP vs attention on a shared multiscale head, 5 seeds, artifacts per run
├── replicate.py    registered classical ladder, controls, diagnostics, H1-H9 on one session
├── final_test.py   the one-time test-segment score for a session
└── run_session.py  eligibility gate -> replicate -> fit_neural -> final_test, each stage once
notebooks/          01_neural_comparison.ipynb — walkthrough of the models from saved artifacts
tests/              recovery-from-known-truth tests
data/               replayed book cache (gitignored)
```

Do not create placeholder modules, interfaces, or abstractions to fill structure. The structure may change as the project evolves.

---

# TickForge Access

MicroFlux will use TickForge's research interface:

```text
/research/{symbol}/sessions
/research/{symbol}/events
/research/{symbol}/export
```

The expected workflow is:

```text
TickForge
    ↓
bulk Parquet export
    ↓
MicroFlux local cache
    ↓
research processing
```

Training should use locally cached historical data rather than repeatedly requesting samples from TickForge.

---

# Research Data

TickForge provides factual market events such as:

```text
BookSnapshot
BookUpdate
Trade
```

MicroFlux will derive research-specific event representations from these.

A possible starting vocabulary is:

```text
BUY_TRADE
SELL_TRADE

BID_DEPTH_INCREASE
BID_DEPTH_DECREASE

ASK_DEPTH_INCREASE
ASK_DEPTH_DECREASE
```

This is provisional.

The final event taxonomy should be chosen after exploring the actual TickForge data.

Possible event information includes:

```text
event type
timestamp
time since previous event
relative price
quantity / magnitude
book level
```

Possible market-state information includes:

```text
spread
midprice
microprice

L1 / L5 / L10 imbalance

bid / ask depth

trade imbalance
order-flow imbalance
realised volatility
```

Which features and representations are used should be decided experimentally.

---

# Research Progression

The current research direction is:

```text
Data Exploration
      ↓
Poisson Baseline
      ↓
Multivariate Hawkes
      ↓
State-Dependent Hawkes
      ↓
Neural / Transformer TPP
      ↓
Optional Generative Model
```

This is a research roadmap, not a fixed implementation architecture.

Models should only be added after the preceding experiments establish a reason for them.

---

# Model Direction

## Classical Point Processes

Initial baselines may include:

```text
Poisson Process
Hawkes Process
State-Dependent Hawkes Process
```

These provide interpretable models of event arrival and temporal excitation.

## Neural Temporal Models

Later experiments may investigate:

```text
Neural Hawkes Processes
Transformer Hawkes Processes
other Temporal Point Processes
```

These should be considered when simpler models cannot capture important nonlinear or long-range behaviour.

## Generative Models

Possible later directions include:

```text
autoregressive event generation
diffusion-based TPPs
flow matching
state-conditioned generation
```

No generative architecture is currently selected.

---

# Evaluation

Evaluation should eventually cover three levels.

## Predictive

Possible measures:

```text
negative log-likelihood
event-type accuracy / F1
event-time error
```

## Information Propagation

Study:

```text
event-to-event excitation
excitation decay
state-dependent behaviour
market-regime differences
```

## Market Realism

If generative models are introduced, compare synthetic and real data through measures such as:

```text
spread
depth
imbalance
liquidity
volatility
inter-arrival times
order-flow persistence
return distributions
```

The exact evaluation methodology should evolve with the models being tested.

---

# Time and Session Handling

TickForge provides:

```text
session
capture_seq
timestamp_ns
received_ns
```

MicroFlux should preserve these distinctions.

Separate TickForge capture sessions should not automatically be assumed to form a continuous market sequence.

The treatment of session boundaries and event-time representation should be decided during data exploration.

---

# Current Principles

### TickForge is the source of market truth

MicroFlux should not reproduce Binance-specific ingestion or L2 reconstruction.

### Research semantics belong in MicroFlux

Event labels, windows, features, targets, regimes, and modelling assumptions belong downstream.

### Preserve raw data

Downloaded TickForge data should remain unchanged.

Derived datasets should be reproducible from the raw cache.

### Start simple

Complex models should follow working datasets and interpretable baselines.

### Architecture follows research

Do not create abstractions simply because they might be useful later.

---

# Decision Log

Decisions made from measurement. Each records what was decided, why, and what it changes. Newest last.

### D1 — Depth changes are state, not events (2026-09-12)

**Decision.** The event vocabulary is `BUY_ORDER` / `SELL_ORDER` only. `BID_DEPTH_*` / `ASK_DEPTH_*` are dropped as event types; the book is a **state series observed at 1 Hz**.

**Reason.** TickForge subscribes Binance `@depth` at 1000 ms. Measured on the 2026-09-09 BTCUSDT capture: 28,799 book updates in 8.00 h, inter-arrival CV = 0.00, Fano = 0.00. Depth arrival times are a clock, not a process; a Hawkes kernel fitted to them would recover the sampling rate and report it as excitation.

**Consequences.** Stage 3 conditions order-flow intensity on 1 Hz book state (the Morariu-Patrichi & Pakkanen setup) rather than modelling depth as a point process. State resolution is bounded at 1 s until TickForge captures `@depth@100ms`. Trade-to-book alignment error is up to 1 s.

### D2 — Fills collapse to aggressive orders (2026-09-12)

**Decision.** Consecutive trades sharing `(timestamp_ns, aggressor)` are one event, carrying summed quantity and fill count. Implemented in `load.collapse_trades`.

**Reason.** Binance `@trade` reports one row per fill; trade IDs are 100% contiguous. Up to 231 fills share a millisecond — one marketable order sweeping the book. Left raw, 90.8% of consecutive events have Δt = 0 and every point-process likelihood degenerates. Collapsed: 1,599,232 fills → 147,216 orders, 0.31% ties remain, and clustering survives (CV = 1.88, Fano = 11.4 at 1 s bins) — so the excitation is not a reporting artifact.

**Consequences.** This reconstructs `@aggTrade` semantics. Fill count is a natural mark (levels swept). The 0.31% residual ties are opposite-side orders in the same ms and are kept as separate events in capture order.

### D3 — Session continuity is checked, not assumed (2026-09-12)

**Decision.** Files are joined into one sequence iff Binance `first_seq == prev.last_seq + 1` at every boundary. Implemented in `load.is_continuous`.

**Reason.** The 2026-09-09 capture is 34 files with sequential session stamps and one snapshot; `capture_seq` runs 435 → 1,628,029 without restart and Binance sequence is contiguous across all 33 boundaries. These are file rolls, not reconnects. TickForge's rule that sessions must not be *presumed* continuous stands — this is the evidence that replaces the presumption.

### D4 — Baseline Hawkes: exponential kernel, 2 types, MLE on train only (2026-09-12)

**Decision.** `hawkes.py` implements a K-type exponential-kernel Hawkes process with Poisson as the α = 0 special case, sharing one likelihood. Fit on the first 70% by time; evaluated by held-out NLL/event and Ogata time-rescaling KS on the last 15%.

**Result on 2026-09-09.** Test NLL/event: Poisson 0.999 → Hawkes 0.006. Self-excitation is fast and strong (branching 0.44 / 0.46, half-life ≈ 6 ms). Cross-excitation is slow (half-life ≈ 21 s). Spectral radius 0.67. KS on test 0.15 for both types — Hawkes fits much better than Poisson and still does not fit.

**Open.** The 21 s cross-kernels coincide with a measured activity drift (6.0 → 3.2 orders/s across the capture) and may be absorbing non-stationarity rather than excitation. Two-timescale structure (ms and tens of seconds) is what a single exponential cannot represent. Next: time-varying baseline μ(t) as the non-stationarity control, then sum-of-exponentials kernels. → Resolved in D5.

### D5 — Cross-excitation was drift; self-excitation is multi-scale (2026-09-12)

**Decision.** The working order-flow model is `Hawkes x5 + μ(t)`: five fixed exponential scales per pair (half-lives 5 ms, 50 ms, 0.5 s, 5 s, 50 s), weights fit by MLE with analytic gradients, baseline piecewise-constant per 15 min. Held-out evaluation freezes μ at its train mean (`hawkes.extrapolate`).

**Reason — A, the drift control.** Letting μ(t) vary (it spans 1.02–3.54 /s across train) collapses the 21 s `SELL←BUY` kernel from branching 0.160 to **0.007**. The slow cross-excitation reported by the single-exponential model was non-stationarity wearing a kernel. The free-β model is also non-convex: two runs differing only in initialisation landed in different optima (train NLL −1.156 vs −1.148), one with the slow kernel on `BUY←SELL`, the other on `SELL←BUY`. With β fixed the likelihood is concave and this cannot happen.

**Reason — B, the timescales.** Self-excitation is real at every decade: `BUY←BUY` branching 0.37 (5 ms), 0.08 (50 ms), 0.21 (0.5 s), 0.09 (5 s), 0.03 (50 s), total 0.77; `SELL←SELL` totals 0.71 with the same shape. Cross-excitation totals 0.04–0.07 and lives only at 5–50 s. Spectral radius 0.80: four in five orders are triggered by earlier orders.

**Held-out (test NLL/event).** Poisson 0.999 → Hawkes 0.045 → Hawkes x5 −0.099. KS(Exp1) on test: 0.20/0.24 → 0.17/0.15 → 0.15/0.14. Each rung helps; none fits. Note the free-β `Hawkes + μ(t)` forecasts *worse* than `Hawkes` (0.076 vs 0.045): the spurious 21 s kernel was a crude rate tracker, and a train-mean baseline is not. A misspecified model can out-forecast a controlled one — the control is for interpretation, not prediction.

**Consequences.** (1) Any claim about cross-side propagation must be made with a drifting baseline or it is not a claim about excitation. (2) The dominant structure is same-side, millisecond-scale, and near-critical. (3) Remaining misfit candidates, in order: millisecond timestamp quantisation against a 5 ms kernel, unmodelled marks (fill count / size), and intraday seasonality inside 15-min blocks.

### D6 — State enters as the exciting type; imbalance modulates propagation (2026-09-12)

**Decision.** State-dependence follows Morariu-Patrichi & Pakkanen: the kernel from an order depends on the book state when that order happened. Implemented by widening the *exciting* type to `(side, state)` while the *excited* type stays `side` (`hawkes.Events`, `hawkes.events`). The recursion, concavity and analytic gradient are unchanged. State = L5-imbalance tercile at the last book row at or before the order, cuts fit on train. Spread is dropped as a state variable: one tick 99.9% of the time on this capture. Book state is replayed from snapshot + diffs in `book.py` and cached.

**Control.** A shuffled-state fit has the state fit's 106 parameters and none of its information. Held-out test gain over the state-free model: **state +0.0176 NLL/event, shuffled +0.0001.** The state carries real information; the parameters alone buy nothing.

**Result.** Same-side excitation is stronger when the book is stacked on the aggressor's side. `BUY←BUY` branching 0.69 / 0.78 / 0.82 across ask-heavy / balanced / bid-heavy; `SELL←SELL` 0.73 / 0.75 / 0.68 — mirrored. The effect is largest at the **5 ms** scale (`BUY←BUY` 0.23 → 0.45; `SELL←SELL` 0.48 → 0.22), which corrects D5's expectation that 1 Hz state could not reach the fast scale: the state is a persistent condition, and a persistent condition modulates fast dynamics regardless of how often it is sampled.

**Confound — open.** The baseline μ_i(t) is not state-dependent, yet the BUY share is 36% in ask-heavy books and 71% in bid-heavy ones. With the state persistent over many seconds, a state-dependent *rate* and a state-dependent *kernel* look alike: the only way this model can raise the buy rate in bid-heavy periods is through buy-in-bid-heavy events exciting more buys. Part of the kernel result may be baseline in disguise. → Resolved in D7.

### D7 — State acts on the kernel, not only the rate; the sign flips with timescale (2026-09-12)

**Decision.** The baseline is now μ_i(block, state) — `Params.mu` is `(B, S, K)`, and `Events` carries the state as a step function of time so the compensator can integrate it between events. State in the baseline and state in the kernel are independent switches (`fit_hawkes(state_baseline=)`, `events(kernel_state=)`). Tercile cuts now come from train **book rows** (time-weighted) rather than train orders: −0.425 / +0.600. A recovery test (`test_fit_separates_state_rate_from_state_kernel`) simulates a persistent state that doubles the rate and triples the excitation and checks the fit attributes each correctly.

**Held-out gain over `Hawkes x5 + μ(t)`, test NLL/event:**

```
state in baseline only        +0.0027
state in kernel only   (D6)   +0.0170
both                          +0.0200
both, state shifted by 2 h    −0.0006     ← same 198 parameters, no information
```

Nearly additive, and the shifted control is exactly zero. **The kernel effect is real and carries ~85% of what the state knows.** The confound was real in principle and small in practice. The exogenous rate does depend on state — BUY μ 0.50 / 0.63 / 0.90 /s, SELL 0.82 / 0.54 / 0.40 across ask-heavy / balanced / bid-heavy — and it now sits in μ where it belongs.

**Result — the totals mislead; the scales do not.** With μ(t, s) free, total `BUY←BUY` branching is 0.71 / 0.81 / 0.76 — non-monotonic, and the D6 "monotonic with the book" story on totals does not survive (it was partly the confound and partly the order-based cuts). Per scale it is clean and mirrored across sides:

```
BUY<-BUY        ask-heavy  balanced  bid-heavy      SELL<-SELL     ask-heavy  balanced  bid-heavy
    5 ms          0.231     0.334     0.448             5 ms          0.479     0.346     0.232
    5 s           0.172     0.132     0.054            50 s           0.029     0.114     0.186
```

**Fast same-side excitation (5 ms) is strongest when the book is stacked on the aggressor's side** — a buy into a bid-heavy book roughly doubles its immediate follow-on. **Slow same-side excitation (5–50 s) is strongest when the book is stacked against the aggressor** — a buy into an ask-heavy book spawns three times the follow-on over seconds. The two effects have opposite signs and partially cancel in the total, which is why D6's totals were ambiguous. The reading: fast follow-on is momentum into thin liquidity; slow follow-on is patient execution against resistance — a metaorder working through a book that is fighting it.

**Consequences.** (1) Any state-dependence claim must be made per timescale; totals hide sign flips. (2) The exogenous-rate dependence on imbalance (the classic queue-imbalance result) is reproduced and is separate from the propagation effect. (3) One 8-hour session; none of this is replicated. (4) KS is unmoved (0.147 / 0.128): the state is not what the model is missing. Marks and millisecond quantisation remain the candidates.

---

### D8 — PyTorch for the objective; numpy/numba for everything else (2026-09-12)

**Decision.** The fixed-β objective in `mle.fixed_objective` is torch: the recursion R, cell times and compensator kernel sums are precomputed once (numba/numpy), and the log-likelihood on top of them is a few tensor operations that autograd differentiates. scipy L-BFGS-B stays as the optimiser, fed the torch value and gradient. Data, book replay, the recursion, residuals and all evaluation stay numpy. Free-β keeps numerical gradients. float64 throughout so torch, numpy and scipy agree exactly.

**Reason.** Marks make the intensity bilinear in `(α, κ)` and every extension after them — neural TPP, generative — needs autograd. Hand-written gradients were the right call while the likelihood was concave and linear in its parameters; they stop being the right call at the next step. Verified: torch NLL equals the numpy reference to 1e-9 (`test_torch_objective_matches_numpy_loglik_and_finite_differences`), every recovery test lands where it did, and the Stage 3 ladder reproduces to four decimals.

**Consequences.** Fits are ~5× faster (heaviest 78 s → 15 s; torch einsum backward beats a numpy `add.at` loop). `torch` (~200 MB, CPU build) is a hard dependency. CPU is sufficient at this scale — 147k events × 5 scales × 12 exciting types is 9M floats.

---

### D9 — Marks: the multi-scale kernel was two populations (2026-09-12)

**Decision.** The fill count of an aggressive order is its mark, in classes 1 / 2–4 / 5–19 / 20+ (46 / 15 / 19 / 20% of orders). *Fill count is the number of trade rows in a same-millisecond same-side run* (`load.collapse_trades`): resting orders matched, not distinct price levels, and not verified to be one taker order. It is a proxy for how much liquidity was consumed; what is directly supported is that it predicts subsequent activity. The mark class of the *exciting* order widens its exciting type, exactly as state does — non-parametric, still concave, and the shape of the effect is read off rather than assumed. `Events` carries `c` and `C`; `e = m + K (s + S c)`. The likelihood is for times and sides given marks; a mark density that is iid given side would not depend on the kernel and cannot change the fit.

**Held-out gain over `Hawkes x5 + μ(t)`, test NLL/event:**

```
marks in kernel                    +0.1545      KS 0.147 / 0.143  →  0.055 / 0.061
marks, shuffled                    +0.0002
state (D7)                         +0.0197
state and marks                    +0.1593
```

Marks are **8× more informative than state**, the shuffled control is zero, and KS drops by two-thirds — the marks were what the model was missing. Given marks, state adds only +0.005 of its solo +0.020: three-quarters of what imbalance "knew" was correlated with sweep depth.

**Result — the kernel depends strongly on the mark, and the timescale shifts with it.** `BUY←BUY` branching by mark of the exciting order:

```
 half-life    1 fill     2-4    5-19     20+
    5 ms       0.078   0.385   0.510   0.876
   50 ms       0.000   0.000   0.164   0.138
  500 ms       0.356   0.441   0.022   0.048
    5 s        0.157   0.210   0.000   0.000
   50 s        0.054   0.001   0.000   0.000
```

Excitation from **5+ fill orders concentrates at the 5–50 ms scales** with little beyond; excitation from **1-fill orders concentrates at 0.5–50 s** with little at 5 ms. The 2–4 class carries both (0.39 at 5 ms, 0.44 at 500 ms), so the separation is a gradient, not two disjoint populations. A 5 ms half-life places half of that component's integrated effect inside 5 ms, not almost all of it. D5's "self-excitation at every decade" was the mark-average of these. *Mechanisms — a fast reaction to visible consumption of liquidity, a slow train of small child orders — are hypotheses, not findings.* D7's state effect at 5 ms is confined to small orders (1-fill `BUY←BUY` 0.03 / 0.07 / 0.13 across ask-heavy / balanced / bid-heavy); for 20+ sweeps it is flat (0.94 / 0.89 / 0.86).

**Artifact check.** A sweep crossing a millisecond boundary would be split by `collapse_trades` into two orders 1 ms apart and masquerade as fast excitation. Measured: after a 20+ order, a same-side order follows at exactly +1 ms 21% of the time and at +2 ms 18% — a smooth decay, not a spike — and the +1 ms / (+2..5 ms) ratio is *lowest* for the 20+ class (2.65 vs 3.36 for single fills). Not splitting.

**Consequences.** (1) Every propagation claim must condition on mark; unmarked kernels average over it. (2) The working model is `Hawkes x5 + μ(t,s) + k(s,c)`: 378 parameters, J = 24, 81 s to fit, test NLL −0.256, KS 0.055 / 0.059. (3) Quantity (BTC) is a second candidate mark, correlated 0.53 with fills. (4) Still one session.

---

### D10 — Diagnostics calibrated against a simulated null; protocol frozen (2026-09-13)

**Decision.** Every goodness-of-fit statistic is reported next to a null: the fitted model simulated on the real state function with train mark frequencies, passed through the observation process (`simulate.observe`: ms ticks, same-tick same-side merge), refitted, and diagnosed identically. Residual times are jittered within their tick (`residuals.jitter`, the discrete time-rescaling correction). Conditional residual means condition **only on features of the event that opened the interval** — the arriving event's mark or state is end-of-interval information and would select waiting times even under a correct model. Held-out gains carry 95% block-bootstrap intervals (60 s blocks). Controls shuffle marks within (split, side) over five seeds. The 2026-09-09 test segment is **exploratory**; `PROTOCOL.md` fixes what will be computed on fresh sessions.

**What the null showed.** The diagnostics are unbiased (every simulated bin 1.00 ± s.e., autocorrelation ≈ 0). Under a correct model the 1% residual quantile is inflated **2.6–2.8×** and KS is **0.028** — a *simulated null reference* from two replicates with an approximate mark aggregation in `observe`, not a universal floor. Without calibration the low-quantile excess on real data would have been misread as a short-lag misfit.

**What is left, on validation (M4).**
- Residual mean ≈ 1.10 across every bin, rising 1.04 → 1.12 through the window: the train-mean baseline **over-predicts a quieter period**. Rate tracking, not kernel shape. Dominant.
- Autocorrelation 0.08–0.10 at lags 1–2, gone by lag 50 (null ≈ 0): unmodelled clustering at 0.2–2 s.
- Openers with no events in the prior 100 ms: 1.15 / 1.24; with 2–4 events: 0.97 / 0.89. Quiet contexts over-predicted, moderately busy under-predicted. Consistent with the two above.
- **No saturation** (10+ event bursts within 2 s.e. of null) and **no opposite-side inhibition** (opposite-side gap bins flat). The two things a Hawkes structurally cannot represent are not indicated.
- Mark and state residual patterns are within ±0.05 of the offset: largely captured.

**Validation gains with intervals.** Marks +0.1515 [+0.140, +0.164]; state +0.0182 [+0.014, +0.023]; both +0.1585 [+0.149, +0.169]; shuffled-mark control −0.0008 ± 0.0023 (5 seeds).

**Mark audit.** `fills` counts trade rows in a same-ms same-side run; 91% of 2–4 fill orders touched one price. Merging same-side runs 1 ms apart removes 13% of orders, keeps the pattern, and lowers the 5 ms excitation of 20+ orders 0.88 → 0.64 — that fraction of the fastest component is indistinguishable from same-order continuation. Distinct price levels as the mark: +0.094, less informative than fills.

**Consequences.** (1) Pre-registered classical extensions E1 (slow scales, rate tracking) and E2 (dense grid) target the two residuals found; results in D11. (2) The neural design brief (`docs/ml-design-brief.md`) is written so that any neural gain is attributable to kernel shape, interactions, inputs, or memory separately. (3) Confirmatory claims wait for fresh sessions under `PROTOCOL.md`.

---

### D11 — The classical extensions do not reach the residual; the baseline cannot go down (2026-09-13)

**Result (validation, exploratory session).** Gains over M4 with 95% intervals: **E1** (+500 s, +5000 s scales) +0.0011 [+0.0007, +0.0015]; **E2** (ten half-decade scales) +0.0008 [+0.0000, +0.0015]; E1+E2 +0.0014. KS 0.055 → 0.053. The residual mean by hour is **unchanged**: 1.04 / 1.10 / 1.12 under every grid. H7 refuted for E1 as registered; H8 not confirmed.

**Why E1 does not help — corrected in D12.** This entry originally argued that the frozen baseline is a floor the intensity cannot fall below, so no kernel could track a quieter window. That is true as a statement about the structure and wrong as an explanation of the residual: the decomposition in D12 shows the observed validation count sits far *above* the baseline alone, so the floor never binds and the over-prediction is in the excitation term. What is true is that the slow scales fitted alongside 15-minute blocks are small (branching 0.05–0.35) because the blocks absorb the level in-sample, and out of sample they have nothing to track with.

**What is left, restated.** Against the simulated null reference (KS 0.028), the working model leaves ≈ 0.025 of KS and a ~8% over-prediction on the quieter window, plus short-lag residual autocorrelation that E2 shows is not a matter of kernel resolution. The over-prediction is excitation that does not materialise (D12). Two families can respond to that — an excitation-carried level with no time blocks (E1′), and any model whose history representation can lower the intensity — and E1′ is run in D12.

**Consequences.** (1) Under `PROTOCOL.md` §7 the neural benchmark is whichever classical model wins on validation of the session under test; on this session that is E1+E2 by +0.0014 — statistically nonzero, practically negligible next to the gains that would matter. (2) The neural design brief is sharpened: not "interactions" in the abstract, but whether letting history lower the intensity explains the remaining residual. (3) Slow cross-side branching in E1 (SELL←BUY 0.35 at 500 s) is small in likelihood terms and is treated as drift absorption until a fresh session says otherwise.

---

### D12 — The floor does not bind; E1′ is the best classical model (2026-09-13)

**Decomposition (validation, benchmark M4).** Predicted count split into baseline and excitation, against the observed:

```
            observed   baseline   excitation   predicted
BUY            7,573      2,241        5,954       8,195
SELL           6,087      2,209        4,482       6,691
```

The observed count is more than three times the baseline alone. A lower intensity was always reachable with the same baseline and less excitation, so the frozen baseline is not what over-predicts — **the excitation term does**, by ~8%. D11's floor argument is withdrawn as an explanation. Rate over-prediction on a quiet window says the model expects more follow-on than materialises, which is a statement about the kernels or about their conditioning, not about μ.

**E1′ — slow scales, single time block per state.** Validation gain over M4 **+0.0039** [+0.0025, +0.0055] at 60 s blocks, [+0.0030, +0.0050] at 300 s, [+0.0032, +0.0049] at 900 s. Best classical model on this session. Residual mean by hour 1.04 / 1.10 / 1.12 → **1.04 / 1.07 / 1.08**; predicted BUY count 8,195 → 8,012; KS BUY 0.042 → 0.038. It gets there by carrying the level in excitation rather than the baseline: baseline BUY 2,241 → 1,492, and very slow same-side branching of 0.40 (500 s) and 0.46 (5000 s). Because that component decays when activity falls, the intensity follows a quiet window down where a frozen baseline cannot. H9 partially met on the exploratory session: the gain interval excludes zero; the residual drift is reduced, not removed.

**Consequences.** (1) E1′ is the classical reference for the neural pilot on this session. (2) Roughly a third of the validation over-prediction is addressable by letting recent activity set the level; the rest is not explained by any exponential-kernel Hawkes tried. (3) The very slow branching in E1′ is an adaptive baseline in kernel form, not a claim about 5000-second propagation, and should be read that way.

---

### D13 — Neural pilot (2026-09-13) — **SUPERSEDED; see D14**

*Review of this pilot found four implementation defects. The numbers below were produced with them and are kept for the record only; `docs/pilot-neural-2026-09-09-SUPERSEDED.log`. The defects: (1) activity summaries counted later events sharing the anchor's timestamp — the MLP and attention inputs were not strictly causal; (2) block log-likelihoods assigned whole compensator pieces to their starting block — the paired intervals were approximate; (3) the training objective dropped the clipped tail through T_train; (4) `evaluate` scored test NLL in validation scripts (no number in D13 used it). All four are fixed with regression tests that fail on the pilot code (`tests/test_regressions.py`). The corrected comparison is D14.*

**Design.** Two encoders on one intensity head, one likelihood, matched categorical inputs (side, mark class, state). Head: `λ_i(τ) = Σ_ℓ [b_iℓ + (a_iℓ − b_iℓ) e^{−β_ℓ τ}]` with the five benchmark decays, `a, b > 0` from the encoder and the latest causally available state, integrated exactly piece by piece across book rows inside each inter-event interval. Encoders: an MLP over causal activity counts per side at 1 s, 10 s, 60 s, 300 s, 1800 s plus 20+ fill counts at 1 s and 10 s, plus the last event; and a 2-layer, 4-head, width-64 transformer over the last N event tokens (side, mark, state, log gap, log age) with the same summaries. Likelihood on the same events and half-open windows as `hawkes.loglik` (`tests/test_neural.py`: Poisson equals the head with a = b to 1e-5). Adam, 40k training events per epoch, early stopping on validation, three seeds, float32, CPU.

**Validation gains over E1′ (the best classical model on this session), 95% block-bootstrap intervals at 60 s; 300 s and 900 s agree throughout:**

```
                        seed 0                  seed 1                  seed 2                  mean ± sd
MLP (summaries)        −0.0023 [−.010, +.005]  −0.0052 [−.013, +.003]  +0.0014 [−.006, +.009]  −0.0020 ± 0.0034
attention N = 64       +0.0246 [+.018, +.032]  +0.0078 [+.001, +.015]  +0.0169 [+.010, +.024]  +0.0164 ± 0.0084
attention N = 128      +0.0023 [−.005, +.010]  +0.0246 [+.018, +.032]  +0.0186 [+.012, +.025]  +0.0151 ± 0.0115
```

Validation KS (jittered): E1′ 0.039 / 0.044; attention 0.014–0.030 / 0.022–0.037.

**Reading (as written at the time; provisional).** (1) *This* MLP over *these* summaries did not improve on E1′ on any seed — which does not show that activity summaries carry no useful information, only that this control did not extract it. (2) The last 64 event tokens do: every N = 64 seed clears zero at every block size; N = 128 adds nothing over N = 64 within seed noise. (3) The best runs of both horizons land at the same −0.4681 (+0.0246), and the weaker seeds stopped at epochs 11–12 — the spread across seeds (sd ≈ 0.01, comparable to the mean) is optimisation variability, not a property of the data, and the pilot's early stopping is too impatient. (4) Runtime: 12–52 min per seed on CPU.

**What this is and is not.** A predictive gain of roughly +0.01 to +0.025 NLL/event over the best classical model, from access to the identity, mark, state and timing of individual recent events rather than their counts. It is one session, validation only, three seeds. It is not a mechanism, and it is not yet a claim: `PROTOCOL.md` §7 requires the direction to replicate on ≥ 2 fresh sessions before it enters the analysis.

**Consequences.** (1) N = 64 is a provisional practical choice for the corrected rerun, not a finding about memory. (2) Before fresh-session evaluation: patience 8, a fixed epoch floor, and 5 seeds, so the seed spread reflects the model rather than the stopping rule. (3) The next two experiments in the brief — continuous marks, then richer book features — are run against this attention model with the same head, each reported as its own increment. (4) The test segment of 2026-09-09 remains untouched by every neural model.


### D14 — Corrected neural comparison, and the first fresh session (2026-09-13)

Full report: `docs/report-2026-09-13.md`. Configuration frozen in `PROTOCOL.md` §7; artifacts under `runs/`.

**Exploratory session (`BTCUSDT-2026-09-09`, validation, 72 / 15 / 5 blocks).** Reference E1′. Five seeds, patience 8, epochs 10–40. MLP over activity summaries: −0.0040 vs E1′, seed s.d. 0.0040, no seed better. Attention N = 64: **+0.0257 vs E1′**, seed s.d. **0.0029**, 5/5 seeds clear of zero at every block size; attention − MLP paired within seed +0.0297, s.d. 0.0031. The pilot's seed spread was its stopping rule.

**Fresh session 1 (`BTCUSDT-2026-09-09-early`, 2.00 h, 18 / 4 / 2 blocks — 300 s and 900 s intervals insufficient).** Classical: **H1 confirmed** (marks +0.1416 [+0.126, +0.158], control −0.0006 ± 0.0014); the ladder's spine replicates; H2 holds in direction and fails strict monotonicity; H3–H9 not confirmed as registered; E1′ worse than M4, so M4 is the reference. Neural, validation: MLP +0.0064, attention +0.0090, attention − MLP +0.0026 ± 0.0065 — no interval clear of zero. **Final test, scored once:** MLP −0.0140 ± 0.0030, attention −0.0057 ± 0.0100 (3/5 seeds negative). **The neural advantage was not replicated here.** That result stands in the evidence; the test segment is not reused.

**What this does and does not say.** It does not say summaries carry no information (this MLP did not extract it), that attention is ineffective (it is robustly better on the exploratory session), or that training-set size caused the fresh-session result (untested). It says the protocol's replication requirement — the direction on ≥ 2 fresh sessions with intervals clear of zero — is not met after one.

**Decision.** The classical model (M4, or E1′ where it wins on validation) remains the working reference. The attention result is **inconclusive**. Collection batch 1 (`docs/collection-batch-1.md`) — two further 8-hour sessions on different UTC dates, evaluated in capture order under the frozen protocol — answers the one open question: whether the exploratory-session gain appears on independent 8-hour data and survives its untouched test segment. All model extensions wait for it.
---

# Open Decisions

Major unresolved questions currently include:

```text
continuous vs discretized event marks

which L2 features matter

normalization strategy

Hawkes kernel: time-varying baseline, multi-scale kernels

market-state conditioning

neural TPP architecture

event-time modelling

generative formulation

synthetic trajectory construction

single-asset vs multi-asset scope
```

These are intentionally unresolved.

---

# Architecture Decision Process

When an important design decision is reached, the AI coding agent should:

```text
identify the problem
      ↓
research viable approaches
      ↓
explain the main options
      ↓
compare trade-offs
      ↓
ask the project owner
      ↓
decide together
      ↓
implement
```

Relevant research papers and existing implementations should be referenced when useful.

The agent should particularly ask for input when decisions affect:

* event representation
* dataset construction
* model architecture
* research methodology
* training objectives
* evaluation
* future project direction

Routine implementation details do not require unnecessary approval.

---

# Expected Evolution

MicroFlux may broadly progress through:

```text
V0
TickForge integration + data exploration

        ↓

V1
Research dataset + classical baselines

        ↓

V2
State-dependent modelling

        ↓

V3
Neural temporal modelling

        ↓

V4
Generative experiments
```

This progression is indicative rather than fixed.

`ARCHITECTURE.md` should evolve alongside the research whenever meaningful architectural decisions are made.
