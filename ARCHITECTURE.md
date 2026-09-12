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

Flat modules, one job each. Files appear when they have a purpose; the six planned subdirectories (`tickforge/ data/ models/ training/ evaluation/ analysis/`) are unused until a second module of the same kind exists.

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

explore.py          Stage 1 measurements
fit.py              Stage 2 six-model ladder
fit_state.py        Stage 3 state-dependent excitation, with permutation control
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
