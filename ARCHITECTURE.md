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

# Initial Repository Structure

Create only the broad architectural directories:

```text
microflux/
│
├── src/
│   └── microflux/
│       ├── tickforge/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       └── analysis/
│
├── configs/
├── tests/
├── notebooks/
└── docs/
```

**Keep these folders empty initially.**

Do not create placeholder Python modules, model files, interfaces, or abstractions simply to fill the structure.

Files should be introduced only when their purpose and design have been discussed and decided.

The directory structure itself may also change as the project evolves.

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

**Open.** The 21 s cross-kernels coincide with a measured activity drift (6.0 → 3.2 orders/s across the capture) and may be absorbing non-stationarity rather than excitation. Two-timescale structure (ms and tens of seconds) is what a single exponential cannot represent. Next: time-varying baseline μ(t) as the non-stationarity control, then sum-of-exponentials kernels.

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
