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

# Open Decisions

Major unresolved questions currently include:

```text
event taxonomy

continuous vs discretized event marks

event-count vs time-based windows

which L2 features matter

normalization strategy

session handling

Hawkes parameterization

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
