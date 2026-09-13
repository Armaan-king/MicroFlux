# MicroFlux — Goal

## Objective

MicroFlux is a quantitative ML research project studying **how information propagates through high-frequency market activity and the Level-2 order book over time**.

It builds on **TickForge**, which already provides deterministic historical market events, L2 order-book reconstruction, microstructure features, replay, and a research API.

MicroFlux owns the **research, modelling, training, and evaluation layer**.

---

## Project Documents

| Document | Status | Purpose |
|---|---|---|
| `GOAL.md` | **Stable** | Research intent, scope boundaries, success criteria. Changes rarely and deliberately. |
| `ARCHITECTURE.md` | **Flexible / living** | How the project is currently built. |
| `PROTOCOL.md` | **Frozen** | What is computed on a fresh session, the controls, the pre-registered hypotheses. Changes are logged and apply only forward. |
| `CLAUDE.md` / `AGENTS.md` | Working notes | Instructions for AI coding agents. |

**`ARCHITECTURE.md` is explicitly a flexible document.**

It is expected to be modified whenever the project's prospect, technology, scope, or research direction changes. It records the *current* architecture, not a fixed contract, and it should never be treated as a commitment that constrains a better research finding.

When evidence from experiments contradicts the documented architecture, update `ARCHITECTURE.md` — do not preserve it for consistency's sake.

`GOAL.md` describes *what* MicroFlux is trying to learn. `ARCHITECTURE.md` describes *how* it currently does that, and the "how" is allowed to change.

---

## Core Research Question

How does the current market state affect the way trading activity propagates into future market events?

For example:

```text
BUY TRADE
    ↓
ASK DEPTH DECREASE
    ↓
FURTHER BUY ACTIVITY
    ↓
MARKET STATE CHANGES
```

The project studies:

* **event propagation** — which events influence later events and over what timescale
* **state dependence** — how spread, depth, imbalance, liquidity, and volatility affect those relationships
* **generative dynamics** — whether realistic future event sequences can eventually be generated

The goal is not simply to predict price direction.

---

# TickForge Integration

TickForge remains the upstream source of market truth.

```text
Binance
   ↓
TickForge
   ↓
Research API
   ↓
MicroFlux
```

MicroFlux should use:

```text
/research/{symbol}/sessions
/research/{symbol}/events
/research/{symbol}/export
```

Bulk historical data should be downloaded from TickForge, cached locally, and transformed into research datasets.

TickForge continues to own:

* Binance ingestion
* normalized events
* L2 reconstruction
* sequence correctness
* storage
* replay
* capture ordering
* generic microstructure features

MicroFlux owns:

* research event definitions
* dataset construction
* feature selection
* temporal modelling
* model training
* generative modelling
* evaluation
* experiments
* research analysis

---

# Research Data

TickForge events will be transformed into a model-friendly event representation.

An initial event vocabulary may include:

```text
BUY_TRADE
SELL_TRADE

BID_DEPTH_INCREASE
BID_DEPTH_DECREASE

ASK_DEPTH_INCREASE
ASK_DEPTH_DECREASE
```

Each event may contain:

```text
event type
timestamp
time since previous event
relative price
quantity / magnitude
book level
```

The associated market state may contain:

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

The exact event vocabulary, feature set, time representation, and sequence/window design should be chosen **after inspecting the data**.

---

# Research Progression

## 1. Data Exploration

First study the TickForge dataset.

Understand:

* event frequencies
* event-type balance
* inter-arrival times
* depth-change distributions
* session lengths
* spread and liquidity behaviour
* order-flow behaviour
* volatility regimes

Use chronological train, validation, and test splits.

This stage should determine the event representation and whether sequences are better represented using fixed event counts, time windows, or another approach.

---

## 2. Point-Process Baselines

Establish simple interpretable baselines:

```text
Poisson Process
      ↓
Multivariate Hawkes Process
```

Use them to measure self- and cross-excitation between event types.

Example questions:

```text
Does BUY_TRADE increase future BUY_TRADE intensity?

Does ASK_DEPTH_DECREASE increase future aggressive buying?

How quickly does an excitation effect decay?
```

These relationships represent statistical temporal influence rather than proven causality.

---

## 3. State-Dependent Hawkes Modelling

Extend the event model so excitation can depend on the current L2 state.

```text
Event History
      +
L2 Market State
      ↓
Future Event Intensities
```

This stage should investigate whether propagation differs across:

```text
high / low liquidity
balanced / imbalanced books
high / low volatility
tight / wide spreads
```

This is one of the core research directions for MicroFlux.

---

## 4. Neural Temporal Point Processes

Compare classical models with more expressive neural temporal models.

Candidate approaches include:

* Neural Hawkes Process
* Transformer Hawkes Process
* other suitable Temporal Point Process models

A possible architecture is:

```text
Event History
     ↓
Event + Time Encoding
     ↓
Transformer / Neural Encoder
          ↑
          │
     L2 State Encoder
          ↓
Event-Type + Event-Time Distribution
```

The neural architecture should be chosen only after comparing the available approaches and understanding what additional capability it provides over the simpler models.

---

## 5. Generative Extension

After reliable predictive and point-process baselines exist, investigate long-horizon event generation.

Candidate approaches include:

* asynchronous diffusion for Temporal Point Processes
* flow matching
* autoregressive event generation
* market-state-conditioned diffusion

Conceptually:

```text
Historical Events
       +
Current L2 State
       ↓
Generative Model
       ↓
Future Event Sequence
       ↓
Synthetic Market Trajectory
```

This stage should investigate whether a generative model can reproduce realistic temporal and microstructure behaviour.

---

# Research References

The following papers and implementations are starting references rather than fixed architectural requirements.

### State-Dependent Hawkes Processes and Their Application to Limit Order Book Modelling

**Morariu-Patrichi & Pakkanen**

Primary reference for modelling the feedback between order flow and LOB state.

Relevant ideas:

```text
state-dependent excitation
spread conditioning
queue-imbalance conditioning
maximum-likelihood estimation
```

---

### The Neural Hawkes Process

**Mei & Eisner — NeurIPS 2017**

Reference for replacing fixed Hawkes dynamics with learned neural temporal representations.

Relevant ideas:

```text
continuous-time neural event modelling
multivariate event sequences
event generation
```

---

### Transformer Hawkes Process

**Zuo et al. — ICML 2020**

Primary Transformer-based TPP reference.

Relevant ideas:

```text
self-attention over event histories
temporal encoding
event-type prediction
event-time modelling
point-process likelihood
```

The official implementation should also be examined for its preprocessing, likelihood calculation, integration strategy, and training loop.

---

### EasyTPP

**ICLR 2024**

Use as a reference implementation framework for comparing Temporal Point Process models.

Relevant ideas:

```text
common dataset interfaces
configurable experiments
model abstraction
training/evaluation runners
reproducible benchmarking
```

MicroFlux does not need to copy EasyTPP's framework, but should learn from its approach to comparing different TPP models consistently.

---

### ADiff4TPP — Asynchronous Diffusion Models for Temporal Point Processes

Reference for the advanced generative stage.

Relevant ideas:

```text
diffusion / flow-based event generation
asynchronous temporal sequences
next-event prediction
long-horizon sequence generation
latent event representations
```

This should be investigated only after simpler temporal models are working.

---

### TRADES / DeepMarket

Reference for diffusion-based market simulation and practical ML project structure.

Relevant ideas:

```text
LOB-conditioned generation
market simulation
PyTorch training
preprocessing pipelines
experiment management
synthetic-vs-real evaluation
```

It is particularly useful for studying how generative market models are evaluated beyond ordinary prediction accuracy.

---

# Paper Implementation Review

Before implementing a model from or inspired by a paper, the AI coding agent should discuss the paper with the project owner.

For each major paper/model, answer:

1. **What exact problem does the paper solve?**
2. **What are its inputs and outputs?**
3. **How are events and time represented?**
4. **What model components are essential versus implementation choices?**
5. **What loss/objective is used and why?**
6. **How is the model evaluated?**
7. **What assumptions does the paper make about its dataset?**
8. **Which assumptions do not hold for TickForge's Binance L2 data?**
9. **What parts should MicroFlux reproduce faithfully?**
10. **What parts should we adapt, simplify, or deliberately avoid?**
11. **What experiment would demonstrate that the added model complexity is worthwhile?**

Do not silently translate a paper architecture into code.

When multiple implementations are possible, present the main options and let the project owner decide before making a consequential architectural choice.

---

# Evaluation

MicroFlux should evaluate models at three levels.

## Predictive Quality

Possible metrics include:

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
differences across market regimes
```

## Generated Market Realism

For generative models, compare synthetic and real data using:

```text
spread
depth
imbalance
inter-arrival times
order flow
liquidity
volatility
return distributions
temporal correlations
```

---

# Experiments

Models should be compared progressively.

A likely research ladder is:

```text
Poisson
    ↓
Hawkes
    ↓
State-Dependent Hawkes
    ↓
Neural / Transformer TPP
    ↓
Generative TPP
```

Possible feature ablations should investigate whether the model actually benefits from:

```text
L1 vs L5 vs L10 state
imbalance
depth
volatility
trade information
market-state conditioning
```

The most complicated model does not need to win.

Understanding **why models behave differently** is part of the project.

---

# Engineering Direction

Keep the research code organized around:

```text
data/
models/
training/
evaluation/
analysis/
configs/
tests/
```

Use Python modules for core modelling logic.

Use notebooks mainly for:

* exploratory analysis
* visualisation
* interpreting experiments

Experiments should be reproducible through configuration where useful.

PyTorch will be the primary neural modelling framework unless research findings give a strong reason to choose otherwise.

The concrete shape of this structure lives in `ARCHITECTURE.md`, which is a **flexible document** and is expected to change as the project's technology, scope, and research direction evolve.

---

# Decision-Making Principle

Important research and architecture decisions should be made deliberately with the project owner.

Before deciding things such as:

* event taxonomy
* temporal representation
* sequence/window construction
* state features
* Hawkes parameterisation
* Transformer architecture
* point-process likelihood
* diffusion/flow formulation
* generation method
* evaluation methodology

the AI agent should explain the main viable choices, their trade-offs, and how the relevant papers approach the problem.

Then ask the project owner what direction they want to explore.

Prefer:

```text
research
   ↓
compare approaches
   ↓
discuss
   ↓
decide
   ↓
implement
   ↓
evaluate
```

The architecture should evolve from evidence and discussion rather than being silently assumed at the beginning.

---

# Success Criteria

MicroFlux succeeds if it:

1. Builds a reliable research dataset from TickForge.
2. Establishes interpretable event-propagation baselines.
3. Measures how L2 state changes temporal event dynamics.
4. Compares classical and neural temporal models rigorously.
5. Uses chronological, reproducible validation.
6. Produces meaningful analysis of information propagation.
7. Investigates generative event modelling if the earlier results justify it.
8. Clearly explains which research ideas worked, which did not, and why.

The final MicroFlux architecture should emerge through research and experimentation rather than being predetermined.
