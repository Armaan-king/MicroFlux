# ML Design Brief — Neural Temporal Point Process for MicroFlux

**For decision, not implementation.** Each component below lists the realistic options, the trade-offs on *this* data, a recommendation, and the ablation that would test the choice. The benchmark throughout is M4 or its best pre-registered extension (`PROTOCOL.md` §3).

What the classical work has established, and what the neural model must respect:

- Events are aggressive orders at ~5/s with ms timestamps; 46% are single fills. Marks (fill count) carry ~8× the held-out information of book state. Excitation is same-side, near-critical (spectral radius ≈ 0.8), and its timescale shifts with mark class.
- The calibrated residual diagnostics on validation found **no saturation and no opposite-side inhibition** beyond the null. What they found is **rate drift** the frozen baseline cannot track, and **short-lag residual autocorrelation** (0.2–2 s). The pre-registered classical extensions did not reach either (D11): adding slow scales gains +0.001 and leaves the residual mean untouched, because a Hawkes intensity is `μ + positive kernels` and **cannot go below its frozen baseline** on a quieter window; a denser scale grid gains nothing. The remaining structure therefore needs an intensity that can *fall* as a function of recent history — which is exactly what N2's history representation provides and N1 does not.
- Therefore the honest question for a neural model is not "can it fit better" — with these inputs and a fair likelihood it probably will, somewhat — but **which capability accounts for the gain**: kernel shape, interactions between events, richer continuous inputs, or memory. The design must make that attributable.

---

## 1. Inputs and marks

**Options**

| | input set | what it tests |
|---|---|---|
| A | side only | architecture alone (no) |
| **B — matched** | side, mark class (4), state class (3) at the event | exactly the benchmark's information: any gain is the encoder's |
| **C — extended** | B + continuous `log fills`, `log quantity`, `levels`, continuous `imb1`, `imb5`, `spread`, `bid/ask depth`, mid-return since last event | the value of continuous marks and richer state, which Hawkes can only bin |
| D | C + the previous event's mark and side as explicit features | usually redundant given history; cheap sanity check |

**Trade-offs.** B is the only configuration in which a neural-vs-Hawkes comparison is clean. C is where a neural model has a structural advantage (Hawkes needs classes; a network takes a continuum). Feeding C first and reporting one number would silently credit the architecture with the inputs.

**Recommendation.** Run B, then C, always reporting `C − B` for the same architecture separately from `B − benchmark`. Marks are **inputs only** for the primary comparison; predicting the mark is a secondary head, off by default.

**Ablation.** `B − benchmark` (architecture), `C − B` (information). Within C, drop-one-feature on validation.

## 2. History horizon

**Options.** (A) fixed event count *N* ∈ {32, 64, 128, 256, 512}; (B) fixed time window (30 s, 120 s); (C) unbounded via a recurrent state (Neural Hawkes CT-LSTM).

**Trade-offs.** At 5 orders/s, 128 events ≈ 25 s and 512 ≈ 100 s. The benchmark's kernels place most mass inside 5 s (~25 events) with a 50 s tail; E1 tests whether minutes matter. Attention cost is quadratic in *N*; a recurrent state has no window but trains sequentially over 120k events (slow, and truncated BPTT reintroduces a window anyway). Time windows give variable sequence lengths and complicate batching for little gain at this rate.

**Recommendation.** Event count, **N = 128** as the default, with the window forming the sequence for both training and evaluation. Every event's history is the *N* events before it regardless of split boundary (history is observed, not fitted).

**Ablation.** Validation NLL against *N* ∈ {32, 64, 128, 256, 512}. Flat beyond some *N* is the memory the data needs; if E1 wins classically, expect the curve to keep improving to 512.

## 3. Time representation

**Options.** (A) raw Δt in seconds — spans 1 ms to 15 s, heavy-tailed, poorly conditioned; (B) `log(Δt)` — the standard, but 0.3% of gaps are exactly 0 and 1 ms is a hard floor; (C) **`log(Δt + ε)`, ε = 0.5 ms** — half a tick, treating the quantisation as what it is; (D) absolute-time sinusoidal encodings (THP) in addition to Δt; (E) both a Δt feature and a learned encoding.

**Trade-offs.** (C) is (B) with the floor made explicit. Absolute time carries intraday information the benchmark gets from μ(t); adding it to the neural model is a fair match to the baseline blocks, but it must be the *session-relative* hour, never wall-clock, so that fresh sessions are comparable.

**Recommendation.** (C) as the per-event feature; plus **session hour** as one scalar input, matching μ(t). No THP-style positional encoding of absolute time as an attention key — it is what made THP's intensity depend on the absolute time of the last event, which has no meaning here.

**Ablation.** ε ∈ {0.1, 0.5, 1} ms; with/without session hour.

## 4. Encoder

This is the decision that matters. Ordered from least to most expressive; each rung is chosen to answer one question.

| | model | intensity between events | what it adds over the rung below | cost |
|---|---|---|---|---|
| **N1** | **Neural-kernel Hawkes** — keep the additive Hawkes form; replace the fixed-scale weights with a small MLP: `α_l(i, j, mark, state)` for each of the L fixed scales | closed form (sum of exponentials) | kernel shape as a smooth function of mark and state instead of a grid of classes; **still additive, still interpretable, integral still exact** | trivial |
| **N2** | **Attentive TPP with exponential-decay intensity** (SAHP-style: `λ_i(t) = softplus(μ_i + (η_i − μ_i) e^{−γ_i (t − t_last)})`, with μ, η, γ produced by self-attention over the window) | closed-form decay between events; softplus keeps it positive and lets history *lower* it | **interactions**: the effect of an event can depend on what else is in the window; inhibition is representable | moderate |
| N3 | Transformer Hawkes (THP, Zuo et al.) | `softplus(α (t − t_j)/t_j + w·h_j + b)` | attention as in N2, but the time term is odd (scales with absolute t_j) and EasyTPP found its gains fragile | moderate |
| N4 | Neural Hawkes (Mei & Eisner) — continuous-time LSTM | decaying cell state, softplus readout | unbounded memory; fully continuous-time | slow (sequential over 120k events) |
| N5 | Intensity-free (Shchur et al.) — log-normal mixture over Δt, categorical over type | none: models the next-event distribution directly | exact likelihood, fastest; **no λ(t)**, so no compensator, no residual diagnostics, no propagation analysis | low |

**Trade-offs on this data.** N1 is the smallest step above the benchmark and isolates *kernel shape*. N2 is the smallest step that lets history **lower** the intensity — the capability D11 shows the remaining residual needs and the one Hawkes structurally lacks — and, with it, interactions between events. N3's specific intensity form is not worth reproducing; its attention is N2's. N4 answers a memory question that E1 and the horizon ablation answer more cheaply. N5 is a strong predictive model but cannot produce the propagation analysis that is the project's objective; it is useful only as a cross-check that our likelihood numbers are sane.

**Recommendation.** **N1 then N2.** Report `N1 − benchmark` (is the grid of classes the limitation?) and `N2 − N1` (are interactions the limitation?). Skip N3 and N4 unless N2 wins by a margin that memory or a different time term could plausibly widen. N5 optional as a likelihood cross-check.

**Ablation.** N2 depth (1, 2 layers), heads (1, 4), width (32, 64, 128); attention masked to the last 16 events (does interaction need reach?).

## 5. State conditioning

**Options.** (A) state features on each event's input token (= benchmark `k(s)`); (B) the book row current at prediction time injected into the intensity head (= benchmark `μ(t, s)`); **(C) both**; (D) a separate encoder over the recent **book history** — e.g. the last 30 rows of the 1 Hz series — the "L2 state encoder" of GOAL.md.

**Trade-offs.** (C) matches the benchmark. (D) gives the neural model information the benchmark does not have — the *trajectory* of the book, not its current value — so any gain from (D) is information, not architecture, and must be reported as such.

**Recommendation.** (C) for the matched comparison; (D) as the single extension most likely to matter, run last, reported as `D − C`.

**Ablation.** State shifted within split (protocol control); `D − C`; within D, history length 5 / 30 / 120 rows.

## 6. Outputs / prediction targets

**Options.** (A) conditional intensity per side, λ_BUY(t), λ_SELL(t); (B) next-event side and time as point predictions; (C) the next-event distribution directly (N5); (D) a mark head predicting the next order's fill class.

**Recommendation.** **(A)** — it is what the benchmark produces, what the likelihood needs, and what the propagation analysis reads off. (B) is derived from (A) and reported as a secondary metric. (D) off by default: the benchmark conditions on marks and does not predict them, and adding a mark likelihood term would make NLL/event incomparable.

## 7. Loss and integration

The TPP log-likelihood is `Σ_k log λ_{m_k}(t_k) − Σ_i ∫ λ_i(t) dt`. Everything difficult is in the integral.

| encoder | integral | error |
|---|---|---|
| N1 | closed form — it is still a sum of exponentials | none |
| N2 | closed form — `∫ softplus(a + b e^{−γτ}) dτ` has no elementary antiderivative, **but** with `λ = μ + (η − μ) e^{−γτ}` and positivity enforced on μ, η by softplus *on the parameters* rather than on λ, the integral is exact | none |
| N3 / N4 | Monte Carlo over each inter-event interval | variance; bias if few samples |
| N5 | none | none |

**Recommendation.** Use the exact forms for N1 and N2 — that is a reason to prefer the exponential-decay parameterisation over THP's. If Monte Carlo is ever used, ≥ 20 samples per interval and a check that doubling them moves validation NLL by less than the bootstrap interval width.

**Two rules that make NLL/event comparable to the benchmark at all.** (1) The likelihood is computed on **exactly the same events and the same half-open window** `[a, b)` as `hawkes.loglik`, with the same history convention. (2) Training and evaluation use the **recorded ms timestamps**, not jittered ones; jitter is for residual diagnostics only, applied identically to every model.

**Ablation.** For N2, the exact integral against a 100-sample Monte Carlo estimate on validation — they must agree, which is also a unit test of the implementation.

## 8. Evaluation

Per `PROTOCOL.md`:

- **Primary** — validation NLL/event gain with a 95% block-bootstrap interval; test once; ≥ 2 fresh sessions to confirm.
- **Goodness of fit** — time-rescaling residuals on jittered times, KS and conditional means by interval opener, against a simulated null. For N1 the null simulates by thinning as now. For N2 the intensity is bounded on each interval by `max(μ, η)`, so thinning works; the null is simulated from the fitted N2.
- **Propagation** — N1 reads off directly (it is a Hawkes). For N2 the kernel is implicit; extract an **empirical response**: insert a synthetic event of given side/mark/state at lag τ into real histories, measure the change in λ, average over contexts. Compare its shape by mark and state to the benchmark's branching-by-scale. Attention weights reported only as a secondary, with the usual caveat that attention is not attribution.
- **Secondary** — next-event side accuracy, log-time RMSE.
- **Controls** — shuffled marks within (split, side), shifted state within split, both on the neural models exactly as on the classical ones.

---

## The comparison, as it would run

```
benchmark  = M4 (E1 and E2 did not earn their parameters, D11)
E1'        = slow scales, single baseline   gain vs benchmark  →  rate tracking without blocks
N1(B)      matched inputs         gain vs benchmark  →  kernel shape
N2(B)      matched inputs         gain vs N1(B)      →  interactions
N1(C), N2(C)  extended inputs     gain vs same(B)    →  continuous marks + richer state
N2(C)+D    book-history encoder   gain vs N2(C)      →  book trajectory
```

Five numbers with intervals, each attributable to one thing. Decision rule and replication requirement in `PROTOCOL.md` §7.

## What I would decide, if it were mine

Inputs B then C. Horizon 128. Time `log(Δt + 0.5 ms)` plus session hour. Encoder N1 then N2, exact integrals, no THP time term. State C then D. Output λ per side. Evaluation per protocol. The classical control that must run alongside: **E1′**, slow scales with a single baseline and no time blocks, so that recent activity carries the level and the intensity can track a quiet window from above. If E1′ closes the residual, N2's gain over N1 will be small and the story is "rate tracking"; if it does not, N2 is testing something a Hawkes cannot express. Either outcome is informative, which is what makes it the right experiment.
