# Collection batch 1 — prospective plan

**Recorded 2026-09-13, before any session in this batch exists.** Nothing in this
plan changes after the first capture starts. Sessions are evaluated in the
order they are captured, each exactly once, under `PROTOCOL.md`; collection
stops when two eligible sessions have been evaluated, whatever they show.

## What

Two BTCUSDT captures of **at least 8 hours** each, on **two different UTC
dates**, neither overlapping any session already in the evidence
(`BTCUSDT-2026-09-09` 11:32–19:32 UTC, exploratory; `BTCUSDT-2026-09-09-early`
05:04–07:04 UTC, fresh session 1).

## How — the existing TickForge capture, settings unchanged

From the TickForge repository, in PowerShell:

```powershell
cd C:\Users\iidab\OneDrive\Desktop\TickForge
powershell -File scripts\run_capture.ps1 -Hours 8.5
```

`-Hours 8.5` rather than 8: the batch rule is >= 8 h of *usable* span between the
first and last order, and the first seconds go to the snapshot seed. The
half-hour is margin, nothing else.

That is the script the 2026-09-09 session was captured with. Defaults it uses
and that must not change: `-Symbol BTCUSDT`, `-Root C:\tickforge-runs`,
`-HeartbeatSeconds 300`, Binance `@depth` (1000 ms) + `@trade`, file rolling
as configured in TickForge. It suppresses sleep for the duration; the laptop
must stay on mains power and connected. The output partition is
`C:\tickforge-runs\binance\BTCUSDT\<UTC date>\`.

**Timing constraint.** TickForge partitions by the UTC date of each event and
rolls the partition at midnight UTC. For 8.5 hours to land in one partition
the capture must **start between 00:00 and 15:00 UTC**. The laptop is on
Eastern time (EDT = UTC - 4), so that is **20:00 the previous evening to
11:00 local**. A capture that straddles midnight UTC yields two partitions,
each shorter than 8 h; each would be judged separately and would fail the
span rule.

**Schedule.** Recorded 2026-09-13 20:54 UTC; today's partition has too little
left. Capture 1: UTC date **2026-09-14**, start between 20:00 EDT on the 13th
and 11:00 EDT on the 14th. Capture 2: UTC date **2026-09-15**, same window one
day later. The launching PowerShell window holds the sleep suppression for
the whole run, so each capture is started by hand in its own window and that
window stays open until the script prints "sleep suppression released".

Run the two captures on different UTC dates. Do not start a capture on a date
that already has a partition in `C:\tickforge-runs\binance\BTCUSDT\`.

## Eligibility (PROTOCOL.md §1b, checked by `run_session.py`)

`load.is_continuous` true across all files · ≥ 1 snapshot · span ≥ 1 h ·
≥ 10,000 collapsed orders · no overlap with existing sessions. A capture that
fails is recorded in `runs/<session>/eligibility.json` and not evaluated; the
next capture is taken. It is not re-captured on a chosen date.

## What is computed, per session, in this order

```
python run_session.py --date <UTC date> --batch docs/collection-batch-1.json
```

The batch file enforces >= 8 h, distinct UTC dates, no overlap with and no
duplicate of any evaluated data (recognised by fingerprint, not folder name),
capture order, and the batch size; the general one-hour eligibility and the
two-hour session's standing are unchanged (`src/microflux/batch.py`,
`tests/test_batch.py`). Excluded captures and their reasons are recorded in
the batch file.

1. `replicate.py` — the registered classical ladder (M0–M4, M4 without kernel
   state, E1, E2, E1′), paired gains at 60 / 300 / 900 s, mark-shuffle and
   within-split state-shift controls, calibrated diagnostics with simulated
   nulls, H1–H9 with their registered statistics. Validation only.
2. `fit_neural.py` — MLP and attention (N = 64) on the shared head, matched
   inputs, seeds 0–4, patience 8, epochs 10–40, artifacts per run. Paired
   intervals: each vs the classical reference, attention vs MLP within seed.
   Validation only.
3. `final_test.py` — the classical reference and every neural seed scored on
   the test segment, **once**. Logged in `PROTOCOL.md`.

Settings frozen: architecture, inputs, summaries, head, seeds, stopping rule,
block sizes, split fractions, tercile and mark-class rules.

## What would count

- **H1 (marks)**: confirmed on a session if gain > 0.05 with the interval
  clear of zero and the control ≈ 0. Already confirmed on fresh session 1.
- **Neural**: the attention-vs-reference direction on validation *and* on the
  final test, on ≥ 2 of the fresh sessions (session 1 plus these two), with
  60 s intervals clear of zero — per PROTOCOL.md §7. Fresh session 1 did not
  replicate it; that stays in the record.
- **Seed variability** (s.d. across the five seeds) is reported next to, and
  never combined with, the block-bootstrap sampling interval. Every interval
  is reported with the number of blocks behind it; 300 s and 900 s intervals
  on validation or test windows under ~4 h are marked insufficient.

## Recording

- Before the first capture: this file, committed.
- After each capture: `runs/<session>/eligibility.json`, then the three stage
  outputs, committed together with the run logs.
- After the batch: D15 in `ARCHITECTURE.md` and a one-line verdict per session
  in `PROTOCOL.md`'s log.
