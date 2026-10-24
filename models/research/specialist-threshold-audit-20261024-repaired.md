# NIFTY history repair and threshold study — 24 October 2026

Download #10 completed for 25 August–23 October 2026. It added 604 spot,
7,800 futures and 36,076 option candles. The September future had candle
history despite being absent from Groww's contract list. Discovery now
requires real candle evidence for that fallback and looks ahead for monthly
contracts that expire after the requested historical end. Repair requests
can recheck old checkpoints while preserving existing candles.

Database backup: `data/backups/niftypilot-20261024T093727Z-before-history-repair.db`.

## Study scope

38,351 eligible observations on 924 trading days, from 3 January 2023 to
13 October 2026. Repeated intraday observations share a trading day; they are
not 38,351 independent trades. Models are weighted by day, split by day,
and trade performance takes only the first qualifying entry per day.

| Partition | Dates | Observations |
|---|---|---:|
| Fit | 3 Jan 2023–3 Apr 2025 | 23,333 |
| Calibration | 7 Apr–18 Aug 2025 | 3,443 |
| Threshold selection | 20 Aug 2025–6 Jan 2026 | 3,993 |
| Historical evaluation | 8 Jan–13 Oct 2026 | 7,429 |

One whole trading day is purged at each boundary. These historical periods
have already been inspected during earlier development; they are research
evaluation, not a pristine final test. 14–23 October 2026 was not used in
this study and remains reserved for a frozen candidate.

The simulator now checks stops and targets in the entry candle, requires an
exact next-bar entry and a continuous exit path, and squares off at the
15:20 candle open. Volume and OI are excluded from the specialist models
until historical/live field definitions have been reconciled. The current
experiment still uses historical ATM selection; it does not reproduce all
live quote, delta, spread and ITM fallback filters.

## Results

| Specialist | Result |
|---|---|
| ORB CE / PE | No tuning cutoff passed cost, trade-count and neighbouring-cutoff gates |
| Pullback CE / PE | No tuning cutoff passed those gates |
| Failed Breakout CE / PE | Insufficient samples/days in at least one chronological partition |
| VWAP Reclaim CE | 42% cutoff passed tuning, failed later evaluation |
| VWAP Reclaim PE | Negative calibration slope; rejected |

VWAP Reclaim CE illustrates the generalization problem:

| Period | Trades | Wins | Profit factor | Net P&L |
|---|---:|---:|---:|---:|
| Threshold selection | 25 | 60.00% | 1.828 | ₹7,753.88 |
| Later evaluation | 47 | 40.43% | 0.675 | −₹10,414.14 |

No specialist is eligible for promotion. The selected V5 Shadow artifact
was not changed. The study exports a report, not a runtime model bundle.
Two older special sessions (18 May 2024 and 21 October 2025) still have
incomplete regular-hour futures alignment; they are explicitly reported.

Full diagnostics: `specialist-threshold-audit-20261024-repaired.json`.
