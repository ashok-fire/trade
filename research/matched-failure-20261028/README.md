# Matched Shadow failure report — 28 October 2026

Follow-up: the human later authorised protocol alignment and the targeted option
import. Those prerequisites are now resolved; see
`../protocol-alignment-20261028/README.md`. The audit below retains its original
pre-import findings and is not rewritten as a new performance evaluation.

## Decision

**No model replacement, threshold adjustment or retraining is justified.**
The next supported work is protocol/data alignment, not another learner or UI
feature. This diagnostic made no application, registry, mode, risk, broker or
database changes. Model9 and reference11 remain frozen; there is no claim that
the requested >70% net-win goal or live readiness has been demonstrated.

The comparisons below use the existing 26–27 October GROWW/NIFTY evidence.
These dates were already inspected; this is not a new blind final test.

## 1. What reproduced, and what did not

| Check | Result |
|---|---|
| Stored quote-selection replay, original quotes/clock/filters | 116/116 matched, including rejections |
| Reference11 scores on original stored feature vectors | 46/46 matched |
| Model9 scores on scoreable original signal feature vectors | 33/33 matched; 46 other signals lacked scoreable context |
| Closed-trial causality, quote ages, rounded fills, gross/fees/net, exit reason | 46/46 passed all 11 checks |
| Current-database causal feature reconstruction | 212/352 rows matched all features; 140 differed |
| Reconstructed setup direction | 352/352 matched |
| Exact same-contract signal + next-entry historical option candles | 0/46 available pairs |

Recorded quote timestamps are provider/adapter evidence, not proof of exchange
latency or actual executable fills. Successful accounting does not validate a
strategy's predictive edge.

All 140 feature discrepancies are from 26 October, in four fields:
`opening_range_high_distance`, `gap_pct`, `day_open_distance`, and
`opening_range_width_pct`. Reconstructed 27 October features matched.

For the first recorded 26 October boundary, the original features imply an
opening-range high/day open of22524.40. The current stored-candle reconstruction
has opening-range high22517.15 and day open22508.20. These are input differences,
not proof that the model predicted incorrectly. Runtime candle ingestion can
overwrite existing OHLC values with later broker responses (`_store_runtime_rows`
in `app/runtime.py`). A revision or other earlier data change is a plausible
explanation, **not an established root cause**: decision-time raw candle versions
were not preserved. None of these discrepancies changed the reconstructed setup
directions; their effect on alternate scores or profits is not established.

Use original recorded decision features to reproduce scores. Do not quietly
relabel a current-candle reconstruction as the identical original opportunity.

## 2. Research outcomes are not portfolio-eligible trades

There were22 CLOSED research trials on26 October and24 on27 October:46 rows,
but only TWO observed dates. Entry events overlap; family totals cannot be added
as portfolio P&L or treated as46 independent daily decisions.

At fixed capital100000,3% gross premium-stop risk and25% premium stop, a65-unit
lot must have entry premium no more than approximately184.62 per unit. All24
27 October entries exceeded the one-lot risk budget. None failed solely because
of cash affordability. This is a causal risk test using the actual simulated
entry quote/fill and quantity, not the eventual outcome.

All22 eligible observations are from26 October. There are no eligible27 October
entries in either comparison arm. The observer intentionally records one-lot
counterfactuals without portfolio risk filtering; its wins are not evidence that
the configured account could have taken those trades. This finding does NOT
authorise raising capital/risk or narrowing stops after seeing results.

## 3. Matched score-versus-rule comparison

Both arms use exactly the same frozen family/direction/quote protocol, one-lot
cash/risk eligibility, then the first selected entry per date. Selection occurs
before outcome inspection. A pending/incomplete first selection remains unknown;
it is never replaced by a later known winner. Disabled score directions do not
become passing entries. The temporary1% slippage interval stays separate.

| Source protocol | Entered rows | Eligible rows | Rule-first daily decision | Reference11 first-passing decision | Difference |
|---|---:|---:|---|---|---:|
|0.05% / ORB CE|34|20|26 Oct: +1964.51|Same entry: +1964.51|0.00|
|0.05% / Pullback CE|3|2|26 Oct: +2578.80|Same entry: +2578.80|0.00|
|1% / ORB CE|6|0|No entry|No entry|—|
|0.05% / Failed Breakout PE|1|0|No entry|No entry|—|
|0.05% / Pullback PE|1|0|No entry|No entry|—|
|0.05% / VWAP Reclaim CE|1|0|No entry|No entry|—|

Thus there is **no incremental first-daily selection difference for reference11**
in the eligible evidence. The two family rows are the same observed day, not two
independent sessions. A100% fraction over a single selected day is not a reliable
win-rate estimate or sufficient calibration evidence. No uncertainty interval or
family selection is justified from this sample.

Model9's frozen score applied to the SAME reference ORB quote/feature population
selects a later26 October trial with+2283.47 versus rule-first+1964.51: a
descriptive+318.96 difference on ONE date. Those outcomes still use the reference
quote-anchored exit protocol, not model9's signal-close-anchored portfolio rules.
This is neither a validated benefit nor a result authorising model9 promotion.
Details and exact selected IDs: `paired-score-supplement.json`.

The actual selected model9 portfolio has one closed trade across these sessions,
net+611.60 on26 October and no entry on27 October. It used a separate collection
clock and its own entry/risk/exit logic. Do not substitute reference-counterfactual
P&L for that actual simulated portfolio result.

## 4. Confirmed exit-geometry mismatch

Historical training/replay and selected model9 Shadow trades anchor the one-ATR
stop and1.5-ATR target to the SIGNAL candle close. Reference11's parallel observer
anchors both to a LATER underlying quote. All46 stored research geometries
reproduced the quote-anchor rule. The absolute quote-versus-signal difference had
median0.082ATR and maximum0.537ATR.

This is a concrete protocol difference. It can move both barriers enough to
change an exit, but the saved entry/exit endpoints cannot determine when a
different earlier stop/target would have triggered. Do not manufacture an
alternative profitable path or translate the geometry difference into P&L.

Other known differences remain: historical fills use next-bar open / exit candle
proxies; runtime uses later bid/ask plus tick-rounded slippage; research exits
are observed on15-second polls. Historical volume/OI and live cumulative quote
fields are not interchangeable. The earlier decision-aligned research excludes
volume/OI inputs, but the selected legacy artifacts are not those new bundles.

Two underlying-target exits were net option LOSSES. All labels here use net
option P&L after estimated charges, never target touches. Same-exit doubled
slippage figures in the JSON are fill-only sensitivities, not full exit-path
stress replays.

## 5. Full historical-versus-quote replay remains blocked

The database has no exact same-contract historical option signal/entry pairs
for these46 entries on26–27 October. This does not mean2023–2025 data disappeared.
It means these two NEW sessions cannot yet be replayed against stored option
OHLC. No missing path was fabricated, no alternate contract was substituted and
no broker request was made to fill the gap during this read-only audit.

The prior data supplies causal spot/futures warm-up for current feature
reconstruction; reserved14–23 October option outcomes were not used for a final
test, relabelling or candidate selection. Previously searched2026 audit results
also remain exposed research evidence, not fresh validation.

## 6. Evidence-backed next action — narrow alignment, no new model

1. Before another training claim, finish the matched replay: obtain verified
   same-contract26–27 October option history through a separately authorised
   import, retain missing outcomes and compare frozen historical proxies with
   the actual observed quote evidence. No parameter search on these dates.
2. A future observer protocol should use the SAME documented signal-close anchor
   as the selected portfolio/training, and maintain an explicit one-lot
   cash/risk-eligible evaluation view. Keep raw research observations available;
   never overwrite46 existing outcomes or silently mix protocol versions.
3. Diagnose the26 October opening-input revisions before treating reconstructed
   features as original training opportunities. Original saved decisions remain
   the source of truth; this is not evidence that arbitrary new indicators help.
4. Freeze any corrected protocol before later collection. Test reproduction and
   future-input invariance before measuring outcomes. Keep models9/11 and gates
   fixed; no deployment, capital increase or threshold reduction follows from
   the two-session data.

This report narrows a defensible next change to comparability/data integrity.
It does not establish that the change improves win rate or creates a profitable
strategy. There is no supported new learner/cutoff search to start now.

## Reproduction, retained failure and safeguards

`specification.json` was written and source-hashed before this audit's results.
`run-01` retained a complete source snapshot but failed JSON export on a NumPy
boolean. `serialization-amendment.json` changes only scalar normalization and its
regression test/diagnostic plumbing; analytical scope, gates and settings remain
unchanged. Run-02 reproduced the SAME input hash; use its complete report.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research.matched_failure_audit \
  --spec research/matched-failure-20261028/specification.json \
  --amendment research/matched-failure-20261028/serialization-amendment.json \
  --output research/matched-failure-20261028/NEW-UNUSED-RUN
```

Source input records and model bundle hashes are retained in
`run-02/source-snapshot.json`; reconstructed frames retain last-row values and
full-frame hashes. The report records original specification, amendment, input
and reconstructed snapshot hashes. SQLite is opened `mode=ro`, with
`PRAGMA query_only=ON` and one explicit read transaction. No credentials, system
event messages or broker token files were read/exported by the audit.

Verification:8 focused diagnostic tests and the final34-test combined
Shadow/selection/strategy/execution/backtest group passed. Original specification,
amendment, source-file hashes and both evidence snapshot hashes verified. This
was focused regression verification, not a new full-suite or live-trading test.
