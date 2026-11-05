# Research protocol alignment and matched option import — 28 October 2026

## Completed scope

The human explicitly authorised the protocol correction and missing option-data
import after the matched failure audit. No learner, threshold, model activation,
capital/risk setting, broker-order logic or mode change was made. No model was
trained. This corrects comparability, not proven win-rate performance.

## Future research protocol: parallel-shadow-v2

- Underlying stop/target now use the SIGNAL candle close and ATR, matching the
  selected Shadow portfolio and historical training/replay geometry. The later
  underlying quote is still retained as evidence, not used to shift barriers.
- Reward/risk and premium-stop percentages are copied from the saved settings
  into each new observation. At deployment those remain1.5 and25%, respectively;
  slippage remains0.05% per side. Settings are frozen per observation for exits.
- One-lot gross premium-stop eligibility reuses the actual `position_size` rule;
  an explicit cash-affordability check is also recorded at decision time. Cash
  checking is an economic research guard, not a claim that the current portfolio
  implementation enforces that additional check identically.
- Raw counterfactuals remain available even when score/risk eligibility fails.
  New v2 daily comparisons use the first explicitly eligible entry per group/day,
  and separately the first eligible score-pass. Missing eligibility or a pending/
  incomplete first outcome cannot be replaced by a later winner.
- Saved risk settings are part of the protocol identity. Different protocols,
  slippage, capital, stops, models, directions and families are not pooled.
- One-lot evaluation does NOT replay the whole portfolio, sizing across multiple
  lots, simultaneous-position exclusion, or daily-loss/consecutive-loss state.
  The frozen risk snapshot records those settings without claiming to enforce
  portfolio state on independent raw trials. Stops, fees and gaps are not a
  guaranteed loss cap.

Existing352 v1 observation records, including46 completed raw counterfactuals,
are unchanged. V1 geometry and fill settings stay stored as before, and old open
trials continue to be monitored from their own saved barriers/slippage. V1 daily
metrics remain the original raw view; v2 eligibility is never applied
retroactively. Selection-audit IDs now include the new protocol version, so a
new-protocol check cannot accidentally reuse old evidence at the same boundary.
No schema rewrite, deleted observations or broker order is involved.

Implementation: `app/research_protocol.py`, `app/parallel_shadow.py`,
`app/selection_audit.py`, `app/shadow_dashboard.py`. The existing research page's
explanations/counts identify v2 eligible metrics versus legacy raw metrics;
no new trading strategy or model feature was added.

## Targeted historical import

The immutable plan is derived only from the46 observed entries' original
selected contracts. There was no discovery scan, strike substitution, spot/
futures refresh, or reserved14–23 October option-outcome read.

| Trading date | Selected option contracts | New valid five-minute candles |
|---|---|---:|
|26 October|27Oct26:22600CE,22650CE,22700CE|225|
|27 October|03Nov26:22700CE,22750CE,22750PE,22800CE|300|
|Total|7 contract-days|525|

Each exact session has75 unique bars from09:15 through15:25IST. Groww returned
76 rows per request; each15:30 row was deliberately excluded as outside the
regular-session interval, not counted as a missing trading candle. All525 retained
rows passed OHLCV/grid checks, with no in-session gaps or conflicting existing
values. Existing candles are never overwritten. Imported candles retain their
real source volume/OI; no bid/ask, Greeks or intrabar quotes were invented.

Portal research download **#11** completed. Per-request raw responses, filtered
counts and parsed-data hashes are retained in `import-01/`. Only candles,
research chunk checkpoints and that download-run record were added/updated.

SQLite's online backup API created and quick-checked a complete, WAL-consistent,
0600-permission backup before import:

`data/backups/niftypilot-20261028T032243459441Z-before-matched-option-import.db`

This is a pre-import recovery backup, not a post-import sharing export.

## Fixed matched replay result

The replay protocol was hashed before outcome calculation. Both arms use exactly
the same observed signals/contracts, original recorded ATR, historical next-bar
open entry, one lot, each observation's frozen slippage and premium stop,
date-versioned charges, conservative stop-first ordering, and15:20 candle-open
square-off. No fresh model, cutoff, capital, timing or exit grid was searched.
Only the underlying barriers differ: original signal-close geometry versus
stored legacy quote-anchor geometry. Original observations remain untouched.

| Diagnostic | Result |
|---|---:|
|Exact signal/next-entry option candle pairs|46/46|
|Complete contiguous signal-close replay outcomes|46/46|
|Complete legacy-geometry proxy outcomes|46/46|
|Exit reason differs between geometry arms|3|
|Exit candle differs between geometry arms|9|
|Net-win label differs between geometry arms|2|
|Aligned candle net-win label differs from recorded quote outcome|5|

This confirms that the protocol difference can affect labels. It does NOT show
that alignment raises win rate: these are overlapping observations across only
two dates, already inspected, not46 independent trades or new model validation.
Historical bar-open entries precede observed quote entries by seconds; candle
closes do not reproduce the option quote at an intrabar underlying touch.
Current underlying candles also cannot recover all original decision-time raw
versions; the26 October opening-input discrepancy remains documented, not
silently corrected or declared solved. Original decision features are preserved.

`replay-01/report.json` and `source-candles.json` retain every paired result and
its source hash. The aligned replay was regression-checked against the existing
training simulator under its matching0.05% assumptions. Unknown path outcomes
remain unknown; the import/replay never substitutes a later winning contract.

## Verification and readiness decision

-44 focused regression tests passed, covering direction/barriers, shared sizing,
  frozen settings, retained ineligible rows, unknown-first safeguards, legacy
  exits, versioned audit identities, session validation, non-overwrite/idempotent
  import and conservative replay/future-bar invariance.
-Wider regression run:219 passed,1 deselected. The previously timing-out in-process
  HTTP integration test was excluded; this is not a claim that all220 tests ran.
-Pre/post hashes matched for all settings, saved mode, model registry, original
  observations, and original signals/orders/trades. Model9 and reference11 remain
  unchanged. Source/specification, raw import and replay hashes are retained.
-After the user restarted the debug session, health reported SHADOW, artifact9,
  reference11 and `parallel-shadow-v2`, with no runtime error. A final daily-view
  fail-closed guard was regression-tested. The later debug restart did not leave
  port8001 listening; the human explicitly authorised starting the app, and the
  assistant started one local Uvicorn instance after confirming the port was
  free, saved mode was SHADOW and no positions/research trials were open.
  Final health verifies the complete current code with v2 loaded.

The missing option-data prerequisite is now resolved. The fixed replay is ready
for diagnostic review, but **no retraining or model promotion follows
automatically**. Five candle-versus-quote label disagreements and two dates do
not establish predictive improvement. Continue frozen v2 forward collection;
do not use these same dates to tune and then call their rescore independent.
Any subsequent model experiment still needs a distinct justified hypothesis,
chronological separation and genuinely later frozen validation.

## Reproduction and evidence

`specification.json` freezes the initial implementation and protected-record
hashes. `implementation-amendment.json` records the unknown-eligibility
fail-closed correction before any new v2 outcomes, without changing the planned
geometry/gates. `import-plan.json` binds the seven requests to the original
source snapshot. Do not rerun a successful import just to seek different data.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research.import_matched_options \
  --plan research/protocol-alignment-20261028/import-plan.json \
  --output research/protocol-alignment-20261028/NEW-UNUSED-IMPORT

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research.matched_candle_replay \
  --spec research/protocol-alignment-20261028/replay-specification.json \
  --output research/protocol-alignment-20261028/NEW-UNUSED-REPLAY
```

The importer refuses a running download, validates the frozen scope, backs up
before writes, stores raw responses and inserts missing timestamps only. The
replay uses a single read-only SQLite snapshot and prints no model approval or
aggregate overlapping-trial profit claim.
