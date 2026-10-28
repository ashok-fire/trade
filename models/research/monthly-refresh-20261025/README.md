# Monthly-refresh diagnostic — 25 October 2026

Result: **REJECTED_MONTHLY_REFRESH_DIAGNOSTIC**. Monthly retraining did not
establish an improvement or a >70% net-win model. No application settings,
model references, production code, registry rows or orders changed.

## Fixed experiment

Exactly three arms, with no settings search: the preceding frozen 50-input
research estimator, a monthly refreshed 50-input estimator, and a matched
monthly refreshed 44-input control. Profile fixed at research capital ₹200,000,
one historical lot, 3% trade/5% daily risk filters, CE-only, cutoff 0.45,
252-observed-day fitting weight half-life, premium stop25%/target15%/hold60min.
Application capital stayed ₹100,000; this study did not allocate ₹200,000 live.

Each monthly refresh used an expanding fitting period before the preceding
six complete calibration months. Both fitting and calibration purged their
final observed day. All decisions used causal inputs and prior completed
outcomes; positive-slope side calibration and minimum support remained in
place. Jan8–Oct13,2026 was evaluated; October14–23 was never read. Predictions
across months fed ONE continuous capital ledger, not fresh monthly accounts.
Unknown first selected outcomes halt the ledger rather than being replaced.

The specification, complete ten-month schedule, source/reference hashes,
sample gates, risk gates and contrasts were written before reading/relabeling
the evaluated data. The frozen comparator reproduced its earlier audit exactly.
The fixed causal CE/whole-lot evaluation cohort contained 2,767 opportunities
across 109 distinct eligible sessions (first eligible date Jan12), not 2,767
independent trades. The prepared history retained 30,932 rows over 893 days.

## Results

| Measure | Frozen50 | Monthly50 | Monthly44 control |
|---|---:|---:|---:|
| Completed selected trades / distinct selected days |54|26|26|
| Net win rate |42.59%|38.46%|38.46%|
| Profit factor |0.774|0.631|0.631|
| Net P&L after fees |−8397.80|−8507.53|−8507.53|
| Double-slippage net P&L |−10181.67|−8698.41|−8698.41|
| Maximum drawdown |12603.13|12467.83|12467.83|
| Positive / active months |2/10|1/5|1/5|

Monthly50 and Monthly44 produced identical selected trades, though their model
scores differed. Both selected trades only in January–May. All ten CE
calibrators had adequate support and positive slopes; from June onward no
eligible score reached the fixed 0.45 cutoff. Those months are not missing
outcomes or stopped processes. The gate was not lowered after observing this.

The monthly estimator's 95% descriptive day-block interval for mean net P&L
per trade was −981.37 to373.44. Its paired improvement over the frozen model
was −138.02 to126.98 per observed evaluation day. The paired interval against
the monthly44 control was exactly [0,0]. No reliable positive improvement was
demonstrated. Minimum100-trade support, win rate, PF, positive base/stressed
P&L, monthly/quarterly stability and uncertainty gates failed.

This remains repeatedly inspected historical evidence, not a fresh final
test. Candle-based fills are proxies, not executable quotes. A smaller stress
loss or smaller drawdown from trading fewer times is not proof of an edge.

## Verification and continuation

- Eight focused tests cover chronology/purges, future/current-outcome leakage,
  cohort alignment, disabled scores, continuous equity, censored-outcome halt,
  fixed-reference policy validation, cache integrity, reserved-date bounds,
  preregistration and inability to activate a model.
- Full suite: 177 tests passed, with 307 existing deprecation warnings. Source,
  reference, code, specification, prepared-input and prediction hashes verified;
  the two monthly selected-trade files were byte-identical.
- specification.json, report.json, dataset.json and both monthly diagnostic
  files retain all settings, counts, calibration support and rejection reasons.
- prepared-history.pkl and predictions.pkl preserve reproducible research
  inputs/scores; all three selected trade files remain available for audit.
- The app was healthy in SHADOW, artifact9/reference11 unchanged. Read-only
  diagnostics found zero parallel setup observations (including zero CLOSED
  outcomes); Sunday is outside the weekday entry window. Saved capital/risk/
  daily/RR remained100000/3/5/1.5.

Do not repeat this fixed diagnostic or expand its parameter grid. No objectively
supported next offline experiment is queued. Follow-ups should check service
health and meaningful fresh completed quote-based Shadow evidence, staying
quiet when unchanged. New reference11 evidence is evidence for that frozen
compatible reference, not automatic validation of these incompatible rejected
native-exit research models. Any new study needs a concrete preregistered
hypothesis; any final target claim needs genuine forward support.

Research entry point (unused output folder only):

```bash
.venv/bin/python -m research.monthly_refresh_study --output models/research/NEW-UNUSED-MONTHLY-REFRESH
```
