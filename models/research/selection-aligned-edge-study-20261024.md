# Selection-aligned NIFTY study — 24 October 2026

## What changed

Runtime, historical replay and training now share one option candidate policy:
the nearest non-expiry-day contract, ATM first, then ITM1. ATM ties choose
the lower strike. Historical contract versions must be valid on the signal
date. Option fallback decisions use signal-time evidence; a missing or losing
future exit path cannot cause a switch to another strike.

Historical replay now executes the same option that the model scored,
checks stops in the entry candle, requires the exact entry bar and continuous
exit evidence, and squares off at the 15:20 candle open. New runtime entries
stop before the 15:10 boundary.

Full runtime filtering still needs live quote evidence. The historical source
does not provide consistent bid/ask, quote age or delta. The new
`option_selection_audits` table records both candidates, quotes, filters,
selected contract and rejection reasons. The captured decision clock can
then be replayed offline with the same runtime selection function. No
historical quotes or Greeks have been invented.

## Reproducible experiment

The specification was written before model fitting. Eight family/direction
specialists were compared using rule-only baselines, regularized logistic
regression and a shallow CatBoost classifier. The three predefined contexts
were all setups, directional trend alignment, and entries before 12:30 IST.

Models use separate chronological fitting, calibration, threshold-selection
and evaluation periods. Repeated observations are weighted by trading day.
Thresholds must have supporting neighbouring cutoffs and profitable results
in both halves of the tuning period under doubled slippage. Model/context
selection uses tuning results only. The later period has been inspected in
previous research and is not an untouched final test.

Historical observations: **38,381 on 924 days**, through **13 October 2026**.
ATM supplied 38,351 outcomes; signal-time ITM1 fallback supplied 30. These
are repeated setup observations, not 38,381 independent trades.

14–23 October 2026 was excluded and remains reserved.

## Result

The candidate selected on tuning was **morning ORB CE / CatBoost**, with a
research threshold of **42%**. This is a separate experiment and does not
change the selected V5 Shadow model's thresholds.

| Period | Trades | Wins | Profit factor | Net P&L |
|---|---:|---:|---:|---:|
| Tuning | 16 | 68.75% | 3.364 | ₹7,764.43 |
| Later historical evaluation | 22 | 50.00% | 0.993 | −₹86.24 |

Evaluation under doubled slippage lost ₹298.28. Only 3 of 8 active months
were profitable. The candidate did not pass the support, win-rate or cost
gates, so no runtime model was registered or promoted.

Rule-only baselines also showed instability. For example, Pullback CE
fell from 54.69% wins / PF 1.279 in tuning to 32.17% / PF 0.538 later.
Failed Breakout PE was profitable in the later period but failed tuning;
choosing it after seeing evaluation would bias the result.

The current features and setups have not demonstrated a stable replacement
edge. The operational improvement is reliable selection, transparent replay
limits and capture of the real quotes needed to test execution consistently.

Full diagnostics: `selection-aligned-edge-study-20261024.json`.
Frozen experiment specification: `selection-aligned-edge-study-20261024.spec.json`.

Method reference: [scikit-learn threshold selection](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html)
warns against fitting a classifier and tuning its cutoff on the same data.
