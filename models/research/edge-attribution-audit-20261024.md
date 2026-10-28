# NIFTY edge attribution — 24 October 2026

## Conclusion

The evidence does not support adding algorithms or merely changing the cutoff.
There are three distinct weaknesses: unstable directional setup performance,
weak ranking by the selected model on the current replay opportunities, and a
small gross option edge that is largely consumed by estimated trading costs.
Historical/live opportunity and feature consistency must be checked before a
new model is fitted. This audit did not change strategies, settings or models.

## Scope and safeguards

- Provider/index: GROWW / NIFTY. Verified five-minute candles only.
- Source history: 3 January 2023–13 October 2026, 38,381 usable outcomes on
  924 days. These are overlapping observations, not independent trades.
- Diagnostic later period: 8 January–13 October 2026. This history has already
  been inspected in previous research. None of these results is a fresh test.
- Reserved dates 14–23 October were not used.
- Baseline tables take the first usable outcome per family/direction/day.
  Different rows overlap and must not be added together as portfolio P&L.
- The selected-model replay scores all eligible ORB opportunities, applies the
  existing CE 0.52 / PE 0.56 thresholds, then takes the first passing opportunity
  each day. Simultaneous candidates are ordered by score. This is a fixed
  one-trade/day research replay, not the complete live execution/risk engine.
- No fitting, calibration, threshold search or candidate selection was done.
- Database connections used SQLite `PRAGMA query_only=ON`. No broker API was
  called. Only this report and its JSON companion were added to the repository.

## 1. Selected Shadow model #9

Reference: `quality-v5-20261022T153945Z`, schema `cost-aware-option-v5`.
5,973 ORB opportunities were scored. Batched predictions were checked against
the existing `RegimeModel.predict` implementation on 12 observations.

| Measurement | Current diagnostic replay |
|---|---:|
| Selected trials | 112 (58 CE, 54 PE) |
| Net win rate | 45.54% |
| Profit factor before costs on the same exit path | 1.226 |
| Profit factor after estimated slippage and charges | 1.045 |
| Raw option P&L before estimated costs | ₹11,327.93 |
| Slippage deduction | ₹1,089.71 |
| Charges deduction | ₹7,764.87 |
| Net P&L | ₹2,473.35 |
| Net expectancy per trial | ₹22.08 |
| Profitable months / active months | 4 / 10 |
| Mean selected model score | 55.41% |
| Day-weighted ROC AUC, all scored ORB opportunities | 0.5004 |
| Day-weighted Brier score, all scored opportunities | 0.2542 |

The aggregate ROC AUC is approximately chance ranking on this particular
historical opportunity population, not proof about every subgroup or future
performance. AUC measures ranking, not profitability or calibration. See the
[scikit-learn metric definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

Estimated slippage and charges consume 78.17% of the raw positive P&L.
An additional hypothetical ₹25 per completed trial would change the net total
to **−₹326.65**. This is a sensitivity calculation, not a measured live spread.

A descriptive 95% interval for mean net P&L per trade is approximately
**−₹183.16 to +₹249.00**. It uses 10,000 circular moving-block resamples of five
consecutive observed ORB-signal trading days, retaining no-entry days within
that observed population and a fixed seed of 42. It is not an independent fresh
validation or a guarantee that future expectancy lies in this range. The
interval crosses zero; this replay does not establish positive expectancy.

CE contributes −₹2,487.61 net; PE contributes +₹4,960.96. These are descriptive
subgroups, not authorization to choose PE after seeing the evaluation.
Exits: 57 underlying stops, 52 underlying targets, three premium stops.

### Not the legacy 29-trade validation

The dashboard's stored PF 1.926 / 29 trades comes from the older model training
and validation protocol. The 112-trade replay above uses the current repaired
data, option selection, causal exit simulator, all eligible ORB opportunities
and a fixed later-period boundary. It does not reproduce or overwrite that
stored validation. The difference cannot be attributed solely to market decay
or one software change. Legacy validation is not evidence for the new protocol.

## 2. Where each fixed-rule baseline loses its edge

The index reference uses the same signal-anchored one-ATR stop and 1.5-ATR
target, actual next-bar underlying open as its reference entry, conservative
stop-first ordering and square-off at 15:20. Its payoff is in ATR units with
no costs. It is a directional diagnostic, not an executable index strategy.

Option figures use the existing selected option and exit policy. The raw/after
cost bridge holds the replay exit path fixed; it does not rerun a different
zero-cost policy. Profit factor below 1 means losses exceed gains in that layer.

| Setup | Days/trials | Index-reference PF | Raw option PF | Net option PF | Net option P&L |
|---|---:|---:|---:|---:|---:|
| ORB CE | 111 | 0.892 | 1.042 | 0.913 | −₹6,473.44 |
| ORB PE | 119 | 1.208 | 1.004 | 0.884 | −₹9,439.73 |
| Pullback CE | 115 | 0.705 | 0.623 | 0.538 | −₹39,478.67 |
| Pullback PE | 123 | 0.926 | 0.747 | 0.635 | −₹27,201.92 |
| Failed Breakout CE | 34 | 0.667 | 0.773 | 0.673 | −₹7,707.31 |
| Failed Breakout PE | 37 | 1.463 | 1.624 | 1.408 | +₹7,235.01 |
| VWAP Reclaim CE | 85 | 0.828 | 0.716 | 0.623 | −₹23,421.54 |
| VWAP Reclaim PE | 99 | 0.801 | 0.625 | 0.534 | −₹30,726.21 |

- Six of eight index-reference groups have negative point-estimate expectancy.
  That is a directional/setup problem, not merely option fees. Statistical
  significance is not established separately for these eight subgroups.
- Five option groups lose even before costs: both Pullback directions,
  Failed Breakout CE, and both VWAP Reclaim directions. Fee reduction alone
  cannot make those historical cashflows profitable.
- ORB CE has only ₹2,881.32 raw profit, versus ₹9,354.76 in estimated costs.
- ORB PE has only ₹298.08 raw profit, versus ₹9,737.81 in estimated costs.
  Its index-reference PF is 1.208 but option PF before costs is only 1.004,
  illustrating weaker conversion of index movement into option P&L under the
  existing contract and exit protocol.
- Failed Breakout PE is not a demonstrated winner: its earlier development
  period has net PF 0.730 across 143 trials and −₹16,559.15 net. Choosing it
  because the later 37 observations look good would be evaluation-period bias.

## 3. Fixed-clock cross-check

The JSON companion also compares signed index points, raw option P&L and net
option P&L at pre-specified 15- and 30-minute horizons, requiring continuous
paired paths. These are explanatory diagnostics, not new entry/exit proposals.
For example, Pullback CE has a 56.52% favourable index-move rate at 15 minutes
but raw option PF only 0.702. Counting correct direction is insufficient to
establish profitable option outcomes; move size and premium behaviour matter.

This audit cannot isolate time decay from implied-volatility changes or prove
which strike would have done better. The available five-minute candles do not
contain the necessary synchronized quote/Greek evidence. Option prices depend
on multiple inputs beyond direction: [OIC, Option Price Behavior](https://www.optionseducation.org/referencelibrary/faq/option-price-behavior).

## 4. Data and train/runtime consistency

The audit observed no stored `option_selection_audits`, no
`setup_observations`, and no `trades` in the database. Actual spread, quote-age,
delta and fill behaviour therefore remain unverified; blank evidence is not
proof of zero friction or correct operation during a market session.

The training builder defaults to `first_per_day` sampling and `ATM_ONLY`.
The current standard training call does not override those defaults. Runtime
evaluates eligible closed bars repeatedly and uses `ATM_ITM1`. The latest
offline edge study explicitly requested all eligible bars / ATM_ITM1, but it
did not replace the selected older V5 artifact. The sampling and execution
protocols must be explicit and versioned before another training run; the old
877-row training population is not the same population as all runtime checks.
Historical/live option-volume and OI meanings also remain unreconciled.
These discrepancies are investigation targets, not proof of the numerical
contribution of each issue to the losses above.

Coverage exclusions: 753 missing required feature vectors, 27 gaps in option
exit paths, eight missing next-bar entries and 11 missing liquid signal-time
options, from 39,180 candidate events. Five incomplete spot days and four
incomplete futures days were dropped upstream. Futures alignment gap dates
remain 18 May 2024 and 21 October 2025. First-usable-outcome selection conditions
on sufficient future exit evidence, so the baseline is not an unbiased live
opportunity census. No missing exit was manufactured into a labelled return.

The simulator checks underlying stops/targets within a five-minute candle and
uses that option candle's close as a fill proxy. The true option quote at the
instant of the underlying touch is unknown. These P&Ls are candle-replay
estimates, not exact executable returns.

## Decision and next work

No activation, extra algorithm, setup selection or threshold adjustment is
justified by this diagnostic alone. The immediate bounded task is to reconcile
training/runtime opportunity sampling, feature definitions and label/fill
protocols. Then use one frozen experiment to test whether a model can rank
net-positive outcomes on matching opportunities better than the rule baseline.
Apply the previously agreed chronological and forward pass/fail gates. If it
does not pass, retain research-only status instead of lowering the gates.

## Evidence

- Full metrics: `edge-attribution-audit-20261024.json`.
- The diagnostic attribution matched the existing simulator's original net
  P&L to within 1e-6 on all 38,381 accepted observations.
- Existing model manifest SHA-256:
  `20c9f224783ddf7f6fe32530bcac6834f7f70c5e3187731cf454f8f059d658d2`.
- Existing model metadata SHA-256:
  `956b3f9f359bc107ce762cf9925cdc3e53690a663ea96b1558341f1031847603`.
