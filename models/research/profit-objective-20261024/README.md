# Profit-aware entry objective — 24 October 2026

## Decision

**Reject this proposed model/entry-policy change.** This bounded experiment
did not establish an improvement. No live or Shadow model was replaced,
registered or activated. There were no application, UI, indicator, input,
exit-rule, broker or execution-setting changes in this research pass.

The evidence does not justify calling either this candidate or the existing
classifier live-ready. No guarantee of future profitability is made.

## Hypothesis

The classifier learns whether net option P&L is positive, without directly
optimising win/loss size. Test whether learning **expected net premium return**
instead can select more economically valuable entries after estimated costs.

This is an objective/entry-policy experiment, not a new software feature.
Only the learner's target and entry-score units change. The same 40 inputs,
NIFTY ORB CE/PE opportunities, ATM→ITM1 selection, one-ATR stop, 1.5-ATR target,
25% premium stop, cost assumptions and chronological periods remain fixed.

Two direction-specific regressors use the existing shallow CatBoost family:
250 iterations, depth 3, learning rate 0.03, L2 10, seed 42, one thread. The
loss is weighted RMSE rather than Logloss. Targets are original net P&L after
estimated fees/slippage divided by (entry premium × historical lot size).
There is no target clipping, winsorising, class reweighting or exit search.

Weighted linear calibration is fitted on the separate calibration period.
Non-positive slopes disable a direction. Forecasts are expected returns, not
win probabilities; they are not mapped into the app's confidence threshold.
The research policy deliberately uses a separate, unsupported runtime schema.

Technical reference: [CatBoost regression objectives](https://catboost.ai/docs/en/concepts/loss-functions-regression).

## Frozen specification and identical comparison data

`specification.json` was saved before fitting. One objective/configuration was
allowed; there was no model-family, parameter, market-regime, strike or exit
search. Existing classifier-control files were fingerprinted before fitting
and checked unchanged afterward. The saved regression weights were reloaded
before forecasting.

The builder reproduced exactly the classifier control's opportunity table:

- 30,932 decision opportunities; 30,897 complete outcomes, 35 unlabelled.
- 18,784 fitting outcomes, 2,717 calibration outcomes, 3,273 tuning outcomes.
- Later evaluation: 6,004 opportunities, 5,993 complete labels on 181 days.
- Evaluation dates: 8 January–13 October 2026, previously inspected research
  history, not an untouched validation.
- Reserved 14–23 October market data was not used.
- Repeated intraday observations share trading days and are not independent
  trades. Fitting and calibration use equal total weight per day/direction.

The classifier control reproduced its frozen 47-trade evaluation metrics
exactly on these same opportunities. Thus a different data set does not
explain this experiment's outcome.

## Tuning result: no supported economic entry policy

Forecast entry-margin buffers were fixed at 0.25%, 0.50%, 0.75%, 1.00%, 1.50%,
2.00%, 2.50% and 3.00% of entry premium **after estimated costs**. These are
economic return forecasts, not probability cutoffs. The interior chosen buffer
would need at least 30 trades, 99% outcome coverage, 55% wins, PF 1.30, supported
neighbouring buffers and at least ten stressed-profitable trades in each tuning
half. Among qualifying buffers the objective was stressed net contribution,
not the highest win rate. Disabled margins have no forced-trade fallback.

Calibration slopes were positive: CE 1.1482, PE 0.9500. That alone did not
establish a tradeable policy.

| CE forecast buffer | Tuning trades | Wins | PF | Double-slippage net | Reason not accepted |
|---|---:|---:|---:|---:|---|
| 0.25% | 28 | 39.29% | 1.016 | −₹28.18 | Counts, wins, PF and cost-stress gates fail |
| 0.50% | 25 | 40.00% | 1.435 | +₹2,962.79 | Counts/wins/support fail |
| 0.75% | 20 | 50.00% | 2.019 | +₹5,404.93 | Counts/wins/half support fail |
| 1.00% | 18 | 50.00% | 2.437 | +₹5,625.57 | Counts/wins/half support fail |
| 1.50% | 13 | 53.85% | 3.023 | +₹5,985.08 | Counts/wins/half support fail |
| 2.00% | 7 | 85.71% | 66.894 | +₹6,907.31 | Tiny sample and unstable neighbours |
| 2.50% | 5 | 40.00% | 0.550 | −₹1,418.01 | Counts, wins, PF and cost-stress gates fail |
| 3.00% | 3 | 66.67% | 0.949 | −₹118.90 | Counts, PF and cost-stress gates fail |

PE forecasts generated **zero qualifying entries at every positive buffer**
during tuning. No CE or PE buffer met the frozen criteria. In particular,
85.71% wins and PF 66.894 on seven trades are not accepted as evidence of a
superior strategy. The requirements were not lowered after these results.

## Later evaluation and controls

Both economic entry margins were disabled before evaluation. Consequently the
profit-aware policy selected **zero trades**. This is a rejected/no-entry
policy, not demonstrated capital protection, profitability or a measured 0%
win rate. Mechanical zero counts/coverage/P&L in `report.json` reflect that
absence of selections; they do not mean all underlying data were missing or
that zero-return trading was validated.

| Frozen policy | Selected complete trades | Wins | PF | Estimated net |
|---|---:|---:|---:|---:|
| Profit-aware proposal | 0 | Not applicable | Not applicable | No entries |
| Existing aligned classifier control | 47 | 48.94% | 1.205 | +₹4,996.22 |
| Rule-only control | 181 | 38.67% | 0.776 | −₹28,667.54 |

The classifier still fails its own acceptance gates; retaining it as the
comparison control is not an endorsement for live-money deployment.

The return forecasts themselves were evaluated descriptively on all 5,993
labelled later opportunities, independent of the disabled entry policy:

- Day-weighted RMSE: **0.13227** in premium-return units.
- Constant calibration-period mean reference RMSE: **0.13107**.
- Day-weighted R²: **−0.06217** (no improvement over the contemporaneous
  constant-mean reference used in that statistic).
- Forecast-return 5th/50th/95th percentiles: −9.47%, −4.15%, +2.96%.

This one fixed regression configuration did not establish broadly useful
net-return forecasting. It does not prove that no possible model or strategy
could work. No attractive evaluation-only subgroup was chosen afterward.

## Remaining limits and operational state

The historical record still lacks executable bid/ask and stop-touch option
quotes. Entry and exit cashflows remain candle-replay estimates with the
established cost proxy. The comparison uses one historical lot and the first
passing opportunity per day, not full portfolio sizing or multiple-trade
runtime risk management. A genuinely forward frozen validation and execution
evidence remain mandatory before any live deployment.

Application mode remains SHADOW. Existing selected model #9 and research
reference #11 are unchanged. Database access in the experiment was query-only;
no order endpoint or model-registry write was used. New research-only files
are `research/profit_objective.py`, its tests and this experiment directory.

No more parameter/threshold searching is justified as proof from this same
evaluation period. The bounded attempt is complete and rejected. A different
hypothesis would need a new predeclared experiment and independent evidence;
this failed result must not be repackaged as a profitable live model.

## Reproducibility

- Full results: `report.json`; frozen model/entry semantics: `policy.json`.
- Fitted research weights: `ce.cbm`, `pe.cbm`.
- Specification hash:
  `6b3888cd19d1f92ea263409d57efea64354bb050a8e041de5295c15b987edde0`.
- Frozen policy hash:
  `32dce1143e0ec0136823c17618dbb4b4bba00e1ff3fe0db5ff1e739c1dd256a8`.
- Opportunity-table hash, identical to classifier control:
  `8961a01b2377b882f670f376a9e0c3fc193a047cb27ebccbc0ae01f816418cd7`.

The runner refuses to overwrite an existing experiment or fit against a data
fingerprint different from the frozen control. See the specification for the
unchanged acceptance gates and the pre-fitting classifier file fingerprints.
