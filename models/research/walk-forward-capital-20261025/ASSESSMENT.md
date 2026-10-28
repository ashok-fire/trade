# Completed assessment: do not promote

The requested settings were saved: ₹100,000 capital, 3% per-trade gross stop-risk
budget (₹3,000) and 5% daily-loss entry gate (₹5,000). The existing 1.5× underlying
reward/risk target and 25% option-premium emergency stop are unchanged. These
are different price bases; the target is not a guaranteed ₹4,500 option profit.
Actual losses can exceed stop budgets due to fees, slippage or gaps.

Six CatBoost configurations and nine common CE/PE thresholds were compared on
six chronological 2024–2025 validation windows, using models trained on preceding
data from 2023 onward, separately calibrated, with observed-day boundary purges.
The dataset contains 30,932 opportunities (30,897 complete, 35 censored), not
30,932 independent trades. 26,690 opportunities meet signal-time whole-lot risk
eligibility before score selection and cash-reserve checks.

The selected configuration was depth 4, L2 10, 250 iterations, learning rate 0.03,
seed 42, shared CE/PE probability cutoff 0.48. Selection was frozen in
`policy.json` before the 2026 audit. No parameters were changed after that audit.

| Measure | 2024–2025 selection validation | Jan 8–Oct 13, 2026 audit |
|---|---:|---:|
| Completed one-lot trades | 100 | 61 |
| Win rate | 55.00% | 40.98% |
| Net profit factor | 2.173 | 0.848 |
| Net P&L after estimated costs | ₹29,535.02 | -₹4,161.37 |
| Double-slippage net P&L | ₹28,933.12 | -₹4,634.10 |
| Maximum drawdown | ₹3,519.45 | ₹6,598.09 |
| Positive active months | 12/16 | 5/10 |

Validation figures are selection-biased and combine reset quarterly ledgers;
they are not an unbiased estimate or continuous compounded account return.
The 2026 audit also reuses previously inspected history, not a fresh final test.
The reserved October 14–23 data was not used. Historical fills remain candle
proxies, not executable quotes. This experiment uses one historical lot per
trade rather than automatically deploying the full ₹3,000 risk budget.

On the same 2026 capital/risk replay, the frozen earlier aligned classifier
returned -₹666.33 over 31 trades; rule-only returned -₹27,256.24 over 140 trades.
The tuned candidate is worse than that classifier control. Its descriptive
five-day block-bootstrap mean net/trade 95% interval is [-₹298.58, ₹159.21];
paired improvement per observed day versus the classifier is [-₹89.69, ₹48.16].
Neither interval establishes a positive edge. These intervals do not correct
for historical research exposure or model search.

Conclusion: the selected candidate fails profitability, stressed profitability,
sample support, consistency, and positive-edge checks in the later audit.
It is research-only and must not be promoted. The raw policy/report stage status
`SELECTED_FOR_HISTORICAL_AUDIT` is preserved for provenance, not live approval.
The selected Shadow artifact remains #9; the parallel reference remains #11;
the application stays configured for SHADOW. No live order was submitted.
