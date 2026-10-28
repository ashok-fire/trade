# Premium-exit strategy study

Result: **REJECTED_NO_FOUNDATION_EDGE**.

Tested 48 fixed exit/direction/condition combinations. Foundation passers: 0. Model trained: False.

Development selection: six chronological windows in 2024–2025. Reserved October 14–23 history excluded. Any 2026 audit is previously inspected history, not fresh validation.

| Exit rule | Side | Condition | Trades | Wins % | PF | Net ₹ | Stress net ₹ | Foundation |
|---|---|---|---:|---:|---:|---:|---:|---|
| stop15-target15-hold30 | CE | ALL | 202 | 45.05 | 0.87 | -8874.89 | -10796.83 | False |
| stop15-target15-hold30 | CE | TREND_ALIGNED | 137 | 50.36 | 1.34 | 12517.6 | 11892.54 | False |
| stop15-target15-hold30 | CE | WEAK_TREND | 117 | 43.59 | 0.678 | -14994.29 | -16517.65 | False |
| stop15-target15-hold30 | PE | ALL | 199 | 50.25 | 0.935 | -4572.47 | -5336.71 | False |
| stop15-target15-hold30 | PE | TREND_ALIGNED | 121 | 46.28 | 0.828 | -7593.74 | -8124.96 | False |
| stop15-target15-hold30 | PE | WEAK_TREND | 126 | 48.41 | 0.953 | -1983.43 | -2451.51 | False |
| stop15-target15-hold60 | CE | ALL | 202 | 49.5 | 0.885 | -9710.58 | -11129.12 | False |
| stop15-target15-hold60 | CE | TREND_ALIGNED | 137 | 54.01 | 1.294 | 13083.5 | 12543.38 | False |
| stop15-target15-hold60 | CE | WEAK_TREND | 117 | 43.59 | 0.591 | -23995.48 | -25003.58 | False |
| stop15-target15-hold60 | PE | ALL | 199 | 48.74 | 0.957 | -3299.02 | -3982.56 | False |
| stop15-target15-hold60 | PE | TREND_ALIGNED | 121 | 49.59 | 0.891 | -5301.76 | -5757.77 | False |
| stop15-target15-hold60 | PE | WEAK_TREND | 126 | 46.03 | 0.952 | -2214.24 | -2639.8 | False |
| stop15-target25-hold30 | CE | ALL | 202 | 42.08 | 0.927 | -5193.41 | -6162.84 | False |
| stop15-target25-hold30 | CE | TREND_ALIGNED | 137 | 47.45 | 1.34 | 13089.06 | 12381.91 | False |
| stop15-target25-hold30 | CE | WEAK_TREND | 117 | 39.32 | 0.633 | -18035.89 | -18575.14 | False |
| stop15-target25-hold30 | PE | ALL | 199 | 46.73 | 0.934 | -4870.35 | -5750.87 | False |
| stop15-target25-hold30 | PE | TREND_ALIGNED | 121 | 42.15 | 0.86 | -6551.12 | -7137.63 | False |
| stop15-target25-hold30 | PE | WEAK_TREND | 126 | 46.03 | 1.009 | 377.82 | -164.42 | False |
| stop15-target25-hold60 | CE | ALL | 202 | 43.07 | 0.956 | -4017.18 | -4947.21 | False |
| stop15-target25-hold60 | CE | TREND_ALIGNED | 137 | 46.72 | 1.266 | 13960.46 | 13344.59 | False |
| stop15-target25-hold60 | CE | WEAK_TREND | 117 | 35.9 | 0.589 | -26173.45 | -26628.93 | False |
| stop15-target25-hold60 | PE | ALL | 199 | 40.2 | 0.88 | -10491.48 | -11287.19 | False |
| stop15-target25-hold60 | PE | TREND_ALIGNED | 121 | 40.5 | 0.839 | -8759.53 | -9280.64 | False |
| stop15-target25-hold60 | PE | WEAK_TREND | 126 | 40.48 | 0.911 | -4514.26 | -5007.12 | False |
| stop25-target15-hold30 | CE | ALL | 202 | 46.53 | 0.878 | -8394.2 | -10414.66 | False |
| stop25-target15-hold30 | CE | TREND_ALIGNED | 137 | 51.82 | 1.271 | 10515.38 | 9842.73 | False |
| stop25-target15-hold30 | CE | WEAK_TREND | 117 | 46.15 | 0.723 | -12612.42 | -14208.97 | False |
| stop25-target15-hold30 | PE | ALL | 199 | 53.27 | 0.914 | -6718.36 | -7551.64 | False |
| stop25-target15-hold30 | PE | TREND_ALIGNED | 121 | 47.93 | 0.989 | -443.83 | -1033.58 | False |
| stop25-target15-hold30 | PE | WEAK_TREND | 126 | 51.59 | 0.918 | -3845.81 | -4349.96 | False |
| stop25-target15-hold60 | CE | ALL | 202 | 53.96 | 0.852 | -13931.86 | -15345.92 | False |
| stop25-target15-hold60 | CE | TREND_ALIGNED | 137 | 58.39 | 1.206 | 10268.0 | 9687.25 | False |
| stop25-target15-hold60 | CE | WEAK_TREND | 117 | 48.72 | 0.65 | -21023.13 | -22102.42 | False |
| stop25-target15-hold60 | PE | ALL | 199 | 52.76 | 0.889 | -10036.01 | -10770.25 | False |
| stop25-target15-hold60 | PE | TREND_ALIGNED | 121 | 54.55 | 0.919 | -4181.26 | -4688.25 | False |
| stop25-target15-hold60 | PE | WEAK_TREND | 126 | 51.59 | 0.923 | -4043.13 | -4500.25 | False |
| stop25-target25-hold30 | CE | ALL | 202 | 43.56 | 0.924 | -5567.9 | -6635.86 | False |
| stop25-target25-hold30 | CE | TREND_ALIGNED | 137 | 48.91 | 1.274 | 11086.84 | 10332.1 | False |
| stop25-target25-hold30 | CE | WEAK_TREND | 117 | 41.88 | 0.664 | -16567.14 | -17180.22 | False |
| stop25-target25-hold30 | PE | ALL | 199 | 48.74 | 0.875 | -10731.41 | -11695.59 | False |
| stop25-target25-hold30 | PE | TREND_ALIGNED | 121 | 43.8 | 0.992 | -347.95 | -997.99 | False |
| stop25-target25-hold30 | PE | WEAK_TREND | 126 | 48.41 | 0.934 | -3388.53 | -3976.44 | False |
| stop25-target25-hold60 | CE | ALL | 202 | 47.52 | 0.939 | -6217.3 | -7156.72 | False |
| stop25-target25-hold60 | CE | TREND_ALIGNED | 137 | 51.09 | 1.17 | 9964.53 | 9289.76 | False |
| stop25-target25-hold60 | CE | WEAK_TREND | 117 | 41.03 | 0.674 | -21285.88 | -21825.18 | False |
| stop25-target25-hold60 | PE | ALL | 199 | 43.72 | 0.801 | -20961.61 | -21820.46 | False |
| stop25-target25-hold60 | PE | TREND_ALIGNED | 121 | 45.45 | 0.83 | -10105.43 | -10687.2 | False |
| stop25-target25-hold60 | PE | WEAK_TREND | 126 | 45.24 | 0.873 | -7532.04 | -8062.46 | False |

## Selection and audit

{
  "strategy": null,
  "model_trained": false,
  "status": "REJECTED_NO_FOUNDATION_EDGE",
  "historical_audit": null,
  "audit_uncertainty": null
}

## Limits

- Candle-level stop/target execution is a proxy, not guaranteed executable liquidity
- Historical spreads, quote age, IV/delta and live feature semantics are not reconciled
- Selection performance is biased by strategy/model search; 2026 is previously inspected history
- One-lot, reset-quarter development P&L is not full-account compounded return
- Capital, app risk settings, runtime exits and model references unchanged; no live orders or promotion.
- Foundation gates screen economic plausibility, not a >70% claim or live-readiness approval.
- Missing paths remain censored; the first selected unknown outcome halts its quarterly ledger.
