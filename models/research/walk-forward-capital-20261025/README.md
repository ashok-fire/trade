# ₹1 lakh capital-aware walk-forward tuning

**Completed audit conclusion: REJECTED for trading promotion.** The frozen
`SELECTED_FOR_HISTORICAL_AUDIT` policy status below records the earlier selection
stage, not approval. See [ASSESSMENT.md](ASSESSMENT.md) for the final conclusion.

Capital ₹100,000; per-trade gross stop-risk budget ₹3,000 (3.0%); daily entry-gate limit ₹5,000 (5.0%).

Result: SELECTED_FOR_HISTORICAL_AUDIT. No model activated or selected for Shadow.

| Depth | L2 | Qualifying cutoff | Best validation stress net (diagnostic only) |
|---|---|---|---|
| 2 | 10 | 0.44 | ₹20,731.08 |
| 2 | 30 | None | ₹18,549.77 |
| 3 | 10 | 0.44 | ₹28,549.48 |
| 3 | 30 | None | ₹22,465.90 |
| 4 | 10 | 0.48 | ₹28,933.12 |
| 4 | 30 | None | ₹21,150.71 |

## Historical audit

{
  "metrics": {
    "selected": 61,
    "trades": 61,
    "censored": 0,
    "coverage": 1.0,
    "net_pnl": -4161.37,
    "stress_net_pnl": -4634.1,
    "win_rate_pct": 40.98,
    "profit_factor": 0.848,
    "max_drawdown": 6598.09,
    "daily_loss_overshoots": 0,
    "active_months": 10,
    "positive_months": 5,
    "monthly": {
      "2026-01": -3834.0,
      "2026-02": 232.74,
      "2026-03": 1774.09,
      "2026-04": -2054.85,
      "2026-05": -8.15,
      "2026-06": -1921.46,
      "2026-07": 1305.68,
      "2026-08": 853.29,
      "2026-09": -2075.89,
      "2026-10": 1567.19
    },
    "ce_trades": 46,
    "pe_trades": 15
  },
  "calibration": {
    "CE": {
      "intercept": 0.3545667093686884,
      "slope": 1.4457692865179443,
      "enabled": true
    },
    "PE": {
      "intercept": -0.1688874362606002,
      "slope": 0.9164953622495586,
      "enabled": true
    }
  },
  "samples": {
    "CE": {
      "fit_rows": 11536,
      "fit_days": 401,
      "calibration_rows": 1238,
      "calibration_days": 40
    },
    "PE": {
      "fit_rows": 10904,
      "fit_days": 399,
      "calibration_rows": 993,
      "calibration_days": 31
    }
  }
}

Same-capital frozen classifier: {"selected": 31, "trades": 31, "censored": 0, "coverage": 1.0, "net_pnl": -666.33, "stress_net_pnl": -907.89, "win_rate_pct": 48.39, "profit_factor": 0.958, "max_drawdown": 4241.31, "daily_loss_overshoots": 0, "active_months": 10, "positive_months": 5, "monthly": {"2026-01": -3340.7, "2026-02": 2635.94, "2026-03": -530.48, "2026-04": -2518.96, "2026-05": 4075.92, "2026-06": -119.03, "2026-07": 55.63, "2026-08": 869.33, "2026-09": -3788.44, "2026-10": 1994.45}, "ce_trades": 31, "pe_trades": 0}

Same-capital rule-only control: {"selected": 140, "trades": 140, "censored": 0, "coverage": 1.0, "net_pnl": -27256.24, "stress_net_pnl": -28393.61, "win_rate_pct": 37.86, "profit_factor": 0.683, "max_drawdown": 35496.24, "daily_loss_overshoots": 0, "active_months": 10, "positive_months": 3, "monthly": {"2026-01": 2961.13, "2026-02": -360.24, "2026-03": 2497.7, "2026-04": -328.21, "2026-05": -10332.62, "2026-06": 436.26, "2026-07": -9001.32, "2026-08": -713.25, "2026-09": -10485.29, "2026-10": -1930.42}, "ce_trades": 60, "pe_trades": 80}

## Limitations

- Signal-close premium used for affordability; next-bar-open fill may gap
- Stop risk excludes charges and is not a guaranteed maximum loss
- No historical executable quotes, feature parity or full live portfolio evidence
- Validation data are used for selection, not unbiased performance estimation
- 2026 audit is previously inspected history; not fresh validation
