# Pooled day-weighted model study

Result: **REJECTED_NO_SUPPORTED_POOLED_POLICY**.

First CE/PE candidate per day, equal total day weights, six-month calibration; logistic regression, shallow histogram boosting, shallow CatBoost and fixed equal consensus. No production changes.

## Highest win rate with 100+ development trades (selection-biased diagnostic)

{
  "learner": "CONSENSUS",
  "scope": "BOTH",
  "cutoff": 0.4,
  "metrics": {
    "selected": 144,
    "trades": 144,
    "censored": 0,
    "coverage": 1.0,
    "net_pnl": 6953.38,
    "stress_net_pnl": 5998.67,
    "win_rate_pct": 44.44,
    "profit_factor": 1.102,
    "max_drawdown": 13544.32,
    "daily_loss_overshoots": 0,
    "active_months": 12,
    "positive_months": 8,
    "monthly": {
      "2024-07": -2618.58,
      "2024-08": -5202.16,
      "2024-09": -2154.97,
      "2024-10": 754.72,
      "2024-11": 345.62,
      "2024-12": 61.13,
      "2025-04": 4566.79,
      "2025-05": -763.94,
      "2025-06": 2490.28,
      "2025-07": 8578.57,
      "2025-08": 872.74,
      "2025-09": 23.19
    },
    "ce_trades": 77,
    "pe_trades": 67
  },
  "windows": [
    {
      "window": "2024-Q2",
      "selected": 0,
      "trades": 0,
      "censored": 0,
      "coverage": 0.0,
      "net_pnl": 0.0,
      "stress_net_pnl": 0.0,
      "win_rate_pct": null,
      "profit_factor": null,
      "max_drawdown": 0.0,
      "daily_loss_overshoots": 0,
      "active_months": 0,
      "positive_months": 0,
      "monthly": {},
      "ce_trades": 0,
      "pe_trades": 0
    },
    {
      "window": "2024-Q3",
      "selected": 39,
      "trades": 39,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": -9975.72,
      "stress_net_pnl": -10104.22,
      "win_rate_pct": 28.21,
      "profit_factor": 0.286,
      "max_drawdown": 9975.72,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 0,
      "monthly": {
        "2024-07": -2618.58,
        "2024-08": -5202.16,
        "2024-09": -2154.97
      },
      "ce_trades": 22,
      "pe_trades": 17
    },
    {
      "window": "2024-Q4",
      "selected": 11,
      "trades": 11,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": 1161.47,
      "stress_net_pnl": 1119.17,
      "win_rate_pct": 63.64,
      "profit_factor": 1.421,
      "max_drawdown": 1725.97,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 3,
      "monthly": {
        "2024-10": 754.72,
        "2024-11": 345.62,
        "2024-12": 61.13
      },
      "ce_trades": 9,
      "pe_trades": 2
    },
    {
      "window": "2025-Q2",
      "selected": 33,
      "trades": 33,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": 6293.13,
      "stress_net_pnl": 5994.86,
      "win_rate_pct": 45.45,
      "profit_factor": 1.238,
      "max_drawdown": 6439.27,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 2,
      "monthly": {
        "2025-04": 4566.79,
        "2025-05": -763.94,
        "2025-06": 2490.28
      },
      "ce_trades": 19,
      "pe_trades": 14
    },
    {
      "window": "2025-Q3",
      "selected": 61,
      "trades": 61,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": 9474.5,
      "stress_net_pnl": 8988.86,
      "win_rate_pct": 50.82,
      "profit_factor": 1.377,
      "max_drawdown": 7218.4,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 3,
      "monthly": {
        "2025-07": 8578.57,
        "2025-08": 872.74,
        "2025-09": 23.19
      },
      "ce_trades": 27,
      "pe_trades": 34
    },
    {
      "window": "2025-Q4",
      "selected": 0,
      "trades": 0,
      "censored": 0,
      "coverage": 0.0,
      "net_pnl": 0.0,
      "stress_net_pnl": 0.0,
      "win_rate_pct": null,
      "profit_factor": null,
      "max_drawdown": 0.0,
      "daily_loss_overshoots": 0,
      "active_months": 0,
      "positive_months": 0,
      "monthly": {},
      "ce_trades": 0,
      "pe_trades": 0
    }
  ],
  "target_pass": false,
  "economic_pass": false
}

## Selected later historical audit

No supported policy; no extra 2026 candidate audit.

Reserved October 14–23 excluded. Previously inspected 2026 is not fresh evidence. Different first-candidate contract is not runtime-compatible. A separately labelled economic track never counts as reaching >70%. Full portfolio/quote-based forward validation remains required.
