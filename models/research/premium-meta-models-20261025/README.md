# Premium-outcome meta-model study

Result: **REJECTED_NO_SUPPORTED_PREMIUM_META_POLICY**.

Two fixed native-exit targets; three regularised pooled classifiers and fixed consensus. All fitting/calibration weights normalise each day, not each overlapping row. No production changes.

## Highest win rate with 100+ development trades (selection-biased diagnostic)

{
  "key": "stop25-target15-hold60",
  "learner": "HIST_GRADIENT",
  "scope": "BOTH",
  "condition": "ALL",
  "cutoff": 0.55,
  "metrics": {
    "selected": 103,
    "trades": 103,
    "censored": 0,
    "coverage": 1.0,
    "net_pnl": 10334.36,
    "stress_net_pnl": 9952.34,
    "win_rate_pct": 66.99,
    "profit_factor": 1.291,
    "max_drawdown": 6781.14,
    "daily_loss_overshoots": 0,
    "active_months": 15,
    "positive_months": 12,
    "monthly": {
      "2024-07": -642.25,
      "2024-08": 1482.56,
      "2024-09": 1555.69,
      "2024-10": 293.83,
      "2024-11": 729.62,
      "2024-12": 1625.05,
      "2025-04": 3650.2,
      "2025-05": -1752.77,
      "2025-06": 290.32,
      "2025-07": 189.29,
      "2025-08": 466.97,
      "2025-09": 1010.01,
      "2025-10": 4264.17,
      "2025-11": 290.03,
      "2025-12": -3118.37
    },
    "ce_trades": 67,
    "pe_trades": 36
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
      "selected": 24,
      "trades": 24,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": 2396.0,
      "stress_net_pnl": 2358.09,
      "win_rate_pct": 75.0,
      "profit_factor": 1.928,
      "max_drawdown": 1587.9,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 2,
      "monthly": {
        "2024-07": -642.25,
        "2024-08": 1482.56,
        "2024-09": 1555.69
      },
      "ce_trades": 19,
      "pe_trades": 5
    },
    {
      "window": "2024-Q4",
      "selected": 7,
      "trades": 7,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": 2648.5,
      "stress_net_pnl": 2634.47,
      "win_rate_pct": 85.71,
      "profit_factor": 8.277,
      "max_drawdown": 363.95,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 3,
      "monthly": {
        "2024-10": 293.83,
        "2024-11": 729.62,
        "2024-12": 1625.05
      },
      "ce_trades": 4,
      "pe_trades": 3
    },
    {
      "window": "2025-Q2",
      "selected": 21,
      "trades": 21,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": 2187.75,
      "stress_net_pnl": 2091.03,
      "win_rate_pct": 57.14,
      "profit_factor": 1.177,
      "max_drawdown": 6781.14,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 2,
      "monthly": {
        "2025-04": 3650.2,
        "2025-05": -1752.77,
        "2025-06": 290.32
      },
      "ce_trades": 15,
      "pe_trades": 6
    },
    {
      "window": "2025-Q3",
      "selected": 22,
      "trades": 22,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": 1666.27,
      "stress_net_pnl": 1574.79,
      "win_rate_pct": 63.64,
      "profit_factor": 1.254,
      "max_drawdown": 1962.88,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 3,
      "monthly": {
        "2025-07": 189.29,
        "2025-08": 466.97,
        "2025-09": 1010.01
      },
      "ce_trades": 8,
      "pe_trades": 14
    },
    {
      "window": "2025-Q4",
      "selected": 29,
      "trades": 29,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": 1435.83,
      "stress_net_pnl": 1293.96,
      "win_rate_pct": 65.52,
      "profit_factor": 1.105,
      "max_drawdown": 6574.77,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 2,
      "monthly": {
        "2025-10": 4264.17,
        "2025-11": 290.03,
        "2025-12": -3118.37
      },
      "ce_trades": 21,
      "pe_trades": 8
    }
  ],
  "target_pass": false,
  "economic_pass": false
}

## Selected later historical audit

No supported policy; no extra 2026 candidate audit.

## Limitations

- Candidate selection reuses development history and is biased by repeated research
- Native stops/targets differ from runtime; research models are not compatible with active execution
- Candle fills, historical feature meanings and estimated charges need genuine forward quote-based validation
- One-lot, reset-quarter P&L is not a full-account compounded return
- Reserved October 14–23 excluded. A partial economic result is not a >70% claim. Forward quote validation remains mandatory.
