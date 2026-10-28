# Effective-day and causal-cost model study

Result: **REJECTED_NO_EFFECTIVE_DAY_POLICY**.

No application settings or live model changed. Native option exits, causal risk filters and day-normalised regularisation; historical selection remains biased, and forward quote validation is required. Reserved October14–23 excluded.

## Highest win rate with100+ development trades (diagnostic only)

{
  "variant": [
    "day_risk_cost_h252",
    true,
    252
  ],
  "scope": "BOTH",
  "cutoff": 0.45,
  "metrics": {
    "selected": 191,
    "trades": 191,
    "censored": 0,
    "coverage": 1.0,
    "net_pnl": 12188.7,
    "stress_net_pnl": 11466.94,
    "win_rate_pct": 60.73,
    "profit_factor": 1.189,
    "max_drawdown": 11156.86,
    "daily_loss_overshoots": 0,
    "active_months": 18,
    "positive_months": 11,
    "monthly": {
      "2024-04": -607.39,
      "2024-05": 1198.07,
      "2024-06": -1053.09,
      "2024-07": 1030.79,
      "2024-08": -393.34,
      "2024-09": 1310.38,
      "2024-10": 711.12,
      "2024-11": 327.29,
      "2024-12": 2310.39,
      "2025-04": 1718.86,
      "2025-05": 3048.61,
      "2025-06": 2237.7,
      "2025-07": -4564.25,
      "2025-08": -2454.05,
      "2025-09": -1693.22,
      "2025-10": 3093.3,
      "2025-11": 7587.52,
      "2025-12": -1620.0
    },
    "ce_trades": 128,
    "pe_trades": 63
  },
  "windows": [
    {
      "window": "2024-Q2",
      "selected": 18,
      "trades": 18,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": -462.41,
      "stress_net_pnl": -506.81,
      "win_rate_pct": 61.11,
      "profit_factor": 0.892,
      "max_drawdown": 2170.86,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 1,
      "monthly": {
        "2024-04": -607.39,
        "2024-05": 1198.07,
        "2024-06": -1053.09
      },
      "ce_trades": 15,
      "pe_trades": 3
    },
    {
      "window": "2024-Q3",
      "selected": 34,
      "trades": 34,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": 1947.83,
      "stress_net_pnl": 1887.0,
      "win_rate_pct": 70.59,
      "profit_factor": 1.44,
      "max_drawdown": 2647.65,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 2,
      "monthly": {
        "2024-07": 1030.79,
        "2024-08": -393.34,
        "2024-09": 1310.38
      },
      "ce_trades": 27,
      "pe_trades": 7
    },
    {
      "window": "2024-Q4",
      "selected": 29,
      "trades": 29,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": 3348.81,
      "stress_net_pnl": 3288.92,
      "win_rate_pct": 62.07,
      "profit_factor": 1.639,
      "max_drawdown": 2020.67,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 3,
      "monthly": {
        "2024-10": 711.12,
        "2024-11": 327.29,
        "2024-12": 2310.39
      },
      "ce_trades": 21,
      "pe_trades": 8
    },
    {
      "window": "2025-Q2",
      "selected": 26,
      "trades": 26,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": 7005.17,
      "stress_net_pnl": 6876.95,
      "win_rate_pct": 61.54,
      "profit_factor": 1.549,
      "max_drawdown": 3655.39,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 3,
      "monthly": {
        "2025-04": 1718.86,
        "2025-05": 3048.61,
        "2025-06": 2237.7
      },
      "ce_trades": 17,
      "pe_trades": 9
    },
    {
      "window": "2025-Q3",
      "selected": 41,
      "trades": 41,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": -8711.52,
      "stress_net_pnl": -8920.54,
      "win_rate_pct": 46.34,
      "profit_factor": 0.628,
      "max_drawdown": 11156.86,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 0,
      "monthly": {
        "2025-07": -4564.25,
        "2025-08": -2454.05,
        "2025-09": -1693.22
      },
      "ce_trades": 22,
      "pe_trades": 19
    },
    {
      "window": "2025-Q4",
      "selected": 43,
      "trades": 43,
      "censored": 0,
      "coverage": 1.0,
      "net_pnl": 9060.83,
      "stress_net_pnl": 8841.42,
      "win_rate_pct": 65.12,
      "profit_factor": 1.624,
      "max_drawdown": 4183.66,
      "daily_loss_overshoots": 0,
      "active_months": 3,
      "positive_months": 2,
      "monthly": {
        "2025-10": 3093.3,
        "2025-11": 7587.52,
        "2025-12": -1620.0
      },
      "ce_trades": 26,
      "pe_trades": 17
    }
  ],
  "target_pass": false,
  "economic_pass": false
}

## Later historical audit

null
