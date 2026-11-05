"""Versioned geometry and causal one-lot eligibility; no portfolio execution."""
from decimal import Decimal

from .risk import limits_from_settings, position_size

PROTOCOL = "parallel-shadow-v2"
LEGACY_PROTOCOL = "parallel-shadow-v1"


def risk_snapshot(settings):
    return {name: settings[name] for name in (
        "paper_capital", "risk_per_trade_pct", "daily_loss_limit_pct",
        "max_trades_per_day", "max_consecutive_losses", "max_open_positions",
        "live_max_lots", "premium_emergency_stop_pct", "reward_risk",
    )}


def one_lot_eligibility(entry_price, lot_size, settings):
    """Use the same premium-stop sizing rule, plus explicit cash affordability.

    The observer intentionally retains ineligible raw counterfactuals. This is
    not a portfolio simulation: open positions/daily losses/entry ordering are
    not enforced here, and stops/fees/gaps are not a guaranteed loss cap.
    """
    limits = limits_from_settings(settings)
    plan = position_size(entry_price, lot_size, limits, "SHADOW")
    cash_required = entry_price * lot_size
    stop_risk = cash_required * limits.premium_emergency_stop_pct / 100
    affordable = cash_required <= limits.capital
    return {"eligible": plan.allowed and affordable,
            "reason": plan.reason if not plan.allowed else ("One lot exceeds cash capital" if not affordable else "One lot fits risk and cash budgets"),
            "capital": str(limits.capital), "risk_budget": str(plan.risk_budget),
            "one_lot_quantity": lot_size, "cash_required": str(cash_required),
            "gross_premium_stop_risk": str(stop_risk), "cash_affordable": affordable,
            "portfolio_state_replayed": False,
            "limitation": "One-lot gross premium-stop/cash eligibility only; not a full portfolio or guaranteed loss cap"}


def signal_barriers(signal_close, atr, direction, reward_risk):
    if direction not in {"CE", "PE"} or atr <= 0 or reward_risk <= 0:
        raise ValueError("Invalid research barrier inputs")
    sign = Decimal(1 if direction == "CE" else -1)
    return signal_close-sign*atr, signal_close+sign*atr*reward_risk


def recorded_eligible(row):
    """Only an explicit, decision-time v2 snapshot may qualify new metrics."""
    import json
    if row.protocol_version != PROTOCOL:
        return None
    evidence = json.loads(row.selection_json or "{}").get("economic_eligibility", {})
    value = evidence.get("eligible")
    return value if isinstance(value, bool) else None
