from datetime import date
from decimal import Decimal

from app.risk import RiskLimits, can_enter, option_charges, paper_fill, position_size


def limits() -> RiskLimits:
    return RiskLimits(Decimal("500000"), Decimal("0.25"), Decimal("0.75"), 5, 2, 1, 1, Decimal("25"), Decimal("1.5"))


def test_paper_fills_use_ask_for_buy_and_bid_for_sell():
    assert paper_fill("BUY", Decimal("100"), Decimal("101"), Decimal("0.1"), Decimal("0.05")) == Decimal("101.10")
    assert paper_fill("SELL", Decimal("100"), Decimal("101"), Decimal("0.1"), Decimal("0.05")) == Decimal("99.90")


def test_position_size_is_whole_lot_and_live_is_capped():
    plan = position_size(Decimal("100"), 50, limits(), "LIVE")
    assert plan.allowed and plan.lots == 1 and plan.quantity == 50


def test_one_lot_above_risk_budget_is_rejected():
    plan = position_size(Decimal("500"), 50, limits(), "PAPER")
    assert not plan.allowed
    assert "One lot" in plan.reason


def test_position_size_caps_premium_cash_and_reports_tail_exposure():
    constrained = RiskLimits(Decimal("1000"), Decimal("100"), Decimal("100"), 5, 2, 1, 10, Decimal("25"), Decimal("1.5"))
    rejected = position_size(Decimal("500"), 3, constrained, "PAPER")
    assert not rejected.allowed
    assert rejected.reason == "One lot exceeds cash capital"
    plan = position_size(Decimal("100"), 5, constrained, "PAPER")
    assert plan.allowed
    assert plan.cash_required <= constrained.capital
    assert plan.planned_loss == (Decimal("100") - plan.premium_stop) * plan.quantity
    assert plan.maximum_premium_loss == plan.cash_required


def test_daily_loss_gate_blocks_entry():
    allowed, reason = can_enter(limits=limits(), open_positions=0, trades_today=1, consecutive_losses=0, realized_pnl_today=Decimal("-4000"), quote_fresh=True, broker_connected=True, reconciled=True)
    assert not allowed and "Daily-loss" in reason


def test_projected_stop_must_fit_remaining_daily_loss_budget():
    allowed, reason = can_enter(
        limits=limits(), open_positions=0, trades_today=1, consecutive_losses=1,
        realized_pnl_today=Decimal("-3000"), planned_loss=Decimal("1250"),
        quote_fresh=True, broker_connected=True, reconciled=True,
    )
    assert not allowed
    assert reason == "Planned trade risk exceeds remaining daily-loss budget"


def test_option_charge_model_versions_the_2026_stt_change():
    before = option_charges(Decimal("100"), Decimal("110"), 75, "GROWW", date(2026, 3, 31))
    after = option_charges(Decimal("100"), Decimal("110"), 75, "GROWW", date(2026, 4, 1))
    assert after > before
