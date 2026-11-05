from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
import json
import numpy as np
from types import SimpleNamespace as NS

from research.matched_failure_audit import (
    affordable, audit_closed, clean, daily_comparison, feature_differences, first_daily,
    geometry, group_key, selected_metrics,
)
from app.risk import option_charges, paper_fill


NOW = datetime(2026, 10, 27, 5, 0, 25, tzinfo=UTC)


def test_numpy_diagnostic_values_serialize_without_nan_or_bool_failures():
    assert json.loads(json.dumps(clean({"match": np.bool_(True), "n": np.int64(3), "missing": np.float64("nan")}))) == {"match": True, "n": 3, "missing": None}


def trial(identity="a", **changes):
    base = dict(id=identity, provider="GROWW", underlying_code="NIFTY", model_version="test",
        protocol_version="v1", protocol_json=json.dumps({"slippage_pct": "0.05", "premium_stop_pct": "25", "atr_target": 1.5}),
        setup_family="ORB", direction="CE", entry_time=NOW, observed_at=NOW,
        entry_price=D("100"), quantity=65, status="CLOSED", net_pnl=D("100"), quality_passed=True)
    return NS(**{**base, **changes})


def test_unknown_first_is_not_replaced_and_paired_result_stays_unknown():
    first = trial(status="INCOMPLETE", net_pnl=None)
    winner = trial("winner", observed_at=NOW+timedelta(minutes=5), net_pnl=D("900"))
    metrics = selected_metrics(first_daily([winner, first]))
    assert metrics["unknown"] == 1 and metrics["completed"] == 0
    assert metrics["entries"][0]["id"] == "a"
    comparison = daily_comparison([winner, first])
    assert comparison["paired_days"][0]["known"] is False
    assert comparison["paired_days"][0]["net_difference"] is None


def test_cash_and_risk_eligibility_does_not_use_future_label():
    cheap = trial(net_pnl=D("-800"))
    expensive = trial("expensive", entry_price=D("250"), net_pnl=D("5000"))
    assert affordable(cheap)
    assert not affordable(expensive)  # 4,062.50 premium-stop risk > 3,000 budget.
    assert not affordable(trial(entry_price=D("2000")))
    expensive.net_pnl = D("-5000")
    assert not affordable(expensive)
    comparison = daily_comparison([expensive, cheap])
    assert comparison["one_lot_risk_cash_eligible"] == 1


def test_groups_separate_slippage_and_direction():
    original = trial()
    stress = trial(protocol_json=json.dumps({"slippage_pct": "1", "premium_stop_pct": "25", "atr_target": 1.5}))
    assert group_key(original) != group_key(stress)
    assert group_key(original) != group_key(trial(direction="PE"))
    assert group_key(original) == group_key(trial(protocol_json=json.dumps(json.loads(original.protocol_json), sort_keys=True)))


def test_score_filter_is_not_selected_by_profit_and_no_entry_means_zero_delta():
    loser = trial(quality_passed=False, net_pnl=D("-100"))
    comparison = daily_comparison([loser])
    assert comparison["frozen_score"]["selected_days"] == 0
    assert comparison["paired_days"][0]["net_difference"] == 100
    loser.net_pnl = D("100")
    assert daily_comparison([loser])["frozen_score"]["selected_days"] == 0


def test_missing_features_are_not_fabricated_as_zero():
    assert feature_differences({"a": None}, {"a": float("nan")}) == {}
    assert "a" in feature_differences({"a": None}, {"a": 0})
    assert feature_differences({"a": 100}, {"a": 100.0000001}) == {}


def test_geometry_quantifies_anchor_difference_without_fabricating_pnl():
    row = trial(features_json=json.dumps({"atr_pct": .001}),
        selection_json=json.dumps({"spot": "20000", "underlying_quote": {"last": "20010"}}),
        underlying_stop=D("19990"), underlying_target=D("20040"))
    result = geometry(row)
    assert result["anchor_difference_atr"] == .5
    assert result["signal_anchor_stop"] == "19980.000"
    assert result["quote_anchor_matches_stored"]
    assert "net" not in result


def test_frozen_slippage_and_option_net_label_not_underlying_touch():
    exit_time = NOW+timedelta(minutes=5)
    eq = {"bid": "100", "ask": "101", "timestamp": NOW.isoformat()}
    xq = {"bid": "99", "ask": "100", "timestamp": exit_time.isoformat()}
    entry = paper_fill("BUY", D("100"), D("101"), D(1), D(".05"))
    exit_ = paper_fill("SELL", D("99"), D("100"), D(1), D(".05"))
    fees = option_charges(entry, exit_, 65, "GROWW", NOW.date(), "NSE")
    row = trial(instrument_key="key", exchange="NSE", tick_size=D(".05"),
        protocol_json=json.dumps({"slippage_pct": "1", "premium_stop_pct": "25", "max_quote_age_seconds": 3}),
        selection_json=json.dumps({"candidates": [{"instrument_key": "key", "quote": eq}]}),
        exit_evidence_json=json.dumps({"option_quote": xq, "underlying_quote": {"last": "20040", "timestamp": exit_time.isoformat()}}),
        signal_time=NOW-timedelta(minutes=5), exit_time=exit_time, entry_price=entry, exit_price=exit_,
        premium_stop=entry*D(".75"), underlying_stop=D("19980"), underlying_target=D("20030"),
        gross_pnl=(exit_-entry)*65, charges=fees, net_pnl=(exit_-entry)*65-fees, exit_reason="UNDERLYING_TARGET")
    result = audit_closed(row)
    assert result["failed"] == []
    assert result["net_win"] is False
    row.charges += 1
    assert "fees" in audit_closed(row)["failed"]
