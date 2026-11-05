from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from types import SimpleNamespace as NS
from collections import Counter

import pandas as pd

from research.matched_candle_replay import replay

SIGNAL = datetime(2026, 10, 26, 6, 0, tzinfo=UTC)
ENTRY = SIGNAL+timedelta(minutes=5)


def bar(open_=100, high=120, low=95, close=110):
    return NS(open=D(open_), high=D(high), low=D(low), close=D(close))


def run(options, spots):
    return replay(SIGNAL, options, spots, stop=D(19980), target=D(20030), direction="CE",
                  quantity=65, slip_pct=D(".05"), premium_stop_pct=D(25))


def test_matched_barriers_can_change_outcome_without_reselecting_entries():
    options = {SIGNAL: bar(), ENTRY: bar()}
    spots = {ENTRY: bar(20000, 20035, 19995, 20030)}
    result = run(options, spots)
    assert result["status"] == "COMPLETE" and result["exit_reason"] == "UNDERLYING_TARGET"
    # Future bars cannot change an outcome already determined in the entry bar.
    options[ENTRY+timedelta(minutes=5)] = bar(1, 2, 1, 1)
    assert run(options, spots) == result


def test_gaps_remain_unknown_not_later_winners():
    assert run({}, {})["reason"] == "missing_signal_option_bar"
    assert run({SIGNAL: bar()}, {})["reason"] == "missing_exact_entry_bar"
    assert run({SIGNAL: bar(), ENTRY: bar()}, {})["reason"] == "underlying_path_gap"
    options = {SIGNAL: bar(), ENTRY: bar()}
    spots = {ENTRY: bar(20000, 20010, 19995, 20005)}
    result = run(options, spots)
    assert result["status"] == "INCOMPLETE" and result["reason"] == "option_path_gap"


def test_stop_first_order_and_net_labels_are_preserved():
    options = {SIGNAL: bar(), ENTRY: bar(100, 102, 95, 99)}
    spots = {ENTRY: bar(20000, 20035, 19995, 20030)}
    result = run(options, spots)
    assert result["exit_reason"] == "UNDERLYING_TARGET" and result["net_win"] is False
    spots[ENTRY] = bar(20000, 20035, 19970, 20030)
    result = run(options, spots)
    assert result["exit_reason"] == "UNDERLYING_STOP"
    assert result["ambiguous_stop_target_bars"] == 1


def test_square_off_uses_open_before_future_candle_high_low():
    signal = datetime(2026, 10, 26, 9, 45, tzinfo=UTC)
    square_off = signal+timedelta(minutes=5)
    options = {signal: bar(), square_off: bar(110, 500, 1, 1)}
    result = replay(signal, options, {}, stop=D(19980), target=D(20030), direction="CE", quantity=65,
                    slip_pct=D(".05"), premium_stop_pct=D(25))
    assert result["exit_reason"] == "TIME_EXIT"
    assert result["exit_proxy"] == D(110)*D(".9995")


def test_aligned_proxy_matches_existing_training_simulator(monkeypatch):
    from app.domain import Contract
    from app.v4_training import _simulate_event
    options = {SIGNAL: bar(), ENTRY: bar()}
    for stamp, value in options.items():
        value.timestamp = stamp
        value.volume, value.open_interest = 1000, 1000
    spots = {ENTRY: pd.Series({"low": D(19995), "high": D(20035)})}
    monkeypatch.setattr("app.v4_training.feature_values", lambda *args, **kwargs: {})
    contract = Contract("same-key", "same-key", "NSE", "NIFTY", SIGNAL.date()+timedelta(days=1), D(20000), "CE", 65, D(".05"))
    event = {"time": pd.Timestamp(SIGNAL), "day": SIGNAL.date(), "instrument": contract,
             "row": pd.Series({"close": 20000, "atr": 20}), "direction": "CE", "setup_family": "ORB"}
    historical = _simulate_event(event, list(options.values()), spots, "GROWW", rejection_counts=Counter(), required_features=[])
    paired = run(options, spots)
    assert historical["exit_reason"] == paired["exit_reason"]
    assert abs(historical["net_pnl"]-float(paired["net"])) < 1e-7
    assert historical["label"] == int(paired["net_win"])
