from collections import Counter
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from app.strategy import QUALITY_FEATURES
from app.v4_training import _simulate_event, option_quality_context


def test_option_quality_context_uses_only_signal_time_values():
    values = option_quality_context(
        premium=120.0,
        volume=999,
        open_interest=4999,
        expiry=date(2026, 1, 8),
        strike=Decimal("24000"),
        spot=23900.0,
        trading_day=date(2026, 1, 5),
    )

    assert values["option_premium_pct"] == pytest.approx(120 / 23900 * 100)
    assert values["days_to_expiry"] == 3.0
    assert values["moneyness_pct"] == pytest.approx(100 / 23900 * 100)
    assert values["option_volume_log"] > 0
    assert values["option_oi_log"] > values["option_volume_log"]


def _simulation_case():
    stamp = pd.Timestamp(datetime(2026, 10, 23, 4, 0, tzinfo=UTC))
    row = pd.Series({"close": 100, "atr": 10})
    instrument = SimpleNamespace(
        lot_size=50, exchange="NSE", expiry=date(2026, 10, 27), strike=100
    )
    event = {"time": stamp, "day": date(2026, 10, 23), "direction": "CE", "row": row, "instrument": instrument}
    signal = SimpleNamespace(timestamp=stamp, open=100, close=100, low=99, volume=100, open_interest=1000)
    entry = SimpleNamespace(timestamp=stamp + pd.Timedelta(minutes=5), open=100, close=80, low=70)
    return event, signal, entry


def test_option_label_checks_premium_stop_in_entry_bar(monkeypatch):
    event, signal, entry = _simulation_case()
    monkeypatch.setattr("app.v4_training.feature_values", lambda *_args: {key: 1.0 for key in QUALITY_FEATURES})
    underlying = {entry.timestamp: pd.Series({"low": 99, "high": 101})}
    result = _simulate_event(event, [signal, entry], underlying, "GROWW")
    assert result["exit_reason"] == "PREMIUM_STOP"
    assert result["label"] == 0
    assert result["net_pnl"] < 0


def test_option_label_rejects_delayed_entry_instead_of_using_later_candle():
    event, signal, entry = _simulation_case()
    entry.timestamp += pd.Timedelta(minutes=5)
    reasons = Counter()
    assert _simulate_event(event, [signal, entry], {}, "GROWW", rejection_counts=reasons) is None
    assert reasons == {"missing_next_bar_entry": 1}
