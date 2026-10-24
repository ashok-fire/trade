from datetime import UTC, datetime, timedelta

import pandas as pd

from app.strategy import OPTION_META_FEATURES, PRICE_FEATURES, decide, indicators, regular_session, setup_direction
from app.training import labelled_samples


def test_previous_day_levels_do_not_use_current_day_data():
    start = datetime(2025, 1, 1, 3, 45, tzinfo=UTC)
    rows = []
    for day in range(2):
        for bar in range(6):
            price = 100 + day * 20 + bar
            rows.append({"timestamp": start + timedelta(days=day, minutes=bar * 5), "open": price, "high": price + 2, "low": price - 2, "close": price + 1, "volume": 100})
    frame = indicators(pd.DataFrame(rows))
    first_second_day = frame.iloc[6]
    assert first_second_day.previous_day_high == 107
    assert first_second_day.previous_day_low == 98
    assert first_second_day.previous_day_close == 106


def test_regular_session_filter_and_opening_range_are_causal():
    rows = [
        {"timestamp": datetime(2025, 1, 2, 3, 40, tzinfo=UTC), "open": 90, "high": 200, "low": 50, "close": 90, "volume": 100},
        {"timestamp": datetime(2025, 1, 2, 3, 45, tzinfo=UTC), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100},
        {"timestamp": datetime(2025, 1, 2, 3, 50, tzinfo=UTC), "open": 100, "high": 110, "low": 98, "close": 108, "volume": 100},
        {"timestamp": datetime(2025, 1, 2, 10, 0, tzinfo=UTC), "open": 108, "high": 109, "low": 107, "close": 108, "volume": 100},
        {"timestamp": datetime(2025, 1, 2, 10, 5, tzinfo=UTC), "open": 108, "high": 250, "low": 20, "close": 108, "volume": 100},
    ]

    regular = regular_session(pd.DataFrame(rows))
    frame = indicators(pd.DataFrame(rows))

    assert len(regular) == 3
    assert frame.iloc[0].opening_range_high == 101
    assert frame.iloc[1].opening_range_high == 110
    assert frame.iloc[-1].opening_range_high == 110


def test_training_labels_never_cross_market_days():
    start = datetime(2025, 1, 1, 3, 45, tzinfo=UTC)
    rows = []
    for day in range(3):
        for bar in range(20):
            price = 100 + day * 10 + bar
            rows.append({"timestamp": start + timedelta(days=day, minutes=bar * 5), "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000})
    samples = labelled_samples(pd.DataFrame(rows), horizon=6)
    local = pd.to_datetime(samples.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    counts = samples.assign(day=local.dt.date).groupby("day").size()
    assert all(counts <= 14)
    assert set(samples.label).issubset({"TRADE", "NO_TRADE"})


def test_zero_volume_index_candles_use_futures_liquidity_features():
    start = datetime(2025, 1, 1, 3, 45, tzinfo=UTC)
    spot_rows = []
    future_rows = []
    for day in range(4):
        for bar in range(40):
            stamp = start + timedelta(days=day, minutes=bar * 5)
            spot_price = 24_000 + day * 25 + bar * 2
            future_price = spot_price + 18 + (bar % 5)
            spot_rows.append(
                {
                    "timestamp": stamp,
                    "open": spot_price,
                    "high": spot_price + 4,
                    "low": spot_price - 4,
                    "close": spot_price + 1,
                    "volume": 0,
                }
            )
            future_rows.append(
                {
                    "timestamp": stamp,
                    "open": future_price,
                    "high": future_price + 5,
                    "low": future_price - 5,
                    "close": future_price + 1,
                    "volume": 10_000 + bar * 125,
                }
            )

    samples = labelled_samples(
        pd.DataFrame(spot_rows),
        horizon=6,
        futures_frame=pd.DataFrame(future_rows),
    )

    assert not samples.empty
    assert samples[PRICE_FEATURES].notna().all().all()
    assert (samples["volume_ratio"] > 0).all()


def test_quality_rejection_explains_score_and_direction_threshold(monkeypatch):
    features = {"setup_direction": -1.0, **{name: 1.0 for name in OPTION_META_FEATURES}}
    monkeypatch.setattr("app.strategy.feature_values", lambda _row, _context: features)

    class Model:
        available = True
        requires_option_context = True
        required_features = list(features)
        metadata = {"direction_thresholds": {"CE": 0.52, "PE": 0.56}}

        @staticmethod
        def predict(_features):
            return "TRADE", 0.54

    decision = decide(pd.Series({"close": 25000}), {}, Model())

    assert decision.direction == "NO_TRADE"
    assert decision.reason == "PE quality 54.0% below 56.0% threshold"


def test_additional_setup_families_use_causal_futures_confirmation():
    failed_breakout = pd.Series(
        {
            "close": 109,
            "previous_spot_close": 111,
            "opening_range_high": 110,
            "opening_range_low": 90,
            "futures_close": 108,
            "futures_vwap": 109,
            "ema9": 108,
            "ema20": 107,
            "ema50": 106,
            "rsi": 50,
            "high": 111,
            "low": 108,
        }
    )
    vwap_reclaim = pd.Series(
        {
            "close": 105,
            "previous_spot_close": 101,
            "opening_range_high": 110,
            "opening_range_low": 90,
            "futures_close": 106,
            "futures_vwap": 104,
            "previous_futures_close": 102,
            "previous_futures_vwap": 103,
            "ema9": 104,
            "ema20": 103,
            "ema50": 102,
            "rsi": 58,
            "high": 106,
            "low": 101,
        }
    )

    assert setup_direction(failed_breakout, "FAILED_BREAKOUT") == "PE"
    assert setup_direction(vwap_reclaim, "VWAP_RECLAIM") == "CE"
