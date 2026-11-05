from datetime import UTC, date, datetime, timedelta

import pandas as pd

from research.multi_strategy_study import first_signals, paired_by_day, setup_masks


def synthetic_sessions():
    rows = []
    for day in (2, 3):
        first = datetime(2026, 2, day, 3, 45, tzinfo=UTC)
        for index in range(75):
            breakout = index == 20
            rows.append({
                "timestamp": first + timedelta(minutes=index * 5),
                "trading_day": date(2026, 2, day),
                "open": 100.0, "high": 106.0 if breakout else 101.0,
                "low": 99.0, "close": 105.0 if breakout else 100.0,
                "atr": 2.0, "futures_close": 102.0, "futures_vwap": 100.0,
                "ema9": 101.0, "ema20": 100.0, "ema50": 99.0,
                "opening_range_high": 110.0, "opening_range_low": 90.0,
                "previous_spot_close": 100.0, "vwap": 100.0, "rsi": 60.0,
                "day_open": 100.0, "previous_day_close": 100.0,
            })
    return pd.DataFrame(rows)


def test_rolling_breakout_is_inside_opening_range_and_causal():
    frame = synthetic_sessions()
    masks, days = setup_masks(frame)
    assert days == [date(2026, 2, 2), date(2026, 2, 3)]
    assert bool(masks["ROLLING_30M"][0].iloc[20])
    assert not bool(masks["ORB_CONTROL"][0].iloc[20])
    before = first_signals(frame, masks)["ROLLING_30M"]
    assert before.timestamp.tolist() == [frame.timestamp.iloc[20], frame.timestamp.iloc[95]]
    frame.loc[21, "high"] = 300.0  # Information after the first signal.
    after = first_signals(frame, setup_masks(frame)[0])["ROLLING_30M"]
    assert after.timestamp.tolist() == before.timestamp.tolist()


def test_unknown_first_daily_outcome_is_not_replaced_by_a_winner():
    result = paired_by_day(
        [{"day": "2026-02-02", "status": "INCOMPLETE"}],
        [{"day": "2026-02-02", "status": "COMPLETE", "net": -10.0}],
        ["2026-02-02"],
    )
    assert result["known_days"] == 0
    assert result["unknown_days"] == 1
