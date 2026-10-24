from datetime import date

import pandas as pd

from app.edge_research import fixed_partitions, regime_mask


def test_reserved_october_period_is_never_used_for_model_selection():
    frame = pd.DataFrame({"trading_day": [date(2026, 10, 13), date(2026, 10, 14), date(2026, 10, 23)]})
    partitions = fixed_partitions(frame)
    assert partitions["evaluation"].trading_day.tolist() == [date(2026, 10, 13)]
    assert sum(len(value) for value in partitions.values()) == 1


def test_trend_filter_requires_consistent_directional_confirmation():
    frame = pd.DataFrame({
        "direction": ["CE", "PE", "CE", "PE"],
        "trend_atr_strength": [1.0, -1.0, 1.0, -1.0],
        "ema20_50_distance": [0.01, -0.01, 0.01, 0.01],
        "futures_vwap_distance": [0.01, -0.01, -0.01, -0.01],
    })
    assert regime_mask(frame, "TREND_ALIGNED").tolist() == [True, True, False, False]
