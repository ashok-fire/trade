from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.specialist_research import chronological_partitions, stable_threshold


def test_calibration_and_threshold_selection_never_share_days():
    days = [date(2023, 1, 1) + timedelta(days=offset) for offset in range(120)]
    frame = pd.DataFrame({"trading_day": np.repeat(days, 3)})
    partitions = chronological_partitions(frame)
    previous = None
    for name in ("fit", "calibration", "tuning", "evaluation"):
        partition_days = set(partitions[name].trading_day)
        if previous is not None:
            assert max(previous) + timedelta(days=1) < min(partition_days)
        previous = partition_days
    sets = [set(partition.trading_day) for partition in partitions.values()]
    assert sum(len(value) for value in sets) == len(set.union(*sets))


def test_threshold_selection_rejects_isolated_lucky_cutoff(monkeypatch):
    def performance(_frame, _probabilities, threshold):
        lucky = abs(threshold - 0.56) < 1e-9
        return {
            "threshold": threshold, "trades": 20,
            "win_rate_pct": 80 if lucky else 40,
            "profit_factor": 2 if lucky else 0.8,
            "double_slippage_net_pnl": 100 if lucky else -100,
            "net_pnl": 150 if lucky else -50,
            "active_months": 4,
        }

    monkeypatch.setattr("app.specialist_research._performance", performance)
    cutoff, _curve = stable_threshold(pd.DataFrame(), np.array([]))
    assert cutoff is None
