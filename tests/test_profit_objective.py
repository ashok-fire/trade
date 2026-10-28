import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.aligned_training import FIXED_PERIODS, daily_metrics
from app.training_contract import ALIGNED_FEATURES, contract_spec
from research.profit_objective import forecast, numeric_thresholds, run, select_margin


def test_expected_return_is_not_a_probability_and_disabled_is_never_one_point_zero_one():
    model = SimpleNamespace(predict=lambda values: np.repeat(2.0, len(values)))
    frame = pd.DataFrame({"direction": ["CE", "PE", "CE"], "x": [1., 1., np.nan]})
    calibration = {"CE": {"enabled": True, "intercept": .1, "slope": .8},
                   "PE": {"enabled": False}}
    predictions = forecast({"CE": model, "PE": model}, calibration, frame, features=["x"])
    assert predictions[0] == pytest.approx(1.7)  # Return may legitimately exceed 100%.
    assert np.isnan(predictions[1:]).all()
    thresholds = numeric_thresholds({"CE": None, "PE": .01})
    assert thresholds["CE"] == np.inf and thresholds["PE"] == .01


def test_no_supported_economic_margin_has_no_forced_trade_fallback():
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-09-01", periods=60, tz="UTC"),
        "trading_day": pd.date_range("2025-09-01", periods=60).date,
        "direction": ["CE"]*60, "label_status": ["COMPLETE"]*60,
        "label": [0]*60, "net_pnl": [-100.]*60, "stress_net_pnl": [-120.]*60})
    margin, curve = select_margin(frame, np.ones(60)*.05)
    assert margin is None and len(curve) == 8


def test_isolated_lucky_margin_is_rejected(monkeypatch):
    frame = pd.DataFrame({"trading_day": pd.date_range("2025-09-01", periods=60).date})
    def metrics(frame, scores, thresholds):
        lucky = thresholds["CE"] == .01
        return {"trades": 40, "coverage": 1., "win_rate_pct": 80 if lucky else 40,
                "profit_factor": 2 if lucky else .8, "double_slippage_net_pnl": 100 if lucky else -100,
                "active_months": 3}
    monkeypatch.setattr("research.profit_objective.daily_metrics", metrics)
    assert select_margin(frame, np.ones(60))[0] is None


def test_economic_selection_does_not_maximise_win_rate(monkeypatch):
    frame = pd.DataFrame({"trading_day": pd.date_range("2025-09-01", periods=60).date})
    def metrics(frame, scores, thresholds):
        margin = thresholds["CE"]
        return {"trades": 40, "coverage": 1., "win_rate_pct": 80 if margin == .005 else 55,
                "profit_factor": 1.5, "double_slippage_net_pnl": 100+margin*1000, "active_months": 3}
    monkeypatch.setattr("research.profit_objective.daily_metrics", metrics)
    margin, _ = select_margin(frame, np.ones(60))
    assert margin == .025  # Largest supported economic contribution, not 80% wins.


def synthetic_frame():
    rng = np.random.default_rng(42)
    rows = []
    for start, end in FIXED_PERIODS.values():
        for day in pd.bdate_range(start, end):
            for offset, side in enumerate(("CE", "PE")):
                values = dict(zip(ALIGNED_FEATURES, rng.normal(size=len(ALIGNED_FEATURES))))
                net_return = .05 * values["rsi"] + rng.normal(scale=.02)
                rows.append({**values, "setup_direction": 1. if side == "CE" else -1.,
                    "timestamp": day.tz_localize("UTC") + pd.Timedelta(hours=4, minutes=5*offset),
                    "trading_day": day.date(), "direction": side, "setup_family": "ORB",
                    "label_status": "COMPLETE", "label": int(net_return > 0), "net_return": net_return,
                    "net_pnl": net_return*1000, "stress_net_pnl": net_return*1000-5})
    return pd.DataFrame(rows).sort_values(["timestamp", "setup_family"], kind="stable").reset_index(drop=True)


def make_control(tmp_path, frame):
    directory = tmp_path / "control"
    directory.mkdir()
    path = directory / "candidate.json"
    path.write_text(json.dumps({"quality_models": {"CE": "ce.cbm", "PE": "pe.cbm"}, "ranker": "ranker.json"}))
    for filename in ("ce.cbm", "pe.cbm", "ranker.json"):
        (directory / filename).write_bytes(b"unchanged synthetic classifier control")
    evaluation = frame.loc[(frame.trading_day >= FIXED_PERIODS["evaluation"][0]) & (frame.trading_day <= FIXED_PERIODS["evaluation"][1])]
    expected = daily_metrics(evaluation, np.repeat(.6, len(evaluation)), {"CE": .5, "PE": .5})
    metadata = {"training_contract": contract_spec(("ORB",)), "features": ALIGNED_FEATURES,
        "training_end": "2026-10-13", "direction_thresholds": {"CE": .5, "PE": .5},
        "locked_validation": expected,
        "dataset_hash": hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()}
    path.with_name("candidate.metadata.json").write_text(json.dumps(metadata))
    return path


def test_research_run_preserves_controls_and_has_no_runtime_compatibility(tmp_path, monkeypatch):
    frame = synthetic_frame()
    control = make_control(tmp_path, frame)
    source_hash = hashlib.sha256(control.read_bytes()).hexdigest()
    monkeypatch.setattr("research.profit_objective.config", SimpleNamespace(database_url="sqlite://"))
    monkeypatch.setattr("research.profit_objective.build_option_quality_samples", lambda *args, **kwargs: (frame.copy(), {}))
    monkeypatch.setattr("research.profit_objective.RegimeModel", lambda path: SimpleNamespace(
        available=True, quality_probabilities=lambda frame: np.repeat(.6, len(frame))))
    monkeypatch.setitem(__import__("research.profit_objective", fromlist=["REGRESSION_PARAMETERS"]).REGRESSION_PARAMETERS,
                        "iterations", 10)
    output = tmp_path / "experiment"
    report = run(output, control)
    assert report["control_reproduced_exactly"] and not report["changes_applied_to_application"]
    assert report["activation_eligible"] is False
    assert hashlib.sha256(control.read_bytes()).hexdigest() == source_hash
    policy = json.loads((output / "policy.json").read_text())
    assert policy["schema"] == "net-return-research-v1" and "not probability" in policy["score_unit"]
    assert "feature_schema" not in policy  # Cannot masquerade as a runtime classifier.
    assert json.loads((output / "specification.json").read_text())["target"].endswith("(entry premium * one lot)")
    with pytest.raises(ValueError, match="refusing overwrite"):
        run(output, control)


def test_changed_opportunity_table_stops_before_fit(tmp_path, monkeypatch):
    frame = synthetic_frame()
    control = make_control(tmp_path, frame)
    changed = frame.copy()
    changed.loc[0, "net_return"] += .01
    monkeypatch.setattr("research.profit_objective.config", SimpleNamespace(database_url="sqlite://"))
    monkeypatch.setattr("research.profit_objective.build_option_quality_samples", lambda *args, **kwargs: (changed, {}))
    with pytest.raises(ValueError, match="differs from frozen control"):
        run(tmp_path / "mismatched", control)
    assert not list((tmp_path / "mismatched").glob("*.cbm"))
