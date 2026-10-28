from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from app.aligned_training import FIXED_PERIODS, daily_metrics, partitions_for, select_daily, train_aligned_model
from app.models import Candle, Instrument
from app.strategy import FEATURE_SCHEMA_V6, QUALITY_FEATURES, RegimeModel, decide
from app.training_contract import ALIGNED_FEATURES, contract_spec, entry_window_open, runtime_contract_error
from app.v4_training import build_option_quality_samples, evaluate_option_quality_holdout
from app.database import DEFAULTS


def test_contract_excludes_unreconciled_features_without_changing_legacy_schema():
    assert {"option_volume_log", "option_oi_log"}.isdisjoint(ALIGNED_FEATURES)
    assert {"option_volume_log", "option_oi_log"}.issubset(QUALITY_FEATURES)
    spec = contract_spec(("ORB",))
    assert spec["sampling_policy"] == "all_eligible_bars"
    assert spec["strike_policy"] == "ATM_ITM1" and spec["setup_policy"] == "first_confirmed"
    assert spec["retain_unlabelled"] is True
    assert not entry_window_open(datetime(2026, 10, 23, 9, 40, tzinfo=UTC))
    assert entry_window_open(datetime(2026, 10, 23, 4, 5, tzinfo=UTC))


def test_runtime_contract_rejects_geometry_and_scope_drift_but_preserves_legacy():
    metadata = {"training_contract": contract_spec(("ORB",)), "indices": ["NIFTY"],
                "compatible_live_providers": ["GROWW"]}
    settings = {**DEFAULTS, "active_broker": "GROWW", "enabled_indices": "NIFTY"}
    assert runtime_contract_error(metadata, settings) is None
    assert "reward_risk" in runtime_contract_error(metadata, {**settings, "reward_risk": "2"})
    assert "premium_emergency" in runtime_contract_error(metadata, {**settings, "premium_emergency_stop_pct": "20"})
    assert "indices" in runtime_contract_error(metadata, {**settings, "enabled_indices": "BANK_NIFTY"})
    assert "Provider" in runtime_contract_error(metadata, {**settings, "active_broker": "UPSTOX"})
    assert runtime_contract_error({}, settings) is None


def opportunities():
    return pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-08T04:00Z", "2026-01-08T04:05Z"]),
        "trading_day": [date(2026, 1, 8)]*2, "direction": ["CE"]*2,
        "label_status": ["CENSORED", "COMPLETE"], "label": [None, 1],
        "net_pnl": [None, 1000], "stress_net_pnl": [None, 900]})


def test_daily_selection_does_not_replace_unknown_first_outcome():
    frame = opportunities()
    selected = select_daily(frame, [.8, .9], {"CE": .7})
    assert len(selected) == 1 and selected.iloc[0].label_status == "CENSORED"
    result = daily_metrics(frame, [.8, .9], {"CE": .7})
    assert result["trades"] == 0 and result["censored"] == 1 and result["net_pnl"] == 0


def test_threshold_rejection_may_choose_later_but_missing_label_may_not():
    frame = opportunities()
    result = daily_metrics(frame, [.6, .9], {"CE": .7})
    assert result["trades"] == 1 and result["censored"] == 0
    assert result["net_pnl"] == 1000
    with pytest.raises(ValueError, match="align exactly"):
        select_daily(frame, [.9], {"CE": .7})


def test_frozen_periods_never_include_reserved_dates():
    frame = pd.DataFrame({"trading_day": [date(2026, 10, 13), date(2026, 10, 14), date(2026, 10, 23)]})
    parts = partitions_for(frame, FIXED_PERIODS)
    assert sum(len(part) for part in parts.values()) == 1
    assert parts["evaluation"].trading_day.tolist() == [date(2026, 10, 13)]
    assert len(partitions_for(pd.DataFrame({"trading_day": []}), FIXED_PERIODS)) == 4
    overlapping = {**FIXED_PERIODS, "tuning": FIXED_PERIODS["calibration"]}
    with pytest.raises(ValueError, match="ordered and disjoint"):
        partitions_for(frame, overlapping)


def test_batch_scores_respect_bundle_features_disabled_directions_and_legacy_ranker():
    model = object.__new__(RegimeModel)
    model.model = {"bundle": True}
    model.required_features = ["setup_direction", "x"]
    model.metadata = {"feature_schema": FEATURE_SCHEMA_V6, "enabled_directions": ["CE"],
        "calibration": {"CE": {"intercept": .1, "quality_coefficient": .5, "rank_coefficient": .7}}}
    model.quality_models = {"CE": SimpleNamespace(predict_proba=lambda matrix: np.tile([.2, .8], (len(matrix), 1)))}
    model.ranker = SimpleNamespace(predict=lambda matrix: np.repeat(.3, matrix.num_row()))
    frame = pd.DataFrame({"setup_direction": [1, -1, 0, 1], "x": [2, 2, 2, np.nan],
                          "option_volume_log": [np.nan]*4})
    probabilities = model.quality_probabilities(frame)
    assert probabilities[0] == pytest.approx(1/(1+np.exp(-(.1+.5*.8+.7*.3))))
    assert np.isnan(probabilities[1:]).all()
    assert model.predict(frame.iloc[0].to_dict())[1] == pytest.approx(probabilities[0])


def test_new_feature_gate_does_not_demand_excluded_option_fields(monkeypatch):
    features = {name: 1.0 for name in ALIGNED_FEATURES}
    features.update(option_volume_log=np.nan, option_oi_log=np.nan, setup_direction=1,
                    setup_orb=1, setup_pullback=0, setup_failed_breakout=0, setup_vwap_reclaim=0)
    monkeypatch.setattr("app.strategy.feature_values", lambda *args: features)
    model = SimpleNamespace(available=True, requires_option_context=True, required_features=ALIGNED_FEATURES,
        metadata={"feature_schema": FEATURE_SCHEMA_V6, "setup_families": ["ORB"],
                  "direction_thresholds": {"CE": .6}}, predict=lambda values: ("TRADE", .7))
    assert decide(pd.Series({"close": 25000}), {}, model).direction == "CE"


def seed_market(session):
    rows = []
    for day in (22, 23):
        for index in range(75):
            stamp = datetime(2026, 10, day, 3, 45, tzinfo=UTC) + timedelta(minutes=index*5)
            rows.append(dict(timestamp=stamp, open=100, high=101, low=99, close=100, volume=1000, atr=1))
            for role, key in (("SPOT", "INDEX"), ("FUTURE", "FUT")):
                if day == 23 and index == 74 and role == "FUTURE":
                    continue  # A missing AFTERNOON bar must not erase morning opportunities.
                session.add(Candle(provider="GROWW", underlying_code="NIFTY", instrument_key=key,
                    data_role=role, timestamp=stamp, open=100, high=101, low=99, close=100,
                    volume=1000, source_verified=True))
    for key, kind, side in (("FUT", "FUTURE", "FU"), ("OPTION", "OPTION", "CE")):
        session.add(Instrument(provider="GROWW", underlying_code="NIFTY", instrument_key=key,
            trading_symbol=key, instrument_kind=kind, option_type=side, strike=100,
            expiry=date(2026, 10, 29), valid_from=date(2026, 10, 1), exchange="NSE", lot_size=50))
    session.commit()
    return pd.DataFrame(rows)


def test_builder_keeps_repeated_priority_opportunities_and_censored_first(session, monkeypatch):
    frame = seed_market(session)
    monkeypatch.setattr("app.v4_training.with_futures_features", lambda *args: frame)
    monkeypatch.setattr("app.v4_training.selected_setup", lambda row, families: (families[0], "CE"))
    monkeypatch.setattr("app.v4_training._setup_direction", lambda *args: "CE")
    monkeypatch.setattr("app.v4_training.feature_values", lambda *args: {name: 1.0 for name in ALIGNED_FEATURES})
    options = [SimpleNamespace(timestamp=row.timestamp, close=Decimal(100), volume=100,
        open_interest=0) for row in frame.itertuples()]
    monkeypatch.setattr("app.v4_training._option_rows_by_key", lambda *args: {"OPTION": options})
    seen = set()
    def simulate(event, rows, underlying, provider, **kwargs):
        assert "option_oi_log" not in kwargs["required_features"]
        if event["day"] not in seen:
            seen.add(event["day"])
            kwargs["rejection_counts"]["missing_next_bar_entry"] += 1
            return None
        return {**{name: 1.0 for name in ALIGNED_FEATURES}, "timestamp": event["time"],
            "trading_day": event["day"], "direction": "CE", "setup_family": event["setup_family"],
            "label_status": "COMPLETE", "label": 1, "net_pnl": 1000, "stress_net_pnl": 900}
    monkeypatch.setattr("app.v4_training._simulate_event", simulate)
    dataset, coverage = build_option_quality_samples(session, provider="GROWW", code="NIFTY",
        start_date=date(2026, 10, 22), end_date=date(2026, 10, 23), setup_families=("ORB", "PULLBACK"),
        sampling_policy="all_eligible_bars", strike_policy="ATM_ITM1", setup_policy="first_confirmed",
        causal_opportunities=True, retain_unlabelled=True, required_features=ALIGNED_FEATURES,
        include_decision_context=True)
    assert set(dataset.setup_family) == {"ORB"}
    assert (dataset.decision_premium == 100).all() and (dataset.lot_size == 50).all()
    assert dataset.groupby("trading_day").size().min() > 1
    assert coverage["dropped_future_days"] == 0 and coverage["unlabelled_opportunities"] == 2
    result = daily_metrics(dataset, np.ones(len(dataset)), {"CE": .5})
    assert result["censored"] == 2 and result["trades"] == 0


def synthetic_opportunities():
    rng = np.random.default_rng(42)
    rows = []
    for day in pd.bdate_range("2020-01-01", periods=1000):
        for offset, side in enumerate(("CE", "PE")):
            values = dict(zip(ALIGNED_FEATURES, rng.normal(size=len(ALIGNED_FEATURES))))
            label = int(values["rsi"] + rng.normal(scale=.5) > 0)
            rows.append({**values, "setup_direction": 1 if side == "CE" else -1,
                "timestamp": day.tz_localize("UTC") + pd.Timedelta(hours=4, minutes=offset*5),
                "trading_day": day.date(), "direction": side, "setup_family": "ORB",
                "label_status": "COMPLETE", "label": label, "net_pnl": 100 if label else -90,
                "stress_net_pnl": 80 if label else -110, "net_return": .1 if label else -.09})
    return pd.DataFrame(rows)


def test_training_pipeline_freezes_contract_and_loadable_bundle(session, tmp_path, monkeypatch):
    dataset = synthetic_opportunities()
    calls = []
    def build(*args, **kwargs):
        calls.append(kwargs)
        return dataset.copy(), {"usable_option_outcomes": len(dataset)}
    monkeypatch.setattr("app.aligned_training.build_option_quality_samples", build)
    monkeypatch.setitem(__import__("app.aligned_training", fromlist=["PARAMETERS"]).PARAMETERS, "iterations", 10)
    path = tmp_path / "candidate.json"
    metadata = train_aligned_model(str(session.bind.url), path, ["NIFTY"], date(2020, 1, 1), date(2024, 1, 1),
        setup_families=("ORB",))
    assert calls[0]["required_features"] == ALIGNED_FEATURES
    assert calls[0]["retain_unlabelled"] and calls[0]["causal_opportunities"]
    assert metadata["activation_eligible"] is False and metadata["requires_forward_shadow"] is True
    assert metadata["calibration_samples"] > 0 and metadata["training_contract"]["version"] == "decision-aligned-v1"
    counts = metadata["partition_counts"]
    assert counts["fit"]["end"] < counts["calibration"]["start"] < counts["tuning"]["start"] < counts["evaluation"]["start"]
    model = RegimeModel(path)
    assert model.available and model.required_features == ALIGNED_FEATURES
    probabilities = model.quality_probabilities(dataset.iloc[:4])
    for index, row in dataset.iloc[:4].iterrows():
        if np.isfinite(probabilities[index]):
            assert model.predict(row.to_dict())[1] == pytest.approx(probabilities[index], abs=1e-6)
    with pytest.raises(ValueError, match="overwrite"):
        train_aligned_model(str(session.bind.url), path, ["NIFTY"], date(2020, 1, 1), date(2024, 1, 1))
    # Frozen evaluation must use the actual bundle feature list and opportunity
    # contract, not the module-global legacy 42-feature/default sampling policy.
    holdout = dataset.iloc[-40:].copy()
    holdout["timestamp"] = holdout.timestamp + pd.DateOffset(years=2)
    holdout["trading_day"] = holdout.timestamp.dt.date
    holdout_calls = []
    def build_holdout(*args, **kwargs):
        holdout_calls.append(kwargs)
        return holdout.copy(), {}
    monkeypatch.setattr("app.v4_training.build_option_quality_samples", build_holdout)
    metrics = evaluate_option_quality_holdout(str(session.bind.url), path, ["NIFTY"],
        date(2025, 1, 1), date(2025, 12, 31), "GROWW")
    assert holdout_calls[0]["required_features"] == ALIGNED_FEATURES
    assert holdout_calls[0]["sampling_policy"] == "all_eligible_bars" and holdout_calls[0]["retain_unlabelled"]
    assert metrics["frozen_model_version"] == metadata["version"]
    with pytest.raises(ValueError, match="must follow"):
        evaluate_option_quality_holdout(str(session.bind.url), path, ["NIFTY"], date(2023, 1, 1), date(2023, 12, 31))


def test_fixed_study_rejects_reserved_range_before_reading_history(tmp_path):
    with pytest.raises(ValueError, match="frozen study range"):
        train_aligned_model("sqlite://", tmp_path/"candidate.json", ["NIFTY"], date(2023, 1, 1),
            date(2026, 10, 23), setup_families=("ORB",), periods=FIXED_PERIODS)
