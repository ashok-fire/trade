from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from research import option_path_models as study
from research.exit_strategy_study import Bar
from test_premium_meta_models import repeated_data


def history():
    stamp = pd.Timestamp("2025-02-03 04:15:00Z")
    day = stamp.tz_convert("Asia/Kolkata").date()
    path = {}
    for i, value in enumerate((100, 102, 101, 105, 999)):
        time = stamp+pd.Timedelta(minutes=(i-3)*5)
        price = Decimal(value)
        path[time] = Bar(time, price-1, price+2, price-3, price)
    frame = pd.DataFrame({"timestamp":[stamp], "trading_day":[day],
                          "instrument_key":["OPTION"], "decision_premium":[105]}, index=[42])
    return frame, {("OPTION", day):path}, stamp


def test_option_features_are_exact_and_future_outcome_invariant():
    frame, paths, stamp = history()
    result = study.option_features(frame, paths)
    row = result.loc[42]
    assert row.option_return_1bar==pytest.approx(105/101-1)
    assert row.option_return_3bars==pytest.approx(.05)
    assert row.option_log_vol_3bars==pytest.approx(np.std(np.diff(np.log([100, 102, 101, 105]))))
    assert row.option_range_fraction==pytest.approx(5/105)
    assert row.option_body_fraction==pytest.approx(1/104)
    assert row.option_close_location==pytest.approx(3/5)
    path = next(iter(paths.values()))
    path[stamp+pd.Timedelta(minutes=5)] = replace(path[stamp], close=Decimal(999999))
    frame["label"], frame["net_pnl"] = 0, -999999
    pd.testing.assert_frame_equal(result, study.option_features(frame, paths))


def test_option_lags_do_not_bridge_gaps_other_contracts_or_days():
    frame, paths, stamp = history()
    path = next(iter(paths.values()))
    missing = path.pop(stamp-pd.Timedelta(minutes=10))
    paths[("OTHER", frame.trading_day.iloc[0])] = {missing.timestamp:missing}
    result = study.option_features(frame, paths).iloc[0]
    assert np.isfinite(result.option_return_1bar)
    assert np.isnan(result.option_return_3bars) and np.isnan(result.option_log_vol_3bars)
    del path[stamp-pd.Timedelta(minutes=5)]
    assert study.option_features(frame, paths).option_return_1bar.isna().all()
    assert study.option_features(frame, {}).isna().all().all()


def test_invalid_candles_missing_stays_nan_flat_location_is_neutral():
    frame, paths, stamp = history()
    path = next(iter(paths.values()))
    path[stamp] = Bar(stamp, Decimal(105), Decimal(105), Decimal(105), Decimal(105))
    assert study.option_features(frame, paths).option_close_location.iloc[0]==.5
    path[stamp] = replace(path[stamp], high=Decimal(104))
    assert study.option_features(frame, paths).isna().all().all()
    path[stamp] = Bar(stamp, Decimal(100), Decimal(110), Decimal(90), Decimal(106))
    with pytest.raises(ValueError, match="premium"):
        study.option_features(frame, paths)
    frame["trading_day"] = pd.Timestamp("2025-02-04").date()
    with pytest.raises(ValueError, match="trading day"):
        study.option_features(frame, paths)


def test_capital_changes_risk_utilization_but_not_source_or_costs():
    frame = study.enrich(repeated_data().iloc[:3])
    original = frame.copy()
    small = study.prepare(frame, study.Profile(100000, None, False))
    large = study.prepare(frame, study.Profile(200000, None, True))
    assert np.allclose(small.risk_utilization, large.risk_utilization*2)
    assert small.estimated_cost_pct.equals(large.estimated_cost_pct)
    pd.testing.assert_frame_equal(frame, original)


def test_present_path_inputs_use_full_50_column_schema_with_partial_missing_values():
    frame = study.enrich(repeated_data())
    for index, column in enumerate(study.PATH_FEATURES):
        frame[column] = np.sin(np.arange(len(frame))*.01+index)
        frame.loc[frame.index[::31], column] = np.nan
    val, scores, detail = study.block(frame, study.FOLDS[0], study.Profile(100000, None, True))
    assert len(detail["model_features"])==50
    assert detail["all_missing_fit_columns"]==[]
    assert len(val)==len(scores)


@pytest.mark.parametrize("path", [False, True])
def test_model_uses_only_declared_inputs_effective_day_weights_and_keeps_unknowns(path):
    frame = study.enrich(repeated_data())
    frame[study.PATH_FEATURES] = np.nan
    frame.loc[frame.index[:3], "decision_premium"] = 10000
    val_index = study.raw_parts(frame, study.longer_fold(study.FOLDS[0]))[2].index[0]
    frame.loc[val_index, "label_status"] = "CENSORED"
    frame.loc[val_index, "label"] = np.nan
    profile = study.Profile(200000, 252, path)
    val, scores, detail = study.block(frame, study.FOLDS[0], profile)
    assert len(detail["features"])==(50 if path else 44)
    assert detail["all_missing_fit_columns"]==(study.PATH_FEATURES if path else [])
    assert not set(study.PATH_FEATURES).intersection(detail["model_features"])
    assert not set(study.OUTCOMES).intersection(detail["features"])
    assert 0<detail["fit_weight_sum"]<=detail["fit_days"]
    assert study.risk_eligible(val, profile.capital, 3, 5).all()
    assert val.loc[val_index, "label_status"]=="CENSORED"
    assert len(val)==len(scores)
    for info in detail["calibration"].values():
        if info["enabled"]:
            assert info["weight_sum"]==pytest.approx(info["days"])


def test_drawdown_gates_scale_only_with_research_capital():
    metrics = {"trades":100, "coverage":1., "win_rate_pct":71., "profit_factor":1.5,
               "net_pnl":20000., "stress_net_pnl":19000., "max_drawdown":15000.,
               "daily_loss_overshoots":0, "active_months":10, "positive_months":7}
    assert study.rejection_reasons(metrics, 100000, win_floor=70, strict=True)==["drawdown_10pct_capital"]
    assert study.rejection_reasons(metrics, 200000, win_floor=70, strict=True)==[]
    metrics["win_rate_pct"] = 70
    assert "win_rate" in study.rejection_reasons(metrics, 200000, win_floor=70, strict=True)


@pytest.mark.parametrize("profitable,path_winner", [(False, False), (True, False), (True, True)])
def test_run_freezes_before_audit_preserves_reserved_days_and_cannot_promote(tmp_path, monkeypatch, profitable, path_winner):
    source, labels, output = tmp_path/"source", tmp_path/"labels", tmp_path/"run"
    source.mkdir()
    labels.mkdir()
    (source/"report.json").write_text("{}")
    (labels/"report.json").write_text("{}")
    frame = repeated_data()
    if not profitable:
        frame["net_pnl"], frame["stress_net_pnl"], frame["label"] = -100., -110., 0
    frame.loc[frame.trading_day<=pd.Timestamp("2025-12-31").date()].to_pickle(labels/f"{study.RULE.key}-labels.pkl")
    monkeypatch.setattr(study, "SOURCE", source)
    monkeypatch.setattr(study, "LABEL_SOURCE", labels)
    monkeypatch.setattr(study, "PROFILES", study.PROFILES[:2])
    monkeypatch.setattr(study, "load_source", lambda *args:(frame.copy(), {"dataset_hash":"frozen"}))
    reads = []
    def paths(frame):
        if str(frame.trading_day.min())>="2026-01-08":
            assert (output/"policy.json").exists()
            assert str(frame.trading_day.max())<="2026-10-13"
        else:
            assert str(frame.trading_day.max())<="2025-12-31"
        reads.append(True)
        return {}
    monkeypatch.setattr(study, "load_paths", paths)
    monkeypatch.setattr(study, "relabel", lambda frame, *args:frame.copy())
    def block(frame, fold, profile, output=None):
        if fold==study.FINAL_FOLD:
            assert (tmp_path/"run"/"policy.json").exists()
        val = study.raw_parts(frame, study.longer_fold(fold))[2]
        return val, np.repeat(np.nan if path_winner and not profile.option_path else .6, len(val)), {}
    monkeypatch.setattr(study, "block", block)
    monkeypatch.setattr(study, "bootstrap_comparison", lambda *args:{
        "mean_net_per_trade_95pct":[1., 2.], "paired_net_improvement_per_day_95pct":[1., 2.]})
    report = study.run(output)
    assert bool(report["audit"])==profitable
    if path_winner:
        assert report["selection"]["profile"]["option_path"]
        assert report["audit"]["matched_control"] is not None
    assert len(reads)==(2 if profitable else 1)
    assert not report["activation_eligible"] and not report["app_model_selection_changed"]
    with pytest.raises(ValueError, match="overwrite"):
        study.run(output)
