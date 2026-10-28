import numpy as np
import pandas as pd
import pytest

from research import premium_meta_models as study
from test_decision_day_specialists import data


def repeated_data():
    pieces = []
    for minute in (0,5,10,15,20):
        part = data()
        part["timestamp"] += pd.Timedelta(minutes=minute)
        part["instrument_key"] = "OPTION"
        part["net_return"] = part.net_pnl/5000
        part["exit_reason"] = "TIME_EXIT"
        part["exit_timestamp"] = part.timestamp+pd.Timedelta(minutes=30)
        pieces.append(part)
    return pd.concat(pieces,ignore_index=True).sort_values("timestamp",kind="stable")


def test_scope_uses_only_known_direction_and_trend_features():
    frame = pd.DataFrame({"direction":["CE","PE","CE"],"setup_direction":[1,-1,1],
        "trend_atr_strength":[2.5,-3.,-4.]})
    assert study.eligible(frame,"CE","ALIGNED").tolist()==[True,False,False]
    assert study.eligible(frame,"BOTH","ALIGNED").tolist()==[True,True,False]
    with pytest.raises(ValueError):
        study.eligible(frame,"BOTH","unknown")


def test_meta_fit_keeps_rows_but_weights_days_and_has_calibration_support():
    frame = repeated_data()
    validation,scores,metadata = study.block(frame,study.FOLDS[0])
    for name in study.LEARNERS:
        assert metadata[name]["fit_rows"]==5*metadata[name]["fit_days"]
        assert metadata[name]["calibration_rows"]==5*metadata[name]["calibration_days"]
        assert len(scores[name])==len(validation)
    weights = study.day_weights(frame)
    totals = pd.Series(weights,index=frame.index).groupby(frame.trading_day).sum()
    assert totals.max()==pytest.approx(totals.min())


@pytest.mark.parametrize("profitable",[False,True])
def test_meta_run_preserves_source_and_freezes_before_2026(tmp_path,monkeypatch,profitable):
    source,output = tmp_path/"source",tmp_path/"run"
    source.mkdir()
    (source/"report.json").write_text("{}")
    frame = repeated_data()
    if not profitable:
        frame["net_pnl"],frame["stress_net_pnl"],frame["label"] = -100.,-110.,0
    monkeypatch.setattr(study,"SOURCE",source)
    monkeypatch.setattr(study,"RULES",study.RULES[:1])
    monkeypatch.setattr(study,"load_source",lambda *args:(frame.copy(),{"dataset_hash":"frozen"}))
    reads = []
    def load_paths(frame):
        if str(frame.trading_day.min())>="2026-01-08":
            assert (output/"policy.json").exists()
        else:
            assert str(frame.trading_day.max())<="2025-12-31"
        reads.append(True)
        return {}
    monkeypatch.setattr(study,"load_paths",load_paths)
    monkeypatch.setattr(study,"relabel",lambda frame,*args:frame.copy())
    def block(frame,fold,save=None):
        validation = study.raw_parts(frame,study.longer_fold(fold))[2]
        return validation,{name:np.repeat(.6,len(validation)) for name in (*study.LEARNERS,"CONSENSUS")},{}
    monkeypatch.setattr(study,"block",block)
    monkeypatch.setattr(study,"bootstrap_comparison",lambda *args:{
        "mean_net_per_trade_95pct":[1.,2.],"paired_net_improvement_per_day_95pct":[1.,2.]})
    report = study.run(output)
    assert bool(report["historical_audit"])==profitable
    assert len(reads)==(2 if profitable else 1)
    assert not report["activation_eligible"] and not report["app_model_selection_changed"]
    with pytest.raises(ValueError,match="overwrite"):
        study.run(output)
