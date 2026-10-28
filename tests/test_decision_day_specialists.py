from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.training_contract import ALIGNED_FEATURES
from research import decision_day_specialists as study


def data():
    days = pd.bdate_range("2023-01-02","2026-10-13")
    rng = np.random.default_rng(42)
    values = rng.normal(size=(len(days),len(ALIGNED_FEATURES)))
    frame = pd.DataFrame(values,columns=ALIGNED_FEATURES)
    wins = np.arange(len(days))%5 != 0
    frame["timestamp"] = days.tz_localize("UTC")+pd.Timedelta(hours=4)
    frame["trading_day"],frame["direction"] = days.date,"CE"
    frame["label_status"],frame["label"] = "COMPLETE",wins.astype(int)
    frame["net_pnl"],frame["stress_net_pnl"] = np.where(wins,200.,-100.),np.where(wins,190.,-110.)
    frame["decision_premium"],frame["lot_size"] = 100.,50
    frame["atr_pct"] = .001+np.arange(len(days))*.000001
    frame["rsi"] = wins.astype(float)*2-1
    return frame


def test_first_unknown_is_not_replaced_and_low_score_cannot_choose_later_bar():
    frame = data().iloc[:2].copy()
    frame.loc[1,"timestamp"] = frame.loc[0,"timestamp"]+pd.Timedelta(minutes=5)
    frame.loc[1,"trading_day"] = frame.loc[0,"trading_day"]
    frame.loc[0,["label_status","label","net_pnl"]] = ["CENSORED",None,None]
    profile = study.Profile("CE","EARLY","ALL","LOGISTIC")
    chosen = study.first_decisions(frame,profile)
    assert chosen.index.tolist()==[0] and chosen.iloc[0].label_status=="CENSORED"
    assert study.labelled(chosen).empty
    assert study.select_capital(chosen,[.4],{"CE":.5},100000,3,5).empty


def test_clock_uses_entry_boundary_ist_not_signal_start():
    frame = data().iloc[:2].copy()
    frame.loc[0,"timestamp"] = pd.Timestamp("2023-01-02T05:25Z")  # 10:55 signal, 11:00 boundary.
    frame.loc[1,"timestamp"] = pd.Timestamp("2023-01-03T03:55Z")  # 09:25 signal, too early.
    assert study.first_decisions(frame,study.Profile("CE","EARLY","ALL","LOGISTIC")).empty
    assert study.first_decisions(frame,study.Profile("CE","MID","ALL","LOGISTIC")).index.tolist()==[0]


def test_volatility_threshold_uses_fit_only_and_unknown_rows_stay_in_validation():
    frame = data()
    profile = study.Profile("CE","EARLY","LOW","LOGISTIC")
    _,_,first = study.block(frame,study.FOLDS[0],profile)
    altered = frame.copy()
    mask = altered.trading_day>=date(2024,1,1)
    altered.loc[mask,"atr_pct"] = .000001
    altered.loc[mask,"label_status"] = "CENSORED"
    altered.loc[mask,["label","net_pnl"]] = np.nan
    val,scores,second = study.block(altered,study.FOLDS[0],profile)
    assert first["volatility_cutoff_fit_only"]==second["volatility_cutoff_fit_only"]
    assert first["fit_days"]==second["fit_days"]
    assert not second["enabled"] and np.isnan(scores).all()
    assert (val.label_status=="CENSORED").all()


@pytest.mark.parametrize("name",study.LEARNERS)
def test_fixed_estimators_can_fit_and_calibrate_chronologically(name):
    frame = data()
    profile = study.Profile("CE","EARLY","ALL",name)
    val,scores,metadata = study.block(frame,study.FOLDS[0],profile)
    assert metadata["fit_days"]>=100 and metadata["calibration_days"]>=20
    assert len(val)==len(scores) and val.trading_day.min()>=date(2024,4,1)
    if metadata["enabled"]:
        assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    else:
        assert np.isnan(scores).all()
    if name=="HIST_GRADIENT":
        assert study.make_estimator(name).early_stopping is False


@pytest.mark.parametrize("profitable",[False,True])
def test_run_freezes_policy_before_audit_without_runtime_changes(tmp_path,monkeypatch,profitable):
    source,output = tmp_path/"source",tmp_path/"run"
    source.mkdir()
    (source/"report.json").write_text("{}")
    frame = data()
    if not profitable:
        frame["net_pnl"],frame["stress_net_pnl"],frame["label"] = -100.,-110.,0
    monkeypatch.setattr(study,"SOURCE",source)
    monkeypatch.setattr(study,"PROFILES",study.PROFILES[:1])
    monkeypatch.setattr(study,"load_source",lambda *args:(frame.copy(),{"dataset_hash":"frozen"}))
    audits = []
    def block(frame,fold,profile):
        if fold==study.FINAL_FOLD:
            assert (output/"policy.json").exists()
            audits.append(True)
        val = study.first_decisions(study.raw_parts(frame,fold)[2],profile)
        return val,np.repeat(.6,len(val)),{}
    monkeypatch.setattr(study,"block",block)
    monkeypatch.setattr(study,"bootstrap_comparison",lambda *args:{
        "mean_net_per_trade_95pct":[1.,2.],"paired_net_improvement_per_day_95pct":[1.,2.]})
    result = study.run(output)
    assert bool(audits)==profitable
    assert not result["activation_eligible"] and not result["app_model_selection_changed"]
    assert (result["status"]=="HISTORICAL_TARGET_MET_FORWARD_REQUIRED")==profitable
    with pytest.raises(ValueError,match="overwrite"):
        study.run(output)
