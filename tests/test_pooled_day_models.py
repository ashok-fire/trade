import numpy as np
import pandas as pd
import pytest

from research import pooled_day_models as study
from test_decision_day_specialists import data


def test_long_calibration_precedes_validation_and_audit():
    assert study.longer_fold(study.FOLDS[0])[1:3]==("2023-10-01","2024-03-31")
    assert study.longer_fold(study.FINAL_FOLD)[1:3]==("2025-07-01","2025-12-31")


def test_pool_keeps_first_unknown_per_direction_not_later_known_winner():
    frame = data().iloc[:3].copy()
    frame.loc[1,"timestamp"] = frame.loc[0,"timestamp"]+pd.Timedelta(minutes=5)
    frame.loc[2,"timestamp"] = frame.loc[0,"timestamp"]+pd.Timedelta(minutes=10)
    frame["trading_day"] = frame.loc[0,"trading_day"]
    frame.loc[0,["label_status","label","net_pnl"]] = ["CENSORED",None,None]
    frame.loc[2,"direction"] = "PE"
    first = study.first_side_decisions(frame)
    assert first.index.tolist()==[0,2]
    assert study.labelled(first).index.tolist()==[2]
    assert study.day_weights(first).sum()==pytest.approx(2)


def test_consensus_requires_two_available_models():
    values = {"LOGISTIC":np.array([.8,.8,np.nan]),"HIST_GRADIENT":np.array([.6,np.nan,np.nan]),
              "CATBOOST":np.array([np.nan,np.nan,np.nan])}
    scores = study.consensus(values)
    assert scores[0]==pytest.approx(.7) and np.isnan(scores[1:]).all()


def test_real_pooled_estimators_and_weighted_pipeline_fit():
    frame = data()
    other = frame.copy()
    other["direction"] = "PE"
    other["timestamp"] += pd.Timedelta(minutes=5)
    source = pd.concat([frame,other],ignore_index=True).sort_values("timestamp",kind="stable")
    validation,scores,metadata = study.block(source,study.FOLDS[0])
    assert len(validation)==len(scores["CONSENSUS"])
    for name in study.LEARNERS:
        assert metadata[name]["fit_rows"]>metadata[name]["fit_days"]
        assert metadata[name]["calibration_rows"]>=40
        assert len(scores[name])==len(validation)
        if metadata[name]["enabled"]:
            assert np.isfinite(scores[name]).all()


@pytest.mark.parametrize("profitable",[False,True])
def test_run_freezes_selection_before_final_block_without_promotion(tmp_path,monkeypatch,profitable):
    source,output = tmp_path/"source",tmp_path/"run"
    source.mkdir()
    (source/"report.json").write_text("{}")
    frame = data()
    if not profitable:
        frame["net_pnl"],frame["stress_net_pnl"],frame["label"] = -100.,-110.,0
    monkeypatch.setattr(study,"SOURCE",source)
    monkeypatch.setattr(study,"load_source",lambda *args:(frame.copy(),{"dataset_hash":"frozen"}))
    audits = []
    def block(frame,fold,output_models=None):
        if fold==study.FINAL_FOLD:
            assert (output/"policy.json").exists()
            audits.append(True)
        val = study.first_side_decisions(study.raw_parts(frame,study.longer_fold(fold))[2])
        return val,{name:np.repeat(.6,len(val)) for name in (*study.LEARNERS,"CONSENSUS")},{}
    monkeypatch.setattr(study,"block",block)
    monkeypatch.setattr(study,"bootstrap_comparison",lambda *args:{
        "mean_net_per_trade_95pct":[1.,2.],"paired_net_improvement_per_day_95pct":[1.,2.]})
    result = study.run(output)
    assert bool(audits)==profitable
    assert not result["activation_eligible"] and not result["app_model_selection_changed"]
    assert (result["status"]=="HISTORICAL_TARGET_MET_FORWARD_REQUIRED")==profitable
    with pytest.raises(ValueError,match="overwrite"):
        study.run(output)
