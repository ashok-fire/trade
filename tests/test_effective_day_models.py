import numpy as np
import pandas as pd
import pytest

from research import effective_day_models as study
from test_premium_meta_models import repeated_data


def test_weights_sum_to_days_not_overlapping_rows_and_decay_by_observed_days():
    frame = pd.DataFrame({"trading_day":[1,1,2,2,2,3]})
    weights = study.effective_weights(frame)
    assert weights.sum()==pytest.approx(3)
    assert pd.Series(weights).groupby(frame.trading_day).sum().tolist()==[1,1,1]
    decayed = pd.Series(study.effective_weights(frame,1)).groupby(frame.trading_day).sum()
    assert decayed.tolist()==[.25,.5,1]


def test_cost_inputs_do_not_read_outcomes():
    frame = repeated_data().iloc[:3].copy()
    first = study.enrich(frame)
    frame["net_pnl"],frame["label"] = 1000000,1
    second = study.enrich(frame)
    assert np.allclose(first.estimated_cost_pct,second.estimated_cost_pct)
    assert np.allclose(first.risk_utilization,second.risk_utilization)
    assert (first.estimated_cost_pct>0).all()


@pytest.mark.parametrize("variant",study.VARIANTS)
def test_effective_day_training_has_causal_risk_parity_and_correct_weight_scale(variant):
    source = study.enrich(repeated_data())
    source.loc[source.index[:3],"decision_premium"] = 10000
    val,scores,detail = study.block(source,study.FOLDS[0],variant)
    assert detail["fit_weight_sum"]<=detail["fit_days"]+1e-8
    assert len(detail["features"])==(44 if variant[1] else 40)
    assert study.risk_eligible(val,100000,3,5).all()
    assert len(val)==len(scores)
    for info in detail["calibration"].values():
        if info["enabled"]:
            assert info["weight_sum"]==pytest.approx(info["days"])


@pytest.mark.parametrize("profitable",[False,True])
def test_day_weight_study_cannot_promote_and_freezes_before_audit(tmp_path,monkeypatch,profitable):
    source,label_source,output = tmp_path/"source",tmp_path/"labels",tmp_path/"run"
    source.mkdir()
    label_source.mkdir()
    (source/"report.json").write_text("{}")
    (label_source/"report.json").write_text("{}")
    frame = repeated_data()
    if not profitable:
        frame["net_pnl"],frame["stress_net_pnl"],frame["label"] = -100.,-110.,0
    frame.loc[frame.trading_day<=pd.Timestamp("2025-12-31").date()].to_pickle(label_source/f"{study.RULE.key}-labels.pkl")
    monkeypatch.setattr(study,"SOURCE",source)
    monkeypatch.setattr(study,"LABEL_SOURCE",label_source)
    monkeypatch.setattr(study,"VARIANTS",study.VARIANTS[:1])
    monkeypatch.setattr(study,"load_source",lambda *args:(frame.copy(),{"dataset_hash":"frozen"}))
    def block(frame,fold,variant,save=None):
        if fold==study.FINAL_FOLD:
            assert (output/"policy.json").exists()
        val = study.raw_parts(frame,study.longer_fold(fold))[2]
        return val,np.repeat(.6,len(val)),{}
    monkeypatch.setattr(study,"block",block)
    monkeypatch.setattr(study,"load_paths",lambda *args:{})
    monkeypatch.setattr(study,"relabel",lambda frame,*args:frame.copy())
    monkeypatch.setattr(study,"bootstrap_comparison",lambda *args:{
        "mean_net_per_trade_95pct":[1.,2.],"paired_net_improvement_per_day_95pct":[1.,2.]})
    report = study.run(output)
    assert bool(report["audit"])==profitable
    assert not report["activation_eligible"] and not report["app_model_selection_changed"]
    with pytest.raises(ValueError,match="overwrite"):
        study.run(output)
