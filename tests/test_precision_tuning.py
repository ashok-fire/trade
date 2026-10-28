import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from research import precision_tuning as precision
from research.profit_objective import fingerprint,write_json
from research.walk_forward_tuning import split_fold


def metrics(**changes):
    return {"trades":100,"win_rate_pct":71.,"coverage":1.,"profit_factor":2.,
        "stress_net_pnl":10000.,"net_pnl":11000.,"max_drawdown":3000.,
        "daily_loss_overshoots":0,"positive_months":10,"active_months":15,**changes}


def test_target_is_realised_wins_not_probability_and_needs_sample_and_profit():
    folds = [metrics() for _ in range(6)]
    assert precision.qualifies(metrics(),folds)
    assert not precision.qualifies(metrics(win_rate_pct=70.),folds)
    assert not precision.qualifies(metrics(trades=10,win_rate_pct=100.),folds)
    assert not precision.qualifies(metrics(net_pnl=-1),folds)
    assert not precision.qualifies(metrics(stress_net_pnl=-1),folds)
    assert not precision.qualifies(metrics(max_drawdown=10001),folds)
    assert not precision.qualifies(metrics(),folds[:3])
    assert precision.qualifies(metrics(),[],require_folds=False)


def source_frame():
    days = pd.bdate_range("2023-01-02","2026-10-13")
    wins = np.arange(len(days))%5 != 0
    return pd.DataFrame({"timestamp":days.tz_localize("UTC")+pd.Timedelta(hours=4),
        "trading_day":days.date,"direction":"CE","label_status":"COMPLETE",
        "label":wins.astype(int),"net_pnl":np.where(wins,200.,-100.),
        "stress_net_pnl":np.where(wins,190.,-110.),"decision_premium":100.,"lot_size":50})


def make_source(path,data):
    path.mkdir()
    write_json(path/"specification.json",{"frozen":True})
    write_json(path/"policy.json",{"frozen":True})
    data.to_pickle(path/"opportunities.pkl")
    write_json(path/"report.json",{"specification_hash":fingerprint(path/"specification.json"),
        "policy_hash":fingerprint(path/"policy.json"),
        "dataset_hash":hashlib.sha256(pd.util.hash_pandas_object(data,index=False).to_numpy().tobytes()).hexdigest()})


def test_cached_data_provenance_must_match(tmp_path):
    source = tmp_path/"source"
    data = source_frame()
    make_source(source,data)
    loaded,_ = precision.load_source(source)
    assert len(loaded) == len(data)
    data.loc[0,"net_pnl"] += 1
    data.to_pickle(source/"opportunities.pkl")
    with pytest.raises(ValueError,match="fingerprint"):
        precision.load_source(source)


@pytest.mark.parametrize("passes",[True,False])
def test_precision_run_has_no_fallback_and_freezes_before_audit(tmp_path,monkeypatch,passes):
    source,output = tmp_path/"source",tmp_path/"study"
    data = source_frame()
    if not passes:
        data["net_pnl"],data["stress_net_pnl"],data["label"] = -100.,-110.,0
    make_source(source,data)
    hashes = {p.name:fingerprint(p) for p in source.iterdir()}
    monkeypatch.setattr(precision,"GRID",precision.GRID[:1])
    monkeypatch.setattr(precision,"CUTOFFS",(.4,.425,.45))
    audits = []
    def fit(data,fold,parameters,save=None):
        if fold == precision.FINAL_FOLD:
            assert (output/"policy.json").exists()
            audits.append(True)
        validation = split_fold(data,fold)[2]
        return validation,np.repeat(.6,len(validation)),{},{}
    monkeypatch.setattr(precision,"fit_scores",fit)
    from types import SimpleNamespace
    monkeypatch.setattr(precision,"RegimeModel",lambda *args:SimpleNamespace(available=True,
        metadata={"direction_thresholds":{"CE":.5,"PE":.5}},
        quality_probabilities=lambda values:np.repeat(.6,len(values))))
    monkeypatch.setattr(precision,"bootstrap_comparison",lambda *args:{
        "mean_net_per_trade_95pct":[1.,2.],"paired_net_improvement_per_day_95pct":[1.,2.]})
    result = precision.run(output,source)
    assert bool(audits) == passes
    assert not result["activation_eligible"] and not result["app_model_selection_changed"]
    assert (result["status"] == "HISTORICAL_TARGET_MET_FORWARD_REQUIRED") == passes
    assert {p.name:fingerprint(p) for p in source.iterdir()} == hashes
    with pytest.raises(ValueError,match="overwrite"):
        precision.run(output,source)
