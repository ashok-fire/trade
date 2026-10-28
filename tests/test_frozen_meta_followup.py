import json

import numpy as np
import pandas as pd
import pytest

from research import frozen_meta_followup as study
from research.profit_objective import fingerprint,write_json
from test_premium_meta_models import repeated_data


@pytest.mark.parametrize("profitable",[False,True])
def test_provisional_audit_requires_full_gates_and_keeps_capital_fixed(tmp_path,monkeypatch,profitable):
    previous,source,output = tmp_path/"previous",tmp_path/"source",tmp_path/"run"
    previous.mkdir()
    source.mkdir()
    frame = repeated_data()
    if not profitable:
        frame["net_pnl"],frame["stress_net_pnl"],frame["label"] = -100.,-110.,0
    dev = frame.loc[frame.trading_day<=pd.Timestamp("2025-12-31").date()]
    frames = []
    for fold in study.FOLDS:
        val = study.meta.raw_parts(dev,study.meta.longer_fold(fold))[2].copy()
        val["fold"] = fold[0]
        frames.append(val)
    val = pd.concat(frames)
    expected = study.capital_replay(val,np.repeat(.6,len(val)),100000,"CE")
    write_json(previous/"specification.json",{"frozen":True})
    write_json(previous/"policy.json",{"frozen":True})
    write_json(previous/"report.json",{"specification_hash":fingerprint(previous/"specification.json"),
        "policy_hash":fingerprint(previous/"policy.json")})
    write_json(previous/f"{study.RULE.key}-development.json",{"profiles":[{"learner":"HIST_GRADIENT",
        "scope":"CE","condition":"ALL","curve":[{"cutoff":.55,"metrics":expected}]}]})
    dev.to_pickle(previous/f"{study.RULE.key}-labels.pkl")
    monkeypatch.setattr(study,"PREVIOUS",previous)
    monkeypatch.setattr(study,"SOURCE",source)
    monkeypatch.setattr(study,"load_source",lambda *args:(frame.copy(),{}))
    def block(frame,fold,save=None):
        if fold==study.FINAL_FOLD:
            assert (output/"policy.json").exists()
        val = study.meta.raw_parts(frame,study.meta.longer_fold(fold))[2]
        return val,{"HIST_GRADIENT":np.repeat(.6,len(val))},{}
    monkeypatch.setattr(study.meta,"block",block)
    monkeypatch.setattr(study,"load_paths",lambda *args:{})
    monkeypatch.setattr(study,"relabel",lambda frame,*args:frame.copy())
    monkeypatch.setattr(study,"bootstrap_comparison",lambda *args:{
        "mean_net_per_trade_95pct":[1.,2.],"paired_net_improvement_per_day_95pct":[1.,2.]})
    result = study.run(output)
    assert result["development_reproduced_exactly"] and not result["activation_eligible"]
    policy = json.loads((output/"policy.json").read_text())
    assert policy["capital"]==100000
    assert (result["status"]=="HISTORICAL_TARGET_MET_FORWARD_REQUIRED")==profitable
    assert not result["app_model_selection_changed"]
    with pytest.raises(ValueError,match="overwrite"):
        study.run(output)
