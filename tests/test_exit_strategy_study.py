from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research import exit_strategy_study as study
from research.walk_forward_tuning import split_fold


def case(clock="2025-01-02T04:00Z"):
    stamp = pd.Timestamp(clock)
    opportunity = SimpleNamespace(timestamp=stamp,trading_day=stamp.tz_convert("Asia/Kolkata").date(),
        decision_premium=100.,lot_size=50,instrument_key="OPTION")
    path = {stamp:study.Bar(stamp,Decimal(100),Decimal(101),Decimal(99),Decimal(100))}
    return opportunity,path


def add(path,stamp,opening=100,high=101,low=99,close=100):
    stamp = pd.Timestamp(stamp)
    path[stamp] = study.Bar(stamp,*map(lambda value:Decimal(str(value)),(opening,high,low,close)))


def test_entry_bar_stop_wins_ambiguous_touch_and_costs_are_deducted():
    op,path = case()
    add(path,op.timestamp+pd.Timedelta(minutes=5),high=120,low=80)
    result = study.simulate(op,path,study.ExitRule(15,15,30),Decimal("0.0005"))
    assert result["exit_reason"] == "PREMIUM_STOP"
    assert result["net_pnl"] < -750 and result["net_return"] < -.15


def test_known_opening_target_gap_precedes_later_intrabar_stop():
    op,path = case()
    add(path,op.timestamp+pd.Timedelta(minutes=5))
    add(path,op.timestamp+pd.Timedelta(minutes=10),opening=130,high=132,low=70,close=100)
    result = study.simulate(op,path,study.ExitRule(15,15,30),Decimal("0.0005"))
    assert result["exit_reason"] == "GAP_TARGET" and 0 < result["net_pnl"] < 750


def test_gap_stop_is_not_filled_at_unavailable_stop_price():
    op,path = case()
    add(path,op.timestamp+pd.Timedelta(minutes=5))
    add(path,op.timestamp+pd.Timedelta(minutes=10),opening=70,high=80,low=65,close=75)
    result = study.simulate(op,path,study.ExitRule(15,15,30),Decimal("0.0005"))
    assert result["exit_reason"] == "GAP_STOP" and result["net_pnl"] < -1500


def test_deadline_uses_open_not_future_high_low_or_close():
    op,path = case()
    for offset in range(5,40,5):
        add(path,op.timestamp+pd.Timedelta(minutes=offset))
    deadline = op.timestamp+pd.Timedelta(minutes=35)
    add(path,deadline,opening=110,high=170,low=1,close=150)
    result = study.simulate(op,path,study.ExitRule(25,25,30),Decimal("0.0005"))
    assert result["exit_reason"] == "TIME_EXIT" and result["exit_timestamp"] == deadline
    assert 400 < result["net_pnl"] < 500


def test_missing_next_bar_or_contiguous_path_stays_unknown():
    op,path = case()
    add(path,op.timestamp+pd.Timedelta(minutes=10),high=130)
    assert study.simulate(op,path,study.ExitRule(25,25,30),Decimal("0.0005")) is None
    add(path,op.timestamp+pd.Timedelta(minutes=5))
    del path[op.timestamp+pd.Timedelta(minutes=10)]
    add(path,op.timestamp+pd.Timedelta(minutes=15),high=130)
    assert study.simulate(op,path,study.ExitRule(25,25,30),Decimal("0.0005")) is None


def test_double_slippage_reruns_path_instead_of_only_adjusting_pnl():
    op,path = case()
    add(path,op.timestamp+pd.Timedelta(minutes=5),high=115.08)
    add(path,op.timestamp+pd.Timedelta(minutes=10),low=80)
    rule = study.ExitRule(15,15,30)
    base = study.simulate(op,path,rule,Decimal("0.0005"))
    stress = study.simulate(op,path,rule,Decimal("0.001"))
    assert base["net_pnl"] > 0 and stress["net_pnl"] < 0


def test_square_off_is_1520_ist_and_unknown_label_cannot_be_fabricated():
    op,path = case("2025-01-02T09:30Z")  # 15:00 IST signal, 15:05 entry.
    for offset in range(5,25,5):
        add(path,op.timestamp+pd.Timedelta(minutes=offset))
    result = study.simulate(op,path,study.ExitRule(25,25,60),Decimal("0.0005"))
    assert result["exit_timestamp"].tz_convert("Asia/Kolkata").strftime("%H:%M") == "15:20"
    data = pd.DataFrame([vars(op)])
    relabelled = study.relabel(data,{},study.ExitRule(25,25,60))
    assert relabelled.iloc[0].label_status == "CENSORED" and pd.isna(relabelled.iloc[0].label)


def synthetic_frame(profitable):
    days = pd.bdate_range("2023-01-02","2026-10-13")
    wins = np.arange(len(days))%5 != 0
    net = np.where(wins,200.,-100.) if profitable else np.repeat(-100.,len(days))
    return pd.DataFrame({"timestamp":days.tz_localize("UTC")+pd.Timedelta(hours=4),
        "trading_day":days.date,"direction":"CE","instrument_key":"OPTION","setup_direction":1.,
        "trend_atr_strength":2.5,"label_status":"COMPLETE","label":(net>0).astype(int),
        "net_pnl":net,"stress_net_pnl":net-10,"decision_premium":100.,"lot_size":50})


@pytest.mark.parametrize("profitable",[False,True])
def test_study_freezes_stage_selections_and_never_fits_without_edge(tmp_path,monkeypatch,profitable):
    source,output = tmp_path/"source",tmp_path/"study"
    source.mkdir()
    (source/"report.json").write_text("{}")
    data = synthetic_frame(profitable)
    monkeypatch.setattr(study,"SOURCE",source)
    monkeypatch.setattr(study,"RULES",study.RULES[:1])
    monkeypatch.setattr(study,"load_source",lambda path:(data.copy(),{"dataset_hash":"frozen"}))
    path_reads = []
    def paths(data):
        if data.trading_day.min()>=date(2026,1,8):
            assert (output/"model-policy.json").exists()
        else:
            assert data.trading_day.max()<=date(2025,12,31)
        path_reads.append(data.trading_day.max())
        return {}
    monkeypatch.setattr(study,"load_paths",paths)
    monkeypatch.setattr(study,"relabel",lambda data,*args:data.copy())
    fitted = []
    def fit(data,fold,save=None):
        assert profitable and (output/"strategy-policy.json").exists()
        fitted.append(fold[0])
        val = split_fold(data,fold)[2]
        return val,np.repeat(.6,len(val)),{}
    monkeypatch.setattr(study,"fit_side_scores",fit)
    monkeypatch.setattr(study,"bootstrap_comparison",lambda *args:{
        "mean_net_per_trade_95pct":[1,2],"paired_net_improvement_per_day_95pct":[1,2]})
    report = study.run(output)
    assert report["model_trained"] == profitable and bool(fitted) == profitable
    assert bool(report["historical_audit"]) == profitable
    assert not report["activation_eligible"] and not report["app_model_selection_changed"]
    assert len(path_reads) == (2 if profitable else 1)
    with pytest.raises(ValueError,match="overwrite"):
        study.run(output)
