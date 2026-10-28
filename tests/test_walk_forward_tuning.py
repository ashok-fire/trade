from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.database import DEFAULTS
from research import walk_forward_tuning as tuning


def frame():
    days = pd.bdate_range("2023-01-02", "2026-10-23")
    return pd.DataFrame({"timestamp": days.tz_localize("UTC") + pd.Timedelta(hours=4),
        "trading_day": days.date, "direction": "CE", "setup_family": "ORB",
        "decision_premium": 100., "lot_size": 50,
        "label_status": "COMPLETE", "label": np.where(np.arange(len(days))%3,1,0),
        "net_pnl": np.where(np.arange(len(days))%3,200.,-100.),
        "stress_net_pnl": np.where(np.arange(len(days))%3,190.,-110.)})


def test_requested_capital_and_risk_defaults():
    assert DEFAULTS["paper_capital"] == "100000"
    assert DEFAULTS["risk_per_trade_pct"] == "3"
    assert DEFAULTS["daily_loss_limit_pct"] == "5"
    assert DEFAULTS["reward_risk"] == "1.5"


def test_risk_settings_route_accepts_requested_three_percent_not_more(session):
    from decimal import Decimal
    from app.main import save_risk
    from app.models import Setting
    values = dict(paper_capital=Decimal("100000"),risk_per_trade_pct=Decimal("3"),
        daily_loss_limit_pct=Decimal("5"),max_trades_per_day=5,max_consecutive_losses=2,
        premium_emergency_stop_pct=Decimal("25"),reward_risk=Decimal("1.5"),
        paper_slippage_pct=Decimal("0.05"),session=session)
    response = save_risk(**values)
    assert "Risk+settings+saved" in response.headers["location"]
    assert session.get(Setting,"risk_per_trade_pct").value == "3"
    response = save_risk(**{**values,"risk_per_trade_pct":Decimal("3.1")})
    assert "safe+limits" in response.headers["location"]
    assert session.get(Setting,"risk_per_trade_pct").value == "3"


def test_new_backtest_snapshots_daily_limit_and_defaults(session,monkeypatch):
    from decimal import Decimal
    from app import main
    from app.models import BacktestRun
    import inspect
    import json
    from sqlalchemy import select
    assert inspect.signature(main.start_backtest).parameters["capital"].default.default == Decimal("100000")
    submitted = []
    monkeypatch.setattr(main.backtest_service,"submit",submitted.append)
    response = main.start_backtest(indices=["NIFTY"],start_date=date(2025,1,1),end_date=date(2025,2,1),
        capital=Decimal("100000"),risk_pct=Decimal("3"),slippage_pct=Decimal("0.05"),
        historical_provider="UPSTOX",cost_model="UPSTOX",strategy_mode="RULE_ONLY",
        strike_policy="ATM_ITM1",entry_delay_bars=1,minimum_option_volume=1,
        maximum_spread_pct=Decimal("1.5"),session=session)
    run = session.scalar(select(BacktestRun))
    values = json.loads(run.configuration_json)
    assert response.status_code == 303 and submitted == [run.id]
    assert values["daily_loss_limit_pct"] == "5" and values["risk_pct"] == "3"


def test_splits_group_days_purge_boundaries_and_exclude_reserved_window():
    data = frame()
    for fold in (*tuning.FOLDS,tuning.FINAL_FOLD):
        fit, cal, val = tuning.split_fold(data,fold)
        assert fit.trading_day.max() < cal.trading_day.min() < val.trading_day.min()
        fit_days = sorted(data.loc[data.trading_day < date.fromisoformat(fold[1]),"trading_day"])
        assert fit_days[-1] not in set(fit.trading_day)
        assert val.trading_day.max() <= date(2026,10,13)
        assert not set(fit.trading_day) & set(cal.trading_day)


def test_whole_lot_risk_and_cash_use_no_outcome_fields():
    data = pd.DataFrame({"decision_premium":[100.,300.,100.,float("nan")],
                         "lot_size":[50,50,5000,50]})
    assert tuning.risk_eligible(data,100000,3,5).tolist() == [True,False,False,False]
    assert tuning.risk_eligible(data,100000,.25,.75).tolist() == [False]*4
    with pytest.raises(ValueError):
        tuning.risk_eligible(data,100000,3.1,5)


def test_censored_first_entry_stops_ledger_instead_of_choosing_later_winner():
    data = frame().head(3).copy()
    data.loc[0,["label_status","label","net_pnl","stress_net_pnl"]] = ["CENSORED",None,None,None]
    selected = tuning.select_capital(data,np.repeat(.8,3),{"CE":.5},100000,3,5)
    metrics = tuning.summarize(selected,100000,5)
    assert selected.index.tolist() == [0]
    assert metrics["trades"] == 0 and metrics["censored"] == 1
    assert metrics["win_rate_pct"] is None


def test_lower_threshold_can_select_later_entry_only_if_first_is_rejected():
    data = frame().head(2).copy()
    data.loc[1,"trading_day"] = data.loc[0,"trading_day"]
    selected = tuning.select_capital(data,[.4,.8],{"CE":.5},100000,3,5)
    assert selected.index.tolist() == [1]
    selected = tuning.select_capital(data,[.8,.9],{"CE":.5},100000,3,5)
    assert selected.index.tolist() == [0]


def test_losing_policy_has_no_fallback():
    data = frame()
    validations = []
    for fold in tuning.FOLDS:
        val = tuning.split_fold(data,fold)[2].copy()
        val["fold"] = fold[0]
        validations.append(val)
    data = pd.concat(validations)
    data["net_pnl"] = -100.
    data["stress_net_pnl"] = -110.
    winner,curve = tuning.choose_policy(data,np.repeat(.6,len(data)),100000,3,5)
    assert winner is None and len(curve) == 9


@pytest.mark.parametrize("profitable",[True,False])
def test_run_freezes_policy_before_final_audit_and_preserves_runtime(tmp_path,monkeypatch,profitable):
    data = frame()
    if not profitable:
        data["net_pnl"],data["stress_net_pnl"] = -100.,-110.
    output = tmp_path/"run"
    monkeypatch.setattr(tuning,"config",SimpleNamespace(database_url="sqlite://"))
    def build(*args,**kwargs):
        assert kwargs["end_date"] == date(2026,10,13)
        assert kwargs["include_decision_context"]
        return data.loc[data.trading_day <= kwargs["end_date"]].copy(),{}
    monkeypatch.setattr(tuning,"build_option_quality_samples",build)
    monkeypatch.setattr(tuning,"GRID",(tuning.GRID[0],))
    final_calls = []
    def fit_scores(data,fold,parameters,model_output=None):
        if fold == tuning.FINAL_FOLD:
            assert (output/"policy.json").exists()
            final_calls.append(True)
        validation = tuning.split_fold(data,fold)[2]
        return validation,np.repeat(.6,len(validation)),{},{}
    monkeypatch.setattr(tuning,"fit_scores",fit_scores)
    monkeypatch.setattr(tuning,"RegimeModel",lambda path:SimpleNamespace(available=True,
        metadata={"direction_thresholds":{"CE":.5,"PE":.5}},
        quality_probabilities=lambda values:np.repeat(.6,len(values))))
    monkeypatch.setattr(tuning,"bootstrap_comparison",lambda *args:{"descriptive":True})
    report = tuning.run(output)
    assert report["completed_candidates"] == 1
    assert bool(report["historical_audit"]) == profitable
    assert bool(final_calls) == profitable
    assert not report["activation_eligible"] and not report["app_model_selection_changed"]
    assert (output/"report.json").exists()
    with pytest.raises(ValueError,match="overwrite"):
        tuning.run(output)
