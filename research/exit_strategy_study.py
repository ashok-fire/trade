"""Preregistered premium-exit research on existing causal ORB opportunities.

No production strategy, setting, model selection, or broker order is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from itertools import product
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from app.aligned_training import PARAMETERS, bootstrap_comparison, day_weights, labelled
from app.config import config
from app.models import Candle
from app.risk import option_charges
from app.specialist_research import quality_logit
from app.training_contract import ALIGNED_FEATURES
from research.precision_tuning import load_source
from research.profit_objective import fingerprint, write_json
from research.walk_forward_tuning import FOLDS, FINAL_FOLD, select_capital, split_fold, summarize


@dataclass(frozen=True)
class ExitRule:
    stop_pct: int
    target_pct: int
    hold_minutes: int

    @property
    def key(self):
        return f"stop{self.stop_pct}-target{self.target_pct}-hold{self.hold_minutes}"


@dataclass(frozen=True)
class Bar:
    timestamp: pd.Timestamp
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


RULES = tuple(ExitRule(*values) for values in product((15,25),(15,25),(30,60)))
CONDITIONS = ("ALL","TREND_ALIGNED","WEAK_TREND")
THRESHOLDS = tuple(value/100 for value in range(40,81,5))
MODEL_PARAMETERS = {**PARAMETERS,"l2_leaf_reg":30}
SOURCE = Path("models/research/walk-forward-capital-20261025")


def utc(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def load_paths(frame):
    """Read only verified option OHLC, bounded to this stage's declared days."""
    if frame.empty:
        return {}
    needed = set(zip(frame.instrument_key,frame.trading_day))
    keys = sorted(frame.instrument_key.unique())
    start = datetime.combine(frame.trading_day.min(),time.min,tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(UTC)
    end = datetime.combine(frame.trading_day.max()+timedelta(days=1),time.min,tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(UTC)
    engine = create_engine(config.database_url)
    @event.listens_for(engine,"connect")
    def read_only(connection,_):
        connection.execute("PRAGMA query_only=ON")
    paths = defaultdict(dict)
    try:
        with Session(engine) as session:
            for offset in range(0,len(keys),350):
                rows = session.execute(select(Candle.instrument_key,Candle.timestamp,Candle.open,
                    Candle.high,Candle.low,Candle.close).where(Candle.provider=="GROWW",
                    Candle.data_role=="OPTION",Candle.timeframe=="5minute",Candle.source_verified.is_(True),
                    Candle.instrument_key.in_(keys[offset:offset+350]),Candle.timestamp>=start,Candle.timestamp<end))
                for key,stamp,opening,high,low,close in rows:
                    stamp = utc(stamp)
                    pair = (key,stamp.tz_convert("Asia/Kolkata").date())
                    if pair in needed:
                        paths[pair][stamp] = Bar(stamp,opening,high,low,close)
    finally:
        engine.dispose()
    return dict(paths)


def simulate(opportunity,path,rule,slip):
    """Causal next-open entry; known opening gaps first, otherwise stop-first."""
    signal = utc(opportunity.timestamp)
    signal_bar = path.get(signal)
    if signal_bar is None:
        return None
    if abs(float(signal_bar.close)-float(opportunity.decision_premium)) > .005:
        raise ValueError("Cached signal premium disagrees with stored candle")
    entry_time = signal+pd.Timedelta(minutes=5)
    first = path.get(entry_time)
    if first is None or first.open <= 0:
        return None
    entry = first.open*(1+slip)
    stop = entry*(1-Decimal(rule.stop_pct)/100)
    target = entry*(1+Decimal(rule.target_pct)/100)
    square_off = utc(datetime.combine(opportunity.trading_day,time(15,20),tzinfo=ZoneInfo("Asia/Kolkata")))
    deadline = min(entry_time+pd.Timedelta(minutes=rule.hold_minutes),square_off)
    for stamp in pd.date_range(entry_time,deadline,freq="5min"):
        bar = path.get(stamp)
        if (bar is None or min(bar.open,bar.high,bar.low,bar.close) <= 0
                or bar.low > min(bar.open,bar.close) or bar.high < max(bar.open,bar.close)):
            return None
        if stamp == deadline:
            base,reason,exit_time = bar.open,"TIME_EXIT",stamp
        elif bar.open <= stop:
            base,reason,exit_time = bar.open,"GAP_STOP",stamp
        elif bar.open >= target:
            # Do not award a better-than-limit opening fill.
            base,reason,exit_time = target,"GAP_TARGET",stamp
        elif bar.low <= stop:
            base,reason,exit_time = stop,"PREMIUM_STOP",stamp+pd.Timedelta(minutes=5)
        elif bar.high >= target:
            base,reason,exit_time = target,"PREMIUM_TARGET",stamp+pd.Timedelta(minutes=5)
        else:
            continue
        exit_price = base*(1-slip)
        charges = option_charges(entry,exit_price,int(opportunity.lot_size),"GROWW",opportunity.trading_day)
        net = (exit_price-entry)*int(opportunity.lot_size)-charges
        return {"net_pnl":float(net),"net_return":float(net/(entry*int(opportunity.lot_size))),
                "exit_reason":reason,"exit_timestamp":exit_time}
    return None


def relabel(frame,paths,rule):
    records = []
    for opportunity in frame.itertuples():
        path = paths.get((opportunity.instrument_key,opportunity.trading_day),{})
        base = simulate(opportunity,path,rule,Decimal("0.0005"))
        # Re-run the path, not only the fill deduction: entry-linked levels may change.
        stress = simulate(opportunity,path,rule,Decimal("0.001"))
        if base is None or stress is None:
            records.append({"net_pnl":None,"stress_net_pnl":None,"net_return":None,"label":None,
                "label_status":"CENSORED","exit_reason":"INCOMPLETE_PATH","exit_timestamp":None})
        else:
            records.append({**base,"stress_net_pnl":stress["net_pnl"],"label":int(base["net_pnl"]>0),
                            "label_status":"COMPLETE"})
    result = frame.copy()
    updates = pd.DataFrame(records,index=frame.index)
    for column in updates:
        result[column] = updates[column]
    return result


def scope(frame,side,condition):
    mask = frame.direction == side
    if condition == "TREND_ALIGNED":
        mask &= frame.setup_direction*frame.trend_atr_strength >= 2
    elif condition == "WEAK_TREND":
        mask &= frame.trend_atr_strength.abs() <= 1
    elif condition != "ALL":
        raise ValueError("Unknown condition")
    return frame.loc[mask].copy()


def window_metrics(frame,scores=None,cutoff=0.):
    scores = np.ones(len(frame)) if scores is None else np.asarray(scores)
    selected,windows = [],[]
    for fold in FOLDS:
        if "fold" in frame:
            mask = (frame.fold==fold[0]).to_numpy()
        else:
            a,b = map(date.fromisoformat,fold[3:])
            mask = ((frame.trading_day>=a)&(frame.trading_day<=b)).to_numpy()
        trades = select_capital(frame.loc[mask],scores[mask],{"CE":cutoff,"PE":cutoff},100000,3,5)
        selected.append(trades)
        windows.append({"window":fold[0],**summarize(trades,100000,5)})
    combined = pd.concat(selected)
    return summarize(combined,100000,5),windows,combined


def foundation_pass(metrics,windows):
    return (metrics["trades"]>=100 and metrics["coverage"]>=.99
        and (metrics["profit_factor"] or 0)>=1.3 and metrics["net_pnl"]>0 and metrics["stress_net_pnl"]>0
        and metrics["max_drawdown"]<=10000 and metrics["daily_loss_overshoots"]==0
        and metrics["positive_months"]>=.6*metrics["active_months"]
        and sum(row["trades"]>=10 and row["stress_net_pnl"]>0 for row in windows)>=4)


def fit_side_scores(frame,fold,output=None):
    fit,cal,val = split_fold(frame,fold)
    counts = {"fit_rows":len(fit),"calibration_rows":len(cal),
              "fit_days":fit.trading_day.nunique(),"calibration_days":cal.trading_day.nunique()}
    scores = np.full(len(val),np.nan)
    if len(fit)<100 or len(cal)<40 or fit.label.nunique()<2 or cal.label.nunique()<2:
        return val,scores,{"enabled":False,"reason":"insufficient fitting/calibration data",**counts}
    model = CatBoostClassifier(**MODEL_PARAMETERS)
    model.fit(fit[ALIGNED_FEATURES],fit.label.astype(int),sample_weight=day_weights(fit))
    calibration = LogisticRegression(C=1.,random_state=42).fit(
        quality_logit(model.predict_proba(cal[ALIGNED_FEATURES])[:,1]),cal.label.astype(int),
        sample_weight=day_weights(cal))
    slope = float(calibration.coef_[0,0])
    metadata = {"enabled":slope>0,"slope":slope,"intercept":float(calibration.intercept_[0]),**counts}
    if slope>0 and len(val):
        scores = calibration.predict_proba(quality_logit(model.predict_proba(val[ALIGNED_FEATURES])[:,1]))[:,1]
    if output:
        model.save_model(str(output/"quality.cbm"))
        write_json(output/"calibration.json",metadata)
    return val,scores,metadata


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen exit study")
    output.mkdir(parents=True)
    specification = {"schema":"premium-exit-study-v1","source_report_hash":fingerprint(SOURCE/"report.json"),
        "code_hash":fingerprint(Path(__file__)),"hypothesis":"Premium-native exits and fixed existing trend-strength conditions improve option outcomes before fitting another classifier",
        "rules":[asdict(rule) for rule in RULES],"sides":["CE","PE"],"conditions":CONDITIONS,
        "condition_definitions":{"ALL":"no additional condition","TREND_ALIGNED":"setup_direction * trend_atr_strength >= 2", "WEAK_TREND":"abs(trend_atr_strength) <= 1"},
        "entry":"frozen causal ORB ATM/ITM1 opportunity; exact next-bar open plus 0.05% slippage; no reselection by outcome",
        "exit":"filled premium-linked stop/target; known open gap first, otherwise stop before target; time exit at deadline open, capped at 15:20 IST",
        "stress":"twice slippage per side; re-simulate entry-linked stop/target path; same versioned estimated Groww charges",
        "risk":"unchanged capital 100000 / gross stop-risk budget 3% / daily gate 5%; one historical lot; conservative signal-close 25% risk reservation for every variant; quarter ledgers reset",
        "foundation_gates":"100 trades, >=99% known coverage, PF>=1.3, positive net and stress, DD<=10000, no observed daily overshoot, 4/6 stress-positive windows with >=10 trades, >=60% positive active months",
        "selection":"48 fixed combinations, select maximum stressed net among foundation passers only; freeze before model fitting; no fallback",
        "model":"one fixed day-weighted shallow CatBoost per selected strategy, separate positive-slope Platt calibration; same 40 features, no new indicators",
        "model_parameters":MODEL_PARAMETERS,"thresholds":THRESHOLDS,
        "model_selection":"foundation gates plus neighbouring cutoffs >=60 trades/PF1.1/stress positive; positive paired lower95 improvement versus unscored strategy required",
        "goal":"strictly greater than 70% realised net wins with 100+ complete trades and positive economic gates; foundation passage alone is NOT that goal",
        "folds":FOLDS,"later_audit":FINAL_FOLD,"previously_inspected_history":True,
        "reserved_window_not_used":["2026-10-14","2026-10-23"],
        "limits":"No 2026 relabelling until strategy and model selection frozen; if no foundation, no model fitted; no live activation",
        "limitations":["Candle-level stop/target execution is a proxy, not guaranteed executable liquidity",
            "Historical spreads, quote age, IV/delta and live feature semantics are not reconciled",
            "Selection performance is biased by strategy/model search; 2026 is previously inspected history",
            "One-lot, reset-quarter development P&L is not full-account compounded return"],
        "activation_eligible":False,"application_or_registry_changes":False}
    write_json(output/"specification.json",specification)
    spec_hash = fingerprint(output/"specification.json")
    source,source_report = load_source(SOURCE)
    development = source.loc[source.trading_day<=date(2025,12,31)].copy()
    paths = load_paths(development)
    write_json(output/"progress.json",{"status":"STRATEGY_REPLAY","completed_rules":0,"total_rules":len(RULES)})
    results,best,best_data = [],None,None
    for rule in RULES:
        data = relabel(development,paths,rule)
        for side,condition in product(("CE","PE"),CONDITIONS):
            subset = scope(data,side,condition)
            metrics,windows,_ = window_metrics(subset)
            legacy,_,_ = window_metrics(scope(development,side,condition))
            record = {"rule":asdict(rule),"key":rule.key,"side":side,"condition":condition,
                "metrics":metrics,"windows":windows,"same_scope_legacy_exit":legacy,
                "foundation_passed":bool(foundation_pass(metrics,windows))}
            results.append(record)
            if record["foundation_passed"] and (best is None or metrics["stress_net_pnl"]>best["metrics"]["stress_net_pnl"]):
                best,best_data = record,subset
        write_json(output/f"{rule.key}.json",results[-6:])
        print(f"Exit replay {rule.key}: {sum(row['foundation_passed'] for row in results)} qualifying strategy scopes so far",flush=True)
        write_json(output/"progress.json",{"status":"STRATEGY_REPLAY","completed_rules":len(results)//6,"total_rules":len(RULES)})
    strategy = {"selected":best,"specification_hash":spec_hash,"dataset_hash":source_report["dataset_hash"],"activation_eligible":False}
    write_json(output/"strategy-policy.json",strategy)
    strategy_hash = fingerprint(output/"strategy-policy.json")
    report = {"status":"REJECTED_NO_FOUNDATION_EDGE","activation_eligible":False,
        "specification_hash":spec_hash,"strategy_hash":strategy_hash,"dataset_hash":source_report["dataset_hash"],
        "combinations_tested":len(results),"foundation_passers":sum(row["foundation_passed"] for row in results),
        "strategy":best,"model_trained":False,"historical_audit":None,"results":results,
        "app_model_selection_changed":False,"limitations":specification["limitations"]}
    if best:
        best_data.to_pickle(output/"selected-development.pkl")
        frames,predictions,calibrations = [],[],[]
        for fold in FOLDS:
            val,scores,metadata = fit_side_scores(best_data,fold)
            val = val.copy()
            val["fold"] = fold[0]
            frames.append(val)
            predictions.extend(scores)
            calibrations.append({"fold":fold[0],**metadata})
        validation,scores = pd.concat(frames),np.asarray(predictions)
        curve = []
        for cutoff in THRESHOLDS:
            metrics,windows,_ = window_metrics(validation,scores,cutoff)
            curve.append({"cutoff":cutoff,"metrics":metrics,"windows":windows,
                          "passed":bool(foundation_pass(metrics,windows))})
        candidates = [row for i,row in enumerate(curve) if 0<i<len(curve)-1 and row["passed"]
            and all(curve[j]["metrics"]["trades"]>=60 and (curve[j]["metrics"]["profit_factor"] or 0)>=1.1
                    and curve[j]["metrics"]["stress_net_pnl"]>0 for j in (i-1,i+1))]
        chosen = max(candidates,key=lambda row:row["metrics"]["stress_net_pnl"]) if candidates else None
        unscored = window_metrics(validation)[2]
        uncertainty = None
        if chosen:
            selected = window_metrics(validation,scores,chosen["cutoff"])[2]
            uncertainty = bootstrap_comparison(validation,selected,unscored)
            if uncertainty.get("paired_net_improvement_per_day_95pct",[-1])[0]<=0:
                chosen = None
        policy = {"strategy_hash":strategy_hash,"selected_cutoff":chosen["cutoff"] if chosen else None,
                  "parameters":MODEL_PARAMETERS,"activation_eligible":False}
        write_json(output/"model-policy.json",policy)
        policy_hash = fingerprint(output/"model-policy.json")
        report.update({"model_trained":True,"model_curve":curve,"calibrations":calibrations,
            "validation_uncertainty":uncertainty,"model_policy_hash":policy_hash,
            "status":"REJECTED_NO_MODEL_IMPROVEMENT"})
        if chosen:
            # Only now re-label the selected exit rule for the later 2026 audit.
            later = source.loc[(source.trading_day>=date(2026,1,8))&(source.trading_day<=date(2026,10,13))].copy()
            later = relabel(later,load_paths(later),ExitRule(**best["rule"]))
            combined = pd.concat([best_data,scope(later,best["side"],best["condition"])])
            evaluation,eval_scores,calibration = fit_side_scores(combined,FINAL_FOLD,output)
            selected = select_capital(evaluation,eval_scores,{"CE":chosen["cutoff"],"PE":chosen["cutoff"]},100000,3,5)
            baseline = select_capital(evaluation,np.ones(len(evaluation)),{"CE":0,"PE":0},100000,3,5)
            audit = summarize(selected,100000,5)
            comparison = bootstrap_comparison(evaluation,selected,baseline)
            target = (audit["trades"]>=100 and (audit["win_rate_pct"] or 0)>70 and audit["coverage"]>=.99
                and (audit["profit_factor"] or 0)>=1.3 and audit["net_pnl"]>0 and audit["stress_net_pnl"]>0
                and audit["max_drawdown"]<=10000 and audit["daily_loss_overshoots"]==0
                and audit["positive_months"]>=.6*audit["active_months"]
                and comparison.get("mean_net_per_trade_95pct",[-1])[0]>0
                and comparison.get("paired_net_improvement_per_day_95pct",[-1])[0]>0)
            report.update({"historical_audit":audit,"audit_uncertainty":comparison,
                "unscored_audit":summarize(baseline,100000,5),
                "status":"HISTORICAL_TARGET_MET_FORWARD_REQUIRED" if target else "REJECTED_LATER_AUDIT_OR_70_PERCENT_TARGET"})
            selected.to_json(output/"audit-trades.json",orient="records",date_format="iso",indent=2)
        if fingerprint(output/"model-policy.json")!=policy_hash:
            raise ValueError("Model selection changed after audit")
    if fingerprint(output/"specification.json")!=spec_hash or fingerprint(output/"strategy-policy.json")!=strategy_hash:
        raise ValueError("Frozen study changed")
    write_json(output/"report.json",report)
    write_json(output/"progress.json",{"status":"COMPLETE","result":report["status"]})
    lines = ["# Premium-exit strategy study", "",f"Result: **{report['status']}**.",
        "",f"Tested {len(results)} fixed exit/direction/condition combinations. Foundation passers: {report['foundation_passers']}. Model trained: {report['model_trained']}.",
        "", "Development selection: six chronological windows in 2024–2025. Reserved October 14–23 history excluded. Any 2026 audit is previously inspected history, not fresh validation.",
        "", "| Exit rule | Side | Condition | Trades | Wins % | PF | Net ₹ | Stress net ₹ | Foundation |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in results:
        m = row["metrics"]
        lines.append(f"| {row['key']} | {row['side']} | {row['condition']} | {m['trades']} | {m['win_rate_pct']} | {m['profit_factor']} | {m['net_pnl']} | {m['stress_net_pnl']} | {row['foundation_passed']} |")
    lines += ["", "## Selection and audit", "",json.dumps({key:report.get(key) for key in
        ("strategy","model_trained","status","historical_audit","audit_uncertainty")},indent=2),
        "", "## Limits", "", *["- "+item for item in specification["limitations"]],
        "- Capital, app risk settings, runtime exits and model references unchanged; no live orders or promotion.",
        "- Foundation gates screen economic plausibility, not a >70% claim or live-readiness approval.",
        "- Missing paths remain censored; the first selected unknown outcome halts its quarterly ledger."]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({key:report[key] for key in ("status","foundation_passers","model_trained","historical_audit")},indent=2),flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    run(args.output)
