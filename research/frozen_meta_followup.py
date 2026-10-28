"""Investigate one explicitly underpowered CE lead; do not relax final gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.aligned_training import bootstrap_comparison
from research import premium_meta_models as meta
from research.exit_strategy_study import ExitRule,load_paths,relabel
from research.precision_tuning import load_source
from research.profit_objective import fingerprint,write_json
from research.walk_forward_tuning import FOLDS,FINAL_FOLD,select_capital,summarize

PREVIOUS = Path("models/research/premium-meta-models-20261025")
SOURCE = Path("models/research/walk-forward-capital-20261025")
RULE = ExitRule(25,15,60)
CUTOFF = .55


def capital_replay(frame,scores,capital,side):
    mask = np.ones(len(frame),dtype=bool) if side=="BOTH" else (frame.direction==side).to_numpy()
    data,values = frame.loc[mask],np.asarray(scores)[mask]
    selected = []
    for fold in FOLDS:
        window = (data.fold==fold[0]).to_numpy()
        selected.append(select_capital(data.loc[window],values[window],{"CE":CUTOFF,"PE":CUTOFF},capital,3,5))
    return summarize(pd.concat(selected),capital,5)


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen follow-up")
    previous = json.loads((PREVIOUS/"report.json").read_text())
    for filename,key in (("specification.json","specification_hash"),("policy.json","policy_hash")):
        if fingerprint(PREVIOUS/filename)!=previous[key]:
            raise ValueError("Previous experiment provenance changed")
    profiles = json.loads((PREVIOUS/f"{RULE.key}-development.json").read_text())["profiles"]
    profile = next(row for row in profiles if row["learner"]=="HIST_GRADIENT" and row["scope"]=="CE" and row["condition"]=="ALL")
    expected = next(point["metrics"] for point in profile["curve"] if point["cutoff"]==CUTOFF)
    output.mkdir(parents=True)
    specification = {"schema":"frozen-provisional-meta-v1","code_hash":fingerprint(Path(__file__)),
        "previous_report_hash":fingerprint(PREVIOUS/"report.json"),
        "purpose":"Investigate the existing >70% CE-only lead despite insufficient development count. This is evidence gathering, NOT acceptance or relaxing final gates",
        "rule":vars(RULE),"learner":"HIST_GRADIENT","scope":"CE","condition":"ALL","cutoff":CUTOFF,
        "expected_development":expected,"capital":100000,"risk_per_trade_pct":3,"daily_loss_limit_pct":5,
        "amount_diagnostics":"Frozen score/cutoff, one historical lot, capitals100000/200000/500000/1000000, CE and BOTH; development diagnostics ONLY. Do not choose the audit capital from those results",
        "primary_audit":"Fixed CE branch, fixed100000 capital, one lot, same six-month calibration, no retraining hyperparameter search; 2026 audit previously inspected",
        "final_gates":"100+ completed audit trades, net wins strictly>70%, coverage99%, PF1.3, positive net/double-slippage, DD<=10000, no daily overshoot, >=60% positive active months, positive lower95 mean expectancy and paired improvement over unscored CE baseline",
        "reserved_dates_excluded":["2026-10-14","2026-10-23"],"activation_eligible":False,
        "app_or_registry_changes":False,"fresh_forward_evidence_required":True}
    write_json(output/"specification.json",specification)
    spec_hash = fingerprint(output/"specification.json")
    source,_ = load_source(SOURCE)
    labels = pd.read_pickle(PREVIOUS/f"{RULE.key}-labels.pkl")
    dev = source.loc[labels.index].copy()
    for column in ("label_status","label","net_pnl","stress_net_pnl","net_return","exit_reason","exit_timestamp"):
        dev[column] = labels[column]
    frames,predictions = [],[]
    for fold in FOLDS:
        val,scores,_ = meta.block(dev,fold)
        val = val.copy()
        val["fold"] = fold[0]
        frames.append(val)
        predictions.extend(scores["HIST_GRADIENT"])
    validation,scores = pd.concat(frames),np.asarray(predictions)
    reproduced = capital_replay(validation,scores,100000,"CE")
    if reproduced!=expected:
        raise ValueError("Frozen CE development result did not reproduce exactly")
    amounts = [{"capital":capital,"scope":side,"metrics":capital_replay(validation,scores,capital,side)}
        for capital in (100000,200000,500000,1000000) for side in ("CE","BOTH")]
    write_json(output/"amount-diagnostics.json",amounts)
    policy = {"schema":"frozen-provisional-meta-v1","specification_hash":spec_hash,"activation_eligible":False,
        "rule":vars(RULE),"learner":"HIST_GRADIENT","scope":"CE","cutoff":CUTOFF,"capital":100000}
    write_json(output/"policy.json",policy)
    policy_hash = fingerprint(output/"policy.json")
    later = source.loc[(source.trading_day>=pd.Timestamp("2026-01-08").date())&
                       (source.trading_day<=pd.Timestamp("2026-10-13").date())].copy()
    later = relabel(later,load_paths(later),RULE)
    evaluation,values,metadata = meta.block(pd.concat([dev,later]),FINAL_FOLD,output)
    mask = (evaluation.direction=="CE").to_numpy()
    evaluation = evaluation.loc[mask]
    selected = select_capital(evaluation,values["HIST_GRADIENT"][mask],{"CE":CUTOFF},100000,3,5)
    baseline = select_capital(evaluation,np.ones(len(evaluation)),{"CE":0.},100000,3,5)
    m = summarize(selected,100000,5)
    uncertainty = bootstrap_comparison(evaluation,selected,baseline)
    passed = (m["trades"]>=100 and (m["win_rate_pct"] or 0)>70 and m["coverage"]>=.99
        and (m["profit_factor"] or 0)>=1.3 and m["net_pnl"]>0 and m["stress_net_pnl"]>0
        and m["max_drawdown"]<=10000 and m["daily_loss_overshoots"]==0
        and m["positive_months"]>=.6*m["active_months"]
        and uncertainty.get("mean_net_per_trade_95pct",[-1])[0]>0
        and uncertainty.get("paired_net_improvement_per_day_95pct",[-1])[0]>0)
    status = "HISTORICAL_TARGET_MET_FORWARD_REQUIRED" if passed else "REJECTED_PROVISIONAL_CE_AUDIT"
    if fingerprint(output/"specification.json")!=spec_hash or fingerprint(output/"policy.json")!=policy_hash:
        raise ValueError("Frozen policy changed during audit")
    report = {"status":status,"specification_hash":spec_hash,"policy_hash":policy_hash,"activation_eligible":False,
        "development_reproduced_exactly":True,"development":reproduced,"audit":m,
        "unscored_CE_audit":summarize(baseline,100000,5),"uncertainty":uncertainty,"models":metadata,
        "amount_diagnostics":amounts,"app_model_selection_changed":False}
    write_json(output/"report.json",report)
    selected.to_json(output/"audit-trades.json",orient="records",date_format="iso",indent=2)
    lines = ["# Frozen provisional CE investigation","",f"Result: **{status}**.",
        "", "The 70.59% development lead has only 68 trades and was not an accepted candidate. This follow-up gathers later evidence without changing the final100-trade/>70%/profit/risk gates.",
        "", "## Later 2026 audit (previously inspected history)","",json.dumps(m,indent=2),
        "", "## Capital sensitivity: unchanged model, cutoff and one-lot sizing","",
        "| Capital | Scope | Trades | Wins % | Net ₹ | PF |","|---:|---|---:|---:|---:|---:|"]
    lines += [f"| {r['capital']} | {r['scope']} | {r['metrics']['trades']} | {r['metrics']['win_rate_pct']} | {r['metrics']['net_pnl']} | {r['metrics']['profit_factor']} |" for r in amounts]
    lines += ["", "Amount diagnostics did not choose or change the fixed ₹100,000 CE audit. Neither amount changes nor reused historical successes establish future performance. Reserved October14–23 excluded; native exits remain incompatible with runtime. No settings, live orders, model references or registry state changed."]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":status,"development":reproduced,"audit":m},indent=2),flush=True)
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    run(args.output)
