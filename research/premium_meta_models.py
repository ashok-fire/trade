"""Cost-aware meta-labels on native option exits, with causal day-grouped validation."""
from __future__ import annotations

import argparse
import json
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from app.aligned_training import bootstrap_comparison,day_weights,labelled
from app.specialist_research import quality_logit
from app.training_contract import ALIGNED_FEATURES
from research.decision_day_specialists import make_estimator,raw_parts,select_policy
from research.exit_strategy_study import ExitRule,load_paths,relabel
from research.pooled_day_models import LEARNERS,consensus,longer_fold
from research.precision_tuning import load_source
from research.profit_objective import fingerprint,write_json
from research.walk_forward_tuning import FOLDS,FINAL_FOLD,select_capital,summarize

SOURCE = Path("models/research/walk-forward-capital-20261025")
RULES = (ExitRule(15,15,30),ExitRule(25,15,60))
SCOPES = ("BOTH","CE","PE")
CONDITIONS = ("ALL","ALIGNED")


def eligible(frame,side,condition):
    mask = np.ones(len(frame),dtype=bool) if side=="BOTH" else (frame.direction==side).to_numpy(copy=True)
    if condition=="ALIGNED":
        mask &= (frame.setup_direction*frame.trend_atr_strength>=2).to_numpy()
    elif condition!="ALL":
        raise ValueError("Unknown condition")
    return mask


def block(frame,fold,output=None):
    raw_fit,raw_cal,val = raw_parts(frame,longer_fold(fold))
    fit,cal = labelled(raw_fit),labelled(raw_cal)
    scores,metadata = {},{}
    for name in LEARNERS:
        scores[name] = np.full(len(val),np.nan)
        detail = {"fit_rows":len(fit),"fit_days":fit.trading_day.nunique(),
            "calibration_rows":len(cal),"calibration_days":cal.trading_day.nunique(),"enabled":False}
        if (len(fit)>=500 and len(cal)>=200 and detail["fit_days"]>=100
                and detail["calibration_days"]>=40 and fit.label.nunique()==2 and cal.label.nunique()==2):
            with threadpool_limits(limits=1):
                model = make_estimator(name)
                fit_weights,cal_weights = day_weights(fit),day_weights(cal)
                if name=="LOGISTIC":
                    model.fit(fit[ALIGNED_FEATURES],fit.label.astype(int),logisticregression__sample_weight=fit_weights)
                else:
                    model.fit(fit[ALIGNED_FEATURES],fit.label.astype(int),sample_weight=fit_weights)
                calibration = LogisticRegression(C=1.,max_iter=2000,random_state=42).fit(
                    quality_logit(model.predict_proba(cal[ALIGNED_FEATURES])[:,1]),
                    cal.label.astype(int),sample_weight=cal_weights)
                slope = float(calibration.coef_[0,0])
                detail.update({"enabled":slope>0,"slope":slope,"intercept":float(calibration.intercept_[0])})
                if slope>0 and len(val):
                    scores[name] = calibration.predict_proba(quality_logit(model.predict_proba(val[ALIGNED_FEATURES])[:,1]))[:,1]
                if output:
                    import joblib
                    joblib.dump({"estimator":model,"calibration":calibration,"features":ALIGNED_FEATURES,
                        "schema":"premium-meta-research-v1","enabled":slope>0},output/f"{name.lower()}.joblib")
        metadata[name] = detail
    scores["CONSENSUS"] = consensus(scores)
    return val,scores,metadata


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen premium-meta study")
    output.mkdir(parents=True)
    specification = {"schema":"premium-meta-research-v1","code_hash":fingerprint(Path(__file__)),
        "source_report_hash":fingerprint(SOURCE/"report.json"),
        "hypothesis":"Economic option-exit labels are more learnable than underlying-touch proxy labels; train a skip/enter meta-model despite unfiltered strategy weakness",
        "rules":[vars(rule) for rule in RULES],"features":ALIGNED_FEATURES,"scopes":SCOPES,"conditions":CONDITIONS,
        "condition_definition":"ALIGNED means setup_direction * trend_atr_strength >=2; evaluation-only fixed condition, not fitted on future outcomes",
        "training":"all labelled causal bar opportunities pooled CE/PE, equal total day weight; overlapping rows are NOT independent trades",
        "calibration":"separate six-month day-grouped Platt, positive slope, fit>=500 rows/100 days and cal>=200 rows/40 days; whole-day boundary purges",
        "models":"same fixed regularised logistic, shallow histogram boosting and shallow CatBoost as decision-day study; fixed equal consensus of >=2 enabled models",
        "folds":[longer_fold(fold) for fold in FOLDS],"audit":longer_fold(FINAL_FOLD),
        "selection":"48 learner/exit/side/condition profiles, nine cutoffs .40-.80; first score-passing eligible bar/day; unknown selected exits never replaced",
        "goal":"strictly >70% realised net wins,100+ complete trades,PF1.3,positive net and double-slip,DD<=10000,coverage99%,4/6 stress-positive windows with>=10 trades,60% positive months; neighbour support",
        "partial_track":"separately labelled >=60% economically robust candidate, never >70%; same fixed select_policy and risk/profit gates; audit at most one selected policy",
        "capital":100000,"risk_per_trade_pct":3,"daily_loss_limit_pct":5,
        "exits":"existing tested native premium simulator, gap-aware, stop-first ambiguous candles, horizon/square-off open; double-slip path re-simulated",
        "2026":"Only selected exit gets 2026 relabelling after policy frozen; previously inspected period, NOT fresh evidence",
        "reserved_dates_excluded":["2026-10-14","2026-10-23"],"activation_eligible":False,"app_or_registry_writes":False,
        "limitations":["Candidate selection reuses development history and is biased by repeated research",
            "Native stops/targets differ from runtime; research models are not compatible with active execution",
            "Candle fills, historical feature meanings and estimated charges need genuine forward quote-based validation",
            "One-lot, reset-quarter P&L is not a full-account compounded return"]}
    write_json(output/"specification.json",specification)
    spec_hash = fingerprint(output/"specification.json")
    source,source_report = load_source(SOURCE)
    development = source.loc[source.trading_day<=date(2025,12,31)].copy()
    paths = load_paths(development)
    results,best,best_data = [],None,None
    for rule in RULES:
        data = relabel(development,paths,rule)
        # Preserve derived outcomes separately for future reproducibility, not in
        # the production DB or by overwriting the original source-cache labels.
        data[["timestamp","instrument_key","label_status","label","net_pnl","stress_net_pnl","net_return","exit_reason","exit_timestamp"]].to_pickle(output/f"{rule.key}-labels.pkl")
        frames,predictions,diagnostics = [],{name:[] for name in (*LEARNERS,"CONSENSUS")},[]
        for fold in FOLDS:
            val,scores,metadata = block(data,fold)
            val = val.copy()
            val["fold"] = fold[0]
            frames.append(val)
            for name in predictions:
                predictions[name].extend(scores[name])
            diagnostics.append({"window":fold[0],"models":metadata})
            print(f"Premium meta {rule.key}: {fold[0]} complete",flush=True)
        validation = pd.concat(frames)
        for name,side,condition in product(predictions,SCOPES,CONDITIONS):
            mask = eligible(validation,side,condition)
            chosen,curve = select_policy(validation.loc[mask],np.asarray(predictions[name])[mask])
            record = {"rule":vars(rule),"key":rule.key,"learner":name,"scope":side,
                "condition":condition,"selected":chosen,"curve":curve}
            results.append(record)
            if chosen and (best is None or (chosen["target_pass"],chosen["metrics"]["stress_net_pnl"])>
                    (best["selected"]["target_pass"],best["selected"]["metrics"]["stress_net_pnl"])):
                best,best_data = record,data
        write_json(output/f"{rule.key}-development.json",{"profiles":results[-24:],"diagnostics":diagnostics})
        write_json(output/"progress.json",{"status":"FITTING","completed_exit_rules":len(results)//24,"total":len(RULES)})
    policy = {"schema":"premium-meta-research-v1","selected":{key:best[key] for key in
        ("rule","learner","scope","condition")} if best else None,"cutoff":best["selected"]["cutoff"] if best else None,
        "development_target_passed":bool(best and best["selected"]["target_pass"]),
        "specification_hash":spec_hash,"dataset_hash":source_report["dataset_hash"],"activation_eligible":False}
    write_json(output/"policy.json",policy)
    policy_hash = fingerprint(output/"policy.json")
    status,audit,uncertainty = "REJECTED_NO_SUPPORTED_PREMIUM_META_POLICY",None,None
    if best:
        later = source.loc[(source.trading_day>=date(2026,1,8))&(source.trading_day<=date(2026,10,13))].copy()
        later = relabel(later,load_paths(later),ExitRule(**best["rule"]))
        evaluation,scores,metadata = block(pd.concat([best_data,later]),FINAL_FOLD,output)
        mask = eligible(evaluation,best["scope"],best["condition"])
        evaluation = evaluation.loc[mask]
        selected = select_capital(evaluation,scores[best["learner"]][mask],{"CE":policy["cutoff"],"PE":policy["cutoff"]},100000,3,5)
        baseline = select_capital(evaluation,np.ones(len(evaluation)),{"CE":0,"PE":0},100000,3,5)
        m = summarize(selected,100000,5)
        uncertainty = bootstrap_comparison(evaluation,selected,baseline)
        economic = (m["trades"]>=100 and m["coverage"]>=.99 and (m["profit_factor"] or 0)>=1.3
            and m["net_pnl"]>0 and m["stress_net_pnl"]>0 and m["max_drawdown"]<=10000
            and m["daily_loss_overshoots"]==0 and m["positive_months"]>=.6*m["active_months"]
            and uncertainty.get("mean_net_per_trade_95pct",[-1])[0]>0
            and uncertainty.get("paired_net_improvement_per_day_95pct",[-1])[0]>0)
        target = economic and (m["win_rate_pct"] or 0)>70
        status = ("HISTORICAL_TARGET_MET_FORWARD_REQUIRED" if target else
            "HISTORICAL_ECONOMIC_IMPROVEMENT_NOT_70_PERCENT" if economic else "REJECTED_LATER_HISTORICAL_AUDIT")
        audit = {"metrics":m,"unscored_same_scope":summarize(baseline,100000,5),"models":metadata}
        selected.to_json(output/"audit-trades.json",orient="records",date_format="iso",indent=2)
    if fingerprint(output/"policy.json")!=policy_hash or fingerprint(output/"specification.json")!=spec_hash:
        raise ValueError("Frozen research changed")
    supported = [dict(key=r["key"],learner=r["learner"],scope=r["scope"],condition=r["condition"],**point)
        for r in results for point in r["curve"] if point["metrics"]["trades"]>=100]
    diagnostic = max(supported,key=lambda row:row["metrics"]["win_rate_pct"] or 0) if supported else None
    report = {"status":status,"profiles":len(results),"specification_hash":spec_hash,"policy_hash":policy_hash,
        "activation_eligible":False,"selection":best,"highest_win_100plus_diagnostic_only":diagnostic,
        "historical_audit":audit,"uncertainty":uncertainty,"app_model_selection_changed":False}
    write_json(output/"report.json",report)
    write_json(output/"progress.json",{"status":"COMPLETE","result":status})
    lines = ["# Premium-outcome meta-model study","",f"Result: **{status}**.",
        "", "Two fixed native-exit targets; three regularised pooled classifiers and fixed consensus. All fitting/calibration weights normalise each day, not each overlapping row. No production changes.",
        "", "## Highest win rate with 100+ development trades (selection-biased diagnostic)","",json.dumps(diagnostic,indent=2),
        "", "## Selected later historical audit","",json.dumps(audit,indent=2) if audit else "No supported policy; no extra 2026 candidate audit.",
        "", "## Limitations","",*["- "+item for item in specification["limitations"]],
        "- Reserved October 14–23 excluded. A partial economic result is not a >70% claim. Forward quote validation remains mandatory."]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":status,"diagnostic":diagnostic["metrics"] if diagnostic else None,"audit":audit},indent=2),flush=True)
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    run(args.output)
