"""Pooled CE/PE day-weighted learning with longer chronological calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from app.aligned_training import bootstrap_comparison, day_weights, labelled
from app.specialist_research import quality_logit
from app.training_contract import ALIGNED_FEATURES
from research.decision_day_specialists import make_estimator, raw_parts, select_policy
from research.precision_tuning import load_source
from research.profit_objective import fingerprint,write_json
from research.walk_forward_tuning import FOLDS,FINAL_FOLD,risk_eligible,select_capital,summarize

SOURCE = Path("models/research/walk-forward-capital-20261025")
LEARNERS = ("LOGISTIC","HIST_GRADIENT","CATBOOST")
SCOPES = ("BOTH","CE","PE")


def longer_fold(fold):
    start = pd.Timestamp(fold[3]).replace(day=1)
    return (fold[0],(start-pd.DateOffset(months=6)).date().isoformat(),
        (start-pd.Timedelta(days=1)).date().isoformat(),fold[3],fold[4])


def first_side_decisions(frame):
    eligible = frame.loc[risk_eligible(frame,100000,3,5)]
    return eligible.sort_values("timestamp",kind="stable").drop_duplicates(["trading_day","direction"])


def consensus(scores):
    matrix = np.stack([scores[name] for name in LEARNERS],axis=1)
    count = np.isfinite(matrix).sum(axis=1)
    result = np.full(len(matrix),np.nan)
    valid = count>=2
    result[valid] = np.nansum(matrix[valid],axis=1)/count[valid]
    return result


def block(frame,fold,output=None):
    raw_fit,raw_cal,raw_val = raw_parts(frame,longer_fold(fold))
    fit,cal = labelled(first_side_decisions(raw_fit)),labelled(first_side_decisions(raw_cal))
    val = first_side_decisions(raw_val).copy()
    scores,metadata = {},{}
    for name in LEARNERS:
        scores[name] = np.full(len(val),np.nan)
        detail = {"fit_rows":len(fit),"fit_days":fit.trading_day.nunique(),
            "calibration_rows":len(cal),"calibration_days":cal.trading_day.nunique(),"enabled":False}
        if len(fit)>=100 and len(cal)>=40 and fit.label.nunique()==2 and cal.label.nunique()==2:
            with threadpool_limits(limits=1):
                model = make_estimator(name)
                fit_weights,cal_weights = day_weights(fit),day_weights(cal)
                if name=="LOGISTIC":
                    # Scale only the fitting population; pass day weights to the
                    # classifier, never compute preprocessing on later blocks.
                    model.fit(fit[ALIGNED_FEATURES],fit.label.astype(int),
                              logisticregression__sample_weight=fit_weights)
                else:
                    model.fit(fit[ALIGNED_FEATURES],fit.label.astype(int),sample_weight=fit_weights)
                calibration = LogisticRegression(C=1.,random_state=42,max_iter=2000).fit(
                    quality_logit(model.predict_proba(cal[ALIGNED_FEATURES])[:,1]),
                    cal.label.astype(int),sample_weight=cal_weights)
                slope = float(calibration.coef_[0,0])
                detail.update({"enabled":slope>0,"slope":slope,"intercept":float(calibration.intercept_[0])})
                if slope>0 and len(val):
                    scores[name] = calibration.predict_proba(quality_logit(model.predict_proba(val[ALIGNED_FEATURES])[:,1]))[:,1]
                if output:
                    import joblib
                    joblib.dump({"estimator":model,"calibration":calibration,"features":ALIGNED_FEATURES,
                        "schema":"pooled-day-research-v1","enabled":slope>0},output/f"{name.lower()}.joblib")
        metadata[name] = detail
    scores["CONSENSUS"] = consensus(scores)
    return val,scores,metadata


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen pooled study")
    output.mkdir(parents=True)
    specification = {"schema":"pooled-day-research-v1","code_hash":fingerprint(Path(__file__)),
        "source_report_hash":fingerprint(SOURCE/"report.json"),"hypothesis":"Pooling first CE/PE opportunities and six-month calibration reduces sparse-specialist instability; cross-model consensus may reduce estimator-specific noise",
        "features":ALIGNED_FEATURES,"learning_unit":"first risk-eligible observation per day per direction, then remove unknown fit/cal labels; equal total day weight; at most two fitting rows/day, not assumed independent",
        "calibration":"six preceding complete calendar months, positive-slope Platt; chronological day purges; min100 fit/40 calibration rows; same regularised estimators as decision-day study",
        "folds":[longer_fold(fold) for fold in FOLDS],"audit":longer_fold(FINAL_FOLD),
        "learners":[*LEARNERS,"CONSENSUS"],"consensus":"equal average when at least two positively calibrated models available; no fitted ensemble weights",
        "scopes":SCOPES,"selection":"12 fixed learner/scope profiles, nine cutoffs .40-.80 step .05; >70% target track, separately labelled >=60% partial economic track; same risk/profit/stability/neighbour support gates as decision-day study",
        "capital":100000,"risk_per_trade_pct":3,"daily_loss_limit_pct":5,
        "trading":"at most one accepted trade/day; each side contributes only its first predetermined opportunity; no later same-side retries after a low score",
        "labels":"unchanged frozen ATR-exit net wins; no new data or exit selection",
        "audit_gates":"100+ trades, >70% net wins for goal; positive net/double-slip, PF1.3, DD<=10000, coverage99%, >=60% positive months, no daily overshoot, positive lower95 mean expectancy and paired daily improvement versus unscored same-scope policy",
        "previously_inspected_history":True,"reserved_window_excluded":["2026-10-14","2026-10-23"],
        "activation_eligible":False,"runtime_compatible":False,"app_or_registry_writes":False}
    write_json(output/"specification.json",specification)
    spec_hash = fingerprint(output/"specification.json")
    source,source_report = load_source(SOURCE)
    dev = source.loc[source.trading_day<=pd.Timestamp("2025-12-31").date()].copy()
    frames,predictions,diagnostics = [],{name:[] for name in (*LEARNERS,"CONSENSUS")},[]
    for fold in FOLDS:
        val,scores,metadata = block(dev,fold)
        val["fold"] = fold[0]
        frames.append(val)
        for name in predictions:
            predictions[name].extend(scores[name])
        diagnostics.append({"fold":fold[0],"models":metadata})
        print(f"Pooled models: {fold[0]} complete; enabled={[k for k,v in metadata.items() if v['enabled']]}",flush=True)
        write_json(output/"progress.json",{"status":"FITTING","completed_windows":len(frames),"total":len(FOLDS)})
    validation = pd.concat(frames)
    results,best = [],None
    for name in predictions:
        for side in SCOPES:
            mask = np.ones(len(validation),dtype=bool) if side=="BOTH" else (validation.direction==side).to_numpy()
            chosen,curve = select_policy(validation.loc[mask],np.asarray(predictions[name])[mask])
            record = {"learner":name,"scope":side,"selected":chosen,"curve":curve}
            results.append(record)
            if chosen and (best is None or (chosen["target_pass"],chosen["metrics"]["stress_net_pnl"])>
                    (best["selected"]["target_pass"],best["selected"]["metrics"]["stress_net_pnl"])):
                best = record
    write_json(output/"development.json",{"profiles":results,"diagnostics":diagnostics})
    policy = {"schema":"pooled-day-research-v1","learner":best["learner"] if best else None,
        "scope":best["scope"] if best else None,"cutoff":best["selected"]["cutoff"] if best else None,
        "development_target_passed":bool(best and best["selected"]["target_pass"]),
        "specification_hash":spec_hash,"dataset_hash":source_report["dataset_hash"],"activation_eligible":False}
    write_json(output/"policy.json",policy)
    policy_hash = fingerprint(output/"policy.json")
    status,audit,uncertainty = "REJECTED_NO_SUPPORTED_POOLED_POLICY",None,None
    if best:
        evaluation,scores,metadata = block(source,FINAL_FOLD,output)
        if best["scope"]!="BOTH":
            mask = (evaluation.direction==best["scope"]).to_numpy()
            evaluation = evaluation.loc[mask]
            predictions_final = scores[best["learner"]][mask]
        else:
            predictions_final = scores[best["learner"]]
        chosen = select_capital(evaluation,predictions_final,{"CE":policy["cutoff"],"PE":policy["cutoff"]},100000,3,5)
        baseline = select_capital(evaluation,np.ones(len(evaluation)),{"CE":0,"PE":0},100000,3,5)
        m = summarize(chosen,100000,5)
        uncertainty = bootstrap_comparison(evaluation,chosen,baseline)
        economic = (m["trades"]>=100 and m["coverage"]>=.99 and (m["profit_factor"] or 0)>=1.3
            and m["net_pnl"]>0 and m["stress_net_pnl"]>0 and m["max_drawdown"]<=10000
            and m["daily_loss_overshoots"]==0 and m["positive_months"]>=.6*m["active_months"]
            and uncertainty.get("mean_net_per_trade_95pct",[-1])[0]>0
            and uncertainty.get("paired_net_improvement_per_day_95pct",[-1])[0]>0)
        target = economic and (m["win_rate_pct"] or 0)>70
        status = ("HISTORICAL_TARGET_MET_FORWARD_REQUIRED" if target else
            "HISTORICAL_ECONOMIC_IMPROVEMENT_NOT_70_PERCENT" if economic else "REJECTED_LATER_HISTORICAL_AUDIT")
        audit = {"metrics":m,"unscored_baseline":summarize(baseline,100000,5),"model_details":metadata}
        chosen.to_json(output/"audit-trades.json",orient="records",date_format="iso",indent=2)
    if fingerprint(output/"policy.json")!=policy_hash or fingerprint(output/"specification.json")!=spec_hash:
        raise ValueError("Frozen selection changed")
    adequate = [dict(learner=r["learner"],scope=r["scope"],**point) for r in results for point in r["curve"] if point["metrics"]["trades"]>=100]
    diagnostic = max(adequate,key=lambda row:row["metrics"]["win_rate_pct"] or 0) if adequate else None
    report = {"status":status,"profiles":len(results),"specification_hash":spec_hash,"policy_hash":policy_hash,
        "activation_eligible":False,"selection":best,"highest_win_100plus_diagnostic_only":diagnostic,
        "historical_audit":audit,"uncertainty":uncertainty,"app_model_selection_changed":False}
    write_json(output/"report.json",report)
    write_json(output/"progress.json",{"status":"COMPLETE","result":status})
    lines = ["# Pooled day-weighted model study","",f"Result: **{status}**.",
        "", "First CE/PE candidate per day, equal total day weights, six-month calibration; logistic regression, shallow histogram boosting, shallow CatBoost and fixed equal consensus. No production changes.",
        "", "## Highest win rate with 100+ development trades (selection-biased diagnostic)","",json.dumps(diagnostic,indent=2),
        "", "## Selected later historical audit","",json.dumps(audit,indent=2) if audit else "No supported policy; no extra 2026 candidate audit.",
        "", "Reserved October 14–23 excluded. Previously inspected 2026 is not fresh evidence. Different first-candidate contract is not runtime-compatible. A separately labelled economic track never counts as reaching >70%. Full portfolio/quote-based forward validation remains required."]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":status,"diagnostic":diagnostic["metrics"] if diagnostic else None,"audit":audit},indent=2),flush=True)
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    run(args.output)
