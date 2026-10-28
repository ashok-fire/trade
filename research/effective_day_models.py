"""Risk-aligned, effective-day regularisation and causal option-cost inputs."""
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from app.aligned_training import bootstrap_comparison,labelled
from app.risk import option_charges
from app.specialist_research import quality_logit
from app.training_contract import ALIGNED_FEATURES
from research.decision_day_specialists import raw_parts,select_policy
from research.exit_strategy_study import ExitRule,load_paths,relabel
from research.pooled_day_models import longer_fold
from research.precision_tuning import load_source
from research.profit_objective import fingerprint,write_json
from research.walk_forward_tuning import FOLDS,FINAL_FOLD,risk_eligible,select_capital,summarize

SOURCE = Path("models/research/walk-forward-capital-20261025")
LABEL_SOURCE = Path("models/research/premium-meta-models-20261025")
RULE = ExitRule(25,15,60)
EXTRA_FEATURES = ["decision_premium","lot_size","estimated_cost_pct","risk_utilization"]
VARIANTS = (("day_risk",False,None),("day_risk_cost",True,None),
            ("day_risk_cost_h126",True,126),("day_risk_cost_h252",True,252))


def enrich(frame):
    frame = frame.copy()
    costs = []
    for row in frame.itertuples():
        entry = Decimal(str(row.decision_premium))*Decimal("1.0005")
        charge = option_charges(entry,entry,int(row.lot_size),"GROWW",row.trading_day)
        costs.append(float(charge/(entry*int(row.lot_size))*100))
    frame["estimated_cost_pct"] = costs
    frame["risk_utilization"] = frame.decision_premium*1.0005*frame.lot_size*.25/3000
    return frame


def effective_weights(frame,half_life=None):
    weights = (1/frame.groupby("trading_day").trading_day.transform("size")).to_numpy(copy=True)
    if half_life:
        days = sorted(frame.trading_day.unique())
        age = frame.trading_day.map({day:len(days)-1-i for i,day in enumerate(days)}).to_numpy()
        weights *= np.power(.5,age/half_life)
    return weights  # Deliberately NOT rescaled to the number of candle rows.


def block(frame,fold,variant,output=None):
    name,cost_features,half_life = variant
    fit,cal,val = raw_parts(frame,longer_fold(fold))
    fit = labelled(fit.loc[risk_eligible(fit,100000,3,5)])
    cal = labelled(cal.loc[risk_eligible(cal,100000,3,5)])
    # Risk eligibility is causal; do not remove unknown exits from validation.
    val = val.loc[risk_eligible(val,100000,3,5)].copy()
    features = ALIGNED_FEATURES+EXTRA_FEATURES if cost_features else ALIGNED_FEATURES
    scores = np.full(len(val),np.nan)
    detail = {"variant":name,"fit_rows":len(fit),"fit_days":fit.trading_day.nunique(),
              "features":features,"calibration":{}}
    if len(fit)<500 or detail["fit_days"]<100 or fit.label.nunique()<2:
        detail["disabled"] = "insufficient fitting support"
        return val,scores,detail
    with threadpool_limits(limits=1):
        model = HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,learning_rate=.05,
            min_samples_leaf=20,l2_regularization=10,early_stopping=False,random_state=42)
        weights = effective_weights(fit,half_life)
        detail["fit_weight_sum"] = float(weights.sum())
        model.fit(fit[features],fit.label.astype(int),sample_weight=weights)
        calibrators = {}
        for side in ("CE","PE"):
            side_cal = cal.loc[cal.direction==side]
            mask = (val.direction==side).to_numpy()
            info = {"days":side_cal.trading_day.nunique(),"rows":len(side_cal),"enabled":False}
            if info["days"]>=40 and len(side_cal)>=100 and side_cal.label.nunique()==2:
                cal_weights = effective_weights(side_cal)
                calibrator = LogisticRegression(C=1.,max_iter=2000,random_state=42).fit(
                    quality_logit(model.predict_proba(side_cal[features])[:,1]),side_cal.label.astype(int),
                    sample_weight=cal_weights)
                slope = float(calibrator.coef_[0,0])
                info.update({"enabled":slope>0,"slope":slope,"weight_sum":float(cal_weights.sum())})
                calibrators[side] = calibrator
                if slope>0 and mask.any():
                    scores[mask] = calibrator.predict_proba(quality_logit(model.predict_proba(val.loc[mask,features])[:,1]))[:,1]
            detail["calibration"][side] = info
        if output:
            import joblib
            joblib.dump({"schema":"effective-day-research-v1","estimator":model,
                "calibrators":calibrators,"metadata":detail},output/"candidate.joblib")
    return val,scores,detail


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite frozen effective-day research")
    output.mkdir(parents=True)
    specification = {"schema":"effective-day-research-v1","code_hash":fingerprint(Path(__file__)),
        "source_report_hash":fingerprint(SOURCE/"report.json"),"label_source_report_hash":fingerprint(LABEL_SOURCE/"report.json"),
        "hypothesis":"Correct effective-day regularisation, risk-parity sampling and causal cost/lot inputs reduce calibration drift under changed option-contract economics",
        "variants":VARIANTS,"features":ALIGNED_FEATURES,"optional_inputs":EXTRA_FEATURES,
        "cost_inputs":"signal premium, historical contract lot size, estimated round-trip charges at unchanged signal premium as percent notional, and25% premium-stop risk /3000 budget; no future entry/exit prices",
        "weights":"each day sums to1 before optional126/252 observed-day half-life decay; never rescale to candle-row count",
        "risk_filter":"same causal whole-lot100000 capital/3% risk/5% daily eligibility in fit, calibration and validation; unknown validation outcomes retained",
        "calibration":"separate CE/PE positive-slope regularised Platt on prior6 complete months, each calibrated day total weight1, >=40 days/100 rows per side",
        "model":"fixed histogram boosting100 iterations/7leaves/lr.05/min_leaf20/L2=10/no random early-stopping; seed42",
        "rule":vars(RULE),"folds":[longer_fold(fold) for fold in FOLDS],"audit":longer_fold(FINAL_FOLD),
        "selection":"4 variants x CE/BOTH x9 cutoffs; unchanged target>70%/100trades and economic>=60% partial gates; freeze at most one candidate before audit",
        "audit_gates":"100 trades/>70% net wins/PF1.3/positive net and stress/DD<=10000/coverage99%/60% positive months/no daily overshoot/positive lower95 mean expectancy and paired improvement",
        "previously_inspected_history":True,"reserved_dates_excluded":["2026-10-14","2026-10-23"],
        "activation_eligible":False,"app_or_registry_changes":False}
    write_json(output/"specification.json",specification)
    spec_hash = fingerprint(output/"specification.json")
    source,source_report = load_source(SOURCE)
    labels = pd.read_pickle(LABEL_SOURCE/f"{RULE.key}-labels.pkl")
    dev = source.loc[labels.index].copy()
    for column in ("label_status","label","net_pnl","stress_net_pnl","net_return","exit_reason","exit_timestamp"):
        dev[column] = labels[column]
    dev = enrich(dev)
    results,best = [],None
    for variant in VARIANTS:
        frames,predictions,diagnostics = [],[],[]
        for fold in FOLDS:
            val,scores,detail = block(dev,fold,variant)
            val["fold"] = fold[0]
            frames.append(val)
            predictions.extend(scores)
            diagnostics.append({"fold":fold[0],**detail})
        validation,scores = pd.concat(frames),np.asarray(predictions)
        for side in ("BOTH","CE"):
            mask = np.ones(len(validation),dtype=bool) if side=="BOTH" else (validation.direction==side).to_numpy()
            selected,curve = select_policy(validation.loc[mask],scores[mask])
            record = {"variant":variant,"scope":side,"selected":selected,"curve":curve,"diagnostics":diagnostics}
            results.append(record)
            if selected and (best is None or (selected["target_pass"],selected["metrics"]["stress_net_pnl"])>
                    (best["selected"]["target_pass"],best["selected"]["metrics"]["stress_net_pnl"])):
                best = record
        write_json(output/f"{variant[0]}.json",results[-2:])
        print(f"Effective-day variant {variant[0]} complete",flush=True)
    policy = {"schema":"effective-day-research-v1","variant":best["variant"] if best else None,
        "scope":best["scope"] if best else None,"cutoff":best["selected"]["cutoff"] if best else None,
        "specification_hash":spec_hash,"dataset_hash":source_report["dataset_hash"],"activation_eligible":False}
    write_json(output/"policy.json",policy)
    policy_hash = fingerprint(output/"policy.json")
    status,audit,uncertainty = "REJECTED_NO_EFFECTIVE_DAY_POLICY",None,None
    if best:
        later = source.loc[(source.trading_day>=pd.Timestamp("2026-01-08").date())&(source.trading_day<=pd.Timestamp("2026-10-13").date())].copy()
        later = enrich(relabel(later,load_paths(later),RULE))
        evaluation,scores,detail = block(pd.concat([dev,later]),FINAL_FOLD,best["variant"],output)
        mask = np.ones(len(evaluation),dtype=bool) if best["scope"]=="BOTH" else (evaluation.direction==best["scope"]).to_numpy()
        evaluation,scores = evaluation.loc[mask],scores[mask]
        selected = select_capital(evaluation,scores,{"CE":policy["cutoff"],"PE":policy["cutoff"]},100000,3,5)
        baseline = select_capital(evaluation,np.ones(len(evaluation)),{"CE":0.,"PE":0.},100000,3,5)
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
        audit = {"metrics":m,"calibration":detail,"unscored":summarize(baseline,100000,5)}
        selected.to_json(output/"audit-trades.json",orient="records",date_format="iso",indent=2)
    if fingerprint(output/"specification.json")!=spec_hash or fingerprint(output/"policy.json")!=policy_hash:
        raise ValueError("Frozen study changed")
    adequate = [dict(variant=r["variant"],scope=r["scope"],**point) for r in results for point in r["curve"] if point["metrics"]["trades"]>=100]
    diagnostic = max(adequate,key=lambda row:row["metrics"]["win_rate_pct"] or 0) if adequate else None
    report = {"status":status,"specification_hash":spec_hash,"policy_hash":policy_hash,"activation_eligible":False,
        "selection":best,"highest_win_100plus_diagnostic_only":diagnostic,"audit":audit,"uncertainty":uncertainty,
        "app_model_selection_changed":False}
    write_json(output/"report.json",report)
    (output/"README.md").write_text("# Effective-day and causal-cost model study\n\nResult: **"+status+"**.\n\n"+
        "No application settings or live model changed. Native option exits, causal risk filters and day-normalised regularisation; historical selection remains biased, and forward quote validation is required. Reserved October14–23 excluded.\n\n"+
        "## Highest win rate with100+ development trades (diagnostic only)\n\n"+json.dumps(diagnostic,indent=2)+
        "\n\n## Later historical audit\n\n"+json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"status":status,"diagnostic":diagnostic["metrics"] if diagnostic else None,"audit":audit},indent=2),flush=True)
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    run(args.output)
