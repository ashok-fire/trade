"""One-decision-per-day specialist study; research only, no runtime writes."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.aligned_training import bootstrap_comparison, labelled
from app.specialist_research import quality_logit
from app.training_contract import ALIGNED_FEATURES
from research.exit_strategy_study import foundation_pass, window_metrics
from research.precision_tuning import load_source, qualifies
from research.profit_objective import fingerprint, write_json
from research.walk_forward_tuning import FOLDS, FINAL_FOLD, risk_eligible, select_capital, summarize

SOURCE = Path("models/research/walk-forward-capital-20261025")
CLOCKS = {"EARLY":(575,660),"MID":(660,780),"LATE":(780,910)}
LEARNERS = ("LOGISTIC","HIST_GRADIENT","CATBOOST")
CUTOFFS = tuple(value/100 for value in range(40,81,5))


@dataclass(frozen=True)
class Profile:
    side: str
    clock: str
    volatility: str
    learner: str


PROFILES = tuple(Profile(*values) for values in product(("CE","PE"),CLOCKS,("ALL","LOW"),LEARNERS))


def raw_parts(frame,fold):
    """Do not drop unknown outcomes before choosing the first daily decision."""
    a,b,c,d = map(date.fromisoformat,fold[1:])
    if not a<=b<c<=d:
        raise ValueError("Fold periods must be chronological and disjoint")
    fit = frame.loc[frame.trading_day<a].copy()
    cal = frame.loc[(frame.trading_day>=a)&(frame.trading_day<=b)].copy()
    val = frame.loc[(frame.trading_day>=c)&(frame.trading_day<=d)].copy()
    for part in (fit,cal):
        if len(part):
            part.drop(part.index[part.trading_day==part.trading_day.max()],inplace=True)
    return fit,cal,val


def first_decisions(frame,profile,volatility_cutoff=None):
    stamp = pd.to_datetime(frame.timestamp,utc=True).dt.tz_convert("Asia/Kolkata")+pd.Timedelta(minutes=5)
    minute = stamp.dt.hour*60+stamp.dt.minute
    lower,upper = CLOCKS[profile.clock]
    mask = (frame.direction==profile.side)&(minute>=lower)&(minute<upper)
    mask &= risk_eligible(frame,100000,3,5)
    if profile.volatility=="LOW" and volatility_cutoff is not None:
        mask &= frame.atr_pct<=volatility_cutoff
    elif profile.volatility not in ("LOW","ALL"):
        raise ValueError("Unknown volatility condition")
    return frame.loc[mask].sort_values("timestamp",kind="stable").drop_duplicates("trading_day")


def make_estimator(name):
    if name=="LOGISTIC":
        return make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=2000,random_state=42))
    if name=="HIST_GRADIENT":
        return HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,learning_rate=.05,
            min_samples_leaf=20,l2_regularization=10,early_stopping=False,random_state=42)
    if name=="CATBOOST":
        return CatBoostClassifier(iterations=300,depth=2,learning_rate=.03,l2_leaf_reg=30,
            loss_function="Logloss",random_seed=42,thread_count=1,verbose=False,
            allow_writing_files=False,has_time=True)
    raise ValueError("Unknown estimator")


def block(frame,fold,profile):
    raw_fit,raw_cal,raw_val = raw_parts(frame,fold)
    # Fit-window median of one candidate/day; never use calibration or validation
    # distributions to define LOW volatility, or known outcomes to pick a row.
    first = first_decisions(raw_fit,profile)
    volatility_cutoff = float(first.atr_pct.median()) if profile.volatility=="LOW" and len(first) else None
    fit = labelled(first_decisions(raw_fit,profile,volatility_cutoff))
    cal = labelled(first_decisions(raw_cal,profile,volatility_cutoff))
    val = first_decisions(raw_val,profile,volatility_cutoff).copy()
    scores = np.full(len(val),np.nan)
    metadata = {"fit_days":len(fit),"calibration_days":len(cal),"validation_days":len(val),
                "volatility_cutoff_fit_only":volatility_cutoff,"enabled":False}
    if len(fit)<100 or len(cal)<20 or fit.label.nunique()<2 or cal.label.nunique()<2:
        metadata["reason"] = "insufficient first-decision fitting/calibration support"
        return val,scores,metadata
    with threadpool_limits(limits=1):
        estimator = make_estimator(profile.learner)
        estimator.fit(fit[ALIGNED_FEATURES],fit.label.astype(int))
        calibration = LogisticRegression(C=1.,max_iter=2000,random_state=42).fit(
            quality_logit(estimator.predict_proba(cal[ALIGNED_FEATURES])[:,1]),cal.label.astype(int))
        slope = float(calibration.coef_[0,0])
        metadata.update({"slope":slope,"intercept":float(calibration.intercept_[0]),"enabled":slope>0})
        if slope>0 and len(val):
            scores = calibration.predict_proba(quality_logit(estimator.predict_proba(val[ALIGNED_FEATURES])[:,1]))[:,1]
    return val,scores,metadata


def select_policy(frame,scores):
    curve = []
    for cutoff in CUTOFFS:
        m,windows,_ = window_metrics(frame,scores,cutoff)
        curve.append({"cutoff":cutoff,"metrics":m,"windows":windows,
            "target_pass":bool(qualifies(m,windows)),
            "economic_pass":bool(foundation_pass(m,windows) and (m["win_rate_pct"] or 0)>=60)})
    supported = []
    for index,row in enumerate(curve):
        if index in (0,len(curve)-1):
            continue
        neighbours = [curve[index-1]["metrics"],curve[index+1]["metrics"]]
        if all(m["trades"]>=60 and (m["profit_factor"] or 0)>=1.1 and m["stress_net_pnl"]>0
               and (m["win_rate_pct"] or 0)>=(65 if row["target_pass"] else 55) for m in neighbours):
            if row["target_pass"] or row["economic_pass"]:
                supported.append(row)
    # Target passage has priority. Economic passage is a separately declared
    # partial-improvement track, never relabelled as meeting the >70% goal.
    chosen = max(supported,key=lambda row:(row["target_pass"],row["metrics"]["stress_net_pnl"])) if supported else None
    return chosen,curve


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen decision-day study")
    output.mkdir(parents=True)
    specification = {"schema":"first-decision-specialists-v1","code_hash":fingerprint(Path(__file__)),
        "source_report_hash":fingerprint(SOURCE/"report.json"),
        "hypothesis":"One predetermined daily decision and clock/volatility-specific models improve sample independence and selective net win prediction over overlapping intraday training",
        "profiles":[asdict(profile) for profile in PROFILES],"clocks_ist_minutes":CLOCKS,
        "features":ALIGNED_FEATURES,"labels":"existing frozen underlying-ATR exits; no relabelling or new data",
        "first_decision":"first risk-eligible opportunity per day in the chosen boundary-clock interval; a low model score means skip that entire day, not choose a later outcome",
        "LOW_volatility":"fit-only median ATR/close on first eligible daily candidates; fixed throughout each calibration/validation block",
        "estimators":{"LOGISTIC":"fit-only StandardScaler + L2 C=.05, max_iter2000",
            "HIST_GRADIENT":"100 iterations, 7 leaves, lr.05, L2=10, min_leaf20, early_stopping=False",
            "CATBOOST":"300 iterations, depth2, lr.03, L2=30, has_time=True; seed42"},
        "calibration":"separate-day regularised Platt C1; positive slope required; min100 fit and20 calibration first decisions; no class balancing or random temporal splits",
        "folds":FOLDS,"final_audit":FINAL_FOLD,"cutoffs":CUTOFFS,
        "goal_gates":"strictly >70% realised net wins, >=100 trades, coverage>=99%, PF>=1.3, positive net/stress, DD<=10000, no daily overshoot, >=4/6 stressed-positive windows with10 trades, >=60% positive months; supported neighbours",
        "partial_improvement_track":"separate >=60% win economic candidate with same profitability/risk/stability gates; NEVER counts as >70%. Select at most one final candidate, prioritising goal passage then stressed net",
        "capital":100000,"risk_per_trade_pct":3,"daily_loss_limit_pct":5,
        "audit":"frozen chosen candidate only on previously inspected 2026; paired comparison with unscored same-clock first-decision strategy",
        "previously_inspected_history":True,"reserved_window_excluded":["2026-10-14","2026-10-23"],
        "limitations":["Many-profile development selection is biased and not fresh performance evidence",
            "Different first-decision contract is not compatible with current all-bars runtime",
            "Existing candle fills/costs and historical feature semantics require fresh quote-based validation",
            "Quarter ledgers reset; one historical lot per trade, not full compounded account sizing"],
        "activation_eligible":False,"application_or_registry_changes":False}
    write_json(output/"specification.json",specification)
    spec_hash = fingerprint(output/"specification.json")
    source,source_report = load_source(SOURCE)
    development = source.loc[source.trading_day<=date(2025,12,31)].copy()
    results,best = [],None
    for index,profile in enumerate(PROFILES):
        frames,predictions,diagnostics = [],[],[]
        for fold in FOLDS:
            val,scores,metadata = block(development,fold,profile)
            val["fold"] = fold[0]
            frames.append(val)
            predictions.extend(scores)
            diagnostics.append({"window":fold[0],**metadata})
        validation = pd.concat(frames)
        chosen,curve = select_policy(validation,np.asarray(predictions))
        record = {"index":index,"profile":asdict(profile),"selected":chosen,"curve":curve,"diagnostics":diagnostics}
        results.append(record)
        write_json(output/f"profile-{index:02d}.json",record)
        if chosen and (best is None or (chosen["target_pass"],chosen["metrics"]["stress_net_pnl"]) >
                (best["selected"]["target_pass"],best["selected"]["metrics"]["stress_net_pnl"])):
            best = record
        write_json(output/"progress.json",{"status":"FITTING","completed_profiles":len(results),"total":len(PROFILES)})
        print(f"Decision-day profile {index+1}/{len(PROFILES)} {profile}: target={bool(chosen and chosen['target_pass'])}; economic={bool(chosen)}",flush=True)
    policy = {"schema":"first-decision-specialists-v1","profile":best["profile"] if best else None,
        "cutoff":best["selected"]["cutoff"] if best else None,
        "development_target_passed":bool(best and best["selected"]["target_pass"]),
        "dataset_hash":source_report["dataset_hash"],"specification_hash":spec_hash,"activation_eligible":False}
    write_json(output/"policy.json",policy)
    policy_hash = fingerprint(output/"policy.json")
    audit,uncertainty = None,None
    status = "REJECTED_NO_SUPPORTED_DAILY_SPECIALIST"
    if best:
        evaluation,scores,metadata = block(source,FINAL_FOLD,Profile(**best["profile"]))
        chosen = select_capital(evaluation,scores,{"CE":policy["cutoff"],"PE":policy["cutoff"]},100000,3,5)
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
        audit = {"metrics":m,"unscored_baseline":summarize(baseline,100000,5),"calibration":metadata}
        chosen.to_json(output/"audit-trades.json",orient="records",date_format="iso",indent=2)
    if fingerprint(output/"specification.json")!=spec_hash or fingerprint(output/"policy.json")!=policy_hash:
        raise ValueError("Frozen study changed after selection")
    adequate = [dict(profile=row["profile"],**point) for row in results for point in row["curve"] if point["metrics"]["trades"]>=100]
    diagnostic = max(adequate,key=lambda row:row["metrics"]["win_rate_pct"] or 0) if adequate else None
    report = {"status":status,"profiles_fitted":len(results),"activation_eligible":False,
        "specification_hash":spec_hash,"policy_hash":policy_hash,"selection":best,
        "highest_win_100plus_trades_diagnostic_only":diagnostic,"historical_audit":audit,
        "uncertainty":uncertainty,"app_model_selection_changed":False,"limitations":specification["limitations"]}
    write_json(output/"report.json",report)
    write_json(output/"progress.json",{"status":"COMPLETE","result":status})
    lines = ["# First-decision/day specialist study", "",f"Result: **{status}**.",
        "",f"Fitted {len(results)} fixed profiles across six chronological development windows; three estimator families, CE/PE, three clocks, two volatility scopes.",
        "", "A low-scored first opportunity means no trade that day. This is a research contract, not the existing runtime all-bars contract. No live model or settings changed."]
    if diagnostic:
        lines += ["", "## Highest displayed win rate with 100+ trades (selection-biased diagnostic only)","",json.dumps(diagnostic,indent=2)]
    lines += ["", "## Selected later historical audit", "",json.dumps(audit,indent=2) if audit else "No supported target or separately declared economic candidate; no extra 2026 candidate audit.",
        "", "## Limitations", "",*["- "+item for item in specification["limitations"]],
        "- Reserved October 14–23 excluded. More searches cannot turn inspected history into fresh evidence."]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":status,"profiles_fitted":len(results),
        "diagnostic":diagnostic["metrics"] if diagnostic else None,"historical_audit":audit},indent=2),flush=True)
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    run(args.output)
