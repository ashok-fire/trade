"""Bounded >70% realised-win research. Never changes application model selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from app.aligned_training import PARAMETERS, bootstrap_comparison
from app.strategy import RegimeModel
from app.training_contract import ALIGNED_FEATURES
from research.profit_objective import fingerprint, write_json
from research.walk_forward_tuning import FOLDS, FINAL_FOLD, fit_scores, select_capital, summarize

GRID = tuple({**PARAMETERS,"depth":depth,"l2_leaf_reg":l2,"iterations":500,"learning_rate":.02}
             for depth,l2 in product((2,3),(30,100)))
CUTOFFS = tuple(round(.40+i*.025,3) for i in range(19))


def qualifies(metrics, fold_metrics, neighbour=False, require_folds=True):
    """A high displayed win rate with a tiny sample is not an acceptable edge."""
    if not (metrics["trades"] >= (60 if neighbour else 100)
            and (metrics["win_rate_pct"] or 0) > (65 if neighbour else 70)
            and metrics["coverage"] >= .99
            and (metrics["profit_factor"] or 0) >= (1.1 if neighbour else 1.3)
            and metrics["stress_net_pnl"] > 0 and metrics["net_pnl"] > 0
            and metrics["max_drawdown"] <= 10000 and not metrics["daily_loss_overshoots"]):
        return False
    return neighbour or ((not require_folds or
        sum(row["trades"] >= 10 and row["stress_net_pnl"] > 0 for row in fold_metrics) >= 4)
        and metrics["positive_months"] >= .6*metrics["active_months"])


def choose(frame, scores):
    curve = []
    for cutoff in CUTOFFS:
        selections, folds = [], []
        for fold in FOLDS:
            mask = (frame.fold == fold[0]).to_numpy()
            selected = select_capital(frame.loc[mask],np.asarray(scores)[mask],
                {"CE":cutoff,"PE":cutoff},100000,3,5)
            selections.append(selected)
            folds.append({"fold":fold[0],**summarize(selected,100000,5)})
        metrics = summarize(pd.concat(selections),100000,5)
        curve.append({"cutoff":cutoff,**metrics,"folds":folds,"passes":qualifies(metrics,folds)})
    supported = [row for i,row in enumerate(curve) if 0 < i < len(curve)-1 and row["passes"]
        and all(qualifies(curve[j],[],True) for j in (i-1,i+1))]
    winner = max(supported,key=lambda row:(row["stress_net_pnl"],-row["max_drawdown"])) if supported else None
    return winner,curve


def load_source(source):
    """Only the known locally generated prior-study cache is used, never remote pickle."""
    source = Path(source)
    report = json.loads((source/"report.json").read_text())
    for filename,key in (("specification.json","specification_hash"),("policy.json","policy_hash")):
        if fingerprint(source/filename) != report[key]:
            raise ValueError("Source study provenance changed")
    frame = pd.read_pickle(source/"opportunities.pkl")
    actual = hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).to_numpy().tobytes()).hexdigest()
    if actual != report["dataset_hash"]:
        raise ValueError("Source dataset fingerprint changed")
    if str(frame.trading_day.max()) != "2026-10-13" or str(frame.trading_day.min()) < "2023-01-01":
        raise ValueError("Source study contains dates outside the declared window")
    return frame,report


def run(output,source=Path("models/research/walk-forward-capital-20261025")):
    output,source = Path(output),Path(source)
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen precision study")
    output.mkdir(parents=True)
    specification = {"schema":"precision-research-v1","goal":"realised net win rate strictly greater than 70%, not probability cutoff",
        "source":str(source),"source_report_hash":fingerprint(source/"report.json"),
        "code_hash":fingerprint(Path(__file__)),"features":ALIGNED_FEATURES,
        "grid":GRID,"cutoffs":CUTOFFS,"folds":FOLDS,"historical_audit":FINAL_FOLD,
        "capital":100000,"risk_per_trade_pct":3,"daily_loss_limit_pct":5,
        "unchanged":"ORB CE/PE, 40 features, causal option selection, exits, fees and slippage, one-lot capital replay",
        "gates":"strictly >70% net wins, >=100 complete trades, coverage>=99%, PF>=1.3, net and double-slippage P&L>0, DD<=10000, no daily overshoot, 4/6 stressed-positive windows with >=10 trades, >=60% positive active months; both neighbours >65% wins/60 trades/PF1.1/stress positive",
        "audit_gates":"same >70%/100-trade/profit/PF/coverage/DD/month gates plus positive lower 95% block-bootstrap mean expectancy and paired improvement over frozen classifier; forward validation mandatory regardless",
        "limits":"four fixed configurations, nineteen common thresholds; no retry, no fallback, no changing gates after outcomes",
        "future_runs":"No automatic repeated search on this unchanged dataset; new evidence or a distinct preregistered hypothesis required",
        "previously_inspected_history":True,"reserved_dates_excluded":["2026-10-14","2026-10-23"],
        "activation_eligible":False,"registry_or_order_writes":False}
    write_json(output/"specification.json",specification)
    spec_hash = fingerprint(output/"specification.json")
    frame,source_report = load_source(source)
    results,best = [],None
    for index,parameters in enumerate(GRID):
        frames,predictions,calibrations = [],[],[]
        for fold in FOLDS:
            validation,scores,calibration,counts = fit_scores(frame,fold,parameters)
            validation = validation.copy()
            validation["fold"] = fold[0]
            frames.append(validation)
            predictions.extend(scores)
            calibrations.append({"fold":fold[0],"calibration":calibration,"counts":counts})
            print(f"Precision candidate {index+1}/{len(GRID)} depth {parameters['depth']} L2 {parameters['l2_leaf_reg']}: {fold[0]} complete",flush=True)
        validation = pd.concat(frames)
        winner,curve = choose(validation,np.asarray(predictions))
        result = {"index":index,"parameters":parameters,"winner":winner,"curve":curve,"calibrations":calibrations}
        write_json(output/f"candidate-{index}.json",result)
        results.append(result)
        if winner and (best is None or winner["stress_net_pnl"] > best["winner"]["stress_net_pnl"]):
            best = result
        write_json(output/"progress.json",{"status":"RUNNING","completed":len(results),"total":len(GRID)})
    policy = {"schema":"precision-research-v1","specification_hash":spec_hash,
        "dataset_hash":source_report["dataset_hash"],"selected_index":best["index"] if best else None,
        "parameters":best["parameters"] if best else None,"cutoff":best["winner"]["cutoff"] if best else None,
        "activation_eligible":False}
    write_json(output/"policy.json",policy)
    policy_hash = fingerprint(output/"policy.json")
    status,audit,uncertainty = "REJECTED_NO_SUPPORTED_70_PERCENT_POLICY",None,None
    if best:
        evaluation,scores,calibration,counts = fit_scores(frame,FINAL_FOLD,best["parameters"],output)
        selected = select_capital(evaluation,scores,{"CE":policy["cutoff"],"PE":policy["cutoff"]},100000,3,5)
        audit = summarize(selected,100000,5)
        control = RegimeModel(Path("models/research/decision-aligned-orb-20261024/candidate.json"))
        if not control.available:
            raise ValueError("Frozen classifier control unavailable")
        control_selected = select_capital(evaluation,control.quality_probabilities(evaluation),
            control.metadata["direction_thresholds"],100000,3,5)
        uncertainty = bootstrap_comparison(evaluation,selected,control_selected)
        passed = (qualifies(audit,[],require_folds=False)
            and uncertainty.get("mean_net_per_trade_95pct",[-1])[0] > 0
            and uncertainty.get("paired_net_improvement_per_day_95pct",[-1])[0] > 0)
        status = "HISTORICAL_TARGET_MET_FORWARD_REQUIRED" if passed else "REJECTED_LATER_HISTORICAL_AUDIT"
        write_json(output/"calibration.json",calibration)
        selected.to_json(output/"audit-trades.json",orient="records",date_format="iso",indent=2)
    if fingerprint(output/"specification.json") != spec_hash or fingerprint(output/"policy.json") != policy_hash:
        raise ValueError("Frozen specification or policy changed")
    supported = [dict(candidate=row["index"],**point) for row in results for point in row["curve"] if point["trades"] >= 100]
    diagnostic = max(supported,key=lambda point:point["win_rate_pct"] or 0) if supported else None
    report = {"status":status,"completed_candidates":len(results),"activation_eligible":False,
        "specification_hash":spec_hash,"policy_hash":policy_hash,"source_report_hash":specification["source_report_hash"],
        "policy":policy,"selected_validation":best["winner"] if best else None,
        "highest_win_with_100_trades_diagnostic_only":diagnostic,
        "historical_audit":audit,"uncertainty":uncertainty,"app_model_selection_changed":False,
        "next_step":"Collect fresh forward evidence; do not run unbounded retries against unchanged history"}
    write_json(output/"report.json",report)
    write_json(output/"progress.json",{"status":"COMPLETE","completed":len(results),"result":status})
    lines = ["# >70% realised-win research", "",f"Result: **{status}**.",
        "", "Four fixed stronger-regularisation CatBoost configurations, six chronological validation windows, nineteen thresholds; no new application features.",
        "", "Capital ₹100,000; 3% per-trade gross risk budget; 5% daily entry-gate limit. Existing exits/costs unchanged. One historical lot per entry; no compounding.",
        "", "Probability cutoff is not realised win rate. Selection validation is biased by model/threshold search, not a fresh test. 2026 history has also been inspected previously. Reserved October 14–23 data excluded."]
    if diagnostic:
        lines += ["",f"Highest diagnostic win rate among policies with >=100 validation trades: {diagnostic['win_rate_pct']}% across {diagnostic['trades']} trades (candidate {diagnostic['candidate']}, cutoff {diagnostic['cutoff']}). This is not a qualifying policy unless every preregistered gate passes."]
    lines += ["", "## Later historical audit", "", json.dumps(audit,indent=2) if audit else "Not performed: no >70% supported policy passed validation. No fallback and no extra 2026 candidate testing.",
        "", "No application model selection, registry entry, risk setting, or order was changed by this command. Fresh executable-quote Shadow evidence is required before any deployment decision.",
        "", "Do not repeat an unchanged failed search. Ongoing follow-up should check new evidence and service health, and run another bounded experiment only for a concrete new hypothesis or meaningful new labelled sessions."]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    run(args.output)
