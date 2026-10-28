"""Bounded, capital-aware offline search. No broker or model-registry writes.

Walk-forward validation scores are used for selection, NOT advertised as an
unbiased final test. The later 2026 audit is already exposed historical data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from decimal import Decimal
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from app.aligned_training import PARAMETERS, bootstrap_comparison, day_weights, labelled
from app.config import config
from app.risk import option_charges
from app.specialist_research import quality_logit
from app.strategy import RegimeModel
from app.training_contract import ALIGNED_FEATURES, contract_spec
from app.v4_training import build_option_quality_samples
from research.profit_objective import fingerprint, write_json

GRID = tuple({**PARAMETERS, "depth": depth, "l2_leaf_reg": l2}
             for depth, l2 in product((2, 3, 4), (10, 30)))
CUTOFFS = (.40, .44, .48, .52, .56, .60, .64, .68, .72)
FOLDS = (
    ("2024-Q2", "2024-01-01", "2024-03-31", "2024-04-01", "2024-06-30"),
    ("2024-Q3", "2024-04-01", "2024-06-30", "2024-07-01", "2024-09-30"),
    ("2024-Q4", "2024-07-01", "2024-09-30", "2024-10-01", "2024-12-31"),
    ("2025-Q2", "2025-01-01", "2025-03-31", "2025-04-01", "2025-06-30"),
    ("2025-Q3", "2025-04-01", "2025-06-30", "2025-07-01", "2025-09-30"),
    ("2025-Q4", "2025-07-01", "2025-09-30", "2025-10-01", "2025-12-31"),
)
FINAL_FOLD = ("2026-audit", "2025-10-01", "2025-12-31", "2026-01-08", "2026-10-13")


def split_fold(frame, fold):
    """Group whole trading days and purge an observed session at each boundary."""
    _, cal_start, cal_end, val_start, val_end = fold
    a, b, c, d = map(date.fromisoformat, (cal_start, cal_end, val_start, val_end))
    if not a <= b < c <= d:
        raise ValueError("Fold periods must be chronological and disjoint")
    fit = frame.loc[frame.trading_day < a].copy()
    cal = frame.loc[(frame.trading_day >= a) & (frame.trading_day <= b)].copy()
    validation = frame.loc[(frame.trading_day >= c) & (frame.trading_day <= d)].copy()
    for part in (fit, cal):
        if not part.empty:
            part.drop(part.index[part.trading_day == part.trading_day.max()], inplace=True)
    return labelled(fit), labelled(cal), validation


def fit_scores(frame, fold, parameters, output=None):
    fitting, calibration, validation = split_fold(frame, fold)
    predictions = np.full(len(validation), np.nan)
    calibrations, counts = {}, {}
    for side in ("CE", "PE"):
        fit = fitting.loc[fitting.direction == side]
        cal = calibration.loc[calibration.direction == side]
        mask = (validation.direction == side).to_numpy()
        if len(fit) < 100 or len(cal) < 40 or fit.label.nunique() < 2 or cal.label.nunique() < 2:
            raise ValueError(f"Insufficient data for {fold[0]} {side}")
        estimator = CatBoostClassifier(**parameters)
        estimator.fit(fit[ALIGNED_FEATURES], fit.label.astype(int), sample_weight=day_weights(fit))
        calibrator = LogisticRegression(C=1., random_state=42).fit(
            quality_logit(estimator.predict_proba(cal[ALIGNED_FEATURES])[:, 1]),
            cal.label.astype(int), sample_weight=day_weights(cal))
        slope = float(calibrator.coef_[0, 0])
        calibrations[side] = {"intercept": float(calibrator.intercept_[0]), "slope": slope,
                              "enabled": slope > 0}
        counts[side] = {"fit_rows": len(fit), "fit_days": fit.trading_day.nunique(),
                       "calibration_rows": len(cal), "calibration_days": cal.trading_day.nunique()}
        if slope > 0 and mask.any():
            predictions[mask] = calibrator.predict_proba(quality_logit(
                estimator.predict_proba(validation.loc[mask, ALIGNED_FEATURES])[:, 1]))[:, 1]
        if output:
            estimator.save_model(str(output / f"{side.lower()}.cbm"))
    return validation, predictions, calibrations, counts


def risk_eligible(frame, capital, risk_pct, daily_pct, premium_stop_pct=25.):
    """Only signal-time fields are read; never inspect a future P&L or label."""
    if capital <= 0 or not 0 < risk_pct <= 3 or not 0 < daily_pct <= 5:
        raise ValueError("Capital/risk limits are outside the requested bounds")
    premium = frame.decision_premium.to_numpy(dtype=float) * 1.0005
    lots = frame.lot_size.to_numpy(dtype=float)
    return (np.isfinite(premium) & np.isfinite(lots) & (premium > 0) & (lots > 0)
            & (lots == np.floor(lots))
            & (premium * lots * premium_stop_pct / 100 <= capital * min(risk_pct, daily_pct) / 100)
            & (premium * lots <= capital))


def select_capital(frame, scores, cutoffs, capital, risk_pct, daily_pct):
    if len(frame) != len(scores):
        raise ValueError("Scores must align with opportunities")
    working = frame.copy()
    working["score"] = scores
    thresholds = working.direction.map(cutoffs).fillna(np.inf).to_numpy()
    eligible = risk_eligible(working, capital, risk_pct, daily_pct)
    working = working.loc[eligible & np.isfinite(scores) & (np.asarray(scores) >= thresholds)]
    equity, selected = capital, []
    # One historical lot and at most one trade/day; not percentage-based sizing.
    for _, day in working.sort_values("timestamp", kind="stable").groupby("trading_day", sort=True):
        for index, row in day.iterrows():
            entry = Decimal(str(row.decision_premium)) * Decimal("1.0005")
            reserve = float(option_charges(entry, entry, int(row.lot_size), "GROWW", row.trading_day))
            if float(entry) * row.lot_size + reserve > min(capital, equity):
                continue
            selected.append(index)
            # Unknown outcomes must not be silently zero-filled or allow a
            # later winning trade: stop this ledger and expose incomplete coverage.
            if row.label_status != "COMPLETE" or not np.isfinite(row.net_pnl):
                return frame.loc[selected].copy()
            equity += float(row.net_pnl)
            break
    return frame.loc[selected].copy()


def summarize(selected, capital, daily_pct):
    complete = labelled(selected).sort_values("timestamp", kind="stable")
    pnl = complete.net_pnl.to_numpy(dtype=float)
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0., equity])[1:]
    losses = -pnl[pnl < 0].sum()
    monthly = complete.groupby(pd.to_datetime(complete.timestamp, utc=True).dt.strftime("%Y-%m")).net_pnl.sum()
    return {"selected": len(selected), "trades": len(complete), "censored": len(selected)-len(complete),
            "coverage": len(complete)/len(selected) if len(selected) else 0.,
            "net_pnl": round(float(pnl.sum()), 2),
            "stress_net_pnl": round(float(complete.stress_net_pnl.sum()), 2),
            "win_rate_pct": round(float((pnl > 0).mean()*100), 2) if len(pnl) else None,
            "profit_factor": round(float(pnl[pnl > 0].sum()/losses), 3) if losses else None,
            "max_drawdown": round(float((peak-equity).max()), 2) if len(pnl) else 0.,
            "daily_loss_overshoots": int((pnl < -capital*daily_pct/100).sum()),
            "active_months": len(monthly), "positive_months": int((monthly > 0).sum()),
            "monthly": {key: round(float(value), 2) for key,value in monthly.items()},
            "ce_trades": int((complete.direction == "CE").sum()),
            "pe_trades": int((complete.direction == "PE").sum())}


def choose_policy(frame, scores, capital, risk_pct, daily_pct, folds=FOLDS):
    curve = []
    # Common cutoff bounds the search: do not independently optimise CE and PE.
    for cutoff in CUTOFFS:
        selected, metrics = [], []
        for fold in folds:
            mask = frame.fold == fold[0]
            trades = select_capital(frame.loc[mask], np.asarray(scores)[mask],
                {"CE": cutoff, "PE": cutoff}, capital, risk_pct, daily_pct)
            selected.append(trades)
            metrics.append({"fold": fold[0], **summarize(trades, capital, daily_pct)})
        all_selected = pd.concat(selected)
        result = summarize(all_selected, capital, daily_pct)
        passed = (result["trades"] >= 100 and result["coverage"] >= .99
            and (result["profit_factor"] or 0) >= 1.3 and result["stress_net_pnl"] > 0
            and result["max_drawdown"] <= capital*.10 and not result["daily_loss_overshoots"]
            and sum(item["trades"] >= 10 and item["stress_net_pnl"] > 0 for item in metrics) >= 4
            and result["positive_months"] >= .6*result["active_months"])
        curve.append({"cutoff": cutoff, **result, "folds": metrics, "passes": bool(passed)})
    # Neighbour support means not choosing a single isolated lucky threshold.
    accepted = [row for i,row in enumerate(curve) if 0 < i < len(curve)-1 and row["passes"]
        and all(curve[j]["trades"] >= 60 and (curve[j]["profit_factor"] or 0) >= 1.1
                and curve[j]["stress_net_pnl"] > 0 for j in (i-1,i+1))]
    winner = max(accepted, key=lambda row: (row["stress_net_pnl"], -row["max_drawdown"])) if accepted else None
    return winner, curve


def run(output, capital=100000., risk_pct=3., daily_pct=5.):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen tuning run")
    # Validate before writing or reading market history.
    risk_eligible(pd.DataFrame({"decision_premium": [], "lot_size": []}), capital, risk_pct, daily_pct)
    output.mkdir(parents=True)
    specification = {"schema": "capital-aware-walk-forward-v1", "capital": capital,
        "risk_per_trade_pct": risk_pct, "daily_loss_limit_pct": daily_pct,
        "features": ALIGNED_FEATURES, "contract": contract_spec(("ORB",)),
        "grid": GRID, "cutoffs": CUTOFFS, "folds": FOLDS, "final_fold": FINAL_FOLD,
        "selection": "6 fixed CatBoost settings x 9 shared thresholds, selection on concatenated forward validation only; positive-slope separate calibration; one observed session purged at each boundary",
        "gates": "100 trades, PF>=1.3, stress positive in >=4/6 folds with >=10 trades each, >=60% positive active months, DD<=10% capital, >=99% known outcomes, no observed daily-limit overshoot; neighbouring cutoffs >=60 trades/PF1.1/stress positive",
        "sizing": "one historical lot; signal premium plus 0.05% slippage; 25% premium-stop gross risk <=min(trade,daily) budget; premium debit plus estimated charges <=min(initial capital,current equity)",
        "capital_ledger": "each validation quarter resets to initial capital; fixed non-compounding budgets; final audit is one continuous ledger; incomplete selected outcome halts ledger",
        "limitations": ["Signal-close premium used for affordability; next-bar-open fill may gap",
            "Stop risk excludes charges and is not a guaranteed maximum loss",
            "No historical executable quotes, feature parity or full live portfolio evidence",
            "Validation data are used for selection, not unbiased performance estimation",
            "2026 audit is previously inspected history; not fresh validation"],
        "reserved_dates_not_used": ["2026-10-14", "2026-10-23"],
        "activation_eligible": False, "broker_or_model_registry_writes": False}
    write_json(output / "specification.json", specification)
    specification_hash = fingerprint(output / "specification.json")
    engine = create_engine(config.database_url)
    @event.listens_for(engine, "connect")
    def read_only(connection, _):
        connection.execute("PRAGMA query_only=ON")
    try:
        with Session(engine) as session:
            frame, coverage = build_option_quality_samples(session, provider="GROWW", code="NIFTY",
                start_date=date(2023,1,1), end_date=date(2026,10,13), setup_families=("ORB",),
                sampling_policy="all_eligible_bars", strike_policy="ATM_ITM1", setup_policy="first_confirmed",
                causal_opportunities=True, retain_unlabelled=True, required_features=ALIGNED_FEATURES,
                include_decision_context=True)
    finally:
        engine.dispose()
    frame = frame.sort_values(["timestamp", "setup_family"], kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No historical opportunities")
    dataset_hash = hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).to_numpy().tobytes()).hexdigest()
    frame.to_pickle(output / "opportunities.pkl")
    write_json(output / "dataset.json", {"hash": dataset_hash, "opportunities": len(frame),
        "coverage": coverage, "risk_eligible": int(risk_eligible(frame,capital,risk_pct,daily_pct).sum())})
    results, best = [], None
    for index, parameters in enumerate(GRID):
        frames, predictions, diagnostics = [], [], []
        for fold in FOLDS:
            validation, scores, calibrations, counts = fit_scores(frame, fold, parameters)
            validation = validation.copy()
            validation["fold"] = fold[0]
            frames.append(validation)
            predictions.extend(scores)
            diagnostics.append({"fold": fold[0], "calibration": calibrations, "samples": counts})
            print(f"Candidate {index+1}/{len(GRID)} depth={parameters['depth']} L2={parameters['l2_leaf_reg']}: {fold[0]} complete", flush=True)
        validation = pd.concat(frames)
        scores = np.asarray(predictions)
        winner, curve = choose_policy(validation,scores,capital,risk_pct,daily_pct)
        result = {"candidate": index, "parameters": parameters, "winner": winner,
                  "curve": curve, "diagnostics": diagnostics}
        results.append(result)
        write_json(output / f"candidate-{index}.json", result)
        if winner and (best is None or winner["stress_net_pnl"] > best["winner"]["stress_net_pnl"]):
            best = result
        write_json(output / "progress.json", {"status": "TUNING", "completed_candidates": len(results),
            "total_candidates": len(GRID), "qualifying_candidates": sum(row["winner"] is not None for row in results)})
    policy = {"schema": "capital-aware-walk-forward-v1", "activation_eligible": False,
        "selected_candidate": best["candidate"] if best else None,
        "parameters": best["parameters"] if best else None,
        "cutoff": best["winner"]["cutoff"] if best else None,
        "capital": capital, "risk_per_trade_pct": risk_pct, "daily_loss_limit_pct": daily_pct,
        "specification_hash": specification_hash, "dataset_hash": dataset_hash,
        "status": "SELECTED_FOR_HISTORICAL_AUDIT" if best else "REJECTED_NO_QUALIFYING_CANDIDATE"}
    # Freeze selection BEFORE fitting or scoring the previously exposed 2026 audit.
    write_json(output / "policy.json", policy)
    policy_hash = fingerprint(output / "policy.json")
    evaluation = split_fold(frame, FINAL_FOLD)[2]
    baseline_scores = np.ones(len(evaluation))
    baseline_selected = select_capital(evaluation, baseline_scores, {"CE":0.,"PE":0.},capital,risk_pct,daily_pct)
    control = RegimeModel(Path("models/research/decision-aligned-orb-20261024/candidate.json"))
    if not control.available:
        raise ValueError("Frozen classifier control unavailable")
    control_selected = select_capital(evaluation, control.quality_probabilities(evaluation),
        control.metadata["direction_thresholds"],capital,risk_pct,daily_pct)
    audit, uncertainty = None, None
    if best:
        evaluation, scores, calibration, counts = fit_scores(frame, FINAL_FOLD,best["parameters"],output)
        selected = select_capital(evaluation,scores,{"CE":policy["cutoff"],"PE":policy["cutoff"]},capital,risk_pct,daily_pct)
        audit = {"metrics": summarize(selected,capital,daily_pct), "calibration": calibration, "samples": counts}
        uncertainty = bootstrap_comparison(evaluation,selected,control_selected)
        write_json(output / "calibration.json",calibration)
        selected.to_json(output / "audit-trades.json",orient="records",date_format="iso",indent=2)
    if fingerprint(output/"policy.json") != policy_hash or fingerprint(output/"specification.json") != specification_hash:
        raise ValueError("Frozen specification or selection changed during evaluation")
    report = {"status": policy["status"], "activation_eligible": False,
        "capital":capital,"risk_per_trade_pct":risk_pct,"daily_loss_limit_pct":daily_pct,
        "specification_hash":specification_hash,"policy_hash":policy_hash,"dataset_hash":dataset_hash,
        "opportunities":len(frame),"completed_candidates":len(results),
        "qualifying_candidates":sum(row["winner"] is not None for row in results),
        "selection":policy,"candidate_results":results,"historical_audit":audit,"uncertainty":uncertainty,
        "same_capital_rule_baseline":summarize(baseline_selected,capital,daily_pct),
        "same_capital_frozen_classifier":summarize(control_selected,capital,daily_pct),
        "limitations":specification["limitations"],"app_model_selection_changed":False}
    write_json(output/"report.json",report)
    write_json(output/"progress.json",{"status":"COMPLETE","completed_candidates":len(results),"result":policy["status"]})
    lines = ["# ₹1 lakh capital-aware walk-forward tuning", "",
        f"Capital ₹{capital:,.0f}; per-trade gross stop-risk budget ₹{capital*risk_pct/100:,.0f} ({risk_pct}%); daily entry-gate limit ₹{capital*daily_pct/100:,.0f} ({daily_pct}%).",
        "",f"Result: {policy['status']}. No model activated or selected for Shadow.","",
        "| Depth | L2 | Qualifying cutoff | Best validation stress net (diagnostic only) |",
        "|---|---|---|---|"]
    for row in results:
        best_stress = max(item["stress_net_pnl"] for item in row["curve"])
        lines.append(f"| {row['parameters']['depth']} | {row['parameters']['l2_leaf_reg']} | {row['winner']['cutoff'] if row['winner'] else 'None'} | ₹{best_stress:,.2f} |")
    lines += ["", "## Historical audit", "", json.dumps(audit,indent=2) if audit else "No qualifying candidate: no candidate fitted or scored on 2026; no fallback.",
        "", "Same-capital frozen classifier: "+json.dumps(report["same_capital_frozen_classifier"]),
        "", "Same-capital rule-only control: "+json.dumps(report["same_capital_rule_baseline"]),
        "", "## Limitations", "", *["- "+line for line in specification["limitations"]]]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({key:report[key] for key in ("status","completed_candidates","qualifying_candidates","historical_audit")},indent=2),flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    run(args.output)
