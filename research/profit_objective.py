"""Offline-only test of net-return learning; no application or order integration.

This research schema intentionally cannot be loaded as a win-probability model.
No threshold, model family, indicator, exit or strike search uses evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from app.aligned_training import (
    FIXED_PERIODS, PARAMETERS, bootstrap_comparison, daily_metrics,
    day_weights, labelled, partitions_for, select_daily,
)
from app.config import config
from app.strategy import RegimeModel
from app.training_contract import ALIGNED_FEATURES, contract_spec
from app.v4_training import build_option_quality_samples

SCHEMA = "net-return-research-v1"
MARGINS = (.0025, .005, .0075, .01, .015, .02, .025, .03)
REGRESSION_PARAMETERS = {**PARAMETERS, "loss_function": "RMSE"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forecast(models, calibrations, frame, features=ALIGNED_FEATURES):
    """Expected net premium return, never a probability or confidence score."""
    values = np.full(len(frame), np.nan)
    matrix = frame.reindex(columns=features).to_numpy(dtype=float)
    finite = np.isfinite(matrix).all(axis=1)
    for side, model in models.items():
        calibration = calibrations[side]
        if not calibration["enabled"]:
            continue
        positions = np.flatnonzero(finite & (frame.direction.to_numpy() == side))
        if len(positions):
            values[positions] = calibration["intercept"] + calibration["slope"] * model.predict(matrix[positions])
    return values


def select_margin(frame, forecasts):
    """Economic buffers selected only on tuning; no forced-trade fallback."""
    if frame.empty:
        return None, []
    days = sorted(frame.trading_day.unique())
    middle = days[len(days)//2]
    first = (frame.trading_day < middle).to_numpy()
    curve = []
    for margin in MARGINS:
        thresholds = {side: margin for side in ("CE", "PE")}
        metrics = daily_metrics(frame, forecasts, thresholds)
        halves = [daily_metrics(frame.loc[mask], np.asarray(forecasts)[mask], thresholds) for mask in (first, ~first)]
        curve.append({"margin_net_premium_return": margin, **metrics, "halves": halves})
    def passes(item, neighbour=False):
        return (item["trades"] >= (15 if neighbour else 30) and item["coverage"] >= .99
            and item["win_rate_pct"] >= (50 if neighbour else 55)
            and (item["profit_factor"] or 0) >= (1.1 if neighbour else 1.3)
            and item["double_slippage_net_pnl"] > 0 and item["active_months"] >= 3)
    eligible = [item for index, item in enumerate(curve) if 0 < index < len(curve)-1
        and passes(item) and passes(curve[index-1], True) and passes(curve[index+1], True)
        and all(half["trades"] >= 10 and half["double_slippage_net_pnl"] > 0 for half in item["halves"])]
    # Same observation dates, so maximising stressed total selects by net
    # economic contribution, not the highest win rate on a tiny trade subset.
    chosen = max(eligible, key=lambda item: (item["double_slippage_net_pnl"], item["trades"])) if eligible else None
    return (chosen["margin_net_premium_return"] if chosen else None), curve


def numeric_thresholds(margins):
    return {side: (float("inf") if margin is None else margin) for side, margin in margins.items()}


def diagnostics(frame, predictions, calibration_reference):
    complete = frame.label_status.eq("COMPLETE").to_numpy() & np.isfinite(predictions)
    if not complete.any():
        return {"samples": 0}
    observed = frame.loc[complete]
    targets = observed.net_return.to_numpy(dtype=float)
    weights = day_weights(observed)
    predicted = predictions[complete]
    return {"samples": len(observed), "days": int(observed.trading_day.nunique()),
        "forecast_return_p05_p50_p95": np.quantile(predicted, [.05, .5, .95]).round(6).tolist(),
        "actual_mean_net_return": float(np.average(targets, weights=weights)),
        "weighted_rmse": float(np.sqrt(mean_squared_error(targets, predicted, sample_weight=weights))),
        "weighted_r2": float(r2_score(targets, predicted, sample_weight=weights)),
        "constant_calibration_mean_rmse": float(np.sqrt(mean_squared_error(
            targets, observed.direction.map(calibration_reference).to_numpy(), sample_weight=weights)))}


def run(output: Path, control: Path):
    output, control = Path(output), Path(control)
    if output.exists():
        raise ValueError("Frozen research output already exists; refusing overwrite")
    control_metadata_path = control.with_name(f"{control.stem}.metadata.json")
    control_metadata = json.loads(control_metadata_path.read_text())
    if (control_metadata.get("training_contract") != contract_spec(("ORB",))
            or control_metadata.get("features") != ALIGNED_FEATURES
            or control_metadata.get("training_end") != "2026-10-13"):
        raise ValueError("Control is not the frozen, decision-aligned NIFTY ORB study")
    manifest = json.loads(control.read_text())
    control_files = [control, control_metadata_path,
        *(control.parent / name for name in manifest["quality_models"].values()), control.parent / manifest["ranker"]]
    control_hashes = {str(path): fingerprint(path) for path in control_files}
    specification = {"hypothesis": "Learning the size of net option returns improves economic entry selection over win/loss classification",
        "schema": SCHEMA, "features": ALIGNED_FEATURES, "contract": contract_spec(("ORB",)),
        "unchanged": ["NIFTY", "ORB CE/PE", "data", "feature inputs", "exits", "strike selection", "cost assumptions", "chronological partitions"],
        "target": "net_return = option P&L after estimated fees/slippage / (entry premium * one lot)",
        "target_processing": "no winsorising, class weights, target clipping or evaluation-driven changes",
        "estimator": "existing shallow CatBoost, regression objective instead of Logloss",
        "parameters": REGRESSION_PARAMETERS,
        "calibration": "positive-slope weighted ordinary least squares on separate calibration days; otherwise disable",
        "margin_grid_net_premium_return": list(MARGINS),
        "selection": "tuning only; min 30 trades, >=99% coverage, >=55% wins, PF>=1.30, both halves >=10 stressed-profitable trades, supported neighbours; maximise stressed net total",
        "evaluation": "first passing observation per day; retain censored first selections; one historical lot",
        "periods": {key: [str(a), str(b)] for key,(a,b) in FIXED_PERIODS.items()},
        "control_hashes": control_hashes, "expected_dataset_hash": control_metadata["dataset_hash"],
        "acceptance": {"complete_trades": 100, "observed_sessions": 60, "win_rate_pct": 55, "pf": 1.3,
            "coverage": .99, "active_months": 6, "positive_month_fraction": .6,
            "stress_net_positive": True, "positive_lower_95pct_expectancy": True,
            "positive_lower_95pct_paired_improvement_vs_both_controls": True,
            "drawdown_no_greater_than_classifier_control": True},
        "historical_evaluation_previously_exposed": True, "reserved_dates_not_used": ["2026-10-14", "2026-10-23"],
        "activation_eligible": False, "attempt_budget": "one objective/configuration; no hyperparameter, exit, regime or family search"}
    output.mkdir(parents=True)
    write_json(output / "specification.json", specification)
    spec_hash = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    engine = create_engine(config.database_url)
    @event.listens_for(engine, "connect")
    def read_only(connection, _):
        connection.execute("PRAGMA query_only=ON")
    try:
        with Session(engine) as session:
            dataset, coverage = build_option_quality_samples(session, provider="GROWW", code="NIFTY",
                start_date=FIXED_PERIODS["fit"][0], end_date=FIXED_PERIODS["evaluation"][1], setup_families=("ORB",),
                sampling_policy="all_eligible_bars", strike_policy="ATM_ITM1", setup_policy="first_confirmed",
                causal_opportunities=True, retain_unlabelled=True, required_features=ALIGNED_FEATURES)
    finally:
        engine.dispose()
    dataset = dataset.sort_values(["timestamp", "setup_family"], kind="stable").reset_index(drop=True)
    dataset_hash = hashlib.sha256(pd.util.hash_pandas_object(dataset, index=False).to_numpy().tobytes()).hexdigest()
    if dataset_hash != specification["expected_dataset_hash"]:
        raise ValueError("Opportunity table differs from frozen control; stopping before fitting")
    parts = partitions_for(dataset, FIXED_PERIODS)
    fitting, calibration = labelled(parts["fit"]), labelled(parts["calibration"])
    tuning, evaluation = parts["tuning"], parts["evaluation"]
    models, calibrations, references = {}, {}, {}
    for side in ("CE", "PE"):
        fit, cal = fitting.loc[fitting.direction == side], calibration.loc[calibration.direction == side]
        if len(fit) < 100 or len(cal) < 40:
            raise ValueError(f"Insufficient fit/calibration days for {side}")
        model = CatBoostRegressor(**REGRESSION_PARAMETERS)
        model.fit(fit[ALIGNED_FEATURES], fit.net_return, sample_weight=day_weights(fit))
        models[side] = model
        raw = model.predict(cal[ALIGNED_FEATURES])
        linear = LinearRegression().fit(raw.reshape(-1,1), cal.net_return, sample_weight=day_weights(cal))
        slope = float(linear.coef_[0])
        calibrations[side] = {"intercept": float(linear.intercept_), "slope": slope, "enabled": slope > 0,
            "fit_samples": len(fit), "calibration_samples": len(cal),
            "reason": "positive calibration slope" if slope > 0 else "non-positive calibration slope; direction disabled"}
        references[side] = float(np.average(cal.net_return, weights=day_weights(cal)))
        model.save_model(str(output / f"{side.lower()}.cbm"))
        print(f"Profit-aware {side}: calibration slope {slope:.4f}; enabled {slope > 0}", flush=True)
    # Reload actual saved weights before the tuning or evaluation forecasts.
    for side in models:
        loaded = CatBoostRegressor()
        loaded.load_model(str(output / f"{side.lower()}.cbm"))
        models[side] = loaded
    tune_forecasts = forecast(models, calibrations, tuning)
    margins, curves = {}, {}
    for side in ("CE", "PE"):
        mask = (tuning.direction == side).to_numpy()
        margins[side], curves[side] = select_margin(tuning.loc[mask], tune_forecasts[mask]) if calibrations[side]["enabled"] else (None, [])
    frozen_policy = {"schema": SCHEMA, "score_unit": "expected net return on entry premium, not probability",
        "features": ALIGNED_FEATURES, "models": {side: f"{side.lower()}.cbm" for side in models},
        "calibration": calibrations, "margin_thresholds": margins,
        "specification_hash": spec_hash, "activation_eligible": False, "research_only": True}
    write_json(output / "policy.json", frozen_policy)
    # From here the policy is immutable. No evaluation result changes it.
    policy_hash = fingerprint(output / "policy.json")
    eval_forecasts = forecast(models, calibrations, evaluation)
    thresholds = numeric_thresholds(margins)
    profit_metrics = daily_metrics(evaluation, eval_forecasts, thresholds)
    selected = select_daily(evaluation, eval_forecasts, thresholds)
    baseline = select_daily(evaluation, np.ones(len(evaluation)), {"CE": 0, "PE": 0})
    control_model = RegimeModel(control)
    if not control_model.available:
        raise ValueError("Frozen classifier control cannot be loaded")
    classifier_scores = control_model.quality_probabilities(evaluation)
    classifier_thresholds = control_metadata["direction_thresholds"]
    classifier_selected = select_daily(evaluation, classifier_scores, classifier_thresholds)
    classifier_metrics = daily_metrics(evaluation, classifier_scores, classifier_thresholds)
    if classifier_metrics != control_metadata["locked_validation"]:
        raise ValueError("Control replay does not exactly reproduce its frozen metrics")
    versus_rules = bootstrap_comparison(evaluation, selected, baseline)
    versus_classifier = bootstrap_comparison(evaluation, selected, classifier_selected)
    blockers = []
    if all(value is None for value in margins.values()):
        blockers.append("No economic entry margin passed the frozen tuning gates; no fallback policy")
    if profit_metrics["trades"] < 100 or evaluation.trading_day.nunique() < 60:
        blockers.append("Fewer than 100 completed trades or 60 observed evaluation sessions")
    if profit_metrics["coverage"] < .99:
        blockers.append("Selected outcome coverage is below 99%")
    if profit_metrics["win_rate_pct"] < 55 or (profit_metrics["profit_factor"] or 0) < 1.3:
        blockers.append("Win-rate/profit-factor gates failed")
    if profit_metrics["double_slippage_net_pnl"] <= 0:
        blockers.append("Double-slippage net result is not positive")
    if profit_metrics["active_months"] < 6 or profit_metrics["positive_months"] / max(profit_metrics["active_months"],1) < .6:
        blockers.append("Monthly profitability stability gate failed")
    if versus_classifier.get("mean_net_per_trade_95pct", [-1])[0] <= 0:
        blockers.append("Positive expectancy is not supported by a positive lower 95% bound")
    for name, comparison in (("rules", versus_rules), ("classifier", versus_classifier)):
        if comparison.get("paired_net_improvement_per_day_95pct", [-1])[0] <= 0:
            blockers.append(f"Improvement versus {name} is not supported by a positive paired lower 95% bound")
    if profit_metrics["max_drawdown"] > classifier_metrics["max_drawdown"]:
        blockers.append("Drawdown exceeds the frozen classifier control")
    if any(fingerprint(path) != control_hashes[str(path)] for path in control_files):
        raise ValueError("Frozen classifier control changed during this experiment")
    if fingerprint(output / "policy.json") != policy_hash:
        raise ValueError("Policy changed after evaluation began")
    historical_passed = not blockers
    blockers.extend(["Previously inspected historical evaluation is not a fresh test",
        "Executable quote/fill and full portfolio-risk evidence are not established",
        "A frozen genuinely forward validation period remains mandatory"])
    report = {"created_at": datetime.now(UTC).isoformat(), "specification": specification,
        "specification_hash": spec_hash, "frozen_policy_hash": policy_hash, "dataset_hash": dataset_hash,
        "coverage": coverage, "opportunities": len(dataset), "labelled_outcomes": len(labelled(dataset)),
        "calibration": calibrations, "entry_margin_thresholds": margins, "tuning_curves": curves,
        "tuning": daily_metrics(tuning, tune_forecasts, thresholds),
        "historical_evaluation": profit_metrics, "classifier_control": classifier_metrics,
        "rule_control": daily_metrics(evaluation, np.ones(len(evaluation)), {"CE": 0, "PE": 0}),
        "return_forecast_diagnostics": diagnostics(evaluation, eval_forecasts, references),
        "uncertainty_vs_rules": versus_rules, "uncertainty_vs_classifier": versus_classifier,
        "historical_gates_passed": historical_passed, "activation_eligible": False,
        "activation_blockers": blockers, "control_reproduced_exactly": True,
        "changes_applied_to_application": False}
    write_json(output / "report.json", report)
    print(json.dumps({key: report[key] for key in ("opportunities", "entry_margin_thresholds", "historical_evaluation",
        "classifier_control", "return_forecast_diagnostics", "historical_gates_passed", "activation_blockers")}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, default=Path("models/research/decision-aligned-orb-20261024/candidate.json"))
    arguments = parser.parse_args()
    run(arguments.output, arguments.control)
