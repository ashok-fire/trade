"""Fixed-specification comparison of simple models and causal setup filters.

Exports research diagnostics only; it cannot select or activate runtime models.
The last historical evaluation period has been seen during prior research.
14--23 October 2026 remains reserved and is not queried here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from .config import config
from .specialist_research import SPECIALIST_FEATURES, quality_logit, score_metrics, stable_threshold
from .strategy import SETUP_FAMILIES
from .v4_training import _performance, build_option_quality_samples


PERIODS = {
    "fit": (date(2023, 1, 1), date(2025, 4, 3)),
    "calibration": (date(2025, 4, 7), date(2025, 8, 18)),
    "tuning": (date(2025, 8, 20), date(2026, 1, 6)),
    "evaluation": (date(2026, 1, 8), date(2026, 10, 13)),
}
REGIMES = ("ALL", "TREND_ALIGNED", "MORNING")
ESTIMATORS = ("LOGISTIC", "CATBOOST")


def regime_mask(frame: pd.DataFrame, regime: str) -> np.ndarray:
    if regime == "ALL":
        return np.ones(len(frame), dtype=bool)
    if regime == "TREND_ALIGNED":
        side = np.where(frame.direction.to_numpy() == "CE", 1.0, -1.0)
        return (
            (frame.trend_atr_strength.to_numpy() * side >= 0.5)
            & (frame.ema20_50_distance.to_numpy() * side > 0)
            & (frame.futures_vwap_distance.to_numpy() * side > 0)
        )
    if regime == "MORNING":
        entry = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert("Asia/Kolkata") + pd.Timedelta(minutes=5)
        return (entry.dt.hour * 60 + entry.dt.minute < 12 * 60 + 30).to_numpy()
    raise ValueError("Unknown regime")


def day_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = (1 / frame.groupby("trading_day").trading_day.transform("size")).to_numpy()
    return weights * len(frame) / weights.sum()


def fixed_partitions(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        name: dataset.loc[(dataset.trading_day >= start) & (dataset.trading_day <= end)].copy()
        for name, (start, end) in PERIODS.items()
    }


def make_estimator(name: str):
    if name == "LOGISTIC":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=42))
    return CatBoostClassifier(
        iterations=250, depth=3, learning_rate=0.03, l2_leaf_reg=10,
        loss_function="Logloss", random_seed=42, thread_count=1,
        verbose=False, allow_writing_files=False, has_time=True,
    )


def run(output: Path) -> dict:
    specification = {
        "periods": {name: [a.isoformat(), b.isoformat()] for name, (a, b) in PERIODS.items()},
        "features": SPECIALIST_FEATURES,
        "estimators": list(ESTIMATORS), "regimes": list(REGIMES),
        "regime_definitions": {
            "ALL": "Every causal setup within the entry window",
            "TREND_ALIGNED": "Directional trend/ATR >= 0.5, EMA20/50 and futures/VWAP aligned",
            "MORNING": "Entry before 12:30 IST",
        },
        "strike_policy": "ATM_ITM1",
        "selection": "Tuning-only; stable neighbouring thresholds, both tuning halves profitable under double slippage; maximize stressed expectancy with trade support",
        "historical_evaluation_previously_exposed": True,
        "reserved_holdout_start": "2026-10-14",
        "runtime_filters_missing_in_history": ["bid_ask_spread", "quote_age", "delta", "live_volume_and_oi_semantics"],
    }
    specification_hash = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".spec.json").write_text(json.dumps(specification, indent=2, sort_keys=True))
    engine = create_engine(config.database_url)
    try:
        with Session(engine) as session:
            dataset, coverage = build_option_quality_samples(
                session, provider="GROWW", code="NIFTY", start_date=PERIODS["fit"][0],
                end_date=PERIODS["evaluation"][1], sampling_policy="all_eligible_bars",
                strike_policy="ATM_ITM1",
            )
    finally:
        engine.dispose()
    dataset = dataset.sort_values(["timestamp", "setup_family"], kind="stable").reset_index(drop=True)
    partitions = fixed_partitions(dataset)
    report = {
        "created_at": datetime.now(UTC).isoformat(), "specification_hash": specification_hash,
        "specification": specification, "coverage": coverage,
        "samples": len(dataset), "distinct_days": int(dataset.trading_day.nunique()),
        "actual_sample_start": str(dataset.trading_day.min()),
        "actual_sample_end": str(dataset.trading_day.max()),
        "baselines": [], "experiments": [], "selected_on_tuning": None,
        "activation_eligible": False,
    }
    for family in SETUP_FAMILIES:
        for side in ("CE", "PE"):
            frames = {
                name: frame.loc[(frame.setup_family == family) & (frame.direction == side)].copy()
                for name, frame in partitions.items()
            }
            fit, calibration, tuning, evaluation = [frames[name] for name in PERIODS]
            baseline = {"setup": f"{family}:{side}"}
            for name in ("tuning", "evaluation"):
                base = frames[name]
                baseline[name] = _performance(base, np.ones(len(base)), 0.0)
            report["baselines"].append(baseline)
            if fit.trading_day.nunique() < 100 or any(len(frame) < 40 or frame.label.nunique() < 2 for frame in (fit, calibration, tuning)):
                report["experiments"].append({"setup": baseline["setup"], "status": "INSUFFICIENT_DEVELOPMENT_DATA"})
                continue
            for estimator_name in ESTIMATORS:
                model = make_estimator(estimator_name)
                if estimator_name == "LOGISTIC":
                    model.fit(fit[SPECIALIST_FEATURES], fit.label, logisticregression__sample_weight=day_weights(fit))
                else:
                    model.fit(fit[SPECIALIST_FEATURES], fit.label, sample_weight=day_weights(fit))
                calibrator = LogisticRegression(C=1.0, random_state=42)
                calibrator.fit(
                    quality_logit(model.predict_proba(calibration[SPECIALIST_FEATURES])[:, 1]),
                    calibration.label, sample_weight=day_weights(calibration),
                )
                slope = float(calibrator.coef_[0, 0])
                if slope <= 0:
                    report["experiments"].append({"setup": baseline["setup"], "estimator": estimator_name, "status": "NEGATIVE_CALIBRATION_SLOPE", "slope": slope})
                    continue
                tune_probability = calibrator.predict_proba(quality_logit(model.predict_proba(tuning[SPECIALIST_FEATURES])[:, 1]))[:, 1]
                supported = []
                for regime in REGIMES:
                    mask = regime_mask(tuning, regime)
                    subset, probability = tuning.loc[mask].copy(), tune_probability[mask]
                    experiment = {"setup": baseline["setup"], "estimator": estimator_name, "regime": regime, "slope": round(slope, 4)}
                    if subset.empty:
                        experiment["status"] = "NO_TUNING_SETUPS"
                        report["experiments"].append(experiment)
                        continue
                    cutoff, curve = stable_threshold(subset, probability)
                    experiment.update({"threshold": cutoff, "tuning_curve": curve, "tuning_scores": score_metrics(subset, probability)})
                    if cutoff is None:
                        experiment["status"] = "NO_STABLE_THRESHOLD"
                    else:
                        performance = _performance(subset, probability, cutoff)
                        experiment["tuning"] = performance
                        days = sorted(tuning.trading_day.unique())
                        middle = days[len(days) // 2]
                        first = subset.trading_day.to_numpy() < middle
                        halves = [_performance(subset.loc[mask], probability[mask], cutoff) for mask in (first, ~first)]
                        experiment["tuning_halves"] = halves
                        if all(item["trades"] >= 5 and item["double_slippage_net_pnl"] > 0 for item in halves):
                            experiment["status"] = "SUPPORTED_ON_TUNING"
                            supported.append(experiment)
                        else:
                            experiment["status"] = "UNSTABLE_TUNING_HALVES"
                    report["experiments"].append(experiment)
                # Choose at most one regime per fitted estimator using tuning.
                # Keep its model in memory; evaluation is never used to tune it.
                for experiment in supported:
                    experiment["_model"] = model
                    experiment["_calibrator"] = calibrator
            print(f"Completed tuning comparisons: {family}:{side}", flush=True)
    supported = [item for item in report["experiments"] if item.get("status") == "SUPPORTED_ON_TUNING"]
    if supported:
        selected = max(supported, key=lambda item: item["tuning"]["double_slippage_net_pnl"] / item["tuning"]["trades"])
        family, side = selected["setup"].split(":")
        evaluation = partitions["evaluation"]
        evaluation = evaluation.loc[(evaluation.setup_family == family) & (evaluation.direction == side)].copy()
        evaluation = evaluation.loc[regime_mask(evaluation, selected["regime"])].copy()
        probability = selected["_calibrator"].predict_proba(quality_logit(selected["_model"].predict_proba(evaluation[SPECIALIST_FEATURES])[:, 1]))[:, 1]
        report["selected_on_tuning"] = {
            key: selected[key] for key in ("setup", "estimator", "regime", "threshold", "tuning", "tuning_halves")
        }
        performance = _performance(evaluation, probability, selected["threshold"])
        report["selected_on_tuning"]["historical_evaluation"] = performance
        report["selected_on_tuning"]["evaluation_scores"] = score_metrics(evaluation, probability)
        report["selected_on_tuning"]["historical_gates_passed"] = (
            performance["trades"] >= 25 and performance["win_rate_pct"] >= 55
            and (performance["profit_factor"] or 0) >= 1.3 and performance["double_slippage_net_pnl"] > 0
            and performance["active_months"] >= 6
            and performance["positive_months"] / max(performance["active_months"], 1) >= 0.60
        )
    for item in report["experiments"]:
        item.pop("_model", None)
        item.pop("_calibrator", None)
    report["blockers"] = [
        "Full quote-filter parity requires recorded decision-time snapshots",
        "Fresh holdout and forward Shadow validation remain required",
    ]
    if report["selected_on_tuning"] is None:
        report["blockers"].append("No model/regime/threshold met tuning stability requirements")
    elif not report["selected_on_tuning"]["historical_gates_passed"]:
        report["blockers"].append("Tuning-selected candidate failed later historical gates")
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"samples": len(dataset), "coverage": coverage, "selected": report["selected_on_tuning"], "blockers": report["blockers"], "report": str(output.resolve())}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output)
