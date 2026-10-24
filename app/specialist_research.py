"""Offline setup research with separate fit/calibrate/tune/evaluate periods.

This command writes a research report only. It cannot register or select a
runtime model. Historical evaluation has already been inspected during V5/V6
development and therefore cannot be described as an untouched final test.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from .config import config
from .strategy import QUALITY_FEATURES, SETUP_FAMILIES, SETUP_FEATURES
from .v4_training import _performance, build_option_quality_samples


# Five-minute candle volume and a live quote's volume have not been reconciled.
# OI units across the two sources have not been verified either. Keep them out
# of this experiment rather than assume a conversion or fill synthetic values.
SPECIALIST_FEATURES = [
    name for name in QUALITY_FEATURES
    if name not in {"option_volume_log", "option_oi_log", *SETUP_FEATURES}
]


def chronological_partitions(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    days = sorted(dataset.trading_day.unique())
    if len(days) < 100:
        raise ValueError("At least 100 trading days are required")
    boundaries = [int(len(days) * fraction) for fraction in (0.60, 0.70, 0.80)]
    ranges = {
        "fit": days[:boundaries[0] - 1],
        "calibration": days[boundaries[0]:boundaries[1] - 1],
        "tuning": days[boundaries[1]:boundaries[2] - 1],
        "evaluation": days[boundaries[2]:],
    }
    return {
        name: dataset.loc[dataset.trading_day.isin(values)].copy()
        for name, values in ranges.items()
    }


def quality_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def score_metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict:
    if frame.empty:
        return {"samples": 0, "days": 0}
    # Repeated five-minute candidates share a day; evaluate probability quality
    # with equal total weight per day, not as independent Bernoulli trials.
    weights = 1.0 / frame.groupby("trading_day").trading_day.transform("size")
    return {
        "samples": len(frame),
        "days": int(frame.trading_day.nunique()),
        "score_p05_p50_p95": np.quantile(probabilities, [0.05, 0.5, 0.95]).round(4).tolist(),
        "auc": round(float(roc_auc_score(frame.label, probabilities, sample_weight=weights)), 4)
        if frame.label.nunique() == 2 else None,
        "brier": round(float(brier_score_loss(frame.label, probabilities, sample_weight=weights)), 4),
    }


def stable_threshold(frame: pd.DataFrame, probabilities: np.ndarray) -> tuple[float | None, list[dict]]:
    curve = [_performance(frame, probabilities, cutoff / 100) for cutoff in range(40, 77, 2)]

    def passes(item: dict, *, neighbour: bool = False) -> bool:
        pf = item["profit_factor"]
        return (
            item["trades"] >= (10 if neighbour else 15)
            and item["win_rate_pct"] >= (50 if neighbour else 55)
            and pf is not None and pf >= (1.10 if neighbour else 1.30)
            and item["double_slippage_net_pnl"] > 0
            and item["active_months"] >= 3
        )

    eligible = [
        item for index, item in enumerate(curve)
        if 0 < index < len(curve) - 1
        and passes(item)
        and passes(curve[index - 1], neighbour=True)
        and passes(curve[index + 1], neighbour=True)
    ]
    # Prefer support across more days over the highest win rate in a tiny tail.
    chosen = max(eligible, key=lambda item: (item["trades"], item["net_pnl"])) if eligible else None
    return (chosen["threshold"] if chosen else None), curve


def research_specialist(partitions: dict[str, pd.DataFrame], family: str, side: str) -> tuple[dict, pd.DataFrame | None]:
    frames = {
        name: frame.loc[(frame.setup_family == family) & (frame.direction == side)].copy()
        for name, frame in partitions.items()
    }
    result = {"setup": f"{family}:{side}", "status": "RESEARCH_ONLY", "blockers": []}
    result["partition_counts"] = {
        name: {"samples": len(frame), "days": int(frame.trading_day.nunique())}
        for name, frame in frames.items()
    }
    fit, calibration, tuning, evaluation = [frames[name] for name in ("fit", "calibration", "tuning", "evaluation")]
    if any(len(frame) < 40 or frame.label.nunique() < 2 for frame in frames.values()) or fit.trading_day.nunique() < 100:
        result["blockers"].append("insufficient distinct days or mixed outcomes in a chronological partition")
        return result, None
    model = CatBoostClassifier(
        iterations=250, depth=3, learning_rate=0.03, l2_leaf_reg=10,
        loss_function="Logloss", random_seed=42, thread_count=1,
        verbose=False, allow_writing_files=False, has_time=True,
    )
    weights = 1.0 / fit.groupby("trading_day").trading_day.transform("size")
    weights *= len(fit) / weights.sum()
    model.fit(fit[SPECIALIST_FEATURES], fit.label, sample_weight=weights)
    raw_calibration = model.predict_proba(calibration[SPECIALIST_FEATURES])[:, 1]
    calibrator = LogisticRegression(C=1.0, random_state=42)
    calibration_weights = 1.0 / calibration.groupby("trading_day").trading_day.transform("size")
    calibration_weights *= len(calibration) / calibration_weights.sum()
    calibrator.fit(quality_logit(raw_calibration), calibration.label, sample_weight=calibration_weights)
    slope = float(calibrator.coef_[0, 0])
    result["calibration_slope"] = round(slope, 4)
    if slope <= 0:
        result["blockers"].append("quality predictions do not rank outcomes positively on separate calibration days")
        return result, None
    tune_probability = calibrator.predict_proba(quality_logit(model.predict_proba(tuning[SPECIALIST_FEATURES])[:, 1]))[:, 1]
    cutoff, curve = stable_threshold(tuning, tune_probability)
    result["threshold"] = cutoff
    result["tuning_curve"] = curve
    result["tuning_scores"] = score_metrics(tuning, tune_probability)
    if cutoff is None:
        result["blockers"].append("no tuning threshold passed the trade-count, cost and neighbouring-cutoff gates")
        return result, None
    result["tuning"] = _performance(tuning, tune_probability, cutoff)
    probability = calibrator.predict_proba(quality_logit(model.predict_proba(evaluation[SPECIALIST_FEATURES])[:, 1]))[:, 1]
    result["evaluation_scores"] = score_metrics(evaluation, probability)
    performance = _performance(evaluation, probability, cutoff)
    result["historical_evaluation"] = performance
    if performance["trades"] < 25:
        result["blockers"].append("fewer than 25 evaluation trades")
    if performance["win_rate_pct"] < 55:
        result["blockers"].append("evaluation win rate below 55%")
    if (performance["profit_factor"] or 0) < 1.30 or performance["double_slippage_net_pnl"] <= 0:
        result["blockers"].append("evaluation lacks a sufficient edge after costs")
    result["blockers"].append("fresh future data and parallel Shadow evaluation are still required")
    scored = evaluation.copy()
    scored["probability"] = probability
    scored["specialist_threshold"] = cutoff
    return result, scored


def run(start: date, end: date, output: Path) -> dict:
    engine = create_engine(config.database_url)
    try:
        with Session(engine) as session:
            dataset, coverage = build_option_quality_samples(
                session, provider="GROWW", code="NIFTY", start_date=start, end_date=end,
                sampling_policy="all_eligible_bars",
            )
    finally:
        engine.dispose()
    if dataset.empty:
        raise ValueError("No valid option outcomes")
    dataset = dataset.sort_values(["timestamp", "setup_family"], kind="stable").reset_index(drop=True)
    print(f"Built {len(dataset)} eligible outcomes on {dataset.trading_day.nunique()} days", flush=True)
    partitions = chronological_partitions(dataset)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "historical_evaluation_previously_exposed": True,
        "fresh_holdout_required": True,
        "requested_start": start.isoformat(), "requested_end": end.isoformat(),
        "actual_first_sample": str(dataset.trading_day.min()),
        "actual_last_sample": str(dataset.trading_day.max()),
        "coverage": coverage, "features": SPECIALIST_FEATURES,
        "partition_dates": {
            name: {"first": str(frame.trading_day.min()), "last": str(frame.trading_day.max()), "samples": len(frame), "days": int(frame.trading_day.nunique())}
            for name, frame in partitions.items()
        },
        "specialists": [],
    }
    for family in SETUP_FAMILIES:
        for side in ("CE", "PE"):
            result, _scored = research_specialist(partitions, family, side)
            report["specialists"].append(result)
            print(json.dumps({key: result.get(key) for key in ("setup", "threshold", "calibration_slope", "historical_evaluation", "blockers")}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Research report: {output.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 13))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.start, arguments.end, arguments.output)
