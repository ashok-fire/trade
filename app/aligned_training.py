"""One bounded, versioned decision-alignment experiment. Never auto-promotes.

Only known, labelled paths fit models. Unlabelled decision opportunities remain
in daily evaluation and cannot be replaced with later known outcomes.
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
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from .config import config
from .specialist_research import chronological_partitions, quality_logit, score_metrics
from .strategy import FEATURE_SCHEMA_V6, SETUP_FAMILIES, RegimeModel
from .training import model_metadata_path
from .training_contract import ALIGNED_FEATURES, VERSION, contract_spec
from .v4_training import _fit_ranker, build_option_quality_samples

FIXED_PERIODS = {
    "fit": (date(2023, 1, 1), date(2025, 4, 3)),
    "calibration": (date(2025, 4, 7), date(2025, 8, 18)),
    "tuning": (date(2025, 8, 20), date(2026, 1, 6)),
    "evaluation": (date(2026, 1, 8), date(2026, 10, 13)),
}
PARAMETERS = dict(iterations=250, depth=3, learning_rate=0.03, l2_leaf_reg=10,
    loss_function="Logloss", random_seed=42, thread_count=1, verbose=False,
    allow_writing_files=False, has_time=True)


def day_weights(frame):
    weights = (1 / frame.groupby("trading_day").trading_day.transform("size")).to_numpy()
    return weights * len(frame) / weights.sum()


def partitions_for(frame, periods=None):
    if periods is None:
        return chronological_partitions(frame)
    previous_end = None
    for name in ("fit", "calibration", "tuning", "evaluation"):
        start, end = periods[name]
        if start > end or (previous_end is not None and start <= previous_end):
            raise ValueError("Chronological periods must be ordered and disjoint")
        previous_end = end
    if frame.empty:
        return {name: frame.copy() for name in periods}
    return {name: frame.loc[(frame.trading_day >= start) & (frame.trading_day <= end)].copy()
            for name, (start, end) in periods.items()}


def labelled(frame):
    return frame.loc[(frame.label_status == "COMPLETE") & frame.label.notna()].copy()


def select_daily(frame, probabilities, thresholds):
    if len(frame) != len(probabilities):
        raise ValueError("Scores must align exactly with opportunity rows")
    working = frame.copy()
    working["probability"] = probabilities
    cutoffs = np.array([thresholds.get(side, 1.01) for side in working.direction])
    working = working.loc[np.isfinite(probabilities) & (np.asarray(probabilities) >= cutoffs)]
    # Retain the whole original row. groupby.first() can silently replace null
    # exit/P&L values with a different, later observation's known outcome.
    return working.sort_values(["timestamp", "probability"], ascending=[True, False], kind="stable").drop_duplicates("trading_day")


def daily_metrics(frame, probabilities, thresholds):
    selected = select_daily(frame, probabilities, thresholds)
    complete = labelled(selected)
    pnl = complete.net_pnl.to_numpy(dtype=float)
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0, equity])[1:]
    monthly = complete.groupby(pd.to_datetime(complete.timestamp, utc=True).dt.strftime("%Y-%m")).net_pnl.sum()
    return {"selected": len(selected), "trades": len(complete), "censored": len(selected) - len(complete),
        "coverage": len(complete) / len(selected) if len(selected) else 0,
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 2) if len(pnl) else 0,
        "profit_factor": round(float(gains / losses), 3) if losses else None,
        "net_pnl": round(float(pnl.sum()), 2),
        "double_slippage_net_pnl": round(float(complete.stress_net_pnl.sum()), 2),
        "mean_net_pnl": round(float(pnl.mean()), 2) if len(pnl) else None,
        "max_drawdown": round(float((peak - equity).max()), 2) if len(pnl) else 0,
        "active_months": len(monthly), "positive_months": int((monthly > 0).sum()),
        "monthly": {key: round(float(value), 2) for key, value in monthly.items()},
        "ce_trades": int((complete.direction == "CE").sum()), "pe_trades": int((complete.direction == "PE").sum())}


def choose_threshold(frame, probabilities):
    """Fixed tuning-only support/stability gates; disabled means no fallback."""
    if frame.empty:
        return 1.01, []
    curve = []
    days = sorted(frame.trading_day.unique())
    middle = days[len(days) // 2]
    first = (frame.trading_day < middle).to_numpy()
    for integer in range(40, 77, 2):
        cutoff = integer / 100
        limits = {side: cutoff for side in ("CE", "PE")}
        metrics = daily_metrics(frame, probabilities, limits)
        halves = [daily_metrics(frame.loc[mask], np.asarray(probabilities)[mask], limits) for mask in (first, ~first)]
        curve.append({"threshold": cutoff, **metrics, "halves": halves})
    def passed(item, neighbour=False):
        return (item["trades"] >= (10 if neighbour else 15) and item["coverage"] >= .99
                and item["win_rate_pct"] >= (50 if neighbour else 55)
                and (item["profit_factor"] or 0) >= (1.10 if neighbour else 1.30)
                and item["double_slippage_net_pnl"] > 0 and item["active_months"] >= 3)
    supported = [item for index, item in enumerate(curve) if 0 < index < len(curve) - 1
        and passed(item) and passed(curve[index-1], True) and passed(curve[index+1], True)
        and all(half["trades"] >= 5 and half["double_slippage_net_pnl"] > 0 for half in item["halves"])]
    chosen = max(supported, key=lambda item: (item["trades"], item["double_slippage_net_pnl"])) if supported else None
    return (chosen["threshold"] if chosen else 1.01), curve


def bootstrap_comparison(frame, selected, baseline):
    days = sorted(frame.trading_day.unique())
    if not days:
        return {}
    # Censored selected entries are NOT zero returns. Remove their days from
    # this descriptive paired comparison and report the exclusion count.
    censored = set(selected.loc[selected.label_status != "COMPLETE", "trading_day"])
    censored |= set(baseline.loc[baseline.label_status != "COMPLETE", "trading_day"])
    days = [day for day in days if day not in censored]
    if not days or labelled(selected).empty:
        return {"excluded_censored_days": len(censored)}
    model_pnl = labelled(selected).set_index("trading_day").net_pnl.reindex(days, fill_value=0).to_numpy(dtype=float)
    base_pnl = labelled(baseline).set_index("trading_day").net_pnl.reindex(days, fill_value=0).to_numpy(dtype=float)
    trades = labelled(selected).groupby("trading_day").size().reindex(days, fill_value=0).to_numpy()
    rng = np.random.default_rng(42)
    means, improvements = [], []
    for _ in range(10000):
        starts = rng.integers(0, len(days), size=(len(days)+4)//5)
        indexes = ((starts[:, None] + np.arange(5)) % len(days)).reshape(-1)[:len(days)]
        if trades[indexes].sum():
            means.append(model_pnl[indexes].sum() / trades[indexes].sum())
        improvements.append((model_pnl[indexes] - base_pnl[indexes]).mean())
    return {"method": "10,000 five-observed-day circular moving-block resamples; seed 42; descriptive exposed history",
        "excluded_censored_days": len(censored),
        "mean_net_per_trade_95pct": np.quantile(means, [.025, .975]).round(2).tolist(),
        "paired_net_improvement_per_day_95pct": np.quantile(improvements, [.025, .975]).round(2).tolist()}


def train_aligned_model(db_url, model_path, indices, start_date, end_date, provider="GROWW", *,
                        setup_families=SETUP_FAMILIES, periods=None):
    model_path = Path(model_path)
    if start_date >= end_date:
        raise ValueError("Training end date must follow start date")
    if len(indices) != 1:
        raise ValueError("Aligned training requires one index; multi-index portfolio parity is not established")
    if not setup_families or any(family not in SETUP_FAMILIES for family in setup_families):
        raise ValueError("Unsupported setup families")
    if periods is not None:
        # Validate before any history is read; fixed CLI runs cannot consume the
        # reserved October window merely because the database contains it.
        if start_date != periods["fit"][0] or end_date != periods["evaluation"][1]:
            raise ValueError("Requested range must equal the frozen study range")
        partitions_for(pd.DataFrame({"trading_day": []}), periods)
    if model_path.exists() or model_metadata_path(model_path).exists() or model_path.with_suffix(".spec.json").exists():
        raise ValueError("Refusing to overwrite a frozen candidate; use a new output path")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    contract = contract_spec(setup_families)
    specification = {"experiment": VERSION, "provider": provider, "indices": indices,
        "start": str(start_date), "end": str(end_date), "contract": contract,
        "features": ALIGNED_FEATURES, "estimator": "existing shallow CatBoost; no estimator/regime search",
        "parameters": PARAMETERS, "calibration": "separate-day Platt; negative slope disables direction",
        "periods": ({name: [str(a), str(b)] for name, (a,b) in periods.items()} if periods else
                    "60% fit / 10% calibration / 10% tuning / 20% evaluation; one-day boundary purges"),
        "thresholds": "0.40–0.76 in 0.02 steps; neighbours, counts, coverage and both tuning halves must pass",
        "gates": {"min_completed_trades": 100, "min_observed_sessions": 60, "min_pf": 1.3,
                  "min_coverage": .99, "min_win_rate_pct": 55, "min_positive_month_fraction": .6,
                  "positive_double_slippage_pnl": True, "positive_expectancy_and_paired_lower_95pct_bounds": True},
        "historical_evaluation_previously_exposed": True, "activation_eligible": False}
    specification_hash = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    model_path.with_suffix(".spec.json").write_text(json.dumps(specification, indent=2, sort_keys=True))
    engine = create_engine(db_url)
    if db_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def query_only(connection, _):
            connection.execute("PRAGMA query_only=ON")
    try:
        with Session(engine) as session:
            dataset, coverage = build_option_quality_samples(session, provider=provider, code=indices[0],
                start_date=start_date, end_date=end_date, setup_families=setup_families,
                sampling_policy=contract["sampling_policy"], strike_policy=contract["strike_policy"],
                setup_policy=contract["setup_policy"], causal_opportunities=True, retain_unlabelled=True,
                required_features=ALIGNED_FEATURES)
    finally:
        engine.dispose()
    if dataset.empty or len(labelled(dataset)) < 500:
        raise RuntimeError("At least 500 labelled aligned opportunities are required")
    dataset = dataset.sort_values(["timestamp", "setup_family"], kind="stable").reset_index(drop=True)
    partitions = partitions_for(dataset, periods)
    fitting, calibration = labelled(partitions["fit"]), labelled(partitions["calibration"])
    tuning, evaluation = partitions["tuning"], partitions["evaluation"]
    if min(len(fitting), len(calibration), len(labelled(tuning)), len(labelled(evaluation))) < 100:
        raise RuntimeError("Aligned chronological partitions are too small")
    models, calibrations, enabled = {}, {}, []
    for side in ("CE", "PE"):
        fit = fitting.loc[fitting.direction == side]
        cal = calibration.loc[calibration.direction == side]
        if len(fit) < 100 or len(cal) < 40 or fit.label.nunique() < 2 or cal.label.nunique() < 2:
            raise RuntimeError(f"Insufficient fitting/calibration observations for {side}")
        quality = CatBoostClassifier(**PARAMETERS)
        quality.fit(fit[ALIGNED_FEATURES], fit.label.astype(int), sample_weight=day_weights(fit))
        models[side] = quality
        probability = quality.predict_proba(cal[ALIGNED_FEATURES])[:, 1]
        calibrator = LogisticRegression(C=1.0, random_state=42)
        calibrator.fit(quality_logit(probability), cal.label.astype(int), sample_weight=day_weights(cal))
        slope = float(calibrator.coef_[0, 0])
        calibrations[side] = {"method": "platt", "input": "logit_quality",
            "intercept": float(calibrator.intercept_[0]), "quality_coefficient": slope, "rank_coefficient": 0.0}
        if slope > 0:
            enabled.append(side)
        print(f"Aligned {side}: fit {len(fit)}, calibration {len(cal)}, slope {slope:.4f}", flush=True)
    # Keep the established bundle format. The legacy ranker is fit only on fit
    # days; its coefficient is explicitly zero in this controlled experiment.
    ranker = _fit_ranker(fitting, rounds=120, features=ALIGNED_FEATURES)
    manifest = {"quality_models": {}, "ranker": f"{model_path.stem}.ranker.json"}
    for side in ("CE", "PE"):
        filename = f"{model_path.stem}.{side.lower()}.cbm"
        models[side].save_model(str(model_path.parent / filename))
        manifest["quality_models"][side] = filename
    ranker.save_model(model_path.parent / manifest["ranker"])
    metadata = {"version": f"quality-aligned-v1-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "trained_at": datetime.now(UTC).isoformat(), "feature_schema": FEATURE_SCHEMA_V6,
        "features": ALIGNED_FEATURES, "classes": ["NO_TRADE", "TRADE"],
        "provider": provider, "training_data_provider": provider, "compatible_live_providers": [provider],
        "indices": indices, "training_start": str(start_date), "training_end": str(end_date),
        "setup_families": list(setup_families), "enabled_directions": enabled,
        "disabled_directions": [side for side in ("CE", "PE") if side not in enabled],
        "calibration": calibrations, "training_contract": contract, "specification": specification,
        "specification_hash": specification_hash, "coverage": {indices[0]: coverage},
        "samples": len(labelled(dataset)), "opportunities": len(dataset),
        "training_samples": len(fitting), "calibration_samples": len(calibration),
        "tuning_samples": len(labelled(tuning)), "validation_samples": len(labelled(evaluation)),
        "partition_counts": {name: {"opportunities": len(frame), "labelled": len(labelled(frame)),
            "days": int(frame.trading_day.nunique()), "start": str(frame.trading_day.min()),
            "end": str(frame.trading_day.max())} for name, frame in partitions.items()},
        "dataset_hash": hashlib.sha256(pd.util.hash_pandas_object(dataset, index=False).to_numpy().tobytes()).hexdigest(),
        "actual_sample_start": str(dataset.trading_day.min()), "actual_sample_end": str(dataset.trading_day.max()),
        "bundle_files": [*manifest["quality_models"].values(), manifest["ranker"]],
        "label_method": contract["label_source"], "slippage_pct": "0.05", "stress_slippage_pct": "0.10",
        "daily_trade_limit": 1, "research_only": True, "requires_forward_shadow": True,
        "activation_eligible": False, "historical_evaluation_previously_exposed": True,
        "decision_threshold": 1.01, "direction_thresholds": {"CE": 1.01, "PE": 1.01}}
    # Persist and load the actual frozen bundle before scoring. Evaluation does
    # not refit, recalibrate or select models/families/regimes.
    model_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    model_metadata_path(model_path).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    model = RegimeModel(model_path)
    tune_probabilities = model.quality_probabilities(tuning)
    thresholds, curves = {}, {}
    for side in ("CE", "PE"):
        mask = (tuning.direction == side).to_numpy()
        thresholds[side], curves[side] = choose_threshold(tuning.loc[mask], tune_probabilities[mask]) if side in enabled else (1.01, [])
    evaluation_probabilities = model.quality_probabilities(evaluation)
    tuning_metrics = daily_metrics(tuning, tune_probabilities, thresholds)
    metrics = daily_metrics(evaluation, evaluation_probabilities, thresholds)
    baseline = daily_metrics(evaluation, np.ones(len(evaluation)), {"CE": 0, "PE": 0})
    selected = select_daily(evaluation, evaluation_probabilities, thresholds)
    baseline_selected = select_daily(evaluation, np.ones(len(evaluation)), {"CE": 0, "PE": 0})
    uncertainty = bootstrap_comparison(evaluation, selected, baseline_selected)
    scored = np.isfinite(evaluation_probabilities) & evaluation.label.notna().to_numpy()
    quality_metrics = score_metrics(evaluation.loc[scored], evaluation_probabilities[scored])
    blockers = []
    if all(value > 1 for value in thresholds.values()):
        blockers.append("No direction passed the frozen tuning stability gates; no fallback threshold")
    if metrics["trades"] < 100 or evaluation.trading_day.nunique() < 60:
        blockers.append("Fewer than 100 complete trades or 60 observed evaluation sessions")
    if metrics["coverage"] < .99:
        blockers.append("Selected outcome coverage below 99%")
    if metrics["win_rate_pct"] < 55 or (metrics["profit_factor"] or 0) < 1.3:
        blockers.append("Historical win-rate/profit-factor gates failed")
    if metrics["double_slippage_net_pnl"] <= 0:
        blockers.append("Historical double-slippage result is not positive")
    if metrics["active_months"] < 6 or metrics["positive_months"] / max(metrics["active_months"], 1) < .6:
        blockers.append("Historical monthly stability gate failed")
    if (uncertainty.get("mean_net_per_trade_95pct", [-1])[0] <= 0 or
            uncertainty.get("paired_net_improvement_per_day_95pct", [-1])[0] <= 0):
        blockers.append("Positive expectancy and paired baseline improvement are not supported by lower 95% bounds")
    historical_passed = not blockers
    blockers.extend(["Historical evaluation was previously exposed; it is not a fresh validation",
        "Executable quote/fill parity and reconciled live features require forward evidence",
        "A genuinely forward frozen Shadow period and portfolio-risk validation remain required"])
    metadata.update({"direction_thresholds": thresholds,
        "setup_thresholds": {family: thresholds.copy() for family in setup_families},
        "decision_threshold": min(thresholds.values()), "enabled_setups": [f"{family}:{side}"
            for family in setup_families for side in ("CE", "PE") if thresholds[side] <= 1],
        "direction_tuning_curves": curves, "tuning": tuning_metrics, "locked_validation": metrics,
        "historical_gates_passed": historical_passed, "evaluation_scores": quality_metrics,
        "baseline_evaluation": baseline, "uncertainty": uncertainty,
        "activation_blockers": blockers})
    model_metadata_path(model_path).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps({"version": metadata["version"], "opportunities": len(dataset),
        "thresholds": thresholds, "evaluation": metrics, "baseline": baseline,
        "scores": quality_metrics, "blockers": blockers, "model_path": str(model_path)}, indent=2), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_aligned_model(config.database_url, args.output, ["NIFTY"], FIXED_PERIODS["fit"][0],
        FIXED_PERIODS["evaluation"][1], "GROWW", setup_families=("ORB",), periods=FIXED_PERIODS)
