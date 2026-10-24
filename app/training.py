from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from .models import Candle, Instrument
from .strategy import FEATURE_SCHEMA, PRICE_FEATURES, feature_values, regular_session, with_futures_features


LABEL_HORIZON_BARS = 6
PURGE_BARS = 6
MIN_REGULAR_SESSION_BARS = 70


def model_metadata_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}.metadata.json")


def load_model_metadata(model_path: Path) -> dict:
    metadata_path = model_metadata_path(model_path)
    if not model_path.exists() or not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def labelled_samples(
    frame: pd.DataFrame,
    horizon: int = LABEL_HORIZON_BARS,
    futures_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create cost-buffered directional labels while keeping all features causal."""
    enriched = with_futures_features(frame, futures_frame if futures_frame is not None else frame)
    local_day = enriched.timestamp.dt.tz_convert("Asia/Kolkata").dt.date
    labels = np.full(len(enriched), "NO_TRADE", dtype=object)
    future_available = np.zeros(len(enriched), dtype=bool)
    # A direction is tradable only when its ATR/cost-buffered target is reached
    # before the adverse stop. Whipsaws and small moves remain NO_TRADE.
    for positions in pd.Series(np.arange(len(enriched))).groupby(local_day).groups.values():
        day_positions = np.asarray(list(positions), dtype=int)
        seen_directions: set[str] = set()
        for offset in range(0, max(0, len(day_positions) - horizon)):
            position = day_positions[offset]
            path_positions = day_positions[offset + 1 : offset + horizon + 1]
            if len(path_positions) < horizon:
                continue
            close = float(enriched.iloc[position].close)
            atr = float(enriched.iloc[position].atr)
            if not np.isfinite(atr) or atr <= 0 or close <= 0:
                continue
            current = enriched.iloc[position]
            local_stamp = pd.Timestamp(current.timestamp).tz_convert("Asia/Kolkata")
            minutes_from_open = local_stamp.hour * 60 + local_stamp.minute - 555
            bullish_setup = (
                minutes_from_open > 15
                and float(current.futures_close) > float(current.futures_vwap)
                and float(current.ema9) > float(current.ema20)
                and close > float(current.opening_range_high)
            )
            bearish_setup = (
                minutes_from_open > 15
                and float(current.futures_close) < float(current.futures_vwap)
                and float(current.ema9) < float(current.ema20)
                and close < float(current.opening_range_low)
            )
            if bullish_setup == bearish_setup:
                continue
            direction = "CE" if bullish_setup else "PE"
            if direction in seen_directions:
                continue
            seen_directions.add(direction)
            path = enriched.iloc[path_positions]
            target = max(atr * 0.90, close * 0.0012)
            stop = max(atr * 0.55, close * 0.0007)

            def first_hit(mask: np.ndarray) -> int | None:
                matches = np.flatnonzero(mask)
                return int(matches[0]) if len(matches) else None

            up_target = first_hit(path.high.to_numpy(dtype=float) >= close + target)
            up_stop = first_hit(path.low.to_numpy(dtype=float) <= close - stop)
            down_target = first_hit(path.low.to_numpy(dtype=float) <= close - target)
            down_stop = first_hit(path.high.to_numpy(dtype=float) >= close + stop)
            up_valid = up_target is not None and (up_stop is None or up_target < up_stop)
            down_valid = down_target is not None and (down_stop is None or down_target < down_stop)
            terminal = float(path.iloc[-1].close)
            if bullish_setup and up_valid and not down_valid and terminal > close:
                labels[position] = "TRADE"
            elif bearish_setup and down_valid and not up_valid and terminal < close:
                labels[position] = "TRADE"
            future_available[position] = True

    feature_rows = [feature_values(row) for _, row in enriched.iterrows()]
    samples = pd.DataFrame(feature_rows, index=enriched.index)
    samples["label"] = labels
    samples["timestamp"] = enriched.timestamp
    samples["future_close_available"] = future_available
    finite = np.isfinite(samples[PRICE_FEATURES].to_numpy(dtype=float)).all(axis=1) & future_available
    return samples.loc[finite].reset_index(drop=True)


def _complete_regular_days(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    regular = regular_session(frame)
    if regular.empty:
        return regular, 0
    day = regular.timestamp.dt.tz_convert("Asia/Kolkata").dt.date
    counts = regular.groupby(day).size()
    complete_days = set(counts[counts >= MIN_REGULAR_SESSION_BARS].index)
    kept = regular.loc[day.isin(complete_days)].reset_index(drop=True)
    return kept, int(len(counts) - len(complete_days))


def _scale_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not len(probabilities):
        return probabilities
    logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / max(float(temperature), 0.1)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _fit_temperature(probabilities: np.ndarray, expected: np.ndarray) -> float:
    if not len(probabilities):
        return 1.0
    best_temperature, best_loss = 1.0, float("inf")
    rows = np.arange(len(expected))
    for temperature in np.linspace(0.5, 2.5, 81):
        scaled = _scale_probabilities(probabilities, float(temperature))
        loss = float(-np.log(np.clip(scaled[rows, expected], 1e-9, 1.0)).mean())
        if loss < best_loss:
            best_temperature, best_loss = float(temperature), loss
    return best_temperature


def _classification_metrics(
    probabilities: np.ndarray,
    expected: np.ndarray,
    classes: list[str],
    *,
    decision_threshold: float,
) -> dict:
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predictions == expected)) if len(expected) else 0.0
    per_class_accuracy = []
    confusion = [[0 for _ in classes] for _ in classes]
    for actual, predicted in zip(expected.tolist(), predictions.tolist()):
        confusion[int(actual)][int(predicted)] += 1
    per_class_metrics: dict[str, dict[str, float]] = {}
    for class_id, class_name in enumerate(classes):
        mask = expected == class_id
        if mask.any():
            per_class_accuracy.append(float(np.mean(predictions[mask] == expected[mask])))
        true_positive = confusion[class_id][class_id]
        predicted_total = sum(row[class_id] for row in confusion)
        actual_total = sum(confusion[class_id])
        per_class_metrics[class_name] = {
            "precision": round(true_positive / predicted_total, 4) if predicted_total else 0.0,
            "recall": round(true_positive / actual_total, 4) if actual_total else 0.0,
        }
    confidence = np.max(probabilities, axis=1) if len(probabilities) else np.array([])
    directional_ids = {
        index for index, name in enumerate(classes) if name in {"TRADE", "UPTREND", "DOWNTREND"}
    }
    selected = np.array(
        [prediction in directional_ids for prediction in predictions], dtype=bool
    ) & (confidence >= decision_threshold)
    directional_count = int(selected.sum())
    directional_precision = (
        float(np.mean(predictions[selected] == expected[selected])) if directional_count else 0.0
    )
    directional_curve = []
    for threshold in np.arange(0.50, 0.71, 0.02):
        threshold_selected = np.array(
            [prediction in directional_ids for prediction in predictions], dtype=bool
        ) & (confidence >= threshold)
        count = int(threshold_selected.sum())
        directional_curve.append(
            {
                "threshold": round(float(threshold), 2),
                "predictions": count,
                "precision": round(
                    float(np.mean(predictions[threshold_selected] == expected[threshold_selected])), 4
                ) if count else 0.0,
            }
        )
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(per_class_accuracy)) if per_class_accuracy else 0.0,
        "confusion_matrix": confusion,
        "per_class_metrics": per_class_metrics,
        "confidence": confidence,
        "directional_count": directional_count,
        "directional_precision": directional_precision,
        "directional_curve": directional_curve,
    }


def _select_direction_threshold(
    probabilities: np.ndarray,
    expected: np.ndarray,
    classes: list[str],
) -> float:
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    directional_ids = {
        index for index, name in enumerate(classes) if name in {"TRADE", "UPTREND", "DOWNTREND"}
    }
    directional = np.array([prediction in directional_ids for prediction in predictions], dtype=bool)
    # The target/stop label is approximately 1.6:1 before costs, so a 40%
    # precision floor is above its break-even hit rate. The real option
    # backtest remains the final gate before activation.
    for threshold in np.arange(0.50, 0.81, 0.02):
        selected = directional & (confidence >= threshold)
        if int(selected.sum()) >= 25 and float(np.mean(predictions[selected] == expected[selected])) >= 0.38:
            return round(float(threshold), 2)
    return 0.70


def front_future_frame(candles: list[Candle], instruments: list[Instrument]) -> pd.DataFrame:
    expiries = {item.instrument_key: item.expiry for item in instruments}
    rows = []
    for candle in candles:
        expiry = expiries.get(candle.instrument_key)
        if expiry is None:
            continue
        timestamp = pd.Timestamp(candle.timestamp)
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        trading_day = timestamp.tz_convert("Asia/Kolkata").date()
        # Roll at the start of expiry day so training and live inference use the
        # same deterministic contract-selection policy.
        if expiry <= trading_day:
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "expiry": expiry,
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
                "volume": candle.volume,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows).sort_values(["timestamp", "expiry"])
    return frame.drop_duplicates("timestamp", keep="first").drop(columns=["expiry"]).reset_index(drop=True)


def train_regime_model(
    db_url: str,
    model_path: Path,
    indices: list[str],
    start_date: date,
    end_date: date,
    provider: str = "UPSTOX",
) -> dict:
    if start_date >= end_date:
        raise ValueError("Training end date must be after the start date")

    engine = create_engine(db_url, connect_args={"check_same_thread": False, "timeout": 30} if db_url.startswith("sqlite") else {})
    samples: list[pd.DataFrame] = []
    diagnostics: dict[str, dict[str, int]] = {}
    try:
        with Session(engine) as session:
            for code in indices:
                candles = session.scalars(
                    select(Candle).where(
                        Candle.provider == provider,
                        Candle.underlying_code == code,
                        Candle.data_role == "SPOT",
                        Candle.timeframe == "5minute",
                        Candle.timestamp >= datetime.combine(start_date, time.min, tzinfo=UTC),
                        Candle.timestamp <= datetime.combine(end_date, time.max, tzinfo=UTC),
                        Candle.source_verified.is_(True),
                    ).order_by(Candle.timestamp)
                ).all()
                if len(candles) < 100:
                    diagnostics[code] = {"spot_candles": len(candles), "futures_candles": 0, "usable_samples": 0}
                    continue
                future_candles = session.scalars(
                    select(Candle).where(
                        Candle.provider == provider,
                        Candle.underlying_code == code,
                        Candle.data_role == "FUTURE",
                        Candle.timeframe == "5minute",
                        Candle.timestamp >= datetime.combine(start_date, time.min, tzinfo=UTC),
                        Candle.timestamp <= datetime.combine(end_date, time.max, tzinfo=UTC),
                        Candle.source_verified.is_(True),
                    ).order_by(Candle.timestamp)
                ).all()
                future_instruments = session.scalars(
                    select(Instrument).where(
                        Instrument.provider == provider,
                        Instrument.underlying_code == code,
                        Instrument.instrument_kind == "FUTURE",
                    )
                ).all()
                futures_frame = front_future_frame(future_candles, future_instruments)
                if len(futures_frame) < 100:
                    diagnostics[code] = {
                        "spot_candles": len(candles),
                        "futures_candles": len(futures_frame),
                        "usable_samples": 0,
                    }
                    continue
                raw_frame = pd.DataFrame([
                    {"timestamp": row.timestamp, "open": float(row.open), "high": float(row.high), "low": float(row.low), "close": float(row.close), "volume": row.volume}
                    for row in candles
                ])
                frame, dropped_spot_days = _complete_regular_days(raw_frame)
                futures_frame, dropped_future_days = _complete_regular_days(futures_frame)
                if len(frame) < 100 or len(futures_frame) < 100:
                    diagnostics[code] = {
                        "spot_candles": len(candles),
                        "futures_candles": len(futures_frame),
                        "usable_samples": 0,
                        "dropped_spot_days": dropped_spot_days,
                        "dropped_future_days": dropped_future_days,
                    }
                    continue
                index_samples = labelled_samples(frame, futures_frame=futures_frame)
                index_samples["underlying_code"] = code
                diagnostics[code] = {
                    "spot_candles": len(candles),
                    "futures_candles": len(futures_frame),
                    "usable_samples": len(index_samples),
                    "dropped_spot_days": dropped_spot_days,
                    "dropped_future_days": dropped_future_days,
                }
                if not index_samples.empty:
                    samples.append(index_samples)
    finally:
        engine.dispose()

    if not samples:
        detail = "; ".join(
            f"{code}: {values['spot_candles']} spot, {values['futures_candles']} futures, {values['usable_samples']} usable"
            for code, values in diagnostics.items()
        )
        raise RuntimeError(
            "No usable spot/futures training samples. Download verified futures candles; "
            "index candles have no tradable volume."
            + (f" Coverage: {detail}" if detail else "")
        )
    dataset = pd.concat(samples, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if len(dataset) < 500:
        raise RuntimeError(f"Only {len(dataset)} usable samples found; at least 500 are required")

    import xgboost as xgb

    classes = sorted(dataset.label.unique().tolist())
    if len(classes) < 2:
        raise RuntimeError("Training data contains fewer than two trade outcomes; use a wider date range")
    class_to_id = {name: index for index, name in enumerate(classes)}
    labels = dataset.label.map(class_to_id).to_numpy(dtype=int)
    matrix = dataset[PRICE_FEATURES].to_numpy(dtype=float)
    timestamps = pd.to_datetime(dataset.timestamp, utc=True)
    tuning_start = int(len(dataset) * 0.65)
    validation_start = int(len(dataset) * 0.80)
    purge_delta = timedelta(minutes=PURGE_BARS * 5)
    train_cutoff = timestamps.iloc[tuning_start] - purge_delta
    tuning_cutoff = timestamps.iloc[validation_start] - purge_delta
    training_end_index = int((timestamps < train_cutoff).sum())
    tuning_end = int((timestamps < tuning_cutoff).sum())
    if training_end_index < 500 or tuning_end - tuning_start < 100 or len(dataset) - validation_start < 100:
        raise RuntimeError("Not enough chronological samples after applying the purged validation split")

    def sample_weights(values: np.ndarray) -> np.ndarray:
        counts = Counter(values.tolist())
        return np.array(
            [len(values) / (len(counts) * counts[value]) for value in values],
            dtype=float,
        )

    parameters = {
        "objective": "multi:softprob",
        "num_class": len(classes),
        "eval_metric": "mlogloss",
        "eta": 0.03,
        "max_depth": 3,
        "min_child_weight": 8,
        "gamma": 0.05,
        "reg_alpha": 0.10,
        "reg_lambda": 2.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "nthread": 1,
    }
    development_y = labels[:training_end_index]
    tuning_y = labels[tuning_start:tuning_end]
    development_matrix = xgb.DMatrix(
        matrix[:training_end_index],
        label=development_y,
        weight=sample_weights(development_y),
        feature_names=PRICE_FEATURES,
    )
    tuning_matrix = xgb.DMatrix(
        matrix[tuning_start:tuning_end],
        label=tuning_y,
        weight=sample_weights(tuning_y),
        feature_names=PRICE_FEATURES,
    )
    development_model = xgb.train(
        parameters,
        development_matrix,
        num_boost_round=800,
        evals=[(tuning_matrix, "tuning")],
        callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
        verbose_eval=False,
    )
    best_rounds = max(1, int(development_model.best_iteration) + 1)
    raw_tuning_probabilities = development_model.predict(tuning_matrix)
    probability_temperature = _fit_temperature(raw_tuning_probabilities, tuning_y)
    tuning_probabilities = _scale_probabilities(
        raw_tuning_probabilities, probability_temperature
    )
    tuning_metrics = _classification_metrics(
        tuning_probabilities,
        tuning_y,
        classes,
        decision_threshold=0.50,
    )
    decision_threshold = _select_direction_threshold(
        tuning_probabilities, tuning_y, classes
    )

    validation_y = labels[validation_start:]
    validation_matrix = xgb.DMatrix(
        matrix[validation_start:], label=validation_y, feature_names=PRICE_FEATURES
    )
    validation_probabilities = _scale_probabilities(
        development_model.predict(validation_matrix), probability_temperature
    )
    validation_metrics = _classification_metrics(
        validation_probabilities,
        validation_y,
        classes,
        decision_threshold=decision_threshold,
    )
    confidence = validation_metrics["confidence"]

    # Once diagnostics are locked, fit the deployable candidate on the complete
    # 2023-2024 window with the selected round count. The untouched 2025 period
    # remains the true out-of-sample test.
    final_matrix = xgb.DMatrix(
        matrix,
        label=labels,
        weight=sample_weights(labels),
        feature_names=PRICE_FEATURES,
    )
    final_model = xgb.train(
        parameters,
        final_matrix,
        num_boost_round=best_rounds,
        verbose_eval=False,
    )
    importance = final_model.get_score(importance_type="gain")
    walk_forward: list[dict[str, float | int | str]] = []
    for train_fraction in (0.5, 0.6, 0.7):
        fold_validation_start = int(len(dataset) * train_fraction)
        fold_validation_end = min(
            len(dataset), fold_validation_start + max(1, int(len(dataset) * 0.1))
        )
        fold_cutoff = timestamps.iloc[fold_validation_start] - purge_delta
        fold_training_end = int((timestamps < fold_cutoff).sum())
        if fold_training_end < 300 or fold_validation_end - fold_validation_start < 50:
            continue
        fold_y = labels[:fold_training_end]
        fold_train = xgb.DMatrix(
            matrix[:fold_training_end],
            label=fold_y,
            weight=sample_weights(fold_y),
            feature_names=PRICE_FEATURES,
        )
        expected = labels[fold_validation_start:fold_validation_end]
        fold_validation = xgb.DMatrix(
            matrix[fold_validation_start:fold_validation_end],
            label=expected,
            feature_names=PRICE_FEATURES,
        )
        fold_model = xgb.train(
            parameters,
            fold_train,
            num_boost_round=best_rounds,
            verbose_eval=False,
        )
        fold_probabilities = _scale_probabilities(
            fold_model.predict(fold_validation), probability_temperature
        )
        fold_metrics = _classification_metrics(
            fold_probabilities,
            expected,
            classes,
            decision_threshold=decision_threshold,
        )
        walk_forward.append(
            {
                "training_end": pd.Timestamp(dataset.iloc[fold_training_end - 1].timestamp).isoformat(),
                "validation_end": pd.Timestamp(dataset.iloc[fold_validation_end - 1].timestamp).isoformat(),
                "training_samples": fold_training_end,
                "validation_samples": fold_validation_end - fold_validation_start,
                "purge_bars": PURGE_BARS,
                "accuracy": round(float(fold_metrics["accuracy"]), 4),
                "balanced_accuracy": round(float(fold_metrics["balanced_accuracy"]), 4),
                "directional_precision": round(float(fold_metrics["directional_precision"]), 4),
                "directional_predictions": int(fold_metrics["directional_count"]),
            }
        )

    activation_blockers = []
    if float(validation_metrics["balanced_accuracy"]) < 0.55:
        activation_blockers.append("balanced validation accuracy is below 55%")
    if int(validation_metrics["directional_count"]) < 25:
        activation_blockers.append("fewer than 25 qualified directional validation predictions")
    if float(validation_metrics["directional_precision"]) < 0.38:
        activation_blockers.append("qualified directional precision is below the 38% break-even floor")
    stable_folds = sum(
        int(float(fold["balanced_accuracy"]) >= 0.52) for fold in walk_forward
    )
    if stable_folds < 2:
        activation_blockers.append("walk-forward performance is not stable across at least two folds")

    trained_at = datetime.now(UTC)
    metadata = {
        "version": f"regime-{trained_at.strftime('%Y%m%dT%H%M%SZ')}",
        "trained_at": trained_at.isoformat(),
        "training_start": start_date.isoformat(),
        "training_end": end_date.isoformat(),
        "indices": indices,
        "provider": provider,
        "training_data_provider": provider,
        "feature_schema": FEATURE_SCHEMA,
        "compatible_live_providers": ["UPSTOX", "GROWW"],
        "features": PRICE_FEATURES,
        "classes": classes,
        "samples": len(dataset),
        "training_samples": len(dataset),
        "development_training_samples": training_end_index,
        "tuning_samples": tuning_end - tuning_start,
        "validation_samples": len(dataset) - validation_start,
        "validation_accuracy": round(float(validation_metrics["accuracy"]), 4),
        "balanced_accuracy": round(float(validation_metrics["balanced_accuracy"]), 4),
        "confusion_matrix": validation_metrics["confusion_matrix"],
        "per_class_metrics": validation_metrics["per_class_metrics"],
        "feature_importance": {feature: round(float(importance.get(feature, 0.0)), 6) for feature in PRICE_FEATURES},
        "validation_confidence": {
            "mean": round(float(np.mean(confidence)), 4) if len(confidence) else 0.0,
            "above_70_percent": round(float(np.mean(confidence >= 0.70) * 100), 2) if len(confidence) else 0.0,
            "above_decision_threshold_percent": round(
                float(np.mean(confidence >= decision_threshold) * 100), 2
            ) if len(confidence) else 0.0,
        },
        "directional_validation": {
            "threshold": decision_threshold,
            "predictions": int(validation_metrics["directional_count"]),
            "precision": round(float(validation_metrics["directional_precision"]), 4),
            "curve": validation_metrics["directional_curve"],
        },
        "tuning_directional_curve": tuning_metrics["directional_curve"],
        "walk_forward_validation": walk_forward,
        "class_counts": dict(Counter(dataset.label.tolist())),
        "label_horizon_bars": LABEL_HORIZON_BARS,
        "label_method": "Causal breakout setup with ATR/cost-buffered target-before-stop outcome",
        "purge_bars": PURGE_BARS,
        "regular_session_only": True,
        "minimum_regular_session_bars": MIN_REGULAR_SESSION_BARS,
        "probability_temperature": round(probability_temperature, 4),
        "decision_threshold": decision_threshold,
        "early_stopping_best_rounds": best_rounds,
        "model_parameters": parameters,
        "activation_eligible": not activation_blockers,
        "activation_blockers": activation_blockers,
        "uses_live_context": False,
        "coverage": diagnostics,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_name(f".{model_path.stem}.tmp.json")
    temporary_metadata = model_metadata_path(model_path).with_name(f".{model_metadata_path(model_path).name}.tmp")
    final_model.save_model(temporary_model)
    temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    os.replace(temporary_model, model_path)
    os.replace(temporary_metadata, model_metadata_path(model_path))
    return metadata
