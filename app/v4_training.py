from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from .models import Candle, Instrument
from .option_selector import historical_option_choice, option_candidates
from .data_quality import futures_alignment_gaps
from .risk import option_charges
from .strategy import (
    FEATURE_SCHEMA_V4,
    FEATURE_SCHEMA_V5,
    FEATURE_SCHEMA_V6,
    OPTION_META_FEATURES,
    QUALITY_FEATURES,
    SETUP_FAMILIES,
    feature_values,
    regular_session,
    selected_setup,
    setup_direction,
    with_futures_features,
)
from .training import _complete_regular_days, front_future_frame, model_metadata_path


SLIPPAGE_PCT = Decimal("0.05")
PREMIUM_STOP_PCT = Decimal("25")
REWARD_RISK = Decimal("1.5")
MIN_DIRECTION_SAMPLES = 100


def option_quality_context(
    *,
    premium: float,
    volume: int,
    open_interest: int,
    expiry: date,
    strike: Decimal | float,
    spot: float,
    trading_day: date,
) -> dict[str, float]:
    """Return option fields available at signal time."""
    safe_spot = max(float(spot), 1e-9)
    return {
        "option_premium_pct": float(premium) / safe_spot * 100.0,
        "option_volume_log": float(np.log1p(max(int(volume), 0))),
        "option_oi_log": float(np.log1p(max(int(open_interest), 0))),
        "days_to_expiry": float(max((expiry - trading_day).days, 0)),
        "moneyness_pct": (float(strike) - safe_spot) / safe_spot * 100.0,
    }


def _utc_stamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _contract_for_signal(
    instruments: list[Instrument], direction: str, trading_day: date, spot: float
) -> Instrument | None:
    choices = option_candidates(
        instruments, spot=Decimal(str(spot)), direction=direction,
        trading_day=trading_day, policy="ATM_ONLY",
    )
    return choices[0][0] if choices else None


def _setup_direction(row: pd.Series, family: str) -> str | None:
    """Compatibility wrapper around the shared live/research detector."""
    return setup_direction(row, family)


def _option_rows_by_key(
    session: Session,
    provider: str,
    events_by_key: dict[str, list[dict]],
) -> dict[str, list[Candle]]:
    rows: dict[str, list[Candle]] = {}
    for instrument_key, events in events_by_key.items():
        first = min(event["time"] for event in events)
        last_day = max(event["day"] for event in events)
        last = datetime.combine(last_day, time(15, 30), tzinfo=UTC)
        rows[instrument_key] = list(
            session.scalars(
                select(Candle)
                .where(
                    Candle.provider == provider,
                    Candle.instrument_key == instrument_key,
                    Candle.data_role == "OPTION",
                    Candle.timeframe == "5minute",
                    Candle.timestamp >= first,
                    Candle.timestamp <= last,
                    Candle.source_verified.is_(True),
                )
                .order_by(Candle.timestamp)
            ).all()
        )
    return rows


def _simulate_event(
    event: dict,
    option_rows: list[Candle],
    underlying_by_time: dict[pd.Timestamp, pd.Series],
    provider: str,
    reward_risk: Decimal = REWARD_RISK,
    premium_stop_pct: Decimal = PREMIUM_STOP_PCT,
    rejection_counts: Counter | None = None,
    required_features: list[str] | None = None,
) -> dict | None:
    required_features = QUALITY_FEATURES if required_features is None else required_features
    def reject(reason: str):
        if rejection_counts is not None:
            rejection_counts[reason] += 1
        return None

    signal_time = event["time"]
    entry_time = signal_time + pd.Timedelta(minutes=5)
    cutoff = _utc_stamp(
        datetime.combine(
            event["day"], time(15, 20), tzinfo=ZoneInfo("Asia/Kolkata")
        )
    )
    signal_candle = next(
        (row for row in option_rows if _utc_stamp(row.timestamp) == signal_time), None
    )
    bars = [
        row
        for row in option_rows
        if signal_time < _utc_stamp(row.timestamp) <= cutoff
    ]
    if signal_candle is None or not bars:
        return reject("missing_signal_or_future_option_candle")
    if _utc_stamp(bars[0].timestamp) != entry_time:
        return reject("missing_next_bar_entry")
    instrument: Instrument = event["instrument"]
    row: pd.Series = event["row"]
    slip = SLIPPAGE_PCT / 100
    entry_base = Decimal(signal_candle.close if bars[0].open <= 0 else bars[0].open)
    entry = entry_base * (1 + slip)
    if entry <= 0:
        return reject("invalid_entry_price")
    premium_stop = entry * (1 - premium_stop_pct / 100)
    spot = Decimal(str(row.close))
    risk = Decimal(str(row.atr))
    if not risk.is_finite() or risk <= 0:
        return reject("invalid_atr")
    direction = event["direction"]
    stop = spot - risk if direction == "CE" else spot + risk
    target = spot + risk * reward_risk if direction == "CE" else spot - risk * reward_risk
    exit_base = None
    exit_reason = "TIME_EXIT"
    expected_time = entry_time
    for bar in bars:
        bar_time = _utc_stamp(bar.timestamp)
        if bar_time != expected_time:
            return reject("gap_in_option_exit_path")
        expected_time += pd.Timedelta(minutes=5)
        if bar_time == cutoff:
            # Square off at 15:20, before this candle's high/low/close exist.
            exit_base = Decimal(bar.open)
            break
        option_low = Decimal(bar.low)
        underlying = underlying_by_time.get(bar_time)
        if underlying is None:
            return reject("gap_in_underlying_exit_path")
        premium_stop_hit = option_low <= premium_stop
        stop_hit = bool(
            underlying is not None
            and (
                (direction == "CE" and Decimal(str(underlying.low)) <= stop)
                or (direction == "PE" and Decimal(str(underlying.high)) >= stop)
            )
        )
        target_hit = bool(
            underlying is not None
            and (
                (direction == "CE" and Decimal(str(underlying.high)) >= target)
                or (direction == "PE" and Decimal(str(underlying.low)) <= target)
            )
        )
        if premium_stop_hit or stop_hit:
            exit_base = min(Decimal(bar.close), premium_stop) if premium_stop_hit else Decimal(bar.close)
            exit_reason = "PREMIUM_STOP" if premium_stop_hit else "UNDERLYING_STOP"
            break
        if target_hit:
            exit_base = Decimal(bar.close)
            exit_reason = "UNDERLYING_TARGET"
            break
    if exit_base is None:
        return reject("missing_square_off_candle")
    exit_price = exit_base * (1 - slip)
    quantity = int(instrument.lot_size)
    charges = option_charges(
        entry, exit_price, quantity, provider, event["day"], instrument.exchange
    )
    net = (exit_price - entry) * quantity - charges
    double_slip = slip * 2
    stress_entry = entry_base * (1 + double_slip)
    stress_exit = exit_base * (1 - double_slip)
    stress_net = (stress_exit - stress_entry) * quantity - option_charges(
        stress_entry,
        stress_exit,
        quantity,
        provider,
        event["day"],
        instrument.exchange,
    )
    context = option_quality_context(
        premium=float(signal_candle.close),
        volume=signal_candle.volume,
        open_interest=signal_candle.open_interest,
        expiry=instrument.expiry,
        strike=instrument.strike,
        spot=float(row.close),
        trading_day=event["day"],
    )
    values = feature_values(
        row,
        {**context, "setup_family": event.get("setup_family", "ORB")},
    )
    values["setup_direction"] = 1.0 if direction == "CE" else -1.0
    if not np.isfinite(np.array([values[name] for name in required_features], dtype=float)).all():
        return reject("missing_required_features")
    return {
        **{name: values[name] for name in required_features},
        "timestamp": signal_time,
        "trading_day": event["day"],
        "direction": direction,
        "setup_family": event.get("setup_family", "ORB"),
        "net_pnl": float(net),
        "stress_net_pnl": float(stress_net),
        "net_return": float(net / (entry * quantity)),
        "label": int(net > 0),
        "exit_reason": exit_reason,
        "instrument_key": instrument.instrument_key if hasattr(instrument, "instrument_key") else "",
        "moneyness": event.get("moneyness", "ATM"),
        "underlying_risk_pct": float(risk / spot),
        "label_status": "COMPLETE",
        "exit_timestamp": bar_time if bar_time == cutoff else bar_time + pd.Timedelta(minutes=5),
    }


def build_option_quality_samples(
    session: Session,
    *,
    provider: str,
    code: str,
    start_date: date,
    end_date: date,
    reward_risk: Decimal = REWARD_RISK,
    premium_stop_pct: Decimal = PREMIUM_STOP_PCT,
    setup_families: tuple[str, ...] = SETUP_FAMILIES,
    sampling_policy: str = "first_per_day",
    last_entry_time: time = time(15, 10),
    strike_policy: str = "ATM_ONLY",
    setup_policy: str = "independent",
    causal_opportunities: bool = False,
    retain_unlabelled: bool = False,
    required_features: list[str] | None = None,
    include_decision_context: bool = False,
) -> tuple[pd.DataFrame, dict]:
    if sampling_policy not in {"first_per_day", "all_eligible_bars"}:
        raise ValueError("Unsupported sampling policy")
    if setup_policy not in {"independent", "first_confirmed"}:
        raise ValueError("Unsupported setup policy")
    required_features = QUALITY_FEATURES if required_features is None else required_features
    start = datetime.combine(start_date, time.min, tzinfo=UTC)
    end = datetime.combine(end_date, time.max, tzinfo=UTC)
    if causal_opportunities:
        start = datetime.combine(start_date, time.min, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(UTC)
        end = datetime.combine(end_date, time.max, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(UTC)
    spot_rows = list(
        session.scalars(
            select(Candle)
            .where(
                Candle.provider == provider,
                Candle.underlying_code == code,
                Candle.data_role == "SPOT",
                Candle.timeframe == "5minute",
                Candle.timestamp >= start,
                Candle.timestamp <= end,
                Candle.source_verified.is_(True),
            )
            .order_by(Candle.timestamp)
        ).all()
    )
    future_rows = list(
        session.scalars(
            select(Candle)
            .where(
                Candle.provider == provider,
                Candle.underlying_code == code,
                Candle.data_role == "FUTURE",
                Candle.timeframe == "5minute",
                Candle.timestamp >= start,
                Candle.timestamp <= end,
                Candle.source_verified.is_(True),
            )
            .order_by(Candle.timestamp)
        ).all()
    )
    future_instruments = list(
        session.scalars(
            select(Instrument).where(
                Instrument.provider == provider,
                Instrument.underlying_code == code,
                Instrument.instrument_kind == "FUTURE",
            )
        ).all()
    )
    raw_spot = pd.DataFrame(
        [
            {
                "timestamp": item.timestamp,
                "open": float(item.open),
                "high": float(item.high),
                "low": float(item.low),
                "close": float(item.close),
                "volume": item.volume,
            }
            for item in spot_rows
        ]
    )
    futures = front_future_frame(future_rows, future_instruments)
    if causal_opportunities:
        # Do not decide a morning entry using whether the remainder of that
        # session eventually has a complete candle path.
        spot, futures = regular_session(raw_spot), regular_session(futures)
        dropped_spot = dropped_futures = 0
    else:
        spot, dropped_spot = _complete_regular_days(raw_spot)
        futures, dropped_futures = _complete_regular_days(futures)
    if len(spot) < 100 or len(futures) < 100:
        return pd.DataFrame(), {
            "spot_candles": len(spot_rows),
            "futures_candles": len(futures),
            "candidate_setups": 0,
            "usable_option_outcomes": 0,
        }
    frame = with_futures_features(spot, futures)
    option_instruments = list(
        session.scalars(
            select(Instrument).where(
                Instrument.provider == provider,
                Instrument.underlying_code == code,
                Instrument.instrument_kind == "OPTION",
                Instrument.metadata_verified.is_(True),
                Instrument.lot_size > 0,
                Instrument.expiry > start_date,
                Instrument.valid_from <= end_date,
            )
        ).all()
    )
    events: list[dict] = []
    seen_setup_days: set[tuple[date, str, str]] = set()
    for index in range(50, len(frame)):
        row = frame.iloc[index]
        stamp = _utc_stamp(row.timestamp)
        local_stamp = stamp.tz_convert("Asia/Kolkata")
        trading_day = local_stamp.date()
        # A bar starting at 15:05 is acted on at the 15:10 boundary, when
        # runtime no longer opens positions. All setup families share this.
        entry_clock = (local_stamp + pd.Timedelta(minutes=5)).time()
        if local_stamp.weekday() >= 5 or not time(9, 35) <= entry_clock < last_entry_time:
            continue
        if causal_opportunities:
            from .training_contract import entry_window_open
            if not entry_window_open((stamp + pd.Timedelta(minutes=5)).to_pydatetime()):
                continue
        families = setup_families
        if setup_policy == "first_confirmed":
            family, direction = selected_setup(row, setup_families)
            families = (family,) if direction else ()
        for setup_family in families:
            family = setup_family.upper()
            direction = _setup_direction(row, family)
            if direction is None:
                continue
            setup_day = (trading_day, family, direction)
            if sampling_policy == "first_per_day" and setup_day in seen_setup_days:
                continue
            choices = option_candidates(
                option_instruments, spot=Decimal(str(row.close)), direction=direction,
                trading_day=trading_day, policy=strike_policy,
            )
            if not choices:
                continue
            seen_setup_days.add(setup_day)
            events.append(
                {
                    "time": stamp,
                    "day": trading_day,
                    "direction": direction,
                    "setup_family": family,
                    "instrument": choices[0][0],
                    "choices": choices,
                    "row": row,
                }
            )
    events_by_key: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        for instrument, _moneyness in event["choices"]:
            events_by_key[instrument.instrument_key].append(event)
    option_rows = _option_rows_by_key(session, provider, events_by_key)
    underlying_by_time = {
        _utc_stamp(row.timestamp): row for _, row in frame.iterrows()
    }
    records = []
    rejection_counts: Counter = Counter()
    by_key_day: dict[tuple[str, date], list[Candle]] = defaultdict(list)
    signal_candles = {}
    for key, candles in option_rows.items():
        for candle in candles:
            stamp = _utc_stamp(candle.timestamp)
            by_key_day[key, stamp.tz_convert("Asia/Kolkata").date()].append(candle)
            signal_candles[key, stamp] = candle
    moneyness_counts: Counter = Counter()
    for event in events:
        # Fallback decisions use only information available at the signal.
        # Missing or losing future outcomes never trigger an ITM1 retry.
        chosen = historical_option_choice(
            event["choices"],
            {instrument.instrument_key: signal_candles.get((instrument.instrument_key, event["time"])) for instrument, _ in event["choices"]},
        )
        if chosen is None:
            rejection_counts["no_signal_time_liquid_option"] += 1
            continue
        instrument, moneyness = chosen
        event = {**event, "instrument": instrument, "moneyness": moneyness}
        # Optional research-only sizing inputs known at the signal close.
        # Keep the default columns unchanged for frozen dataset fingerprints.
        context = ({"decision_premium": float(signal_candles[instrument.instrument_key, event["time"]].close),
                    "lot_size": int(instrument.lot_size)} if include_decision_context else {})
        result = _simulate_event(
            event, by_key_day.get((instrument.instrument_key, event["day"]), []),
            underlying_by_time, provider, reward_risk=reward_risk,
            premium_stop_pct=premium_stop_pct, rejection_counts=rejection_counts,
            required_features=required_features,
        )
        if result is not None:
            records.append({**result, **context})
            moneyness_counts[moneyness] += 1
        elif retain_unlabelled:
            # Features exist at the decision clock even when an exit path is
            # missing. Keep the opportunity so evaluation cannot jump over it
            # and choose a later winner from the same session.
            signal = signal_candles[instrument.instrument_key, event["time"]]
            values = feature_values(event["row"], {"setup_family": event["setup_family"],
                **option_quality_context(premium=float(signal.close), volume=signal.volume,
                    open_interest=signal.open_interest, expiry=instrument.expiry, strike=instrument.strike,
                    spot=float(event["row"].close), trading_day=event["day"])})
            if np.isfinite(np.array([values[name] for name in required_features], dtype=float)).all():
                records.append({**{name: values[name] for name in required_features},
                    "timestamp": event["time"], "trading_day": event["day"], "direction": event["direction"],
                    "setup_family": event["setup_family"], "instrument_key": instrument.instrument_key,
                    "moneyness": moneyness, "label_status": "CENSORED", "label": None,
                    "net_pnl": None, "stress_net_pnl": None, "net_return": None,
                    "exit_reason": "INCOMPLETE_PATH", "exit_timestamp": None, **context})
    diagnostics = {
        "spot_candles": len(spot_rows),
        "futures_candles": len(futures),
        "option_contracts": len(option_instruments),
        "candidate_setups": len(events),
        "usable_option_outcomes": sum(item.get("label_status") == "COMPLETE" for item in records),
        "unlabelled_opportunities": sum(item.get("label_status") == "CENSORED" for item in records),
        "dropped_spot_days": dropped_spot,
        "dropped_future_days": dropped_futures,
        "reward_risk": str(reward_risk),
        "premium_stop_pct": str(premium_stop_pct),
        "setup_families": [item.upper() for item in setup_families],
        "sampling_policy": sampling_policy,
        "strike_policy": strike_policy,
        "setup_policy": setup_policy, "causal_opportunities": causal_opportunities,
        "required_features": list(required_features),
        "moneyness_counts": dict(moneyness_counts),
        "unreplayed_runtime_filters": ["bid_ask_spread", "quote_age", "delta", "live_volume_and_oi_semantics"],
        "entry_cutoff_ist": last_entry_time.isoformat(),
        "rejections": dict(rejection_counts),
        "simulator_revision": "entry-bar-stops-exact-next-bar-contiguous-path-v2",
        "futures_alignment_gap_days": futures_alignment_gaps(
            [item.timestamp for item in spot_rows],
            [item.timestamp for item in future_rows],
        ),
    }
    return pd.DataFrame(records), diagnostics


def _split_by_day(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    days = sorted(dataset.trading_day.unique().tolist())
    if len(days) < 30:
        raise RuntimeError("Cost-aware training requires at least 30 complete trading days")
    tuning_index = max(1, int(len(days) * 0.60))
    validation_index = max(tuning_index + 1, int(len(days) * 0.80))
    # Purge one complete day on both split boundaries.
    training_days = set(days[: max(0, tuning_index - 1)])
    tuning_days = set(days[tuning_index : max(tuning_index, validation_index - 1)])
    validation_days = set(days[validation_index:])
    return (
        dataset[dataset.trading_day.isin(training_days)].copy(),
        dataset[dataset.trading_day.isin(tuning_days)].copy(),
        dataset[dataset.trading_day.isin(validation_days)].copy(),
    )


def _rank_labels(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame.groupby("trading_day").net_return.rank(pct=True, method="average") * 3
    ).round().to_numpy(dtype=float)


def _fit_ranker(frame: pd.DataFrame, rounds: int = 180, *, features: list[str] | None = None):
    import xgboost as xgb

    ordered = frame.sort_values(["trading_day", "timestamp"]).reset_index(drop=True)
    features = QUALITY_FEATURES if features is None else features
    matrix = xgb.DMatrix(
        ordered[features].to_numpy(dtype=float),
        label=_rank_labels(ordered),
        feature_names=features,
    )
    matrix.set_group(ordered.groupby("trading_day", sort=False).size().tolist())
    model = xgb.train(
        {
            "objective": "rank:pairwise",
            "eta": 0.035,
            "max_depth": 3,
            "min_child_weight": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.15,
            "reg_lambda": 2.0,
            "seed": 42,
            "nthread": 1,
        },
        matrix,
        num_boost_round=rounds,
        verbose_eval=False,
    )
    return model


def _rank_scores(model, frame: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb

    return model.predict(
        xgb.DMatrix(
            frame[QUALITY_FEATURES].to_numpy(dtype=float),
            feature_names=QUALITY_FEATURES,
        )
    )


def _fit_quality(frame: pd.DataFrame, direction: str, iterations: int = 500):
    from catboost import CatBoostClassifier

    subset = frame[frame.direction == direction]
    if len(subset) < MIN_DIRECTION_SAMPLES or subset.label.nunique() < 2:
        raise RuntimeError(
            f"{direction} has {len(subset)} usable samples or only one outcome class; "
            f"at least {MIN_DIRECTION_SAMPLES} mixed outcomes are required"
        )
    model = CatBoostClassifier(
        iterations=iterations,
        depth=5,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        l2_leaf_reg=5.0,
        random_seed=42,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
        has_time=True,
    )
    model.fit(subset[QUALITY_FEATURES], subset.label)
    return model


def _calibrate(
    quality_models: dict[str, object], ranker, frame: pd.DataFrame
) -> tuple[np.ndarray, dict[str, dict]]:
    from sklearn.linear_model import LogisticRegression

    result = np.zeros(len(frame), dtype=float)
    calibration: dict[str, dict] = {}
    for direction in ("CE", "PE"):
        positions = np.flatnonzero(frame.direction.to_numpy() == direction)
        subset = frame.iloc[positions]
        quality = quality_models[direction].predict_proba(subset[QUALITY_FEATURES])[:, 1]
        quality_logit = np.log(np.clip(quality, 1e-6, 1 - 1e-6) / np.clip(1 - quality, 1e-6, 1))
        design = quality_logit.reshape(-1, 1)
        if len(subset) >= 40 and subset.label.nunique() == 2:
            calibrator = LogisticRegression(C=1.0, random_state=42)
            calibrator.fit(design, subset.label.to_numpy(dtype=int))
            coefficient = float(calibrator.coef_[0, 0])
            if coefficient > 0:
                result[positions] = calibrator.predict_proba(design)[:, 1]
                calibration[direction] = {
                    "method": "platt",
                    "input": "logit_quality",
                    "intercept": float(calibrator.intercept_[0]),
                    "quality_coefficient": coefficient,
                    "rank_coefficient": 0.0,
                }
            else:
                result[positions] = quality
                calibration[direction] = {"method": "identity"}
        else:
            result[positions] = quality
            calibration[direction] = {"method": "identity"}
    return result, calibration


def _apply_calibration(
    quality_models: dict[str, object],
    ranker,
    frame: pd.DataFrame,
    calibration: dict[str, dict],
) -> np.ndarray:
    probabilities = np.zeros(len(frame), dtype=float)
    for direction in ("CE", "PE"):
        positions = np.flatnonzero(frame.direction.to_numpy() == direction)
        subset = frame.iloc[positions]
        quality = quality_models[direction].predict_proba(subset[QUALITY_FEATURES])[:, 1]
        values = calibration[direction]
        if values.get("method") == "identity":
            probabilities[positions] = quality
            continue
        quality_input = (
            np.log(np.clip(quality, 1e-6, 1 - 1e-6) / np.clip(1 - quality, 1e-6, 1))
            if values.get("input") == "logit_quality"
            else quality
        )
        logits = (
            values["intercept"]
            + values["quality_coefficient"] * quality_input
        )
        probabilities[positions] = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    return probabilities


def _selection_mask(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float | dict,
) -> np.ndarray:
    if isinstance(threshold, dict):
        if threshold and isinstance(next(iter(threshold.values())), dict):
            cutoffs = np.array(
                [
                    float(threshold.get(family, {}).get(direction, 1.01))
                    for family, direction in zip(frame.setup_family, frame.direction)
                ],
                dtype=float,
            )
        else:
            cutoffs = frame.direction.map(threshold).fillna(1.01).to_numpy(dtype=float)
        return probabilities >= cutoffs
    return probabilities >= threshold


def _performance(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float | dict,
) -> dict:
    mask = _selection_mask(frame, probabilities, threshold)
    selected = frame.loc[mask].copy()
    selected["probability"] = probabilities[mask]
    # Live execution takes the first passing candidate and never knows future
    # scores, so validation deliberately mirrors that causal one-trade/day rule.
    selected = selected.sort_values("timestamp").groupby("trading_day", as_index=False).first()
    pnls = selected.net_pnl.to_numpy(dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:] if len(equity) else np.array([])
    drawdown = peak - equity if len(equity) else np.array([])
    assumed_capital = 500_000.0
    if len(selected):
        month_keys = pd.to_datetime(selected.timestamp, utc=True).dt.tz_convert(
            "Asia/Kolkata"
        ).dt.strftime("%Y-%m")
        monthly = selected.groupby(month_keys).net_pnl.sum().to_dict()
    else:
        monthly = {}
    loss_streak = max_loss_streak = 0
    for pnl in pnls:
        loss_streak = loss_streak + 1 if pnl < 0 else 0
        max_loss_streak = max(max_loss_streak, loss_streak)
    return {
        "threshold": (
            {
                key: (
                    {side: round(float(cutoff), 3) for side, cutoff in value.items()}
                    if isinstance(value, dict)
                    else round(float(value), 3)
                )
                for key, value in threshold.items()
            }
            if isinstance(threshold, dict)
            else round(float(threshold), 3)
        ),
        "trades": int(len(selected)),
        "win_rate_pct": round(float((pnls > 0).mean() * 100), 2) if len(pnls) else 0.0,
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 3) if len(losses) else None,
        "net_pnl": round(float(pnls.sum()), 2),
        "double_slippage_net_pnl": round(float(selected.stress_net_pnl.sum()), 2),
        "max_drawdown": round(float(drawdown.max()), 2) if len(drawdown) else 0.0,
        "max_drawdown_pct": round(float(drawdown.max() / assumed_capital * 100), 3) if len(drawdown) else 0.0,
        "ce_trades": int((selected.direction == "CE").sum()),
        "pe_trades": int((selected.direction == "PE").sum()),
        "active_months": len(monthly),
        "positive_months": sum(value > 0 for value in monthly.values()),
        "monthly": {key: round(float(value), 2) for key, value in monthly.items()},
        "max_consecutive_losses": max_loss_streak,
    }


def _select_threshold(frame: pd.DataFrame, probabilities: np.ndarray) -> tuple[float, list[dict]]:
    curve = [_performance(frame, probabilities, value) for value in np.arange(0.50, 0.86, 0.02)]
    eligible = [
        item
        for item in curve
        if item["trades"] >= 25
        and item["win_rate_pct"] >= 55.0
        and (item["profit_factor"] or 0.0) >= 1.30
        and item["double_slippage_net_pnl"] > 0
    ]
    if eligible:
        chosen = max(eligible, key=lambda item: (item["win_rate_pct"], item["net_pnl"]))
    else:
        minimum_fallback_trades = max(
            5, min(25, int(frame.trading_day.nunique()) // 4)
        )
        viable = [item for item in curve if item["trades"] >= minimum_fallback_trades]
        chosen = max(viable, key=lambda item: item["net_pnl"] / item["trades"]) if viable else curve[0]
    return float(chosen["threshold"]), curve


def _select_direction_thresholds(
    frame: pd.DataFrame, probabilities: np.ndarray
) -> tuple[dict[str, float], dict[str, list[dict]], list[str]]:
    thresholds: dict[str, float] = {}
    curves: dict[str, list[dict]] = {}
    enabled: list[str] = []
    for direction in ("CE", "PE"):
        positions = np.flatnonzero(frame.direction.to_numpy() == direction)
        subset = frame.iloc[positions].reset_index(drop=True)
        subset_probabilities = probabilities[positions]
        curve = [
            _performance(subset, subset_probabilities, value)
            for value in np.arange(0.46, 0.71, 0.02)
        ]
        curves[direction] = curve
        eligible = [
            item
            for item in curve
            if item["trades"] >= 12
            and item["win_rate_pct"] >= 52.0
            and (item["profit_factor"] or 0.0) >= 1.20
            and item["double_slippage_net_pnl"] > 0
        ]
        if eligible:
            chosen = max(
                eligible, key=lambda item: (item["win_rate_pct"], item["net_pnl"])
            )
            enabled.append(direction)
        else:
            # Abstain completely when a direction has no profitable,
            # sufficiently-sized tuning threshold. A lucky validation patch
            # must never rescue a direction that failed model selection.
            chosen = {"threshold": 1.01}
        thresholds[direction] = float(chosen["threshold"])
    return thresholds, curves, enabled


def _select_setup_thresholds(
    frame: pd.DataFrame, probabilities: np.ndarray
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, list[dict]]], list[str]]:
    """Select each family/direction on tuning data and disable weak combinations."""
    thresholds: dict[str, dict[str, float]] = {}
    curves: dict[str, dict[str, list[dict]]] = {}
    enabled: list[str] = []
    families = [str(item) for item in frame.setup_family.drop_duplicates().tolist()]
    for family in families:
        thresholds[family] = {}
        curves[family] = {}
        for direction in ("CE", "PE"):
            mask = (frame.setup_family.to_numpy() == family) & (
                frame.direction.to_numpy() == direction
            )
            subset = frame.loc[mask].reset_index(drop=True)
            subset_probabilities = probabilities[mask]
            curve = [
                _performance(subset, subset_probabilities, value)
                for value in np.arange(0.46, 0.71, 0.02)
            ]
            curves[family][direction] = curve
            eligible = [
                item
                for item in curve
                if item["trades"] >= 10
                and item["win_rate_pct"] >= 52.0
                and (item["profit_factor"] or 0.0) >= 1.20
                and item["double_slippage_net_pnl"] > 0
            ]
            if eligible:
                chosen = max(
                    eligible,
                    key=lambda item: (item["win_rate_pct"], item["net_pnl"]),
                )
                enabled.append(f"{family}:{direction}")
                thresholds[family][direction] = float(chosen["threshold"])
            else:
                thresholds[family][direction] = 1.01
    return thresholds, curves, enabled


def _walk_forward_validation(dataset: pd.DataFrame) -> list[dict]:
    """Run expanding, purged train/calibrate/test folds without future leakage."""
    days = sorted(dataset.trading_day.unique().tolist())
    folds: list[dict] = []
    for fraction in (0.45, 0.60, 0.75):
        validation_start = int(len(days) * fraction)
        validation_end = min(len(days), validation_start + max(20, int(len(days) * 0.10)))
        calibration_start = int(validation_start * 0.80)
        development_days = set(days[: max(0, calibration_start - 1)])
        calibration_days = set(days[calibration_start : max(calibration_start, validation_start - 1)])
        validation_days = set(days[validation_start:validation_end])
        development = dataset[dataset.trading_day.isin(development_days)].copy()
        calibration_frame = dataset[dataset.trading_day.isin(calibration_days)].copy()
        validation = dataset[dataset.trading_day.isin(validation_days)].copy()
        if len(development) < 300 or len(calibration_frame) < 50 or len(validation) < 50:
            continue
        try:
            quality = {
                direction: _fit_quality(development, direction, iterations=300)
                for direction in ("CE", "PE")
            }
            ranker = _fit_ranker(development, rounds=120)
            calibration_probabilities, calibration = _calibrate(
                quality, ranker, calibration_frame
            )
            thresholds, _curves, enabled = _select_setup_thresholds(
                calibration_frame, calibration_probabilities
            )
            validation_probabilities = _apply_calibration(
                quality, ranker, validation, calibration
            )
            performance = _performance(validation, validation_probabilities, thresholds)
            performance.update(
                {
                    "development_start": min(development_days).isoformat(),
                    "development_end": max(development_days).isoformat(),
                    "calibration_start": min(calibration_days).isoformat(),
                    "calibration_end": max(calibration_days).isoformat(),
                    "validation_start": min(validation_days).isoformat(),
                    "validation_end": max(validation_days).isoformat(),
                    "development_samples": len(development),
                    "calibration_samples": len(calibration_frame),
                    "validation_samples": len(validation),
                    "enabled_setups": enabled,
                    "purged_days": 2,
                }
            )
            folds.append(performance)
        except RuntimeError as exc:
            folds.append(
                {
                    "validation_start": min(validation_days).isoformat(),
                    "validation_end": max(validation_days).isoformat(),
                    "error": str(exc),
                }
            )
    return folds


def train_option_quality_model(
    db_url: str, model_path: Path, indices: list[str], start_date: date,
    end_date: date, provider: str = "GROWW",
) -> dict:
    """New standard runs use the versioned, four-period aligned contract."""
    from .aligned_training import train_aligned_model
    return train_aligned_model(db_url, model_path, indices, start_date, end_date, provider)


def train_legacy_option_quality_model(
    db_url: str,
    model_path: Path,
    indices: list[str],
    start_date: date,
    end_date: date,
    provider: str = "GROWW",
) -> dict:
    """Legacy research protocol; the application uses train_option_quality_model."""
    if start_date >= end_date:
        raise ValueError("Training end date must be after the start date")
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False, "timeout": 30}
        if db_url.startswith("sqlite")
        else {},
    )
    samples: list[pd.DataFrame] = []
    coverage: dict[str, dict] = {}
    try:
        with Session(engine) as session:
            for code in indices:
                frame, diagnostics = build_option_quality_samples(
                    session,
                    provider=provider,
                    code=code,
                    start_date=start_date,
                    end_date=end_date,
                )
                coverage[code] = diagnostics
                if not frame.empty:
                    frame["underlying_code"] = code
                    samples.append(frame)
    finally:
        engine.dispose()
    if not samples:
        raise RuntimeError(
            "No cost-aware option outcomes were available. Verified spot, futures, "
            "option candles and contract lot sizes are required for cost-aware training."
        )
    dataset = pd.concat(samples, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if len(dataset) < 500:
        raise RuntimeError(f"Only {len(dataset)} option outcomes found; at least 500 are required")
    training, tuning, validation = _split_by_day(dataset)
    if min(len(training), len(tuning), len(validation)) < 100:
        raise RuntimeError("Chronological train/tune/validation partitions are too small")

    development_quality = {
        direction: _fit_quality(training, direction) for direction in ("CE", "PE")
    }
    development_ranker = _fit_ranker(training)
    tuning_probabilities, calibration = _calibrate(
        development_quality, development_ranker, tuning
    )
    setup_thresholds, setup_tuning_curves, enabled_setups = _select_setup_thresholds(
        tuning, tuning_probabilities
    )
    enabled_directions = sorted({item.split(":", 1)[1] for item in enabled_setups})
    direction_thresholds = {
        direction: min(
            (
                cutoffs[direction]
                for cutoffs in setup_thresholds.values()
                if cutoffs.get(direction, 1.01) <= 1.0
            ),
            default=1.01,
        )
        for direction in ("CE", "PE")
    }
    enabled_cutoffs = [
        cutoff
        for cutoffs in setup_thresholds.values()
        for cutoff in cutoffs.values()
        if cutoff <= 1.0
    ]
    threshold = min(enabled_cutoffs, default=1.01)
    tuning_curve = [_performance(tuning, tuning_probabilities, setup_thresholds)]
    validation_probabilities = _apply_calibration(
        development_quality, development_ranker, validation, calibration
    )
    validation_metrics = _performance(
        validation, validation_probabilities, setup_thresholds
    )
    walk_forward = _walk_forward_validation(dataset)

    # Keep the exact models used to learn calibration and produce the locked
    # validation result.  Refitting here would change score distributions and
    # silently invalidate both the calibrated threshold and validation claim.
    quality_models = development_quality
    ranker = development_ranker
    blockers: list[str] = []
    for code, diagnostics in coverage.items():
        gaps = diagnostics.get("futures_alignment_gap_days", [])
        if gaps:
            blockers.append(
                f"{code} has {len(gaps)} sessions with incomplete futures alignment "
                f"({gaps[0]} to {gaps[-1]})"
            )
    if not enabled_setups:
        blockers.append("tuning validation could not establish any robust setup family/direction")
    if validation_metrics["trades"] < 25:
        blockers.append("fewer than 25 locked validation trades")
    if validation_metrics["win_rate_pct"] < 55.0:
        blockers.append("locked validation win rate is below 55%")
    if (validation_metrics["profit_factor"] or 0.0) < 1.30:
        blockers.append("locked validation profit factor is below 1.30")
    if validation_metrics["net_pnl"] <= 0:
        blockers.append("locked validation net P&L is not positive")
    if validation_metrics["double_slippage_net_pnl"] <= 0:
        blockers.append("locked validation fails double-slippage stress")
    if validation_metrics["max_drawdown_pct"] > 5.0:
        blockers.append("locked validation drawdown exceeds 5% of assumed capital")
    for direction in enabled_directions:
        if validation_metrics[f"{direction.lower()}_trades"] < 5:
            blockers.append(
                f"locked validation lacks at least five {direction} trades"
            )
    if validation_metrics["active_months"] < 6:
        blockers.append("locked validation trades occur in fewer than six months")
    elif validation_metrics["positive_months"] / validation_metrics["active_months"] < 0.60:
        blockers.append("fewer than 60% of active validation months are profitable")
    stable_folds = sum(
        1
        for fold in walk_forward
        if not fold.get("error")
        and fold.get("trades", 0) >= 10
        and fold.get("net_pnl", 0) > 0
        and (fold.get("profit_factor") or 0) >= 1.20
        and fold.get("double_slippage_net_pnl", 0) > 0
    )
    if stable_folds < 2:
        blockers.append("fewer than two profitable walk-forward folds passed cost stress")
    blockers.append(
        "V6 is research-only until a genuinely forward paper-shadow period passes"
    )

    trained_at = datetime.now(UTC)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    ce_path = model_path.with_name(f"{model_path.stem}.ce.cbm")
    pe_path = model_path.with_name(f"{model_path.stem}.pe.cbm")
    ranker_path = model_path.with_name(f"{model_path.stem}.ranker.json")
    quality_models["CE"].save_model(str(ce_path))
    quality_models["PE"].save_model(str(pe_path))
    ranker.save_model(ranker_path)
    manifest = {
        "quality_models": {"CE": ce_path.name, "PE": pe_path.name},
        "ranker": ranker_path.name,
    }
    metadata = {
        "version": f"quality-v6-{trained_at.strftime('%Y%m%dT%H%M%SZ')}",
        "trained_at": trained_at.isoformat(),
        "training_start": start_date.isoformat(),
        "training_end": end_date.isoformat(),
        "actual_sample_start": str(dataset.trading_day.min()),
        "actual_sample_end": str(dataset.trading_day.max()),
        "indices": indices,
        "provider": provider,
        "training_data_provider": provider,
        "feature_schema": FEATURE_SCHEMA_V6,
        "compatible_live_providers": [provider],
        "features": QUALITY_FEATURES,
        "classes": ["NO_TRADE", "TRADE"],
        "samples": len(dataset),
        "training_samples": len(training),
        "tuning_samples": len(tuning),
        "validation_samples": len(validation),
        "deployable_fit_samples": len(training),
        "class_counts": dict(Counter(dataset.label.tolist())),
        "direction_counts": dict(Counter(dataset.direction.tolist())),
        "setup_families": list(SETUP_FAMILIES),
        "setup_family_counts": dict(Counter(dataset.setup_family.tolist())),
        "decision_threshold": threshold,
        "direction_thresholds": direction_thresholds,
        "enabled_directions": enabled_directions,
        "disabled_directions": sorted({"CE", "PE"} - set(enabled_directions)),
        "setup_thresholds": setup_thresholds,
        "enabled_setups": enabled_setups,
        "calibration": calibration,
        "tuning_curve": tuning_curve,
        "direction_tuning_curves": setup_tuning_curves,
        "locked_validation": validation_metrics,
        "walk_forward_validation": walk_forward,
        "stable_walk_forward_folds": stable_folds,
        "balanced_accuracy": round(validation_metrics["win_rate_pct"] / 100.0, 4),
        "validation_confidence": {
            "above_decision_threshold_percent": round(
                float(
                    np.mean(
                        _selection_mask(
                            validation, validation_probabilities, setup_thresholds
                        )
                    )
                    * 100
                ), 2
            )
        },
        "label_method": "Actual next-bar option net P&L after broker charges and 0.05% two-sided slippage",
        "slippage_pct": str(SLIPPAGE_PCT),
        "stress_slippage_pct": str(SLIPPAGE_PCT * 2),
        "daily_trade_limit": 1,
        "activation_eligible": not blockers,
        "activation_blockers": blockers,
        "coverage": coverage,
        "bundle_files": [ce_path.name, pe_path.name, ranker_path.name],
        "uses_live_context": True,
        "research_only": True,
        "requires_forward_shadow": True,
    }
    temporary_manifest = model_path.with_name(f".{model_path.name}.tmp")
    temporary_metadata = model_metadata_path(model_path).with_name(
        f".{model_metadata_path(model_path).name}.tmp"
    )
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    os.replace(temporary_manifest, model_path)
    os.replace(temporary_metadata, model_metadata_path(model_path))
    return metadata


def evaluate_option_quality_holdout(
    db_url: str,
    model_path: Path,
    indices: list[str],
    start_date: date,
    end_date: date,
    provider: str = "GROWW",
) -> dict:
    """Evaluate a frozen cost-aware bundle without fitting or selecting anything."""
    from .strategy import RegimeModel

    metadata = json.loads(model_metadata_path(model_path).read_text())
    if metadata.get("feature_schema") not in {FEATURE_SCHEMA_V4, FEATURE_SCHEMA_V5, FEATURE_SCHEMA_V6}:
        raise ValueError("Holdout evaluation requires a cost-aware option model bundle")
    if start_date <= date.fromisoformat(metadata["training_end"]) or start_date > end_date:
        raise ValueError("Holdout must follow the candidate's complete research date range")
    model = RegimeModel(model_path)
    if not model.available:
        raise ValueError("Frozen candidate cannot be loaded")
    contract = metadata.get("training_contract", {})
    if contract and (provider not in metadata.get("compatible_live_providers", [])
            or not set(indices).issubset(metadata.get("indices", []))):
        raise ValueError("Holdout provider/index is outside the frozen training scope")

    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False, "timeout": 30}
        if db_url.startswith("sqlite")
        else {},
    )
    samples: list[pd.DataFrame] = []
    coverage: dict[str, dict] = {}
    try:
        with Session(engine) as session:
            for code in indices:
                frame, diagnostics = build_option_quality_samples(
                    session,
                    provider=provider,
                    code=code,
                    start_date=start_date,
                    end_date=end_date,
                    setup_families=tuple(metadata.get("setup_families", ("ORB",))),
                    sampling_policy=contract.get("sampling_policy", "first_per_day"),
                    strike_policy=contract.get("strike_policy", "ATM_ONLY"),
                    setup_policy=contract.get("setup_policy", "independent"),
                    causal_opportunities=contract.get("causal_opportunities", False),
                    retain_unlabelled=contract.get("retain_unlabelled", False),
                    required_features=model.required_features,
                )
                coverage[code] = diagnostics
                if not frame.empty:
                    frame["underlying_code"] = code
                    samples.append(frame)
    finally:
        engine.dispose()
    if not samples:
        raise RuntimeError("No usable option outcomes were found in the holdout period")
    dataset = pd.concat(samples, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    probabilities = model.quality_probabilities(dataset)
    thresholds = metadata.get("setup_thresholds") or {
        key: float(value)
        for key, value in metadata["direction_thresholds"].items()
    }
    if contract:
        from .aligned_training import daily_metrics
        metrics = daily_metrics(dataset, probabilities, metadata["direction_thresholds"])
    else:
        metrics = _performance(dataset, probabilities, thresholds)
    metrics.update(
        {
            "samples": len(dataset),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "provider": provider,
            "coverage": coverage,
            "evaluated_at": datetime.now(UTC).isoformat(),
            "frozen_model_version": metadata["version"],
            "training_contract": contract,
        }
    )
    return metrics
