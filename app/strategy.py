from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "vwap_distance", "ema9_20_distance", "ema20_50_distance", "atr_pct", "rsi", "volume_ratio",
    "previous_day_high_distance", "previous_day_low_distance", "previous_day_close_distance",
    "india_vix", "futures_basis_pct", "pcr", "minutes_from_open",
    "spot_vwap_distance", "futures_vwap_distance", "futures_volume_ratio",
    "spot_return_1", "spot_return_3", "spot_return_6", "futures_return_1",
    "futures_return_3", "futures_atr_pct", "opening_range_high_distance",
    "opening_range_low_distance", "gap_pct", "time_sin", "time_cos", "setup_direction",
]

# The first locally-trained model deliberately uses only fields that can be
# reconstructed from historical candles. Live-only context can be added to a
# later model by including those fields in its metadata.
PRICE_FEATURES = [
    "spot_vwap_distance", "futures_vwap_distance", "futures_basis_pct",
    "spot_return_1", "spot_return_3", "spot_return_6", "futures_return_1",
    "futures_return_3", "futures_atr_pct", "futures_volume_ratio",
    "ema9_20_distance", "ema20_50_distance", "atr_pct", "rsi",
    "previous_day_high_distance", "previous_day_low_distance", "previous_day_close_distance",
    "opening_range_high_distance", "opening_range_low_distance", "gap_pct", "time_sin", "time_cos",
    "setup_direction",
]

# V4 is a trade-quality model.  It keeps the causal spot/futures setup features
# and adds only option fields that are observable when the signal is made.
# Bid/ask, IV and delta are intentionally excluded because the historical
# provider does not populate them consistently; training and live inference
# therefore see the same feature contract.
OPTION_META_FEATURES = [
    "option_premium_pct",
    "option_volume_log",
    "option_oi_log",
    "days_to_expiry",
    "moneyness_pct",
]
SETUP_FAMILIES = ("ORB", "PULLBACK", "FAILED_BREAKOUT", "VWAP_RECLAIM")
SETUP_FEATURES = [
    "setup_orb",
    "setup_pullback",
    "setup_failed_breakout",
    "setup_vwap_reclaim",
]
REGIME_STRENGTH_FEATURES = [
    "day_open_distance",
    "opening_range_width_pct",
    "trend_atr_strength",
    "realized_vol_6",
    "rsi_change_3",
    "atr_change_3",
    "basis_change_3",
    "return_spread_3",
    "weekday_sin",
    "weekday_cos",
]
QUALITY_FEATURES_V4 = PRICE_FEATURES + OPTION_META_FEATURES
QUALITY_FEATURES_V5 = QUALITY_FEATURES_V4 + REGIME_STRENGTH_FEATURES
QUALITY_FEATURES = QUALITY_FEATURES_V5 + SETUP_FEATURES

FEATURE_SCHEMA = "spot-futures-v3"
FEATURE_SCHEMA_V4 = "cost-aware-option-v4"
FEATURE_SCHEMA_V5 = "cost-aware-option-v5"
FEATURE_SCHEMA_V6 = "cost-aware-option-v6"
SUPPORTED_FEATURE_SCHEMAS = {
    "spot-futures-v2",
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_V4,
    FEATURE_SCHEMA_V5,
    FEATURE_SCHEMA_V6,
}
MARKET_OPEN_MINUTE = 9 * 60 + 15
MARKET_CLOSE_MINUTE = 15 * 60 + 30


@dataclass(frozen=True)
class Decision:
    regime: str
    confidence: float
    direction: str
    reason: str
    features: dict[str, float]


def regular_session(candles: pd.DataFrame) -> pd.DataFrame:
    """Return unique regular-session candles without changing stored history."""
    frame = candles.copy()
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    local = frame.timestamp.dt.tz_convert("Asia/Kolkata")
    minute = local.dt.hour * 60 + local.dt.minute
    return (
        frame.loc[(minute >= MARKET_OPEN_MINUTE) & (minute <= MARKET_CLOSE_MINUTE)]
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def indicators(candles: pd.DataFrame) -> pd.DataFrame:
    frame = regular_session(candles)
    if frame.empty:
        return frame
    frame["ema9"] = frame.close.ewm(span=9, adjust=False).mean()
    frame["ema20"] = frame.close.ewm(span=20, adjust=False).mean()
    frame["ema50"] = frame.close.ewm(span=50, adjust=False).mean()
    previous = frame.close.shift(1)
    tr = pd.concat(((frame.high - frame.low), (frame.high - previous).abs(), (frame.low - previous).abs()), axis=1).max(axis=1)
    frame["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    delta = frame.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)
    typical = (frame.high + frame.low + frame.close) / 3
    local = frame.timestamp.dt.tz_convert("Asia/Kolkata")
    day = local.dt.date
    grouped_close = frame.close.groupby(day)
    frame["return_1"] = grouped_close.pct_change(1)
    frame["return_3"] = grouped_close.pct_change(3)
    frame["return_6"] = grouped_close.pct_change(6)
    frame["vwap"] = ((typical * frame.volume).groupby(day).cumsum() / frame.volume.groupby(day).cumsum().replace(0, np.nan)).fillna(frame.close)
    frame["volume_ratio"] = frame.volume / frame.volume.rolling(20).mean().replace(0, np.nan)
    opening = local.dt.time <= pd.Timestamp("09:30").time()
    opening_high = frame.high.where(opening).groupby(day).cummax()
    opening_low = frame.low.where(opening).groupby(day).cummin()
    frame["opening_range_high"] = opening_high.groupby(day).ffill()
    frame["opening_range_low"] = opening_low.groupby(day).ffill()
    daily = frame.assign(trading_day=day).groupby("trading_day").agg(day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last")).shift(1)
    frame["previous_day_high"] = day.map(daily.day_high)
    frame["previous_day_low"] = day.map(daily.day_low)
    frame["previous_day_close"] = day.map(daily.day_close)
    first_open = frame.open.groupby(day).transform("first")
    frame["gap_pct"] = (first_open - frame.previous_day_close) / frame.previous_day_close.replace(0, np.nan)
    frame["day_open"] = first_open
    frame["opening_range_width_pct"] = (
        frame.opening_range_high - frame.opening_range_low
    ) / frame.close.replace(0, np.nan)
    frame["realized_vol_6"] = frame.groupby(day)["return_1"].transform(
        lambda values: values.rolling(6, min_periods=3).std()
    )
    frame["rsi_change_3"] = frame.groupby(day)["rsi"].diff(3)
    frame["atr_change_3"] = frame.groupby(day)["atr"].pct_change(3)
    return frame


def with_futures_features(spot_candles: pd.DataFrame, futures_candles: pd.DataFrame) -> pd.DataFrame:
    """Build causal spot indicators and attach exact-timestamp futures liquidity features."""
    spot = indicators(spot_candles)
    futures = indicators(futures_candles)
    futures["atr_pct"] = futures.atr / futures.close.replace(0, np.nan)
    futures = futures[
        ["timestamp", "close", "vwap", "volume_ratio", "return_1", "return_3", "atr_pct"]
    ].rename(
        columns={
            "close": "futures_close",
            "vwap": "futures_vwap",
            "volume_ratio": "futures_volume_ratio",
            "return_1": "futures_return_1",
            "return_3": "futures_return_3",
            "atr_pct": "futures_atr_pct",
        }
    )
    combined = spot.merge(futures, on="timestamp", how="left", validate="one_to_one")
    local_day = combined.timestamp.dt.tz_convert("Asia/Kolkata").dt.date
    combined["previous_spot_close"] = combined.groupby(local_day).close.shift(1)
    combined["previous_ema9"] = combined.groupby(local_day).ema9.shift(1)
    combined["previous_ema20"] = combined.groupby(local_day).ema20.shift(1)
    combined["previous_futures_close"] = combined.groupby(local_day).futures_close.shift(1)
    combined["previous_futures_vwap"] = combined.groupby(local_day).futures_vwap.shift(1)
    combined["feature_vwap_distance"] = (
        combined.futures_close - combined.futures_vwap
    ) / combined.futures_close.replace(0, np.nan)
    combined["feature_volume_ratio"] = combined.futures_volume_ratio
    combined["futures_basis_pct"] = (
        (combined.futures_close - combined.close) / combined.close.replace(0, np.nan) * 100
    )
    combined["spot_vwap_distance"] = (
        combined.close - combined.vwap
    ) / combined.close.replace(0, np.nan)
    combined["futures_vwap_distance"] = combined["feature_vwap_distance"]
    combined["basis_change_3"] = combined.groupby(local_day)["futures_basis_pct"].diff(3)
    combined["return_spread_3"] = combined.return_3 - combined.futures_return_3
    return combined


def setup_direction(row: pd.Series, family: str) -> str | None:
    """Return a causal CE/PE direction for one named setup family."""
    family = family.upper()
    close = float(row.close)
    futures_close = float(row.get("futures_close", np.nan))
    futures_vwap = float(row.get("futures_vwap", np.nan))
    ema9, ema20, ema50 = float(row.ema9), float(row.ema20), float(row.ema50)
    if family == "ORB":
        stamp = pd.Timestamp(row.timestamp)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
        local = stamp.tz_convert("Asia/Kolkata")
        minutes_from_open = local.hour * 60 + local.minute - MARKET_OPEN_MINUTE
        bullish = (
            minutes_from_open > 15
            and futures_close > futures_vwap
            and ema9 > ema20
            and close > float(row.opening_range_high)
        )
        bearish = (
            minutes_from_open > 15
            and futures_close < futures_vwap
            and ema9 < ema20
            and close < float(row.opening_range_low)
        )
        return "CE" if bullish else ("PE" if bearish else None)

    previous_close = float(row.get("previous_spot_close", np.nan))
    if not np.isfinite(previous_close):
        return None
    if family == "PULLBACK":
        bullish = (
            futures_close > futures_vwap
            and ema9 > ema20 > ema50
            and previous_close <= ema9
            and float(row.low) <= ema9
            and close > ema9
            and 45 <= float(row.rsi) <= 72
        )
        bearish = (
            futures_close < futures_vwap
            and ema9 < ema20 < ema50
            and previous_close >= ema9
            and float(row.high) >= ema9
            and close < ema9
            and 28 <= float(row.rsi) <= 55
        )
        return "CE" if bullish else ("PE" if bearish else None)
    if family == "FAILED_BREAKOUT":
        failed_high = (
            previous_close > float(row.opening_range_high)
            and close <= float(row.opening_range_high)
            and futures_close < futures_vwap
        )
        failed_low = (
            previous_close < float(row.opening_range_low)
            and close >= float(row.opening_range_low)
            and futures_close > futures_vwap
        )
        return "PE" if failed_high else ("CE" if failed_low else None)
    if family == "VWAP_RECLAIM":
        previous_futures_close = float(row.get("previous_futures_close", np.nan))
        previous_futures_vwap = float(row.get("previous_futures_vwap", np.nan))
        if not np.isfinite(previous_futures_close) or not np.isfinite(previous_futures_vwap):
            return None
        bullish = (
            previous_futures_close <= previous_futures_vwap
            and futures_close > futures_vwap
            and close > ema9 > ema20
            and float(row.rsi) >= 52
        )
        bearish = (
            previous_futures_close >= previous_futures_vwap
            and futures_close < futures_vwap
            and close < ema9 < ema20
            and float(row.rsi) <= 48
        )
        return "CE" if bullish else ("PE" if bearish else None)
    raise ValueError(f"Unknown setup family: {family}")


def selected_setup(
    row: pd.Series, families: list[str] | tuple[str, ...] = ("ORB",)
) -> tuple[str | None, str | None]:
    """Choose the first causal setup using the model's recorded priority."""
    for family in families:
        normalized = str(family).upper()
        direction = setup_direction(row, normalized)
        if direction:
            return normalized, direction
    return None, None


class RegimeModel:
    def __init__(self, path: Path) -> None:
        self.model = None
        self.classes = ["UPTREND", "DOWNTREND", "RANGE", "HIGH_VOLATILITY", "UNCERTAIN"]
        self.required_features = FEATURES
        self.metadata: dict = {}
        self.quality_models: dict[str, object] = {}
        self.ranker = None
        metadata_path = path.with_name(f"{path.stem}.metadata.json")
        if path.exists() and metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text())
            if self.metadata.get("feature_schema") not in SUPPORTED_FEATURE_SCHEMAS:
                return
            self.classes = list(self.metadata.get("classes") or self.classes)
            self.required_features = list(self.metadata.get("features") or FEATURES)
            if self.metadata.get("feature_schema") in {FEATURE_SCHEMA_V4, FEATURE_SCHEMA_V5, FEATURE_SCHEMA_V6}:
                from catboost import CatBoostClassifier
                import xgboost as xgb

                manifest = json.loads(path.read_text())
                for direction in ("CE", "PE"):
                    filename = manifest.get("quality_models", {}).get(direction)
                    if not filename:
                        return
                    quality = CatBoostClassifier()
                    quality.load_model(str(path.parent / filename))
                    self.quality_models[direction] = quality
                ranker_filename = manifest.get("ranker")
                if not ranker_filename:
                    return
                ranker = xgb.Booster()
                ranker.load_model(path.parent / ranker_filename)
                self.ranker = ranker
                self.model = manifest
                return
            import xgboost as xgb

            model = xgb.Booster()
            model.load_model(path)
            self.model = model

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def requires_option_context(self) -> bool:
        return self.metadata.get("feature_schema") in {FEATURE_SCHEMA_V4, FEATURE_SCHEMA_V5, FEATURE_SCHEMA_V6}

    def predict(self, values: dict[str, float]) -> tuple[str, float]:
        if not self.model:
            return "UNCERTAIN", 0.0
        matrix = np.array([[values.get(key, np.nan) for key in self.required_features]], dtype=float)
        if not np.isfinite(matrix).all():
            return "UNCERTAIN", 0.0
        if self.requires_option_context:
            direction = "CE" if float(values.get("setup_direction", 0.0)) > 0 else "PE"
            if direction not in self.metadata.get("enabled_directions", ["CE", "PE"]):
                return "UNCERTAIN", 0.0
            quality = self.quality_models.get(direction)
            if quality is None or self.ranker is None:
                return "UNCERTAIN", 0.0
            quality_probability = float(quality.predict_proba(matrix)[0, 1])
            import xgboost as xgb

            rank_score = float(
                self.ranker.predict(
                    xgb.DMatrix(matrix, feature_names=self.required_features)
                )[0]
            )
            calibration = self.metadata.get("calibration", {}).get(direction, {})
            if calibration.get("method") == "identity":
                probability = quality_probability
            else:
                quality_input = (
                    float(
                        np.log(
                            np.clip(quality_probability, 1e-6, 1 - 1e-6)
                            / np.clip(1 - quality_probability, 1e-6, 1)
                        )
                    )
                    if calibration.get("input") == "logit_quality"
                    else quality_probability
                )
                logit = (
                    float(calibration.get("intercept", 0.0))
                    + float(calibration.get("quality_coefficient", 1.0)) * quality_input
                    + float(calibration.get("rank_coefficient", 0.0)) * rank_score
                )
                probability = float(1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30))))
            return "TRADE", probability
        import xgboost as xgb

        probabilities = self.model.predict(xgb.DMatrix(matrix, feature_names=self.required_features))[0]
        temperature = max(float(self.metadata.get("probability_temperature", 1.0)), 0.1)
        if temperature != 1.0:
            logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature
            probabilities = np.exp(logits - np.max(logits))
            probabilities = probabilities / probabilities.sum()
        best = int(np.argmax(probabilities))
        if best >= len(self.classes):
            return "UNCERTAIN", 0.0
        return self.classes[best], float(probabilities[best])


def feature_values(row: pd.Series, context: dict | None = None) -> dict[str, float]:
    context = context or {}
    close = float(row.close)
    stamp = pd.Timestamp(row.timestamp)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
    stamp = stamp.tz_convert("Asia/Kolkata")
    minutes_from_open = float(stamp.hour * 60 + stamp.minute - MARKET_OPEN_MINUTE)
    angle = min(max(minutes_from_open, 0.0), float(MARKET_CLOSE_MINUTE - MARKET_OPEN_MINUTE))
    angle = angle / float(MARKET_CLOSE_MINUTE - MARKET_OPEN_MINUTE) * 2 * np.pi
    weekday_angle = float(stamp.weekday()) / 5.0 * 2 * np.pi
    futures_close = float(row.get("futures_close", close))
    futures_vwap = float(row.get("futures_vwap", row.vwap))
    requested_family = context.get("setup_family")
    families = context.get("setup_families") or ("ORB",)
    if requested_family:
        family = str(requested_family).upper()
        direction = setup_direction(row, family)
    else:
        family, direction = selected_setup(row, families)
    values = {
        "vwap_distance": float(row.get("feature_vwap_distance", (close - float(row.vwap)) / close)),
        "ema9_20_distance": (float(row.ema9) - float(row.ema20)) / close,
        "ema20_50_distance": (float(row.ema20) - float(row.ema50)) / close,
        "atr_pct": float(row.atr) / close,
        "rsi": float(row.rsi),
        "volume_ratio": float(row.get("feature_volume_ratio", row.volume_ratio)),
        "previous_day_high_distance": (close - float(row.previous_day_high)) / close,
        "previous_day_low_distance": (close - float(row.previous_day_low)) / close,
        "previous_day_close_distance": (close - float(row.previous_day_close)) / close,
        "india_vix": float(context.get("india_vix") or np.nan),
        "futures_basis_pct": float(row.get("futures_basis_pct", context.get("futures_basis_pct") or np.nan)),
        "pcr": float(context.get("pcr") or np.nan),
        "minutes_from_open": minutes_from_open,
        "spot_vwap_distance": float(row.get("spot_vwap_distance", (close - float(row.vwap)) / close)),
        "futures_vwap_distance": float(row.get("futures_vwap_distance", row.get("feature_vwap_distance", np.nan))),
        "futures_volume_ratio": float(row.get("futures_volume_ratio", row.get("feature_volume_ratio", np.nan))),
        "spot_return_1": float(row.get("return_1", np.nan)),
        "spot_return_3": float(row.get("return_3", np.nan)),
        "spot_return_6": float(row.get("return_6", np.nan)),
        "futures_return_1": float(row.get("futures_return_1", np.nan)),
        "futures_return_3": float(row.get("futures_return_3", np.nan)),
        "futures_atr_pct": float(row.get("futures_atr_pct", np.nan)),
        "opening_range_high_distance": (close - float(row.opening_range_high)) / close,
        "opening_range_low_distance": (close - float(row.opening_range_low)) / close,
        "gap_pct": float(row.get("gap_pct", np.nan)),
        "time_sin": float(np.sin(angle)),
        "time_cos": float(np.cos(angle)),
        "setup_direction": 1.0 if direction == "CE" else (-1.0 if direction == "PE" else 0.0),
        "day_open_distance": (close - float(row.get("day_open", close))) / close,
        "opening_range_width_pct": float(row.get("opening_range_width_pct", np.nan)),
        "trend_atr_strength": (
            (float(row.ema9) - float(row.ema50)) / float(row.atr)
            if float(row.atr) > 0
            else np.nan
        ),
        "realized_vol_6": float(row.get("realized_vol_6", np.nan)),
        "rsi_change_3": float(row.get("rsi_change_3", np.nan)),
        "atr_change_3": float(row.get("atr_change_3", np.nan)),
        "basis_change_3": float(row.get("basis_change_3", np.nan)),
        "return_spread_3": float(row.get("return_spread_3", np.nan)),
        "weekday_sin": float(np.sin(weekday_angle)),
        "weekday_cos": float(np.cos(weekday_angle)),
    }
    for name in OPTION_META_FEATURES:
        values[name] = float(context.get(name) if context.get(name) is not None else np.nan)
    for name in SETUP_FEATURES:
        values[name] = 0.0
    if family:
        key = f"setup_{family.lower()}"
        if key in values:
            values[key] = 1.0
    return values


def decide(row: pd.Series, context: dict[str, float | None], model: RegimeModel) -> Decision:
    context = dict(context or {})
    context.setdefault("setup_families", model.metadata.get("setup_families", ["ORB"]))
    close = float(row.close)
    features = feature_values(row, context)
    family = next(
        (name.removeprefix("setup_").upper() for name in SETUP_FEATURES if features.get(name) == 1.0),
        "ORB",
    )
    family_label = {
        "ORB": "ORB",
        "PULLBACK": "Pullback",
        "FAILED_BREAKOUT": "Failed Breakout",
        "VWAP_RECLAIM": "VWAP Reclaim",
    }.get(family, family.replace("_", " ").title())
    multi_family = model.metadata.get("feature_schema") == FEATURE_SCHEMA_V6
    no_setup_reason = (
        "No confirmed ORB, pullback, failed-breakout or VWAP-reclaim setup"
        if multi_family
        else "No confirmed opening-range breakout or breakdown"
    )
    reason_prefix = f"{family_label} " if multi_family else ""
    if model.requires_option_context and not all(
        np.isfinite(features[name]) for name in OPTION_META_FEATURES
    ):
        setup = features["setup_direction"]
        direction = "CE" if setup > 0 else ("PE" if setup < 0 else "NO_TRADE")
        return Decision(
            "PENDING_OPTION_QUALITY" if direction != "NO_TRADE" else "RANGE",
            0.0,
            direction,
            f"Option quality features required for {family_label} {direction}"
            if direction != "NO_TRADE"
            else no_setup_reason,
            features,
        )
    regime, confidence = model.predict(features)
    if not model.available:
        return Decision("UNCERTAIN", 0, "NO_TRADE", "Model unavailable", features)
    if not all(np.isfinite(features[key]) for key in model.required_features):
        return Decision("UNCERTAIN", 0, "NO_TRADE", "Required market context unavailable", features)
    bullish_setup = features["setup_direction"] > 0
    bearish_setup = features["setup_direction"] < 0
    direction = "CE" if bullish_setup else ("PE" if bearish_setup else "NO_TRADE")
    family_thresholds = model.metadata.get("setup_thresholds", {}).get(family, {})
    threshold = float(
        family_thresholds.get(
            direction,
            model.metadata.get("direction_thresholds", {}).get(
                direction, model.metadata.get("decision_threshold", 0.70)
            ),
        )
    )
    if direction == "NO_TRADE":
        return Decision(
            regime,
            confidence,
            "NO_TRADE",
            no_setup_reason,
            features,
        )
    if confidence < threshold:
        return Decision(
            regime,
            confidence,
            "NO_TRADE",
            f"{reason_prefix}{direction} quality {confidence * 100:.1f}% below {threshold * 100:.1f}% threshold",
            features,
        )
    expected_regimes = {"TRADE", "UPTREND"} if direction == "CE" else {"TRADE", "DOWNTREND"}
    if regime not in expected_regimes:
        return Decision(
            regime,
            confidence,
            "NO_TRADE",
            f"{family_label} {direction} setup not confirmed by model regime {regime}",
            features,
        )
    if regime in {"TRADE", "UPTREND"} and confidence >= threshold and bullish_setup:
        return Decision(regime, confidence, "CE", f"Confirmed {family_label} CE setup", features)
    if regime in {"TRADE", "DOWNTREND"} and confidence >= threshold and bearish_setup:
        return Decision(regime, confidence, "PE", f"Confirmed {family_label} PE setup", features)
    return Decision(regime, confidence, "NO_TRADE", "Directional setup was not confirmed", features)


def rule_only_decide(row: pd.Series) -> Decision:
    """Deterministic comparison strategy using the same causal confirmation rules."""
    features = feature_values(row)
    close = float(row.close)
    if features["setup_direction"] > 0:
        return Decision("UPTREND", 1.0, "CE", "Rule-only opening-range breakout", features)
    if features["setup_direction"] < 0:
        return Decision("DOWNTREND", 1.0, "PE", "Rule-only opening-range breakdown", features)
    return Decision("RANGE", 0.0, "NO_TRADE", "Rule-only conditions not aligned", features)

