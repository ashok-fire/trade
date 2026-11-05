"""Immutable, canonical evidence for reproducing a signal-time decision."""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import select

from .models import DecisionSnapshot

PROTOCOL = "decision-snapshot-v2"

SNAPSHOT_SETTING_KEYS = (
    "paper_capital", "risk_per_trade_pct", "daily_loss_limit_pct",
    "max_trades_per_day", "max_consecutive_losses", "max_open_positions",
    "live_max_lots", "premium_emergency_stop_pct", "reward_risk",
    "max_spread_pct", "max_quote_age_seconds", "delta_min", "delta_max",
    "paper_slippage_pct", "square_off_time", "enabled_indices", "active_broker",
)


def _utc(stamp: datetime) -> datetime:
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)


def canonical(value: Any) -> Any:
    """Convert SQLAlchemy/pandas/domain values into strict, stable JSON values."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [canonical(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        stamp = value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
        return _utc(stamp).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def encode(value: Any) -> str:
    return json.dumps(canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(encode(value).encode()).hexdigest()


def candle_fingerprint(rows: Iterable[Any]) -> dict[str, Any]:
    values = [{
        "id": row.id,
        "instrument_key": row.instrument_key,
        "timestamp": row.timestamp,
        "open": row.open,
        "high": row.high,
        "low": row.low,
        "close": row.close,
        "volume": row.volume,
        "open_interest": row.open_interest,
        "source_verified": row.source_verified,
        "data_role": row.data_role,
    } for row in rows]
    return {
        "count": len(values),
        "first_id": values[0]["id"] if values else None,
        "last_id": values[-1]["id"] if values else None,
        "first_timestamp": values[0]["timestamp"] if values else None,
        "last_timestamp": values[-1]["timestamp"] if values else None,
        "sha256": digest(values),
    }


def row_payload(row: pd.Series) -> dict[str, Any]:
    return {str(key): canonical(value) for key, value in row.items()}


def settings_payload(settings: dict[str, str]) -> dict[str, str | None]:
    return {key: settings.get(key) for key in SNAPSHOT_SETTING_KEYS}


def record_decision_snapshot(
    session,
    *,
    provider: str,
    mode: str,
    underlying_code: str,
    boundary: datetime,
    stage: str,
    model,
    settings: dict[str, str],
    market_evidence: dict[str, Any],
    row: pd.Series,
    context: dict[str, Any] | None = None,
    decision: Any | None = None,
    option_evidence: dict[str, Any] | None = None,
    observed_at: datetime | None = None,
) -> DecisionSnapshot:
    """Insert a content-addressed snapshot; identical calls are idempotent."""
    boundary = _utc(boundary)
    model_info = {
        "version": model.metadata.get("version", "unknown"),
        "feature_schema": model.metadata.get("feature_schema"),
        "required_features": getattr(model, "required_features", model.metadata.get("features", [])),
        "setup_families": model.metadata.get("setup_families", ["ORB"]),
    }
    inputs = {
        "protocol_version": PROTOCOL,
        "provider": provider,
        "mode": mode,
        "underlying_code": underlying_code,
        "boundary": boundary,
        "stage": stage,
        "model": model_info,
        "settings": settings_payload(settings),
        "market": market_evidence,
        "row": row_payload(row),
        "context": context or {},
        "option": option_evidence,
    }
    decision_payload = None if decision is None else {
        "regime": decision.regime,
        "confidence": decision.confidence,
        "direction": decision.direction,
        "reason": decision.reason,
        "features": decision.features,
    }
    payload = {**inputs, "decision": decision_payload}
    input_hash = digest(inputs)
    payload_hash = digest(payload)
    identity = f"{provider}:{mode}:{underlying_code}:{boundary.isoformat()}:{stage}:{model_info['version']}:{input_hash}"
    snapshot_id = uuid.uuid5(uuid.NAMESPACE_URL, identity).hex
    existing = session.get(DecisionSnapshot, snapshot_id)
    if existing is not None:
        if existing.payload_hash != payload_hash:
            raise ValueError("Decision snapshot identity collision")
        return existing
    existing = session.scalar(select(DecisionSnapshot).where(
        DecisionSnapshot.provider == provider,
        DecisionSnapshot.mode == mode,
        DecisionSnapshot.underlying_code == underlying_code,
        DecisionSnapshot.boundary == boundary,
        DecisionSnapshot.stage == stage,
        DecisionSnapshot.model_version == str(model_info["version"]),
        DecisionSnapshot.input_hash == input_hash,
    ))
    if existing is not None:
        return existing
    snapshot = DecisionSnapshot(
        id=snapshot_id,
        protocol_version=PROTOCOL,
        provider=provider,
        mode=mode,
        underlying_code=underlying_code,
        boundary=boundary,
        stage=stage,
        model_version=str(model_info["version"]),
        input_hash=input_hash,
        payload_hash=payload_hash,
        spot_candle_id=market_evidence.get("spot", {}).get("last_id"),
        observed_at=_utc(observed_at or datetime.now(UTC)),
        payload_json=encode(payload),
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def compare_snapshot(snapshot: DecisionSnapshot, market_evidence: dict[str, Any], row: pd.Series) -> dict[str, Any]:
    """Describe reconstruction drift without modifying the recorded snapshot."""
    stored = json.loads(snapshot.payload_json)
    rebuilt_row = row_payload(row)
    stored_row = stored.get("row", {})
    keys = sorted(set(stored_row) | set(rebuilt_row))
    changed = {key: {"stored": stored_row.get(key), "rebuilt": rebuilt_row.get(key)}
               for key in keys if stored_row.get(key) != rebuilt_row.get(key)}
    stored_market = stored.get("market", {})
    return {
        "snapshot_id": snapshot.id,
        "market_hash_match": digest(stored_market) == digest(market_evidence),
        "spot_hash_match": stored_market.get("spot", {}).get("sha256") == market_evidence.get("spot", {}).get("sha256"),
        "futures_hash_match": stored_market.get("futures", {}).get("sha256") == market_evidence.get("futures", {}).get("sha256"),
        "row_match": not changed,
        "changed_fields": changed,
    }
