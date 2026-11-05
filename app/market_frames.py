"""The same causal, verified market frame for execution and research."""
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy import select

from .models import Candle, Instrument
from .decision_snapshot import candle_fingerprint, digest
from .strategy import with_futures_features
from .training import front_future_frame


@dataclass(frozen=True)
class MarketFrameResult:
    frame: pd.DataFrame
    candle: Candle
    evidence: dict

    def __iter__(self):
        # Preserve the established two-value unpacking API.
        yield self.frame
        yield self.candle

    def __getitem__(self, index):
        return (self.frame, self.candle)[index]


def market_frame_for_boundary(session, provider: str, code: str, boundary: datetime):
    boundary = boundary.replace(tzinfo=UTC) if boundary.tzinfo is None else boundary.astimezone(UTC)
    rows = session.scalars(select(Candle).where(
        Candle.provider == provider, Candle.underlying_code == code,
        Candle.data_role == "SPOT", Candle.timeframe == "5minute",
        Candle.source_verified.is_(True),
        Candle.timestamp < boundary,
    ).order_by(Candle.timestamp.desc(), Candle.id.desc()).limit(1500)).all()[::-1]
    if len(rows) < 50:
        return None
    stamp = rows[-1].timestamp
    stamp = stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)
    if boundary - stamp != timedelta(minutes=5):
        return None
    future_rows = session.scalars(select(Candle).where(
        Candle.provider == provider, Candle.underlying_code == code,
        Candle.data_role == "FUTURE", Candle.timeframe == "5minute",
        Candle.source_verified.is_(True),
        Candle.timestamp >= rows[0].timestamp, Candle.timestamp < boundary,
    ).order_by(Candle.timestamp, Candle.instrument_key, Candle.id)).all()
    instruments = session.scalars(select(Instrument).where(
        Instrument.provider == provider, Instrument.underlying_code == code,
        Instrument.instrument_kind == "FUTURE",
        Instrument.metadata_verified.is_(True),
    ).order_by(Instrument.expiry, Instrument.instrument_key, Instrument.valid_from)).all()
    futures = front_future_frame(future_rows, instruments)
    if len(futures) < 50:
        return None
    if pd.Timestamp(futures.iloc[-1].timestamp) != pd.Timestamp(boundary - timedelta(minutes=5)):
        return None
    frame = with_futures_features(pd.DataFrame([
        {"timestamp": item.timestamp, "open": float(item.open), "high": float(item.high),
         "low": float(item.low), "close": float(item.close), "volume": item.volume}
        for item in rows
    ]), futures)
    if frame.empty or pd.isna(frame.iloc[-1].futures_close) or pd.isna(frame.iloc[-1].futures_vwap):
        return None
    evidence = {
        "protocol_version": "market-frame-v2",
        "provider": provider,
        "underlying_code": code,
        "boundary": boundary,
        "window_limit": 1500,
        "verified_only": True,
        "spot": candle_fingerprint(rows),
        "futures": candle_fingerprint(future_rows),
        "front_future": {
            "count": len(futures),
            "sha256": digest(futures.to_dict(orient="records")),
        },
    }
    return MarketFrameResult(frame=frame, candle=rows[-1], evidence=evidence)
