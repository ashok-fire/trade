"""The same causal market frame for execution and independent observations."""
from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy import select

from .models import Candle, Instrument
from .strategy import with_futures_features
from .training import front_future_frame


def market_frame_for_boundary(session, provider: str, code: str, boundary: datetime):
    boundary = boundary.replace(tzinfo=UTC) if boundary.tzinfo is None else boundary.astimezone(UTC)
    rows = session.scalars(select(Candle).where(
        Candle.provider == provider, Candle.underlying_code == code,
        Candle.data_role == "SPOT", Candle.timeframe == "5minute",
        Candle.timestamp < boundary,
    ).order_by(Candle.timestamp.desc()).limit(1500)).all()[::-1]
    if len(rows) < 50:
        return None
    stamp = rows[-1].timestamp
    stamp = stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)
    if boundary - stamp != timedelta(minutes=5):
        return None
    future_rows = session.scalars(select(Candle).where(
        Candle.provider == provider, Candle.underlying_code == code,
        Candle.data_role == "FUTURE", Candle.timeframe == "5minute",
        Candle.timestamp >= rows[0].timestamp, Candle.timestamp < boundary,
    ).order_by(Candle.timestamp)).all()
    instruments = session.scalars(select(Instrument).where(
        Instrument.provider == provider, Instrument.underlying_code == code,
        Instrument.instrument_kind == "FUTURE",
    )).all()
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
    return frame, rows[-1]
