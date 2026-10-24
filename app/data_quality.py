from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import Candle, Instrument


@dataclass(frozen=True)
class DataReadiness:
    provider: str
    underlying_code: str
    spot_count: int
    futures_count: int
    option_count: int
    aligned_count: int
    alignment_percent: float
    zero_volume_percent: float
    missing_intervals: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    verified_contracts: int
    pending_contracts: int
    training_ready: bool
    backtest_ready: bool
    reason: str


def _utc_naive(value: datetime) -> datetime:
    return (value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)).replace(tzinfo=None)


def futures_alignment_gaps(spot_values, future_values) -> list[str]:
    """Find observed spot sessions with less than 90% futures alignment.

    Use actual regular-session spot timestamps, so holidays and pre-open
    records do not become false missing sessions. Duplicate future contracts
    at the same timestamp count only once.
    """
    zone = ZoneInfo("Asia/Kolkata")
    expected = {}
    for value in spot_values:
        stamp = _utc_naive(value)
        local = stamp.replace(tzinfo=UTC).astimezone(zone)
        if time(9, 15) <= local.time() < time(15, 30):
            expected[stamp] = local.date().isoformat()
    futures = {_utc_naive(value) for value in future_values}
    counts = Counter(expected.values())
    matched = Counter(day for stamp, day in expected.items() if stamp in futures)
    return sorted(day for day, total in counts.items() if matched[day] / total < 0.90)


def _missing_intervals(values: list[datetime]) -> int:
    if len(values) < 2:
        return 0
    normalized = sorted(_utc_naive(value) for value in values)
    missing = 0
    for previous, current in zip(normalized, normalized[1:]):
        if previous.date() != current.date():
            continue
        gap = int((current - previous).total_seconds() // 300)
        if gap > 1:
            missing += gap - 1
    return missing


def research_readiness(session: Session) -> list[DataReadiness]:
    rows: list[DataReadiness] = []
    for provider in ("UPSTOX", "GROWW"):
        for code in ("NIFTY", "BANK_NIFTY", "SENSEX"):
            base = (Candle.provider == provider, Candle.underlying_code == code, Candle.source_verified.is_(True))
            spot_values = session.scalars(
                select(Candle.timestamp)
                .where(*base, Candle.data_role == "SPOT", Candle.timeframe == "5minute")
                .order_by(Candle.timestamp)
            ).all()
            future_values = session.scalars(
                select(Candle.timestamp)
                .where(*base, Candle.data_role == "FUTURE", Candle.timeframe == "5minute")
            ).all()
            option_count = session.scalar(
                select(func.count()).select_from(Candle).where(*base, Candle.data_role == "OPTION")
            ) or 0
            zero_volume = session.scalar(
                select(func.count()).select_from(Candle).where(
                    *base,
                    Candle.data_role == "SPOT",
                    Candle.timeframe == "5minute",
                    Candle.volume == 0,
                )
            ) or 0
            verified_contracts = session.scalar(
                select(func.count()).select_from(Instrument).where(
                    Instrument.provider == provider,
                    Instrument.underlying_code == code,
                    Instrument.instrument_kind == "OPTION",
                    Instrument.metadata_verified.is_(True),
                    Instrument.lot_size > 0,
                )
            ) or 0
            pending_contracts = session.scalar(
                select(func.count()).select_from(Instrument).where(
                    Instrument.provider == provider,
                    Instrument.underlying_code == code,
                    Instrument.instrument_kind == "OPTION",
                    Instrument.metadata_verified.is_(False),
                )
            ) or 0
            spot_stamps = {_utc_naive(value) for value in spot_values}
            future_stamps = {_utc_naive(value) for value in future_values}
            aligned = len(spot_stamps & future_stamps)
            gap_days = futures_alignment_gaps(spot_values, future_values)
            training_ready = aligned >= 500 and not gap_days
            backtest_ready = training_ready and option_count > 0 and verified_contracts > 0
            reasons = []
            if len(spot_values) < 500:
                reasons.append("spot history below 500 candles")
            if aligned < 500:
                reasons.append("aligned futures history below 500 candles")
            if option_count == 0:
                reasons.append("no option candles")
            if verified_contracts == 0:
                reasons.append("no verified option lot metadata")
            if gap_days:
                reasons.append(
                    f"{len(gap_days)} spot sessions have less than 90% futures coverage "
                    f"({gap_days[0]} to {gap_days[-1]}); repair gaps or select a complete period"
                )
            rows.append(
                DataReadiness(
                    provider=provider,
                    underlying_code=code,
                    spot_count=len(spot_values),
                    futures_count=len(future_values),
                    option_count=int(option_count),
                    aligned_count=aligned,
                    alignment_percent=round(aligned / len(spot_stamps) * 100, 1) if spot_stamps else 0.0,
                    zero_volume_percent=round(int(zero_volume) / len(spot_values) * 100, 1) if spot_values else 0.0,
                    missing_intervals=_missing_intervals(list(spot_values)),
                    first_timestamp=min(spot_values) if spot_values else None,
                    last_timestamp=max(spot_values) if spot_values else None,
                    verified_contracts=int(verified_contracts),
                    pending_contracts=int(pending_contracts),
                    training_ready=training_ready,
                    backtest_ready=backtest_ready,
                    reason="Ready" if backtest_ready else "; ".join(reasons),
                )
            )
    return rows
