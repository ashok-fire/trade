from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.models import Base, Candle, ResearchChunk
from app.providers import BrokerError, RetryableBrokerError
from app.research import _parse, _store_candles


def test_parse_groww_candle_accepts_null_open_interest():
    candle = _parse(["2025-09-24T10:30:00", 245.95, 246.15, 245.05, 245.6, 735060, None])

    assert candle == (
        datetime(2025, 9, 24, 5, 0, tzinfo=UTC),
        Decimal("245.95"),
        Decimal("246.15"),
        Decimal("245.05"),
        Decimal("245.6"),
        735060,
        0,
    )


def test_parse_groww_candle_rejects_non_numeric_price():
    assert _parse(["2025-09-24T10:30:00", "--", 246.15, 245.05, 245.6, 735060, None]) is None


def test_parse_groww_candle_rejects_non_finite_or_invalid_ohlc():
    assert _parse(["2025-09-24T10:30:00", "NaN", 246.15, 245.05, 245.6, 735060, 0]) is None
    assert _parse(["2025-09-24T10:30:00", 245.95, 244, 245.05, 245.6, 735060, 0]) is None


class _SplittingHistoricalClient:
    provider = "GROWW"

    def __init__(self, failed_day: date | None = None):
        self.failed_day = failed_day
        self.calls: list[tuple[date, date]] = []

    async def historical_candles(self, _key, start, end, _minutes=5):
        self.calls.append((start, end))
        if start < end or start == self.failed_day:
            raise RetryableBrokerError("provider timeout")
        return [[f"{start.isoformat()}T09:15:00", 100, 102, 99, 101, 1000, 0]]


@pytest.mark.asyncio
async def test_existing_candle_does_not_mark_an_entire_unchecked_range_complete(tmp_path):
    class CompleteClient:
        calls = 0

        async def historical_candles(self, _key, _start, _end, _minutes):
            self.calls += 1
            return [
                ["2026-01-02T09:15:00", 100, 102, 99, 101, 1000, 0],
                ["2026-01-05T09:15:00", 100, 102, 99, 101, 1000, 0],
            ]

    engine = create_engine(f"sqlite:///{tmp_path / 'partial.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        db.add(Candle(
            provider="GROWW", instrument_key="spot", underlying_code="NIFTY",
            timeframe="5minute", timestamp=datetime(2026, 1, 2, 3, 45),
            open=100, high=102, low=99, close=101, volume=1000,
            source_verified=True, data_role="SPOT",
        ))
        db.commit()
    client = CompleteClient()
    added = await _store_candles(client, factory, "spot", "NIFTY", date(2026, 1, 2), date(2026, 1, 5), provider="GROWW")
    assert client.calls == 1
    assert added == 1
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Candle)) == 2
    await _store_candles(client, factory, "spot", "NIFTY", date(2026, 1, 2), date(2026, 1, 5), provider="GROWW")
    assert client.calls == 1
    await _store_candles(client, factory, "spot", "NIFTY", date(2026, 1, 2), date(2026, 1, 5), provider="GROWW", refresh_existing=True)
    assert client.calls == 2
    engine.dispose()


@pytest.mark.asyncio
async def test_research_splits_failed_ranges_and_resumes_only_missing_chunks(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'chunks.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    client = _SplittingHistoricalClient(failed_day=date(2026, 1, 2))
    warnings: list[str] = []
    incomplete: list[str] = []

    await _store_candles(
        client,
        factory,
        "NSE-NIFTY",
        "NIFTY",
        date(2026, 1, 1),
        date(2026, 1, 2),
        provider="GROWW",
        warnings=warnings,
        incomplete_reasons=incomplete,
    )

    with factory() as session:
        assert session.scalar(select(func.count()).select_from(Candle)) == 1
        assert session.scalar(
            select(func.count()).select_from(ResearchChunk).where(ResearchChunk.status == "FAILED")
        ) == 2
    assert incomplete and "2026-01-02" in incomplete[0]

    calls_before_resume = len(client.calls)
    client.failed_day = None
    resume_incomplete: list[str] = []
    await _store_candles(
        client,
        factory,
        "NSE-NIFTY",
        "NIFTY",
        date(2026, 1, 1),
        date(2026, 1, 2),
        provider="GROWW",
        warnings=[],
        incomplete_reasons=resume_incomplete,
    )

    with factory() as session:
        assert session.scalar(select(func.count()).select_from(Candle)) == 2
        leaf = session.scalar(
            select(ResearchChunk).where(
                ResearchChunk.start_date == date(2026, 1, 2),
                ResearchChunk.end_date == date(2026, 1, 2),
            )
        )
        assert leaf is not None and leaf.status == "COMPLETE"
    assert len(client.calls) == calls_before_resume + 1
    assert resume_incomplete == []
    engine.dispose()


@pytest.mark.asyncio
async def test_research_does_not_hide_permanent_broker_errors(tmp_path):
    class PermanentFailureClient(_SplittingHistoricalClient):
        async def historical_candles(self, _key, start, end, _minutes=5):
            raise BrokerError("access token expired")

    engine = create_engine(f"sqlite:///{tmp_path / 'permanent.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    with pytest.raises(BrokerError, match="access token expired"):
        await _store_candles(
            PermanentFailureClient(),
            factory,
            "NSE-NIFTY",
            "NIFTY",
            date(2026, 1, 1),
            date(2026, 1, 2),
            provider="GROWW",
        )

    engine.dispose()
