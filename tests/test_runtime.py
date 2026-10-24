from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select

from app.models import Candle
from app.runtime import RuntimeService


def _row(stamp: datetime, close: str):
    value = Decimal(close)
    return (
        stamp,
        value - Decimal("1"),
        value + Decimal("2"),
        value - Decimal("2"),
        value,
        1_000,
        500,
    )


def test_runtime_candle_upsert_normalizes_aware_and_naive_utc(session):
    runtime = object.__new__(RuntimeService)
    aware = datetime(2026, 10, 15, 4, 0, tzinfo=UTC)
    naive = aware.replace(tzinfo=None)

    runtime._store_runtime_rows(
        session,
        provider="UPSTOX",
        code="NIFTY",
        instrument_key="NSE_INDEX|Nifty 50",
        rows=[_row(aware, "25000")],
        data_role="SPOT",
    )
    session.commit()
    runtime._store_runtime_rows(
        session,
        provider="UPSTOX",
        code="NIFTY",
        instrument_key="NSE_INDEX|Nifty 50",
        rows=[_row(naive, "25005")],
        data_role="SPOT",
    )
    session.commit()

    rows = session.scalars(select(Candle)).all()
    assert len(rows) == 1
    assert rows[0].timestamp == naive
    assert rows[0].close == Decimal("25005")


def test_runtime_discards_malformed_broker_candles():
    rows = RuntimeService._parse_rows(
        [
            ["2026-10-15T09:15:00+05:30", "25000", "25010", "24990", "25005", 1000, None],
            ["2026-10-15T09:20:00+05:30", "--", "25010", "24990", "25005", 1000, None],
            ["2026-10-15T09:25:00+05:30", "0", "25010", "24990", "25005", 1000, None],
        ]
    )
    assert len(rows) == 1
    assert rows[0][4] == Decimal("25005")


def test_runtime_blocks_new_entries_at_1510_ist():
    assert RuntimeService._entry_window_open(datetime(2026, 10, 23, 9, 39, tzinfo=UTC))
    assert not RuntimeService._entry_window_open(datetime(2026, 10, 23, 9, 40, tzinfo=UTC))
    assert not RuntimeService._entry_window_open(datetime(2026, 10, 24, 4, 0, tzinfo=UTC))
    assert not RuntimeService._entry_window_open(datetime(2026, 10, 23, 3, 40, tzinfo=UTC))
