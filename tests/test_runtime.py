from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.domain import Contract, Quote
from app.execution import ExecutionService
from app.models import Candle, RuntimeSetting, Trade
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


async def _runtime_with_open_shadow_trade(session, entry_time: datetime):
    runtime_setting = session.get(RuntimeSetting, 1)
    runtime_setting.execution_mode, runtime_setting.entries_paused = "SHADOW", False
    session.commit()
    execution = ExecutionService({"UPSTOX": SimpleNamespace()})
    contract = Contract(
        "OPTION", "NIFTY TEST CE", "NSE", "NIFTY", date(2026, 11, 12),
        Decimal("25000"), "CE", 50, Decimal("0.05"),
    )
    quote = Quote(
        "OPTION", datetime.now(UTC), Decimal("99"), Decimal("100"),
        Decimal("99.5"), 1000, 5000, .5, 15,
    )
    trade = await execution.enter(
        session, underlying_code="NIFTY", candle_time=entry_time,
        regime="TRADE", confidence=.6, direction="CE",
        underlying_price=Decimal("25000"), underlying_stop=Decimal("24980"),
        contract=contract, quote=quote, moneyness="ATM", model_version="shadow-test",
    )
    trade.entry_time = entry_time
    session.commit()
    broker = SimpleNamespace(index_key=lambda _code: "INDEX", quotes=AsyncMock())
    runtime = object.__new__(RuntimeService)
    runtime.session_factory = sessionmaker(bind=session.bind, expire_on_commit=False)
    runtime.market_brokers = {"UPSTOX": broker}
    runtime.execution = execution
    runtime.last_tick = None
    return runtime, broker, trade.id


@pytest.mark.asyncio
async def test_late_restart_marks_shadow_trade_incomplete_without_using_quotes(session):
    entry = datetime(2026, 11, 5, 4, 20, tzinfo=UTC)
    runtime, broker, trade_id = await _runtime_with_open_shadow_trade(session, entry)
    await runtime.monitor(datetime(2026, 11, 5, 15, 1, tzinfo=UTC))  # 20:31 IST
    session.expire_all()
    trade = session.get(Trade, trade_id)
    assert trade.status == "INCOMPLETE"
    assert trade.exit_reason == "MISSED_SAME_DAY_EXIT"
    assert broker.quotes.await_count == 0


@pytest.mark.asyncio
async def test_squareoff_precedes_target_for_fresh_shadow_quotes(session):
    entry = datetime(2026, 11, 5, 4, 20, tzinfo=UTC)
    runtime, broker, trade_id = await _runtime_with_open_shadow_trade(session, entry)
    squareoff = datetime(2026, 11, 5, 9, 50, 15, tzinfo=UTC)
    option = Quote("OPTION", squareoff, Decimal("120"), Decimal("121"), Decimal("120.5"), 1, 1, .5, 15)
    underlying = Quote("INDEX", squareoff, Decimal("25050"), Decimal("25051"), Decimal("25050"), 1, 1, None, None)
    broker.quotes.side_effect = lambda keys: {"OPTION": option, "INDEX": underlying}
    await runtime.monitor(squareoff)
    session.expire_all()
    trade = session.get(Trade, trade_id)
    assert trade.status == "CLOSED" and trade.exit_reason == "TIME_EXIT"


@pytest.mark.asyncio
async def test_stale_target_quote_cannot_close_shadow_trade_before_squareoff(session):
    entry = datetime(2026, 11, 5, 4, 20, tzinfo=UTC)
    runtime, broker, trade_id = await _runtime_with_open_shadow_trade(session, entry)
    observed = datetime(2026, 11, 5, 8, 0, tzinfo=UTC)
    stale = observed - timedelta(minutes=1)
    option = Quote("OPTION", stale, Decimal("120"), Decimal("121"), Decimal("120.5"), 1, 1, .5, 15)
    underlying = Quote("INDEX", stale, Decimal("25050"), Decimal("25051"), Decimal("25050"), 1, 1, None, None)
    broker.quotes.side_effect = lambda keys: {"OPTION": option, "INDEX": underlying}
    await runtime.monitor(observed)
    session.expire_all()
    assert session.get(Trade, trade_id).status == "OPEN"
