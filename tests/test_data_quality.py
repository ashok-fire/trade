from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from app.data_quality import _missing_intervals, futures_alignment_gaps, research_readiness
from app.models import Candle, Instrument


def _candle(stamp, role, key, volume=100):
    return Candle(
        provider="GROWW",
        instrument_key=key,
        underlying_code="NIFTY",
        timeframe="5minute",
        timestamp=stamp,
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("98"),
        close=Decimal("101"),
        volume=volume,
        open_interest=10,
        source_verified=True,
        data_role=role,
    )


def test_research_readiness_reports_alignment_gaps_and_metadata(session):
    start = datetime(2026, 1, 2, 3, 45, tzinfo=UTC)
    session.add_all(
        [
            _candle(start, "SPOT", "spot", 0),
            _candle(start + timedelta(minutes=5), "SPOT", "spot", 0),
            _candle(start + timedelta(minutes=15), "SPOT", "spot", 0),
            _candle(start, "FUTURE", "future"),
            _candle(start + timedelta(minutes=5), "FUTURE", "future"),
            _candle(start, "OPTION", "option"),
            Instrument(
                provider="GROWW",
                instrument_key="option",
                underlying_code="NIFTY",
                exchange="NSE",
                trading_symbol="NIFTY TEST CE",
                option_type="CE",
                strike=Decimal("25000"),
                expiry=date(2026, 1, 29),
                lot_size=65,
                tick_size=Decimal("0.05"),
                valid_from=date(2026, 1, 1),
                instrument_kind="OPTION",
                metadata_verified=True,
                metadata_source="CSV_IMPORT",
            ),
        ]
    )
    session.commit()

    row = next(item for item in research_readiness(session) if item.provider == "GROWW" and item.underlying_code == "NIFTY")
    assert row.spot_count == 3
    assert row.futures_count == 2
    assert row.aligned_count == 2
    assert row.missing_intervals == 1
    assert row.zero_volume_percent == 100
    assert row.option_count == 1
    assert row.verified_contracts == 1


def test_futures_gaps_detect_missing_sessions_despite_large_older_history():
    first = datetime(2026, 8, 24, 3, 45, tzinfo=UTC)
    second = first + timedelta(days=1)
    spots = [first + timedelta(minutes=5 * bar) for bar in range(75)]
    spots += [second + timedelta(minutes=5 * bar) for bar in range(75)]
    # Duplicate contracts cannot mask an entirely missing next session.
    futures = spots[:75] * 3
    assert futures_alignment_gaps(spots, futures) == ["2026-08-25"]


def test_futures_gaps_ignore_preopen_and_allow_small_intraday_gaps():
    start = datetime(2026, 8, 24, 3, 45, tzinfo=UTC)
    regular = [start + timedelta(minutes=5 * bar) for bar in range(75)]
    spots = [start - timedelta(minutes=10), *regular]
    assert futures_alignment_gaps(spots, regular[:-2]) == []


def _special_session(day):
    zone = ZoneInfo("Asia/Kolkata")
    if day == date(2024, 5, 18):
        windows = ((9, 15, 9), (11, 30, 12))
        extras = ((9, 5), (10, 0), (11, 15), (11, 20), (12, 30))
    else:
        windows = ((13, 45, 12),)
        extras = ((13, 30), (13, 35), (13, 40), (14, 45), (14, 50), (14, 55), (15, 0))
    core = [datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone)
            + timedelta(minutes=5 * i) for hour, minute, count in windows for i in range(count)]
    outside = [datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone)
               for hour, minute in extras]
    return core, outside


@pytest.mark.parametrize("day,expected_bars", [(date(2024, 5, 18), 21), (date(2025, 10, 21), 12)])
def test_special_sessions_ignore_preopen_breaks_and_close_boundaries(day, expected_bars):
    core, outside = _special_session(day)
    assert len(core) == expected_bars
    # SQLite-naive UTC and broker-aware IST timestamps must compare identically.
    futures = [stamp.astimezone(UTC).replace(tzinfo=None) for stamp in core]
    assert futures_alignment_gaps([*outside, *core], futures) == []
    assert _missing_intervals([*outside, *core, core[0]]) == 0


@pytest.mark.parametrize("day,removed", [(date(2024, 5, 18), 3), (date(2025, 10, 21), 2)])
def test_special_sessions_still_reject_real_futures_gaps(day, removed):
    core, outside = _special_session(day)
    assert futures_alignment_gaps([*core, *outside], core[:-removed] * 3) == [str(day)]
    assert _missing_intervals([*outside, *core[:2], *core[3:]]) == 1


def test_regular_session_keeps_exact_ninety_percent_gate():
    start = datetime(2026, 8, 24, 3, 45, tzinfo=UTC)
    core = [start + timedelta(minutes=5 * i) for i in range(10)]
    assert futures_alignment_gaps(core, core[:9]) == []
    assert futures_alignment_gaps(core, core[:8]) == ["2026-08-24"]


def test_readiness_uses_special_session_denominator_without_mutating_candles(session):
    core, outside = [], []
    for day in (date(2024, 5, 18), date(2025, 10, 21)):
        bars, other = _special_session(day)
        core.extend(bars)
        outside.extend(other)
    session.add_all([_candle(stamp.astimezone(UTC), "SPOT", "spot") for stamp in [*core, *outside]])
    session.add_all([_candle(stamp.astimezone(UTC), "FUTURE", "future") for stamp in core])
    session.commit()
    before = [(row.id, row.timestamp, row.close) for row in session.query(Candle).order_by(Candle.id)]
    row = next(item for item in research_readiness(session)
               if item.provider == "GROWW" and item.underlying_code == "NIFTY")
    assert row.spot_count == len(core) + len(outside)
    assert row.aligned_count == len(core) == 33
    assert row.alignment_percent == 100.0
    assert row.missing_intervals == 0
    assert "spot sessions" not in row.reason
    # Other readiness gates are not bypassed by the calendar correction.
    assert not row.training_ready
    assert not row.backtest_ready
    assert [(item.id, item.timestamp, item.close) for item in session.query(Candle).order_by(Candle.id)] == before
