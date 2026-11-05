from datetime import UTC, date, datetime
from decimal import Decimal as D

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.database import DEFAULTS
from app.domain import Contract
from app.models import Candle
from app.research_protocol import one_lot_eligibility, signal_barriers
from research.import_matched_options import import_rows, valid_rows


def test_one_lot_cash_and_risk_are_distinct_and_causal():
    assert one_lot_eligibility(D("100"), 65, DEFAULTS)["eligible"]
    assert not one_lot_eligibility(D("250"), 65, DEFAULTS)["eligible"]
    cash_limited = {**DEFAULTS, "paper_capital": "10000", "risk_per_trade_pct": "100"}
    result = one_lot_eligibility(D("250"), 65, cash_limited)
    assert not result["eligible"] and not result["cash_affordable"]
    assert result["reason"] == "One lot exceeds cash capital"
    assert not result["portfolio_state_replayed"]


def test_barriers_are_directional_and_reject_invalid_inputs():
    assert signal_barriers(D("20000"), D(20), "PE", D("1.5")) == (D(20020), D(19970))
    with pytest.raises(ValueError):
        signal_barriers(D("20000"), D(-1), "CE", D("1.5"))


def test_import_validates_exact_session_grid_and_does_not_invent_missing_bars():
    raw = [["2026-10-26T09:15:00", 100, 102, 99, 101, 1000, None],
           ["2026-10-27T09:15:00", 100, 102, 99, 101, 1000, 10],
           ["2026-10-26T09:16:00", 100, 102, 99, 101, 1000, 10],
           ["2026-10-26T09:20:00", 100, 98, 99, 101, 1000, 10]]
    rows, rejected, missing = valid_rows(raw, date(2026, 10, 26))
    assert len(rows) == 1 and len(rejected) == 3 and len(missing) == 74
    assert rows[0][0] == datetime(2026, 10, 26, 3, 45, tzinfo=UTC)


def test_import_never_overwrites_existing_candles_or_duplicates_timestamps(session):
    factory = sessionmaker(bind=session.bind, expire_on_commit=False)
    contract = Contract("key", "key", "NSE", "NIFTY", date(2026, 10, 27), D(22000), "CE", 65, D(".05"), provider="GROWW")
    day = date(2026, 10, 26)
    rows, _, _ = valid_rows([["2026-10-26T09:15:00", 100, 102, 99, 101, 1000, 10]], day)
    assert import_rows(factory, contract, day, rows) == (1, [])
    assert import_rows(factory, contract, day, rows) == (0, [])
    changed, _, _ = valid_rows([["2026-10-26T09:15:00", 100, 104, 99, 103, 1000, 10]], day)
    inserted, conflicts = import_rows(factory, contract, day, changed)
    assert inserted == 0 and len(conflicts) == 1
    session.expire_all()
    assert session.scalar(select(Candle)).close == D(101)


def test_conflicting_provider_duplicates_fail_closed():
    with pytest.raises(ValueError, match="conflicting duplicate"):
        valid_rows([["2026-10-26T09:15:00", 100, 102, 99, 101, 1000, 10],
                    ["2026-10-26T09:15:00", 100, 104, 99, 103, 1000, 10]], date(2026, 10, 26))
