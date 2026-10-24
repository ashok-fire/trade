import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import select

from app.domain import Contract, Quote
from app.models import OptionSelectionAudit
from app.option_selector import select_option
from app.selection_audit import record_option_selection, replay_recorded_selection


def test_recorded_quotes_reproduce_itm_fallback_without_new_prices(session):
    now = datetime(2026, 10, 23, 5, 0, tzinfo=UTC)
    contracts = [
        Contract(str(strike), str(strike), "NSE", "NIFTY", date(2026, 10, 27), Decimal(strike), "CE", 65, Decimal("0.05"))
        for strike in (24950, 25000)
    ]
    quotes = {
        item.instrument_key: Quote(item.instrument_key, now, Decimal(100), Decimal(105 if item.strike == 25000 else 101), Decimal(101), 1000, 2000, 0.52, 15)
        for item in contracts
    }
    filters = {"max_spread_pct": Decimal("1.5"), "max_quote_age_seconds": 3, "delta_min": 0.45, "delta_max": 0.60}
    result = select_option(contracts, quotes, spot=Decimal(25000), direction="CE", trading_day=now.date(), now=now, **filters)
    kwargs = dict(provider="GROWW", mode="SHADOW", underlying_code="NIFTY", candle_timestamp=now - timedelta(minutes=5), model_version="research", direction="CE", spot=Decimal(25000), contracts=contracts, quotes=quotes, selection=result, observed_at=now, filters=filters)
    record_option_selection(session, **kwargs)
    session.commit()
    record_option_selection(session, **kwargs)
    session.commit()
    rows = session.scalars(select(OptionSelectionAudit)).all()
    assert len(rows) == 1
    evidence = json.loads(rows[0].evidence_json)
    replay = replay_recorded_selection(evidence)
    assert replay.contract.instrument_key == result.contract.instrument_key == "24950"
    assert replay.moneyness == "ITM1"
    assert replay.reason == result.reason
    assert any(item["rejection_reason"] == "spread too wide" for item in evidence["candidates"])
