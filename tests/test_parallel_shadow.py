import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal as D
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.domain import Quote
from app.models import Candle, Instrument, ModelArtifact, Order, RuntimeSetting, Setting, SetupObservation, Trade
from app.market_frames import market_frame_for_boundary
from app.parallel_shadow import ParallelShadowService, PROTOCOL
from app.shadow_dashboard import daily_metrics, observation_export, shadow_context
from app.strategy import FEATURE_SCHEMA_V6, SETUP_FAMILIES

NOW = datetime(2026, 10, 23, 4, 30, 25, tzinfo=UTC)  # Friday 10:00:25 IST


def quote(key, now=NOW, bid="100", ask="101", last="100.5"):
    return Quote(key, now, D(bid), D(ask), D(last), 10000, 10000, .5, 15)


@pytest.fixture
def observer(session, monkeypatch):
    session.get(RuntimeSetting, 1).execution_mode = "SHADOW"
    session.get(Setting, "enabled_indices").value = "NIFTY"
    for direction in ("CE", "PE"):
        session.add(Instrument(provider="UPSTOX", instrument_key=direction, trading_symbol=direction,
            underlying_code="NIFTY", exchange="NSE", option_type=direction, strike=D("25000"),
            expiry=date(2026, 10, 29), lot_size=65, tick_size=D(".05"), valid_from=date(2026, 10, 1),
            metadata_verified=True, instrument_kind="OPTION", active=True))
    session.commit()
    broker = SimpleNamespace(index_key=lambda code: "INDEX", quotes=AsyncMock(return_value={
        "CE": quote("CE"), "PE": quote("PE"), "INDEX": quote("INDEX", bid="24999", ask="25001", last="25000")}))
    factory = sessionmaker(bind=session.bind, expire_on_commit=False)
    service = ParallelShadowService({"UPSTOX": broker}, factory)
    service.model = SimpleNamespace(available=True, metadata={"version": "research-test", "setup_families": list(SETUP_FAMILIES),
        "setup_thresholds": {family: {"CE": .7, "PE": 1.01} for family in SETUP_FAMILIES}},
        predict=lambda values: ("TRADE", .3))
    row = pd.Series({"atr": 20})
    monkeypatch.setattr("app.parallel_shadow.market_frame_for_boundary", lambda *args: (
        pd.DataFrame([row]), SimpleNamespace(close=D("25000"), timestamp=NOW - timedelta(minutes=5, seconds=25),
            india_vix=15, pcr=1, futures_basis_pct=.1)))
    monkeypatch.setattr("app.parallel_shadow.setup_direction", lambda row, family: "PE" if family == "FAILED_BREAKOUT" else "CE")
    monkeypatch.setattr("app.parallel_shadow.feature_values", lambda row, context: {"family": context["setup_family"], "missing": float("nan")})
    return service, broker


@pytest.mark.asyncio
async def test_four_independent_families_and_no_orders(session, observer):
    service, broker = observer
    await service.observe(NOW)
    await service.observe(NOW + timedelta(seconds=1))
    session.expire_all()
    observations = session.scalars(select(SetupObservation)).all()
    assert len(observations) == 4
    assert {row.setup_family for row in observations} == set(SETUP_FAMILIES)
    assert all(row.status == "OPEN" for row in observations)
    assert all(row.quality_passed is False for row in observations if row.direction == "CE")
    pe = next(row for row in observations if row.direction == "PE")
    assert pe.quality_passed is None and pe.research_threshold is None
    assert pe.score_status == "RESEARCH_THRESHOLD_DISABLED"
    assert all(json.loads(row.features_json)["missing"] is None for row in observations)
    assert all(json.loads(row.selection_json)["scope"] == row.setup_family for row in observations)
    assert session.scalar(select(func.count(Order.id))) == 0
    assert session.scalar(select(func.count(Trade.id))) == 0
    assert broker.quotes.await_count == 1
    assert set(broker.quotes.call_args.args[0]) == {"CE", "PE", "INDEX"}


@pytest.mark.asyncio
async def test_rejected_score_still_gets_fee_adjusted_outcome(session, observer):
    service, broker = observer
    await service.observe(NOW)
    later = NOW + timedelta(seconds=15)
    broker.quotes.return_value = {"CE": quote("CE", later, "120", "121", "120"),
        "PE": quote("PE", later), "INDEX": quote("INDEX", later, "25039", "25041", "25040")}
    await service.monitor(later)
    session.expire_all()
    row = session.scalar(select(SetupObservation).where(SetupObservation.setup_family == "ORB"))
    assert row.quality_passed is False and row.status == "CLOSED"
    assert row.exit_reason == "UNDERLYING_TARGET"
    assert row.charges > 0 and row.net_pnl == row.gross_pnl - row.charges
    assert observation_export(row)["label_net_win"] == 1
    assert observation_export(row)["entry_time"].endswith("+00:00")
    assert json.loads(row.exit_evidence_json)["option_quote"]["bid"] == "120"


@pytest.mark.asyncio
async def test_stale_or_next_day_never_fills(session, observer):
    service, broker = observer
    await service.observe(NOW)
    # All quotes now stale, even though target-looking underlying supplied.
    broker.quotes.return_value["INDEX"] = quote("INDEX", NOW, "25100", "25101", "25100")
    await service.monitor(NOW + timedelta(minutes=1))
    session.expire_all()
    assert all(row.status == "OPEN" for row in session.scalars(select(SetupObservation)))
    await service.monitor(NOW + timedelta(days=3))
    session.expire_all()
    for row in session.scalars(select(SetupObservation)):
        assert row.status == "INCOMPLETE" and row.net_pnl is None
        assert observation_export(row)["label_net_win"] is None


@pytest.mark.asyncio
async def test_quote_rejection_preserves_evidence(session, observer):
    service, broker = observer
    broker.quotes.return_value["CE"] = quote("CE", NOW - timedelta(minutes=1))
    await service.observe(NOW)
    session.expire_all()
    row = session.scalar(select(SetupObservation).where(SetupObservation.setup_family == "ORB"))
    assert row.status == "QUOTE_REJECTED" and row.entry_price is None
    assert "stale" in row.reason
    assert "stale" in json.loads(row.selection_json)["candidates"][0]["rejection_reason"]


@pytest.mark.asyncio
async def test_disabled_weekend_and_stale_frame_skip(session, observer, monkeypatch):
    service, broker = observer
    await service.observe(NOW + timedelta(days=1))
    session.get(RuntimeSetting, 1).execution_mode = "LIVE"
    session.commit()
    await service.observe(NOW)
    session.get(RuntimeSetting, 1).execution_mode = "SHADOW"
    session.commit()
    monkeypatch.setattr("app.parallel_shadow.market_frame_for_boundary", lambda *args: None)
    await service.observe(NOW)
    assert session.scalar(select(func.count(SetupObservation.id))) == 0
    assert broker.quotes.await_count == 0


@pytest.mark.asyncio
async def test_no_setup_and_unsupported_model_are_honest(session, observer, monkeypatch):
    service, _ = observer
    service.model.metadata["setup_families"] = ["ORB"]
    monkeypatch.setattr("app.parallel_shadow.setup_direction", lambda row, family: None if family == "ORB" else "CE")
    await service.observe(NOW)
    session.expire_all()
    rows = session.scalars(select(SetupObservation)).all()
    assert len(rows) == 4
    assert next(row for row in rows if row.setup_family == "ORB").status == "NO_SETUP"
    assert all(row.score_status == "NO_COMPATIBLE_MODEL" for row in rows if row.setup_family != "ORB")


@pytest.mark.asyncio
async def test_pause_still_monitors_and_exits_at_squareoff(session, observer):
    service, broker = observer
    await service.observe(NOW)
    session.get(Setting, "parallel_shadow_enabled").value = "false"
    session.commit()
    late = datetime(2026, 10, 23, 9, 50, 15, tzinfo=UTC)
    broker.quotes.return_value = {"CE": quote("CE", late), "PE": quote("PE", late),
        "INDEX": quote("INDEX", late, "24999", "25001", "25000")}
    await service.observe(late)
    await service.monitor(late)
    session.expire_all()
    assert all(row.status == "CLOSED" and row.exit_reason == "TIME_EXIT" for row in session.scalars(select(SetupObservation)))


def test_daily_comparison_never_replaces_incomplete_first_with_winner():
    base = dict(quality_passed=True, entry_time=NOW, observed_at=NOW, status="INCOMPLETE", net_pnl=None,
        exit_time=None, charges=None, id="first")
    first = SimpleNamespace(**base)
    later = SimpleNamespace(**{**base, "id": "later", "observed_at": NOW + timedelta(minutes=5),
        "status": "CLOSED", "net_pnl": D(1000), "charges": D(50), "exit_time": NOW + timedelta(hours=1)})
    result = daily_metrics([later, first])
    assert result["selected"] == 1 and result["closed"] == 0 and result["win_rate"] is None


def test_empty_dashboard_context(session, observer):
    service, _ = observer
    context = shadow_context(session, service)
    assert context["total"] == 0 and context["comparisons"] == []


def test_research_reference_freezes_without_promoting(session, observer, monkeypatch):
    service, _ = observer
    for number in (1, 2):
        session.add(ModelArtifact(id=number, version=f"v{number}", status="CANDIDATE", model_path=f"{number}.json",
            metadata_path="unused", metadata_json="{}", created_at=NOW + timedelta(seconds=number)))
    session.commit()
    monkeypatch.setattr("app.parallel_shadow.RegimeModel", lambda path: SimpleNamespace(available=True,
        metadata={"version": path.stem, "feature_schema": FEATURE_SCHEMA_V6, "setup_families": list(SETUP_FAMILIES)}))
    service.reload_model()
    assert service.artifact_id == 2
    session.expire_all()
    assert session.get(Setting, "parallel_shadow_model_artifact_id").value == "2"
    session.add(ModelArtifact(id=3, version="v3", status="CANDIDATE", model_path="3.json", metadata_path="unused",
        metadata_json="{}", created_at=NOW + timedelta(seconds=3)))
    session.commit()
    service.reload_model()
    assert service.artifact_id == 2
    assert all(row.status == "CANDIDATE" for row in session.scalars(select(ModelArtifact)))


@pytest.mark.asyncio
async def test_scope_mismatch_and_model_failure_do_not_block_labels(session, observer):
    service, _ = observer
    service.model.metadata["indices"] = ["BANK_NIFTY"]
    await service.observe(NOW)
    session.expire_all()
    assert all(row.score_status == "OUTSIDE_TRAINED_SCOPE" and row.status == "OPEN" for row in session.scalars(select(SetupObservation)))
    service.model.metadata["indices"] = ["NIFTY"]
    def broken(values):
        raise ValueError("unavailable research bundle")
    service.model.predict = broken
    await service.observe(NOW + timedelta(minutes=5))
    # Current quote timestamps are stale at the next boundary, so invoke scoring
    # on existing evidence directly to isolate model-error handling.
    row = session.scalar(select(SetupObservation).where(SetupObservation.entry_time.is_not(None)))
    service.score(row, {})
    assert row.score_status == "SCORE_ERROR" and row.status == "OPEN"
    assert "bundle" in json.loads(row.selection_json)["scoring_error"]


@pytest.mark.asyncio
async def test_premium_stop_and_future_quote_guard(session, observer):
    service, broker = observer
    await service.observe(NOW)
    later = NOW + timedelta(seconds=15)
    broker.quotes.return_value = {"CE": quote("CE", later + timedelta(seconds=1), "60", "61", "60"),
        "PE": quote("PE", later, "60", "61", "60"),
        "INDEX": quote("INDEX", later, "24999", "25001", "25000")}
    await service.monitor(later)
    session.expire_all()
    rows = session.scalars(select(SetupObservation)).all()
    assert all(row.status == "OPEN" for row in rows if row.direction == "CE")
    assert next(row for row in rows if row.direction == "PE").exit_reason == "PREMIUM_STOP"


def test_market_frame_requires_exact_futures_bar(session):
    session.add(Instrument(provider="UPSTOX", instrument_key="FUT", trading_symbol="FUT", underlying_code="NIFTY",
        exchange="NSE", option_type="FUT", strike=0, expiry=date(2026, 10, 29), lot_size=65,
        valid_from=date(2026, 10, 1), instrument_kind="FUTURE"))
    for day, count in ((22, 75), (23, 9)):
        for index in range(count):
            stamp = datetime(2026, 10, day, 3, 45, tzinfo=UTC) + timedelta(minutes=index * 5)
            for role, key in (("SPOT", "INDEX"), ("FUTURE", "FUT")):
                session.add(Candle(provider="UPSTOX", instrument_key=key, underlying_code="NIFTY", data_role=role,
                    timestamp=stamp, open=25000 + index, high=25002 + index, low=24998 + index,
                    close=25001 + index, volume=1000, source_verified=True))
    session.commit()
    boundary = NOW.replace(second=0)
    result = market_frame_for_boundary(session, "UPSTOX", "NIFTY", boundary)
    assert result is not None and result[1].timestamp == (boundary - timedelta(minutes=5)).replace(tzinfo=None)
    future = session.scalar(select(Candle).where(Candle.instrument_key == "FUT").order_by(Candle.timestamp.desc()))
    session.delete(future)
    session.commit()
    assert market_frame_for_boundary(session, "UPSTOX", "NIFTY", boundary) is None
    assert market_frame_for_boundary(session, "GROWW", "NIFTY", boundary) is None


@pytest.mark.asyncio
async def test_populated_template_renders(session, observer):
    from app.main import app, templates
    from starlette.requests import Request
    service, _ = observer
    await service.observe(NOW)
    request = Request({"type": "http", "app": app, "scheme": "http", "server": ("localhost", 8001),
                       "path": "/shadow-research", "root_path": "", "headers": []})
    html = templates.get_template("shadow-research.html").render(request=request, **shadow_context(session, service))
    assert "RESEARCH_ONLY" in html and "30.0%" in html
    assert "disabled/unavailable" in html and "Export evidence" in html
