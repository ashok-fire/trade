import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
from sqlalchemy import select

from app.database import settings_dict
from app.decision_snapshot import compare_snapshot, record_decision_snapshot
from app.models import DecisionSnapshot
from app.parity import decision_parity_report


def model():
    return SimpleNamespace(
        metadata={"version": "test-v1", "feature_schema": "quality-v5", "setup_families": ["ORB"]},
        required_features=["close", "rsi"],
    )


def row(close=25000.0):
    return pd.Series({
        "timestamp": pd.Timestamp("2026-11-04T07:35:00Z"),
        "close": close,
        "rsi": 55.0,
        "futures_close": 25005.0,
    })


def evidence(spot_hash="spot-a"):
    return {
        "protocol_version": "market-frame-v2",
        "verified_only": True,
        "spot": {"last_id": 123, "sha256": spot_hash},
        "futures": {"last_id": 456, "sha256": "future-a"},
    }


def test_decision_snapshots_are_strict_json_content_addressed_and_idempotent(session):
    boundary = datetime(2026, 11, 4, 7, 40, tzinfo=UTC)
    decision = SimpleNamespace(regime="RANGE", confidence=0.0, direction="NO_TRADE",
                               reason="No setup", features={"close": 25000.0, "missing": float("nan")})
    first = record_decision_snapshot(
        session, provider="GROWW", mode="SHADOW", underlying_code="NIFTY",
        boundary=boundary, stage="MARKET", model=model(), settings=settings_dict(session),
        market_evidence=evidence(), row=row(), context={"india_vix": None}, decision=decision,
    )
    again = record_decision_snapshot(
        session, provider="GROWW", mode="SHADOW", underlying_code="NIFTY",
        boundary=boundary, stage="MARKET", model=model(), settings=settings_dict(session),
        market_evidence=evidence(), row=row(), context={"india_vix": None}, decision=decision,
    )
    session.commit()

    assert first.id == again.id
    assert session.scalars(select(DecisionSnapshot)).all() == [first]
    payload = json.loads(first.payload_json)
    assert payload["decision"]["features"]["missing"] is None
    assert first.spot_candle_id == 123


def test_snapshot_comparison_reports_source_and_feature_drift(session):
    snapshot = record_decision_snapshot(
        session, provider="GROWW", mode="SHADOW", underlying_code="NIFTY",
        boundary=datetime(2026, 11, 4, 7, 40, tzinfo=UTC), stage="MARKET",
        model=model(), settings=settings_dict(session), market_evidence=evidence(), row=row(),
    )
    session.commit()

    same = compare_snapshot(snapshot, evidence(), row())
    assert same["market_hash_match"] and same["row_match"]
    drift = compare_snapshot(snapshot, evidence("spot-revised"), row(25001.0))
    assert not drift["market_hash_match"]
    assert not drift["spot_hash_match"]
    assert drift["changed_fields"]["close"] == {"stored": 25000.0, "rebuilt": 25001.0}


def test_parity_report_requires_exact_reconstruction(session, monkeypatch):
    snapshot = record_decision_snapshot(
        session, provider="GROWW", mode="SHADOW", underlying_code="NIFTY",
        boundary=datetime(2026, 11, 4, 7, 40, tzinfo=UTC), stage="MARKET",
        model=model(), settings=settings_dict(session), market_evidence=evidence(), row=row(),
    )
    session.commit()
    rebuilt = SimpleNamespace(evidence=evidence(), frame=pd.DataFrame([row()]))
    monkeypatch.setattr("app.parity.market_frame_for_boundary", lambda *args: rebuilt)

    report = decision_parity_report(session)

    assert report["captured_records"] == report["captured"] == report["exact"] == 1
    assert report["exact_percent"] == 100
    assert report["distinct_sessions"] == 1
    assert not report["support_ready"]
    assert not report["ready_for_model_selection"]
    assert report["records"][0]["snapshot_id"] == snapshot.id
