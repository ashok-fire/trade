"""Read-only reconstruction audit for immutable live decision snapshots."""
from __future__ import annotations

from datetime import UTC
from zoneinfo import ZoneInfo

from sqlalchemy import select

from .decision_snapshot import PROTOCOL, compare_snapshot
from .market_frames import market_frame_for_boundary
from .models import DecisionSnapshot


MINIMUM_SNAPSHOTS = 100
MINIMUM_SESSIONS = 5


def _session_day(stamp):
    stamp = stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)
    return stamp.astimezone(ZoneInfo("Asia/Kolkata")).date()


def decision_parity_report(session, limit: int = 500) -> dict:
    snapshots = session.scalars(
        select(DecisionSnapshot)
        .where(DecisionSnapshot.stage == "MARKET", DecisionSnapshot.protocol_version == PROTOCOL)
        .order_by(DecisionSnapshot.boundary.desc(), DecisionSnapshot.created_at.desc())
        .limit(max(1, min(limit, 500)))
    ).all()
    records = []
    for snapshot in snapshots:
        result = market_frame_for_boundary(
            session, snapshot.provider, snapshot.underlying_code, snapshot.boundary,
        )
        if result is None:
            records.append({
                "snapshot_id": snapshot.id,
                "boundary": snapshot.boundary,
                "provider": snapshot.provider,
                "mode": snapshot.mode,
                "underlying_code": snapshot.underlying_code,
                "reconstructable": False,
                "exact": False,
                "reason": "Verified spot/futures frame is no longer reconstructable",
            })
            continue
        comparison = compare_snapshot(snapshot, result.evidence, result.frame.iloc[-1])
        exact = comparison["market_hash_match"] and comparison["row_match"]
        records.append({
            "snapshot_id": snapshot.id,
            "boundary": snapshot.boundary,
            "provider": snapshot.provider,
            "mode": snapshot.mode,
            "underlying_code": snapshot.underlying_code,
            "reconstructable": True,
            "exact": exact,
            "market_hash_match": comparison["market_hash_match"],
            "spot_hash_match": comparison["spot_hash_match"],
            "futures_hash_match": comparison["futures_hash_match"],
            "row_match": comparison["row_match"],
            "changed_field_count": len(comparison["changed_fields"]),
            "changed_fields": comparison["changed_fields"],
        })
    independent = {}
    for row in records:
        key = (row["provider"], row["underlying_code"], row["boundary"])
        group = independent.setdefault(key, {"reconstructable": True, "exact": True})
        group["reconstructable"] = group["reconstructable"] and row["reconstructable"]
        group["exact"] = group["exact"] and row["exact"]
    reconstructable = [row for row in independent.values() if row["reconstructable"]]
    exact = [row for row in reconstructable if row["exact"]]
    sessions = {_session_day(snapshot.boundary) for snapshot in snapshots}
    exact_rate = len(exact) / len(reconstructable) if reconstructable else 0
    support_ready = len(reconstructable) >= MINIMUM_SNAPSHOTS and len(sessions) >= MINIMUM_SESSIONS
    return {
        "protocol_version": PROTOCOL,
        "captured_records": len(records),
        "captured": len(independent),
        "reconstructable": len(reconstructable),
        "exact": len(exact),
        "exact_percent": round(exact_rate * 100, 2) if reconstructable else None,
        "distinct_sessions": len(sessions),
        "minimum_snapshots": MINIMUM_SNAPSHOTS,
        "minimum_sessions": MINIMUM_SESSIONS,
        "acceptance_target_percent": 95,
        "support_ready": support_ready,
        "ready_for_model_selection": support_ready and exact_rate >= 0.95,
        "records": records,
    }
