"""Descriptive comparisons, not portfolio backtests or validated probabilities."""
from collections import Counter, defaultdict
import json

from sqlalchemy import select

from .models import RuntimeSetting, SetupObservation
from .database import settings_dict
from .parallel_shadow import IST, utc


def daily_metrics(rows, quality_only=False):
    # Choose before looking at outcome: an unfinished first trial cannot be
    # replaced by a later winner. All rows here belong to one comparison group.
    first = {}
    for row in sorted(rows, key=lambda row: (utc(row.observed_at), row.id)):
        if row.entry_time is None or (quality_only and row.quality_passed is not True):
            continue
        day = utc(row.entry_time).astimezone(IST).date()
        first.setdefault(day, row)
    closed = sorted((row for row in first.values() if row.status == "CLOSED" and row.net_pnl is not None),
                    key=lambda row: (utc(row.exit_time), row.id))
    values = [float(row.net_pnl) for row in closed]
    wins, losses = sum(max(value, 0) for value in values), -sum(min(value, 0) for value in values)
    equity = peak = drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {"selected": len(first), "closed": len(closed), "pending": len(first) - len(closed),
            "win_rate": round(100 * sum(value > 0 for value in values) / len(values), 2) if values else None,
            "pf": round(wins / losses, 3) if losses else ("∞" if wins else "—"),
            "net": sum(values), "charges": sum(float(row.charges or 0) for row in closed), "drawdown": drawdown}


def shadow_context(session, service):
    rows = session.scalars(select(SetupObservation).order_by(SetupObservation.observed_at.desc())).all()
    groups = defaultdict(list)
    for row in rows:
        # Frozen fill/filter settings are also part of the comparison identity.
        groups[(row.provider, row.underlying_code, row.model_version or "none", row.protocol_version,
                row.protocol_json, row.setup_family, row.direction)].append(row)
    comparisons = []
    for key, observations in sorted(groups.items()):
        provider, code, model, protocol, _, family, direction = key
        comparisons.append({"provider": provider, "code": code, "model": model, "protocol": protocol,
            "settings": key[4], "family": family, "direction": direction, "checks": len(observations),
            "candidates": sum(row.direction != "NONE" for row in observations),
            "executable": sum(row.entry_time is not None for row in observations),
            "closed": sum(row.status == "CLOSED" for row in observations),
            "incomplete": sum(row.status == "INCOMPLETE" for row in observations),
            "quote_rejected": sum(row.status == "QUOTE_REJECTED" for row in observations),
            "quality_rejected": sum(row.quality_passed is False for row in observations),
            "all_daily": daily_metrics(observations), "quality_daily": daily_metrics(observations, True)})
    reasons = Counter(row.reason for row in rows if row.status in {"QUOTE_REJECTED", "INCOMPLETE"})
    return {"settings": settings_dict(session), "runtime": session.get(RuntimeSetting, 1),
            "observer": service.health(), "comparisons": comparisons, "recent": rows[:100],
            "total": len(rows), "reasons": reasons.most_common(20)}


def observation_export(row):
    result = {column.name: getattr(row, column.name) for column in row.__table__.columns}
    for field in ("features_json", "selection_json", "protocol_json", "exit_evidence_json"):
        result[field.removesuffix("_json")] = json.loads(result.pop(field) or "null")
    result["label_net_win"] = int(row.net_pnl > 0) if row.status == "CLOSED" and row.net_pnl is not None else None
    result["research_only"] = True
    # SQLite stores naive UTC; export explicit offsets to avoid IST ambiguity.
    for field in ("signal_time", "observed_at", "entry_time", "exit_time"):
        if result[field] is not None:
            result[field] = utc(result[field]).isoformat()
    return result
