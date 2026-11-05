"""Read-only matched Shadow diagnostics. No fitting, broker access or app writes.

Generate a JSON report and its source snapshot in a new output directory.
The frozen two-day scope must not be expanded to optimise a result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal as D
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.config import config
from app.market_frames import market_frame_for_boundary
from app.models import Candle, ModelArtifact, OptionSelectionAudit, RuntimeSetting, Setting, SetupObservation, Signal, Trade
from app.risk import option_charges, paper_fill
from app.selection_audit import replay_recorded_selection
from app.strategy import RegimeModel, feature_values, setup_direction
from app.v4_training import option_quality_context

ROOT = Path(__file__).resolve().parents[1]


def utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def clean(value):
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, datetime):
        return utc(value).isoformat()
    if isinstance(value, (D, date, Path)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def record(row):
    return clean({column.name: getattr(row, column.name) for column in row.__table__.columns})


def missing(value):
    return value is None or not math.isfinite(float(value))


def feature_differences(stored, rebuilt):
    differences = {}
    for name, expected in stored.items():
        actual = rebuilt.get(name)
        if missing(expected) and missing(actual):
            continue
        if missing(expected) or missing(actual) or not math.isclose(
            float(expected), float(actual), abs_tol=1e-9, rel_tol=1e-7
        ):
            differences[name] = {"stored": expected, "rebuilt": actual}
    return clean(differences)


def group_key(row):
    return (row.provider, row.underlying_code, row.model_version, row.protocol_version,
            fingerprint(json.loads(row.protocol_json)), row.setup_family, row.direction)


def affordable(row, capital=D("100000"), risk_pct=D("3")):
    """Causal one-lot eligibility; no exit, score or label influences this test."""
    if row.entry_time is None or row.entry_price is None or not row.quantity:
        return False
    premium = D(str(row.entry_price)) * row.quantity
    protocol = json.loads(row.protocol_json)
    risk = premium * D(str(protocol["premium_stop_pct"])) / 100
    return premium <= capital and risk <= capital * risk_pct / 100


def first_daily(rows, predicate=lambda row: True):
    first = {}
    for row in sorted(rows, key=lambda row: (utc(row.observed_at), row.id)):
        if row.entry_time is not None and predicate(row):
            first.setdefault(utc(row.entry_time).date().isoformat(), row)
    return first


def selected_metrics(first):
    closed = [r for r in first.values() if r.status == "CLOSED" and r.net_pnl is not None]
    values = [D(str(r.net_pnl)) for r in closed]
    positive = sum((max(v, D(0)) for v in values), D(0))
    negative = -sum((min(v, D(0)) for v in values), D(0))
    return {"selected_days": len(first), "completed": len(closed), "unknown": len(first)-len(closed),
            "net": float(sum(values, D(0))),
            "win_pct": 100*sum(v > 0 for v in values)/len(values) if values else None,
            "pf": float(positive/negative) if negative else None,
            "entries": [{"day": day, "id": r.id, "status": r.status,
                         "net": str(r.net_pnl) if r.status == "CLOSED" else None}
                        for day, r in sorted(first.items())]}


def daily_comparison(rows):
    # Eligibility applies equally to both arms and precedes outcome inspection.
    eligible = [r for r in rows if affordable(r)]
    baseline = first_daily(eligible)
    scored = first_daily(eligible, lambda r: r.quality_passed is True)
    paired = []
    for day in sorted(baseline.keys() | scored.keys()):
        left, right = baseline.get(day), scored.get(day)
        known = all(r is None or (r.status == "CLOSED" and r.net_pnl is not None) for r in (left, right))
        delta = (D(str(right.net_pnl)) if right else D(0)) - (D(str(left.net_pnl)) if left else D(0)) if known else None
        paired.append({"day": day, "baseline_id": left.id if left else None,
                       "scored_id": right.id if right else None, "known": known,
                       "net_difference": float(delta) if delta is not None else None})
    return {"entered_trials": sum(r.entry_time is not None for r in rows),
            "one_lot_risk_cash_eligible": len(eligible), "rule_only": selected_metrics(baseline),
            "frozen_score": selected_metrics(scored), "paired_days": paired,
            "limitation": "Separate family/protocol diagnostics, not a combined portfolio or independent model selection"}


def geometry(row):
    features = json.loads(row.features_json)
    selection = json.loads(row.selection_json)
    protocol = json.loads(row.protocol_json)
    signal = D(selection["spot"])
    current = D(selection["underlying_quote"]["last"])
    atr = D(str(features["atr_pct"])) * signal
    sign = D(1 if row.direction == "CE" else -1)
    difference = current-signal
    return {"signal_close": str(signal), "entry_underlying_quote": str(current), "atr": str(atr),
            "anchor_difference_points": float(difference), "anchor_difference_atr": float(difference/atr),
            "signal_anchor_stop": str(signal-sign*atr),
            "signal_anchor_target": str(signal+sign*atr*D(str(protocol["atr_target"]))),
            "quote_anchor_matches_stored": abs(D(row.underlying_stop)-(current-sign*atr)) <= D("0.0002")
                and abs(D(row.underlying_target)-(current+sign*atr*D(str(protocol["atr_target"])))) <= D("0.0002"),
            "limitation": "Different geometry; endpoint evidence cannot determine earlier alternative exit or P&L"}


def audit_closed(row):
    protocol, selection, exit_ = map(json.loads, (row.protocol_json, row.selection_json, row.exit_evidence_json))
    entry_quote = next(c["quote"] for c in selection["candidates"] if c["instrument_key"] == row.instrument_key)
    exit_quote, underlying = exit_["option_quote"], exit_["underlying_quote"]
    slip, age = D(protocol["slippage_pct"]), protocol["max_quote_age_seconds"]
    reasons = []
    if D(exit_quote["bid"]) <= row.premium_stop:
        reasons.append("PREMIUM_STOP")
    if ((row.direction == "CE" and D(underlying["last"]) <= row.underlying_stop)
            or (row.direction == "PE" and D(underlying["last"]) >= row.underlying_stop)):
        reasons.append("UNDERLYING_STOP")
    if ((row.direction == "CE" and D(underlying["last"]) >= row.underlying_target)
            or (row.direction == "PE" and D(underlying["last"]) <= row.underlying_target)):
        reasons.append("UNDERLYING_TARGET")
    if utc(row.exit_time).hour*60+utc(row.exit_time).minute >= 9*60+50:
        reasons.append("TIME_EXIT")  # 15:20 IST
    checks = {
        "causal_signal": utc(row.signal_time)+timedelta(minutes=5) <= utc(row.entry_time),
        "entry_quote_age": 0 <= (utc(row.entry_time)-utc(entry_quote["timestamp"])).total_seconds() <= age,
        "exit_quote_age": 0 <= (utc(row.exit_time)-utc(exit_quote["timestamp"])).total_seconds() <= age,
        "underlying_exit_age": 0 <= (utc(row.exit_time)-utc(underlying["timestamp"])).total_seconds() <= age,
        "exit_after_entry": utc(row.entry_time) <= utc(row.exit_time),
        "entry_fill": paper_fill("BUY", D(entry_quote["bid"]), D(entry_quote["ask"]), slip, row.tick_size) == row.entry_price,
        "exit_fill": paper_fill("SELL", D(exit_quote["bid"]), D(exit_quote["ask"]), slip, row.tick_size) == row.exit_price,
        "gross": (row.exit_price-row.entry_price)*row.quantity == row.gross_pnl,
        "fees": option_charges(row.entry_price, row.exit_price, row.quantity, broker=row.provider,
            trade_date=utc(row.exit_time).date(), exchange_name=row.exchange) == row.charges,
        "net": row.gross_pnl-row.charges == row.net_pnl,
        "exit_reason": bool(reasons) and row.exit_reason == reasons[0],
    }
    # Change only fill costs at the SAME recorded exit, not the stop/exit path.
    stress_entry = paper_fill("BUY", D(entry_quote["bid"]), D(entry_quote["ask"]), slip*2, row.tick_size)
    stress_exit = paper_fill("SELL", D(exit_quote["bid"]), D(exit_quote["ask"]), slip*2, row.tick_size)
    stress = (stress_exit-stress_entry)*row.quantity-option_charges(stress_entry, stress_exit, row.quantity,
        broker=row.provider, trade_date=utc(row.exit_time).date(), exchange_name=row.exchange)
    return {"checks": checks, "failed": [k for k, v in checks.items() if not v],
            "same_exit_double_slippage_net": float(stress), "net_win": row.net_pnl > 0}


def run(spec_path, amendment_path=None):
    spec = json.loads(spec_path.read_text())
    hashes = dict(spec["source_hashes"])
    if amendment_path:
        amendment = json.loads(amendment_path.read_text())
        if hashlib.sha256(spec_path.read_bytes()).hexdigest() != amendment["original_specification_sha256"]:
            raise ValueError("Amendment does not match original specification")
        overrides = amendment["source_hash_overrides"]
        if set(overrides) != {"research/matched_failure_audit.py", "tests/test_matched_failure_audit.py"}:
            raise ValueError("Only diagnostic script/test serialization repair allowed")
        hashes.update(overrides)
    for filename, expected in hashes.items():
        if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed since specification: {filename}")
    scope = spec["scope"]
    if (scope["start_ist"], scope["end_ist"]) != ("2026-10-26", "2026-10-27"):
        raise ValueError("This audit is frozen to the two collected sessions")
    path = Path(make_url(config.database_url).database).resolve()
    engine = create_engine(f"sqlite:///{path.as_uri()}?mode=ro&uri=true")
    event.listen(engine, "connect", lambda connection, _: connection.execute("PRAGMA query_only=ON"))
    start, end = datetime(2026, 10, 26, tzinfo=UTC), datetime(2026, 10, 28, tzinfo=UTC)
    with Session(engine) as session:
        session.execute(text("BEGIN"))  # One read snapshot, not a sequence of changing views.
        rows = session.scalars(select(SetupObservation).where(SetupObservation.provider == scope["provider"],
            SetupObservation.underlying_code == scope["index"], SetupObservation.signal_time >= start,
            SetupObservation.signal_time < end).order_by(SetupObservation.observed_at, SetupObservation.id)).all()
        audits = session.scalars(select(OptionSelectionAudit).where(OptionSelectionAudit.provider == scope["provider"],
            OptionSelectionAudit.underlying_code == scope["index"], OptionSelectionAudit.candle_timestamp >= start,
            OptionSelectionAudit.candle_timestamp < end).order_by(OptionSelectionAudit.id)).all()
        signals = session.scalars(select(Signal).where(Signal.provider == scope["provider"], Signal.underlying_code == scope["index"],
            Signal.candle_timestamp >= start, Signal.candle_timestamp < end).order_by(Signal.candle_timestamp, Signal.id)).all()
        trades = session.scalars(select(Trade).where(Trade.provider == scope["provider"], Trade.underlying_code == scope["index"],
            Trade.entry_time >= start, Trade.entry_time < end).order_by(Trade.entry_time)).all()
        snapshot = {"observations": [record(r) for r in rows], "selection_audits": [record(a) for a in audits],
                    "signals": [record(s) for s in signals], "trades": [record(t) for t in trades]}
        model_rows = [session.get(ModelArtifact, n) for n in (9, 11)]
        model_hashes = {}
        models = {}
        for artifact in model_rows:
            path_ = Path(artifact.model_path)
            manifest = json.loads(path_.read_text())
            files = [path_, Path(artifact.metadata_path)] + [path_.parent/f for f in manifest["quality_models"].values()] + [path_.parent/manifest["ranker"]]
            model_hashes[str(artifact.id)] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            models[artifact.id] = RegimeModel(path_)
        # Input fingerprints are established before analytic calculations below.
        inputs_hash = fingerprint(snapshot)
        selection_results = []
        for audit in audits:
            evidence = json.loads(audit.evidence_json)
            replay = replay_recorded_selection(evidence)
            key = replay.contract.instrument_key if replay.contract else None
            selection_results.append({"id": audit.id, "mode": audit.mode, "match": key == evidence["selected_instrument"]
                and replay.reason == evidence["selection_reason"], "selected": key, "reason": replay.reason})
        groups, details, frames, frame_snapshots = defaultdict(list), [], {}, {}
        market_names = Counter()
        option_names = {"option_premium_pct", "option_volume_log", "option_oi_log", "days_to_expiry", "moneyness_pct"}
        for row in rows:
            stored, selection = json.loads(row.features_json), json.loads(row.selection_json)
            boundary = utc(row.signal_time)+timedelta(minutes=5)
            if boundary not in frames:
                frames[boundary] = market_frame_for_boundary(session, row.provider, row.underlying_code, boundary)
                result = frames[boundary]
                frame_snapshots[boundary.isoformat()] = {"hash": fingerprint(result[0].to_dict("records")), "last_row": clean(result[0].iloc[-1].to_dict()),
                    "candle_context": {k: getattr(result[1], k) for k in ("india_vix", "pcr", "futures_basis_pct")}} if result is not None else None
            result = frames[boundary]
            detail = {"id": row.id, "family": row.setup_family, "direction": row.direction, "signal_time": utc(row.signal_time),
                      "status": row.status, "score_status": row.score_status, "quality_passed": row.quality_passed,
                      "protocol_hash": group_key(row)[4], "net": row.net_pnl}
            if result is None:
                detail["feature_reconstruction"] = "UNAVAILABLE"
            else:
                frame, candle = result
                context = {"setup_family": row.setup_family, "india_vix": candle.india_vix, "pcr": candle.pcr,
                           "futures_basis_pct": candle.futures_basis_pct}
                selected = next((c for c in selection.get("candidates", []) if c["instrument_key"] == row.instrument_key), None)
                if selected is not None:
                    q = selected["quote"]
                    context.update(option_quality_context(premium=float(q["last"]), volume=q["volume"], open_interest=q["open_interest"],
                        expiry=date.fromisoformat(selected["expiry"]), strike=D(selected["strike"]), spot=float(candle.close), trading_day=utc(row.observed_at).date()))
                differences = feature_differences(stored, feature_values(frame.iloc[-1], context))
                market_names.update(k for k in differences if k not in option_names)
                detail["feature_differences"] = differences
                detail["setup_direction_match"] = (setup_direction(frame.iloc[-1], row.setup_family) or "NONE") == row.direction
            if row.entry_time is not None:
                groups[group_key(row)].append(row)
                detail["one_lot_eligible"] = affordable(row)
                detail["geometry"] = geometry(row)
                detail["entry_delay_seconds"] = (utc(row.entry_time)-boundary).total_seconds()
                option_bars = session.scalars(select(Candle).where(Candle.provider == row.provider,
                    Candle.instrument_key == row.instrument_key, Candle.data_role == "OPTION", Candle.timeframe == "5minute",
                    Candle.timestamp.in_([utc(row.signal_time), boundary]))).all()
                detail["same_contract_signal_and_entry_option_bars"] = len(option_bars) == 2
                if row.status == "CLOSED":
                    detail["outcome_audit"] = audit_closed(row)
                if row.setup_family == "ORB":
                    peers = [s for s in signals if s.model_version == models[9].metadata["version"]
                             and utc(s.candle_timestamp) == utc(row.signal_time)]
                    detail["selected_model_matching_signals"] = [{"id": s.id, "reason": s.rejection_reason, "score": s.confidence,
                        "selected_instrument": s.selected_instrument} for s in peers]
            details.append(detail)
        score_input = pd.DataFrame([json.loads(r.features_json) for r in rows], dtype=float)
        score_arrays = {key: model.quality_probabilities(score_input) for key, model in models.items()}
        for i, (row, detail) in enumerate(zip(rows, details)):
            score = score_arrays[11][i]
            if row.quality_score is not None:
                detail["score_replay_match"] = math.isfinite(score) and abs(score-row.quality_score) <= 1e-7
                detail["recomputed_reference_score"] = float(score)
            if row.setup_family == "ORB" and row.entry_time is not None:
                score9 = score_arrays[9][i]
                detail["model9_score_on_reference_quote"] = float(score9)
                detail["model9_threshold"] = models[9].metadata.get("direction_thresholds", {}).get(row.direction, models[9].metadata.get("decision_threshold", .7))
                # This is the same stored reference quote, NOT a replay of a different-time selected-model quote.
                detail["model9_on_reference_quote_passed"] = bool(math.isfinite(score9) and score9 >= detail["model9_threshold"])
        comparisons = [{"identity": key, **daily_comparison(group)} for key, group in sorted(groups.items())]
        closed_details = [d for d in details if "outcome_audit" in d]
        entered = [d for d in details if "geometry" in d]
        audit_failures = [{"id": d["id"], "failed": d["outcome_audit"]["failed"]} for d in closed_details if d["outcome_audit"]["failed"]]
        summary = {"counts_by_day": {day: dict(Counter(r.status for r in rows if str(utc(r.signal_time).date()) == day))
            for day in sorted({str(utc(r.signal_time).date()) for r in rows})},
            "observed_dates": len({utc(r.signal_time).date() for r in rows}), "entered": len(entered), "closed": len(closed_details),
            "audited_quote_selections": len(selection_results), "selection_mismatches": sum(not r["match"] for r in selection_results),
            "score_replays": sum("score_replay_match" in d for d in details), "score_mismatches": sum(d.get("score_replay_match") is False for d in details),
            "feature_reconstructed": sum("feature_differences" in d for d in details),
            "feature_mismatch_rows": sum(bool(d.get("feature_differences")) for d in details), "market_mismatches_by_name": dict(market_names),
            "setup_direction_mismatches": sum(d.get("setup_direction_match") is False for d in details),
            "outcome_audit_failures": audit_failures,
            "anchor_matches": sum(d["geometry"]["quote_anchor_matches_stored"] for d in entered),
            "anchor_difference_atr_abs_max": max((abs(d["geometry"]["anchor_difference_atr"]) for d in entered), default=None),
            "anchor_difference_atr_abs_median": float(pd.Series([abs(d["geometry"]["anchor_difference_atr"]) for d in entered]).median()) if entered else None,
            "same_day_historical_option_pairs": sum(d["same_contract_signal_and_entry_option_bars"] for d in entered),
            "one_lot_eligible_trials": sum(d["one_lot_eligible"] for d in entered),
            "underlying_target_net_losses": sum(d["outcome_audit"]["net_win"] is False for d, r in zip(details, rows) if "outcome_audit" in d and r.exit_reason == "UNDERLYING_TARGET"),
            "selected_portfolio_trades": [record(t) for t in trades],
            "selected_signal_reasons": dict(Counter(s.rejection_reason or s.status for s in signals))}
        snapshot["reconstructed_frames"] = clean(frame_snapshots)
        snapshot["model_hashes"] = model_hashes
        result = {"kind": spec["kind"], "specification_hash": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            "serialization_amendment_hash": hashlib.sha256(amendment_path.read_bytes()).hexdigest() if amendment_path else None,
            "snapshot_hash_before_analysis": inputs_hash, "snapshot_hash_with_reconstructions": fingerprint(snapshot),
            "created_at_utc": datetime.now(UTC), "scope": scope, "summary": summary,
            "selection_replays": selection_results, "observations": details, "daily_comparisons": comparisons,
            "status": "NO_MODEL_OR_THRESHOLD_CHANGE_JUSTIFIED", "activation_eligible": False,
            "alignment_decision": "Parallel observer uses quote anchor; selected model9/runtime and training use signal-close anchor. Treat reference comparisons as their own protocol; propose a future versioned alignment correction, never rewrite old outcomes.",
            "limitations": spec["limitations"], "source_snapshot": snapshot}
        session.rollback()
    engine.dispose()
    return clean(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amendment", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = run(args.spec, args.amendment)
    snapshot = result.pop("source_snapshot")
    for filename, payload in (("source-snapshot.json", snapshot), ("report.json", result)):
        with (args.output/filename).open("x") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "summary": result["summary"], "status": result["status"]}, indent=2))
