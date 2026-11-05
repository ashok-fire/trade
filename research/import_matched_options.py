"""Bounded import of the seven frozen observed option contract-days.

Only historical market-data endpoints are used. No model/setting/order writes.
Preserves original OHLC and observations; inserts missing candle timestamps only.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal as D
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.engine import make_url

from app.config import config
from app.database import SessionLocal
from app.domain import Contract
from app.groww import GrowwClient
from app.models import Candle, ResearchChunk, ResearchSyncRun
from app.research import _parse
from research.matched_failure_audit import fingerprint

IST = ZoneInfo("Asia/Kolkata")


def write_json(path, payload):
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, default=str, allow_nan=False)


def load_plan(snapshot_path):
    snapshot = json.loads(snapshot_path.read_text())
    targets = {}
    for row in snapshot["observations"]:
        if row["entry_time"] is None:
            continue
        day = datetime.fromisoformat(row["entry_time"]).astimezone(IST).date()
        if row["provider"] != "GROWW" or row["underlying_code"] != "NIFTY" or day not in {date(2026, 10, 26), date(2026, 10, 27)}:
            raise ValueError("Frozen import scope changed")
        evidence = json.loads(row["selection_json"])
        item = next(c for c in evidence["candidates"] if c["instrument_key"] == row["instrument_key"])
        contract = {k: item[k] for k in ("instrument_key", "trading_symbol", "exchange", "expiry", "strike", "lot_size", "tick_size")}
        contract.update(underlying_code="NIFTY", option_type=row["direction"], provider="GROWW")
        key = (day.isoformat(), row["instrument_key"])
        if key in targets and targets[key]["contract"] != contract:
            raise ValueError("Conflicting frozen contract metadata")
        target = targets.setdefault(key, {"day": day.isoformat(), "contract": contract, "observation_ids": []})
        target["observation_ids"].append(row["id"])
    if len(targets) != 7:
        raise ValueError("Expected exactly seven frozen contract-days")
    return {"kind": "matched-observed-options-v1", "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            "targets": [targets[key] for key in sorted(targets)], "overwrite_existing": False,
            "scope": "Only observed NIFTY option contract-days on26–27 October2026; no spot/futures refresh or discovery"}


def valid_rows(raw, day):
    rows, rejected, seen = [], [], {}
    for index, item in enumerate(raw):
        parsed = _parse(item)
        if parsed is None:
            rejected.append({"row": index, "reason": "malformed OHLCV"})
            continue
        stamp = parsed[0].astimezone(IST)
        if stamp.date() != day or not time(9, 15) <= stamp.time() < time(15, 30):
            rejected.append({"row": index, "reason": "outside exact requested session"})
            continue
        if stamp.minute % 5 or stamp.second or stamp.microsecond:
            rejected.append({"row": index, "reason": "off five-minute grid"})
            continue
        if parsed[0] in seen and seen[parsed[0]] != parsed:
            raise ValueError("Provider returned conflicting duplicate candle timestamps")
        seen[parsed[0]] = parsed
    rows = [seen[key] for key in sorted(seen)]
    expected = {datetime.combine(day, time(9, 15), tzinfo=IST).astimezone(UTC)+timedelta(minutes=5*i) for i in range(75)}
    missing = sorted(expected-set(seen))
    return rows, rejected, [stamp.isoformat() for stamp in missing]


def import_rows(factory, contract, day, rows):
    inserted, conflicts = 0, []
    with factory() as session:
        existing = {r.timestamp.replace(tzinfo=UTC): r for r in session.scalars(select(Candle).where(
            Candle.provider == "GROWW", Candle.instrument_key == contract.instrument_key,
            Candle.timeframe == "5minute", Candle.timestamp >= datetime.combine(day, time.min),
            Candle.timestamp < datetime.combine(day+timedelta(days=1), time.min))).all()}
        for stamp, open_, high, low, close, volume, oi in rows:
            if stamp in existing:
                old = existing[stamp]
                if ((old.open, old.high, old.low, old.close, old.volume, old.open_interest) != (open_, high, low, close, volume, oi)
                        or old.data_role != "OPTION" or not old.source_verified):
                    conflicts.append(stamp.isoformat())
                continue
            session.add(Candle(provider="GROWW", instrument_key=contract.instrument_key, underlying_code="NIFTY",
                timeframe="5minute", timestamp=stamp, open=open_, high=high, low=low, close=close,
                volume=volume, open_interest=oi, source_verified=True, data_role="OPTION"))
            inserted += 1
        session.commit()
    return inserted, conflicts


def backup_database():
    source = Path(make_url(config.database_url).database).resolve()
    backup_path = config.root/"data/backups"/f"niftypilot-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-before-matched-option-import.db"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(backup_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    with sqlite3.connect(source.as_uri()+"?mode=ro", uri=True) as original, sqlite3.connect(backup_path) as backup:
        original.backup(backup)
        check = backup.execute("PRAGMA quick_check").fetchone()[0]
        if check != "ok":
            raise RuntimeError(f"Backup integrity failed: {check}")
    return backup_path


async def run(plan_path, output):
    plan = json.loads(plan_path.read_text())
    snapshot_path = config.root/"research/matched-failure-20261028/run-02/source-snapshot.json"
    if plan != load_plan(snapshot_path):
        raise ValueError("Import plan no longer matches frozen observed evidence")
    output.mkdir(parents=True, exist_ok=False)
    with SessionLocal() as session:
        if session.scalar(select(ResearchSyncRun.id).where(ResearchSyncRun.status.in_(["RUNNING", "QUEUED"]))):
            raise RuntimeError("Another historical download is running; do not duplicate work")
    client = GrowwClient(config.groww_token_file)
    result = {"status": "RUNNING", "plan_hash": hashlib.sha256(plan_path.read_bytes()).hexdigest(), "targets": []}
    run_id = None
    try:
        connected, detail = await client.market_data_health()
        if not connected:
            raise RuntimeError(f"Groww market-data access unavailable: {detail}")
        backup = backup_database()
        result["backup"] = str(backup)
        print(f"Verified backup: {backup}", flush=True)
        with SessionLocal() as session:
            job = ResearchSyncRun(provider="GROWW", status="RUNNING", configuration_json=json.dumps({
                "kind": plan["kind"], "indices": ["NIFTY"], "start_date": "2026-10-26", "end_date": "2026-10-27",
                "provider": "GROWW", "include_options": True, "targets": plan["targets"], "plan_hash": result["plan_hash"]}),
                progress_percent=1, current_step="Importing seven observed option contract-days only")
            session.add(job)
            session.commit()
            run_id = job.id
        result["run_id"] = run_id
        for number, target in enumerate(plan["targets"], 1):
            day = date.fromisoformat(target["day"])
            fields = target["contract"]
            contract = Contract(**{**fields, "expiry": date.fromisoformat(fields["expiry"]),
                "strike": D(fields["strike"]), "tick_size": D(fields["tick_size"])})
            print(f"{number}/7 {day} {contract.instrument_key}", flush=True)
            raw = await asyncio.wait_for(client.historical_option_candles(contract, day, day, 5), timeout=120)
            write_json(output/f"raw-{number:02d}.json", {"target": target, "candles": raw,
                "retrieved_at_utc": datetime.now(UTC).isoformat(), "source": "Groww historical option candles"})
            rows, rejected, missing = valid_rows(raw, day)
            inserted, conflicts = import_rows(SessionLocal, contract, day, rows)
            result["targets"].append({"day": str(day), "instrument_key": contract.instrument_key,
                "received": len(raw), "valid": len(rows), "inserted": inserted, "rejected": rejected,
                "missing_timestamps": missing, "existing_conflicts_preserved": conflicts,
                "parsed_rows_hash": fingerprint(rows)})
            with SessionLocal() as session:
                marker = session.scalar(select(ResearchChunk).where(ResearchChunk.provider == "GROWW",
                    ResearchChunk.instrument_key == contract.instrument_key, ResearchChunk.timeframe == "5minute",
                    ResearchChunk.start_date == day, ResearchChunk.end_date == day))
                if marker is None:
                    marker = ResearchChunk(provider="GROWW", instrument_key=contract.instrument_key, underlying_code="NIFTY",
                        data_role="OPTION", timeframe="5minute", start_date=day, end_date=day)
                    session.add(marker)
                marker.status = "COMPLETE" if not missing and not conflicts else "PARTIAL"
                marker.attempts = (marker.attempts or 0)+1
                marker.last_error = f"{len(missing)} missing; {len(conflicts)} existing conflicts" if missing or conflicts else None
                job = session.get(ResearchSyncRun, run_id)
                job.progress_percent = int(number/7*99)
                job.current_step = f"Stored {number}/7 observed option contract-days"
                session.commit()
        result["status"] = "COMPLETE" if all(not r["missing_timestamps"] and not r["existing_conflicts_preserved"] for r in result["targets"]) else "PARTIAL"
    except Exception as exc:
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        await client.client.aclose()
        if run_id is not None:
            with SessionLocal() as session:
                job = session.get(ResearchSyncRun, run_id)
                job.status, job.result_json = result["status"], json.dumps(result)
                job.error_message = result.get("error")
                job.current_step = "Matched option import "+result["status"].lower()
                job.progress_percent = 100 if result["status"] == "COMPLETE" else job.progress_percent
                job.completed_at = datetime.now(UTC)
                session.commit()
        write_json(output/"report.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args.plan, args.output))
    if result["status"] != "COMPLETE":
        raise SystemExit(2)
