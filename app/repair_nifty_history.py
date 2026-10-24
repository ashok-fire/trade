"""Repair NIFTY history through a recorded, resumable research download."""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import select

from .config import config
from .contract_metadata import historical_nifty_lot_size
from .database import SessionLocal
from .groww import GrowwClient
from .models import Instrument, ResearchSyncRun
from .research_jobs import ResearchJobService


async def repair(start: date, end: date) -> str:
    client = GrowwClient(config.groww_token_file)
    try:
        connected, detail = await client.health()
        if not connected:
            raise RuntimeError(detail)
        with SessionLocal() as session:
            running = session.scalar(select(ResearchSyncRun.id).where(ResearchSyncRun.status.in_(["QUEUED", "RUNNING"])))
            if running:
                raise RuntimeError(f"Research download #{running} is already running")
        # sqlite backup includes WAL state and preserves all existing history.
        if config.database_url.startswith("sqlite:///"):
            source = config.database_url.removeprefix("sqlite:///")
            destination = config.root / "data" / "backups" / f"niftypilot-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-before-history-repair.db"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(source) as original, sqlite3.connect(destination) as backup:
                original.backup(backup)
            print(f"Backup: {destination}", flush=True)
        with SessionLocal() as session:
            run = ResearchSyncRun(
                provider="GROWW",
                configuration_json=json.dumps({
                    "provider": "GROWW", "indices": ["NIFTY"],
                    "start_date": start.isoformat(), "end_date": end.isoformat(),
                    "include_options": True, "include_one_minute": False,
                    "refresh_existing": True,
                }),
            )
            session.add(run)
            session.commit()
            run_id = run.id
        print(f"Research download #{run_id}: {start} through {end}", flush=True)
        service = ResearchJobService({"GROWW": client}, SessionLocal)
        await service._run(run_id)
        with SessionLocal() as session:
            saved = session.get(ResearchSyncRun, run_id)
            print(json.dumps({"run_id": run_id, "status": saved.status, "result": json.loads(saved.result_json or "{}"), "error": saved.error_message}, indent=2), flush=True)
            status = saved.status
            # Reuse the project's exchange-sourced historical NIFTY lot-size
            # policy only for repaired 2026 options with missing contract size.
            verified = 0
            for item in session.scalars(select(Instrument).where(
                Instrument.provider == "GROWW", Instrument.underlying_code == "NIFTY",
                Instrument.instrument_kind == "OPTION", Instrument.lot_size == 0,
                Instrument.expiry >= start, Instrument.expiry <= end + timedelta(days=45),
                Instrument.valid_from <= end,
            )):
                if item.expiry.year != 2026:
                    continue
                evidence = historical_nifty_lot_size(item.expiry)
                if evidence:
                    item.lot_size, item.metadata_source = evidence
                    item.metadata_verified = True
                    verified += 1
            session.commit()
            print(f"Historical NIFTY lot metadata verified: {verified}", flush=True)
        return status
    finally:
        await client.client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    arguments = parser.parse_args()
    status = asyncio.run(repair(arguments.start, arguments.end))
    if status != "COMPLETE":
        raise SystemExit(2)
