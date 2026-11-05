"""Fixed matched-label diagnostic after the seven-contract historical import.

Same observations, contracts, entry proxy and costs; compare only signal-close
versus stored legacy quote-anchored barriers. No fitting or model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, event, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.config import config
from app.models import Candle
from app.research_protocol import signal_barriers
from app.risk import option_charges
from research.matched_failure_audit import clean, fingerprint, record, utc

IST = ZoneInfo("Asia/Kolkata")


def replay(signal_time, option_bars, spot_bars, *, stop, target, direction, quantity,
           slip_pct, premium_stop_pct, provider="GROWW", exchange="NSE"):
    entry_time = utc(signal_time)+timedelta(minutes=5)
    day = entry_time.astimezone(IST).date()
    square_off = datetime.combine(day, time(15, 20), tzinfo=IST).astimezone(UTC)
    if utc(signal_time) not in option_bars:
        return {"status": "INCOMPLETE", "reason": "missing_signal_option_bar"}
    entry_bar = option_bars.get(entry_time)
    if entry_bar is None:
        return {"status": "INCOMPLETE", "reason": "missing_exact_entry_bar"}
    # Match training's OHLC proxy arithmetic; do not call these executable fills.
    entry = D(str(entry_bar.open))*(1+slip_pct/100)
    premium_stop = entry*(1-premium_stop_pct/100)
    stamp, ambiguous = entry_time, 0
    while stamp <= square_off:
        option = option_bars.get(stamp)
        if option is None:
            return {"status": "INCOMPLETE", "reason": "option_path_gap", "timestamp": stamp}
        if stamp == square_off:
            exit_base, reason = D(str(option.open)), "TIME_EXIT"
            break
        spot = spot_bars.get(stamp)
        if spot is None:
            return {"status": "INCOMPLETE", "reason": "underlying_path_gap", "timestamp": stamp}
        premium_hit = D(str(option.low)) <= premium_stop
        stop_hit = D(str(spot.low)) <= stop if direction == "CE" else D(str(spot.high)) >= stop
        target_hit = D(str(spot.high)) >= target if direction == "CE" else D(str(spot.low)) <= target
        ambiguous += int((premium_hit or stop_hit) and target_hit)
        if premium_hit or stop_hit:
            exit_base = min(D(str(option.close)), premium_stop) if premium_hit else D(str(option.close))
            reason = "PREMIUM_STOP" if premium_hit else "UNDERLYING_STOP"
            break
        if target_hit:
            exit_base, reason = D(str(option.close)), "UNDERLYING_TARGET"
            break
        stamp += timedelta(minutes=5)
    else:
        return {"status": "INCOMPLETE", "reason": "missing_square_off"}
    exit_price = exit_base*(1-slip_pct/100)
    fees = option_charges(entry, exit_price, quantity, provider, day, exchange)
    net = (exit_price-entry)*quantity-fees
    return {"status": "COMPLETE", "entry_proxy": entry, "exit_proxy": exit_price,
            "exit_reason": reason, "exit_candle_start": stamp,
            "outcome_known_by": stamp if reason == "TIME_EXIT" else stamp+timedelta(minutes=5),
            "net": net, "fees": fees, "net_win": net > 0,
            "ambiguous_stop_target_bars": ambiguous,
            "limitation": "Five-minute conservative candle proxy; not quote-at-touch or a broker fill"}


def run(spec_path, output):
    spec = json.loads(spec_path.read_text())
    for name, expected in spec["source_hashes"].items():
        if hashlib.sha256((config.root/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed frozen replay source: {name}")
    snapshot_path = config.root/spec["observations_snapshot"]
    if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != spec["observations_snapshot_sha256"]:
        raise ValueError("Changed original observations snapshot")
    snapshot = json.loads(snapshot_path.read_text())
    observations = [r for r in snapshot["observations"] if r["entry_time"] is not None]
    if len(observations) != 46:
        raise ValueError("Expected the frozen 46 observations, not a reselected sample")
    path = Path(make_url(config.database_url).database).resolve()
    engine = create_engine(f"sqlite:///{path.as_uri()}?mode=ro&uri=true")
    event.listen(engine, "connect", lambda c, _: c.execute("PRAGMA query_only=ON"))
    with Session(engine) as session:
        session.execute(text("BEGIN"))
        keys = sorted({r["instrument_key"] for r in observations})
        options = session.scalars(select(Candle).where(Candle.provider == "GROWW", Candle.data_role == "OPTION",
            Candle.instrument_key.in_(keys), Candle.timeframe == "5minute",
            Candle.source_verified.is_(True), Candle.timestamp >= datetime(2026, 10, 26),
            Candle.timestamp < datetime(2026, 10, 28)).order_by(Candle.instrument_key, Candle.timestamp)).all()
        spots = session.scalars(select(Candle).where(Candle.provider == "GROWW", Candle.data_role == "SPOT",
            Candle.instrument_key == "NSE-NIFTY", Candle.timeframe == "5minute", Candle.source_verified.is_(True),
            Candle.timestamp >= datetime(2026, 10, 26), Candle.timestamp < datetime(2026, 10, 28)).order_by(Candle.timestamp)).all()
        source = {"options": [record(r) for r in options], "spots": [record(r) for r in spots]}
        source_hash = fingerprint(source)  # Frozen before outcome calculation.
        paths = {key: {utc(r.timestamp): r for r in options if r.instrument_key == key} for key in keys}
        spot_path = {utc(r.timestamp): r for r in spots}
        results = []
        for obs in observations:
            signal = utc(obs["signal_time"])
            protocol, evidence, features = (json.loads(obs[k]) for k in ("protocol_json", "selection_json", "features_json"))
            close = D(evidence["spot"])
            atr = D(str(features["atr_pct"]))*close
            stop, target = signal_barriers(close, atr, obs["direction"], D(str(protocol["atr_target"])))
            kwargs = {"direction": obs["direction"], "quantity": obs["quantity"], "slip_pct": D(protocol["slippage_pct"]),
                      "premium_stop_pct": D(protocol["premium_stop_pct"]), "exchange": obs["exchange"]}
            option_path = paths[obs["instrument_key"]]
            aligned = replay(signal, option_path, spot_path, stop=stop, target=target, **kwargs)
            legacy = replay(signal, option_path, spot_path, stop=D(obs["underlying_stop"]), target=D(obs["underlying_target"]), **kwargs)
            results.append({"id": obs["id"], "signal_time": signal, "slippage_pct": protocol["slippage_pct"],
                "family": obs["setup_family"], "direction": obs["direction"], "instrument_key": obs["instrument_key"],
                "signal_and_entry_pair_present": signal in option_path and signal+timedelta(minutes=5) in option_path,
                "original_quote_net": obs["net_pnl"], "original_quote_exit_reason": obs["exit_reason"],
                "original_quote_entry": obs["entry_price"], "entry_delay_seconds": (utc(obs["entry_time"])-signal-timedelta(minutes=5)).total_seconds(),
                "signal_close_candle_proxy": aligned, "legacy_barrier_candle_proxy": legacy,
                "source_revision_warning": "26 October opening inputs were revised; saved original ATR/spot retained, path candles are current stored history"})
        complete = [r for r in results if r["signal_close_candle_proxy"]["status"] == "COMPLETE"]
        paired = [r for r in complete if r["legacy_barrier_candle_proxy"]["status"] == "COMPLETE"]
        report = {"status": "MATCHED_REPLAY_COMPLETE" if len(complete) == 46 else "MATCHED_REPLAY_PARTIAL",
            "specification_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(), "source_candles_hash": source_hash,
            "activation_eligible": False, "trained": False,
            "summary": {"observations": 46, "observed_dates": 2, "option_candles": len(options), "spot_candles": len(spots),
                "signal_entry_pairs": sum(r["signal_and_entry_pair_present"] for r in results), "aligned_complete": len(complete),
                "legacy_geometry_complete": len(paired),
                "changed_exit_reason_between_geometries": sum(r["signal_close_candle_proxy"]["exit_reason"] != r["legacy_barrier_candle_proxy"]["exit_reason"] for r in paired),
                "changed_exit_bar_between_geometries": sum(r["signal_close_candle_proxy"]["exit_candle_start"] != r["legacy_barrier_candle_proxy"]["exit_candle_start"] for r in paired),
                "changed_net_sign_between_geometries": sum(r["signal_close_candle_proxy"]["net_win"] != r["legacy_barrier_candle_proxy"]["net_win"] for r in paired),
                "aligned_net_sign_disagrees_observed_quote": sum(r["signal_close_candle_proxy"]["net_win"] != (D(r["original_quote_net"]) > 0) for r in complete),
                "incomplete_reasons": dict(Counter(r["signal_close_candle_proxy"].get("reason") for r in results if r["signal_close_candle_proxy"]["status"] != "COMPLETE"))},
            "results": results,
            "limitations": ["No training, cutoff or amount search; no new performance estimate",
                "Repeated overlapping observations on two dates, not46 independent sessions",
                "Same proxies in both geometry arms isolate barriers, not executable profit improvement",
                "Entry proxy starts at next-bar open; original observed entry occurs later",
                "No synchronization of option quotes at intrabar stop/target touches",
                "Existing v1 outcomes stay unchanged; no reserved14–23 October option outcomes consumed"]}
        session.rollback()
    engine.dispose()
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("source-candles.json", source), ("report.json", clean(report))):
        with (output/name).open("x") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
    print(json.dumps(clean({"status": report["status"], "summary": report["summary"]}), indent=2))
    return clean(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.spec, args.output)
