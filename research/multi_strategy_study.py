"""Frozen, research-only NIFTY setup comparison on historical option candles.

There is no model fitting, parameter selection, broker call, registry write or
app setting change. Each arm keeps its first signal per complete session even
when the option path or one-lot eligibility is unknown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.option_selector import option_candidates
from app.risk import option_charges
from app.strategy import with_futures_features
from research.matched_candle_replay import replay
from research.matched_failure_audit import clean

IST = ZoneInfo("Asia/Kolkata")
END_EXCLUSIVE = "2026-10-14"
ALTERNATIVES = ("ROLLING_30M", "VWAP_PULLBACK", "RANGE_REJECTION", "VWAP_REVERSION", "GAP_FADE")
ARMS = ("ORB_CONTROL", *ALTERNATIVES, "FIVE_SETUP_FIRST")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def utc(value) -> datetime:
    stamp = pd.Timestamp(value)
    return (stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")).to_pydatetime()


def load_market(conn: sqlite3.Connection) -> pd.DataFrame:
    spot = pd.read_sql_query(
        "SELECT timestamp,open,high,low,close,volume FROM candles WHERE provider='GROWW' "
        "AND underlying_code='NIFTY' AND data_role='SPOT' AND timeframe='5minute' "
        "AND source_verified=1 AND timestamp<? ORDER BY timestamp",
        conn, params=(END_EXCLUSIVE,),
    )
    futures = pd.read_sql_query(
        "SELECT timestamp,instrument_key,open,high,low,close,volume FROM candles "
        "WHERE provider='GROWW' AND underlying_code='NIFTY' AND data_role='FUTURE' "
        "AND timeframe='5minute' AND source_verified=1 AND timestamp<? ORDER BY timestamp",
        conn, params=(END_EXCLUSIVE,),
    )
    expiry = dict(conn.execute(
        "SELECT instrument_key,expiry FROM instruments WHERE provider='GROWW' "
        "AND underlying_code='NIFTY' AND instrument_kind='FUTURE'"
    ))
    if spot.empty or futures.empty:
        raise ValueError("Frozen spot/futures history is empty")
    spot["timestamp"] = pd.to_datetime(spot.timestamp, utc=True)
    futures["timestamp"] = pd.to_datetime(futures.timestamp, utc=True)
    futures["expiry"] = futures.instrument_key.map(expiry)
    futures["trading_day"] = futures.timestamp.dt.tz_convert(IST).dt.date.astype(str)
    futures = futures.loc[futures.expiry.notna() & (futures.expiry > futures.trading_day)]
    futures = futures.sort_values(["timestamp", "expiry"]).drop_duplicates("timestamp")
    frame = with_futures_features(
        spot, futures[["timestamp", "open", "high", "low", "close", "volume"]]
    )
    frame["trading_day"] = frame.timestamp.dt.tz_convert(IST).dt.date
    return frame


def setup_masks(frame: pd.DataFrame) -> tuple[dict[str, tuple[pd.Series, pd.Series]], list[date]]:
    """Vectorized causal setups; all rolling values exclude the current bar."""
    day = frame.trading_day
    count = frame.groupby(day).timestamp.transform("count")
    future_count = frame.futures_close.notna().groupby(day).transform("sum")
    complete = count.ge(75) & future_count.ge(0.9 * count)
    valid_days = sorted(set(day[complete]))
    local = frame.timestamp.dt.tz_convert(IST)
    minute = local.dt.hour * 60 + local.dt.minute
    window = minute.ge(9 * 60 + 45) & minute.lt(14 * 60 + 30)
    base = complete & window & frame.atr.gt(0) & frame.futures_close.notna()
    up = frame.futures_close.gt(frame.futures_vwap) & frame.ema9.gt(frame.ema20)
    down = frame.futures_close.lt(frame.futures_vwap) & frame.ema9.lt(frame.ema20)
    inside = frame.close.le(frame.opening_range_high) & frame.close.ge(frame.opening_range_low)
    prior_high = frame.groupby(day).high.transform(
        lambda rows: rows.shift(1).rolling(6, min_periods=6).max()
    )
    prior_low = frame.groupby(day).low.transform(
        lambda rows: rows.shift(1).rolling(6, min_periods=6).min()
    )
    previous = frame.previous_spot_close
    previous_vwap = frame.groupby(day).vwap.shift(1)
    gap_up = frame.day_open.sub(frame.previous_day_close).gt(frame.atr * 0.5)
    gap_down = frame.previous_day_close.sub(frame.day_open).gt(frame.atr * 0.5)
    masks = {
        "ORB_CONTROL": (
            base & up & frame.close.gt(frame.opening_range_high),
            base & down & frame.close.lt(frame.opening_range_low),
        ),
        "ROLLING_30M": (
            base & up & frame.close.gt(prior_high),
            base & down & frame.close.lt(prior_low),
        ),
        "VWAP_PULLBACK": (
            base & up & frame.ema20.gt(frame.ema50) & previous.le(frame.ema9)
            & frame.low.le(frame.ema9) & frame.close.gt(frame.ema9)
            & frame.rsi.between(45, 72),
            base & down & frame.ema20.lt(frame.ema50) & previous.ge(frame.ema9)
            & frame.high.ge(frame.ema9) & frame.close.lt(frame.ema9)
            & frame.rsi.between(28, 55),
        ),
        "RANGE_REJECTION": (
            base & inside & frame.low.le(frame.opening_range_low)
            & frame.close.gt(frame.opening_range_low) & frame.close.gt(frame.open)
            & frame.futures_close.gt(frame.futures_vwap),
            base & inside & frame.high.ge(frame.opening_range_high)
            & frame.close.lt(frame.opening_range_high) & frame.close.lt(frame.open)
            & frame.futures_close.lt(frame.futures_vwap),
        ),
        "VWAP_REVERSION": (
            base & frame.rsi.le(45) & previous.lt(previous_vwap - frame.atr)
            & frame.close.gt(previous) & frame.close.lt(frame.vwap),
            base & frame.rsi.ge(55) & previous.gt(previous_vwap + frame.atr)
            & frame.close.lt(previous) & frame.close.gt(frame.vwap),
        ),
        "GAP_FADE": (
            base & gap_down & frame.close.gt(frame.day_open)
            & frame.close.lt(frame.previous_day_close) & frame.close.gt(previous)
            & frame.futures_close.gt(frame.futures_vwap),
            base & gap_up & frame.close.lt(frame.day_open)
            & frame.close.gt(frame.previous_day_close) & frame.close.lt(previous)
            & frame.futures_close.lt(frame.futures_vwap),
        ),
    }
    for name, (calls, puts) in masks.items():
        if (calls & puts).any():
            raise ValueError(f"Conflicting sides in {name}")
    return masks, valid_days


def first_signals(frame: pd.DataFrame, masks) -> dict[str, pd.DataFrame]:
    choices = {}
    for name, (calls, puts) in masks.items():
        rows = frame.loc[calls | puts].copy()
        rows["direction"] = np.where(calls.loc[rows.index], "CE", "PE")
        rows["family"] = name
        choices[name] = rows.sort_values("timestamp").drop_duplicates("trading_day")
    combined = pd.concat([choices[name] for name in ALTERNATIVES], ignore_index=True)
    combined["priority"] = combined.family.map({name: index for index, name in enumerate(ALTERNATIVES)})
    choices["FIVE_SETUP_FIRST"] = (
        combined.sort_values(["timestamp", "priority"]).drop_duplicates("trading_day")
    )
    return choices


def contract_catalog(conn: sqlite3.Connection):
    catalog = defaultdict(list)
    query = (
        "SELECT instrument_key,option_type,strike,expiry,lot_size,tick_size,exchange,"
        "valid_from,valid_to,metadata_verified FROM instruments WHERE provider='GROWW' "
        "AND underlying_code='NIFTY' AND instrument_kind='OPTION'"
    )
    for key, side, strike, expiry, lot, tick, exchange, start, end, verified in conn.execute(query):
        catalog[(side, date.fromisoformat(expiry))].append(SimpleNamespace(
            instrument_key=key, option_type=side, strike=D(str(strike)),
            expiry=date.fromisoformat(expiry), lot_size=int(lot), tick_size=D(str(tick)),
            exchange=exchange, valid_from=date.fromisoformat(start),
            valid_to=date.fromisoformat(end) if end else None,
            metadata_verified=bool(verified),
        ))
    expiries = {side: sorted(expiry for name, expiry in catalog if name == side)
                for side in ("CE", "PE")}
    return catalog, expiries


def candidates_for_day(catalog, expiries, side, day):
    for expiry in expiries[side]:
        if expiry <= day:
            continue
        rows = [item for item in catalog[(side, expiry)]
                if item.metadata_verified and item.valid_from <= day
                and (item.valid_to is None or item.valid_to >= day)]
        if rows:
            return rows
    return []


def option_bars(conn, key, signal, square_off):
    result = {}
    for timestamp, opening, high, low, close, volume in conn.execute(
        "SELECT timestamp,open,high,low,close,volume FROM candles WHERE provider='GROWW' "
        "AND instrument_key=? AND data_role='OPTION' AND timeframe='5minute' "
        "AND source_verified=1 AND timestamp>=? AND timestamp<=? ORDER BY timestamp",
        (key, signal.strftime("%Y-%m-%d %H:%M:%S.%f"),
         square_off.strftime("%Y-%m-%d %H:%M:%S.%f")),
    ):
        moment = utc(timestamp)
        result[moment] = SimpleNamespace(timestamp=moment, open=D(str(opening)),
                                         high=D(str(high)), low=D(str(low)),
                                         close=D(str(close)), volume=volume)
    return result


def evaluate(conn, selected, catalog, expiries, spot_by_time, arm, cached_contracts):
    records = []
    for row in selected.itertuples():
        signal, day = utc(row.timestamp), row.trading_day
        square_off = datetime.combine(day, time(15, 20), tzinfo=IST).astimezone(UTC)
        record = {"arm": arm, "family": row.family, "day": day.isoformat(),
                  "signal": signal.isoformat(), "direction": row.direction,
                  "status": "INCOMPLETE", "reason": "no_causal_option_contract"}
        lookup = (day, row.direction)
        if lookup not in cached_contracts:
            cached_contracts[lookup] = candidates_for_day(catalog, expiries, row.direction, day)
        choices = option_candidates(cached_contracts[lookup], spot=D(str(row.close)),
                                    direction=row.direction, trading_day=day)
        chosen = None
        for item, moneyness in choices:
            candle = conn.execute(
                "SELECT close,volume FROM candles WHERE provider='GROWW' AND instrument_key=? "
                "AND data_role='OPTION' AND timeframe='5minute' AND source_verified=1 "
                "AND timestamp=?", (item.instrument_key,
                                    signal.strftime("%Y-%m-%d %H:%M:%S.%f")),
            ).fetchone()
            if candle and candle[0] > 0 and candle[1] > 0:
                chosen = item, moneyness
                break
        if chosen is not None:
            item, moneyness = chosen
            record.update(instrument_key=item.instrument_key, moneyness=moneyness,
                          lot_size=item.lot_size)
            path = option_bars(conn, item.instrument_key, signal, square_off)
            entry = path.get(signal + timedelta(minutes=5))
            if entry is None:
                record["reason"] = "missing_exact_entry_bar"
            elif entry.open <= 0 or entry.volume < 1:
                record["reason"] = "invalid_or_illiquid_entry_bar"
            else:
                entry_price = entry.open * D("1.0005")
                reserve = option_charges(entry_price, entry_price, item.lot_size,
                                         "GROWW", day, item.exchange)
                if entry_price * item.lot_size + reserve > D("100000"):
                    record.update(status="INELIGIBLE", reason="one_lot_cash")
                elif entry_price * item.lot_size * D("0.25") > D("5000"):
                    record.update(status="INELIGIBLE", reason="one_lot_risk")
                else:
                    close, atr = D(str(row.close)), D(str(row.atr))
                    stop = close - atr if row.direction == "CE" else close + atr
                    target = close + D("1.5") * atr if row.direction == "CE" else close - D("1.5") * atr
                    args = dict(stop=stop, target=target, direction=row.direction,
                                quantity=item.lot_size, premium_stop_pct=D("25"),
                                provider="GROWW", exchange=item.exchange)
                    normal = replay(signal, path, spot_by_time, slip_pct=D("0.05"), **args)
                    stressed = replay(signal, path, spot_by_time, slip_pct=D("0.10"), **args)
                    record.update(status=normal["status"], reason=normal.get("reason"),
                                  net=float(normal["net"]) if normal["status"] == "COMPLETE" else None,
                                  charges=float(normal["fees"]) if normal["status"] == "COMPLETE" else None,
                                  entry_proxy=float(normal["entry_proxy"]) if normal["status"] == "COMPLETE" else None,
                                  exit_proxy=float(normal["exit_proxy"]) if normal["status"] == "COMPLETE" else None,
                                  exit_candle=clean(normal.get("exit_candle_start")),
                                  exit_reason=normal.get("exit_reason"),
                                  stress_status=stressed["status"],
                                  stress_net=float(stressed["net"]) if stressed["status"] == "COMPLETE" else None)
        records.append(record)
    return records


def metrics(rows):
    complete = [row for row in rows if row["status"] == "COMPLETE"]
    stressed = [row for row in complete if row.get("stress_status") == "COMPLETE"]
    positive = [row["net"] for row in complete if row["net"] > 0]
    negative = [row["net"] for row in complete if row["net"] < 0]
    equity = peak = 100000.0
    drawdown = 0.0
    monthly = defaultdict(float)
    for row in complete:
        equity += row["net"]
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
        monthly[row["day"][:7]] += row["net"]
    return {"first_daily_signals": len(rows), "completed": len(complete),
            "coverage": len(complete) / len(rows) if rows else None,
            "status_counts": {name: sum(row["status"] == name for row in rows)
                              for name in sorted({row["status"] for row in rows})},
            "net_wins": len(positive),
            "win_rate_pct": 100 * len(positive) / len(complete) if complete else None,
            "net_pnl": sum(row["net"] for row in complete),
            "estimated_charges": sum(row["charges"] for row in complete),
            "double_slippage_net": sum(row["stress_net"] for row in stressed),
            "double_slippage_coverage": len(stressed) / len(complete) if complete else None,
            "profit_factor": sum(positive) / -sum(negative) if negative else None,
            "max_drawdown": drawdown, "positive_months": sum(value > 0 for value in monthly.values()),
            "active_months": len(monthly), "monthly_net": dict(sorted(monthly.items()))}


def paired_by_day(candidate, control, days):
    first = {row["day"]: row for row in candidate}
    second = {row["day"]: row for row in control}
    differences, unknown = [], 0
    for day in days:
        a, b = first.get(day), second.get(day)
        if any(row is not None and row["status"] != "COMPLETE" for row in (a, b)):
            unknown += 1
            continue
        differences.append((a["net"] if a else 0) - (b["net"] if b else 0))
    if not differences:
        return {"known_days": 0, "unknown_days": unknown, "mean_difference": None,
                "bootstrap_95pct": None}
    values = np.asarray(differences)
    rng = np.random.default_rng(42)
    means = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return {"known_days": len(values), "unknown_days": unknown,
            "mean_difference": float(values.mean()),
            "bootstrap_95pct": np.quantile(means, [0.025, 0.975]).tolist()}


def run(spec_path: Path, output: Path):
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    database = Path(spec["source_database"])
    if digest(Path(__file__)) != spec["code_sha256"] or digest(database) != spec["database_sha256"]:
        raise ValueError("Frozen code or database changed")
    if tuple(spec["arms"]) != ARMS or spec["end_exclusive"] != END_EXCLUSIVE:
        raise ValueError("Frozen arm/date specification changed")
    if output.exists():
        raise ValueError("Research output exists; refusing to overwrite")
    conn = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    try:
        frame = load_market(conn)
        masks, valid_days = setup_masks(frame)
        choices = first_signals(frame, masks)
        catalog, expiries = contract_catalog(conn)
        spot_by_time = {utc(row.timestamp): SimpleNamespace(high=D(str(row.high)),
                        low=D(str(row.low))) for row in frame.itertuples()}
        cache = {}
        results = {arm: evaluate(conn, choices[arm], catalog, expiries, spot_by_time, arm, cache)
                   for arm in ARMS}
    finally:
        conn.close()
    periods = {"development_2023_2025": lambda day: day < "2026-01-01",
               "exposed_2026": lambda day: "2026-01-01" <= day < END_EXCLUSIVE}
    report = {"status": "COMPLETE", "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
              "research_only": True, "activation_eligible": False,
              "periods": {}, "limitations": [
                  "All historical periods have been inspected in prior research; no fresh final test.",
                  "Option candle OHLC is not executable bid/ask, quote freshness, delta or IV.",
                  "Conservative five-minute stop-first paths can differ from actual quote exits.",
                  "One lot and one first signal per day; not the full current runtime portfolio.",
                  "The five families were motivated by a visible November 4 market, which was excluded.",
                  "No model trained or selected and no app or broker state changed.",
              ]}
    for period, predicate in periods.items():
        days = [day.isoformat() for day in valid_days if predicate(day.isoformat())]
        subset = {arm: [row for row in results[arm] if predicate(row["day"])] for arm in ARMS}
        report["periods"][period] = {
            "eligible_sessions": len(days),
            "arms": {arm: metrics(subset[arm]) for arm in ARMS},
            "paired_combined_vs_orb": paired_by_day(
                subset["FIVE_SETUP_FIRST"], subset["ORB_CONTROL"], days),
        }
    exposed = report["periods"]["exposed_2026"]
    selected = exposed["arms"]["FIVE_SETUP_FIRST"]
    control = exposed["arms"]["ORB_CONTROL"]
    paired = exposed["paired_combined_vs_orb"]
    gates = {
        "at_least_100_completed": selected["completed"] >= 100,
        "selected_coverage_at_least_99pct": selected["coverage"] is not None and selected["coverage"] >= .99,
        "control_coverage_at_least_99pct": control["coverage"] is not None and control["coverage"] >= .99,
        "net_win_rate_above_70pct": selected["win_rate_pct"] is not None and selected["win_rate_pct"] > 70,
        "profit_factor_at_least_1_3": selected["profit_factor"] is not None and selected["profit_factor"] >= 1.3,
        "positive_net_and_double_slippage": selected["net_pnl"] > 0
            and selected["double_slippage_net"] > 0
            and selected["double_slippage_coverage"] is not None
            and selected["double_slippage_coverage"] >= .99,
        "drawdown_at_most_10pct_capital": selected["max_drawdown"] <= 10000,
        "positive_months_at_least_60pct": selected["active_months"] >= 6
            and selected["positive_months"] / selected["active_months"] >= .6,
        "paired_improvement_lower_95pct_above_zero": paired["bootstrap_95pct"] is not None
            and paired["bootstrap_95pct"][0] > 0,
    }
    report["exposed_2026_gates"] = gates
    report["all_gates_passed"] = all(gates.values())
    output.mkdir(parents=True)
    (output / "specification.json").write_bytes(spec_bytes)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for arm in ARMS:
        with (output / f"{arm.lower()}-first-daily.jsonl").open("x") as handle:
            for row in results[arm]:
                handle.write(json.dumps(clean(row), allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.spec, args.output)
    print(json.dumps({"status": result["status"], "periods": result["periods"],
                      "gates": result["exposed_2026_gates"]}, indent=2))
