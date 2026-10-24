"""Persist decision-time option evidence for exact future selection replay."""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from .models import OptionSelectionAudit
from .domain import Contract, Quote
from .option_selector import Selection, option_candidates, quote_rejection_reason, select_option


def replay_recorded_selection(evidence: dict) -> Selection:
    """Reapply runtime filters at the captured clock, without broker access."""
    contracts, quotes = [], {}
    for item in evidence["candidates"]:
        contract = Contract(
            item["instrument_key"], item["trading_symbol"], item["exchange"], "",
            date.fromisoformat(item["expiry"]), Decimal(item["strike"]),
            evidence["direction"], int(item["lot_size"]), Decimal(item["tick_size"]),
        )
        contracts.append(contract)
        raw = item["quote"]
        if raw:
            quotes[contract.instrument_key] = Quote(
                contract.instrument_key, datetime.fromisoformat(raw["timestamp"]),
                Decimal(raw["bid"]), Decimal(raw["ask"]), Decimal(raw["last"]),
                int(raw["volume"]), int(raw["open_interest"]), raw["delta"], raw["iv"],
            )
    now = datetime.fromisoformat(evidence["decision_time"])
    filters = {**evidence["filters"], "max_spread_pct": Decimal(str(evidence["filters"]["max_spread_pct"]))}
    return select_option(
        contracts, quotes, spot=Decimal(evidence["spot"]), direction=evidence["direction"],
        trading_day=now.astimezone(ZoneInfo("Asia/Kolkata")).date(), now=now, **filters,
    )


def record_option_selection(
    session, *, provider: str, mode: str, underlying_code: str,
    candle_timestamp: datetime, model_version: str, direction: str,
    spot: Decimal, contracts: list, quotes: dict, selection: Selection,
    observed_at: datetime, filters: dict,
) -> None:
    audit_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{provider}:{mode}:{underlying_code}:{candle_timestamp.isoformat()}:{direction}:{model_version}").hex
    if session.get(OptionSelectionAudit, audit_id):
        return
    candidates = option_candidates(
        contracts, spot=spot, direction=direction,
        trading_day=observed_at.astimezone(ZoneInfo("Asia/Kolkata")).date(),
    )
    evidence = {
        "policy": "ATM_ITM1", "spot": str(spot), "direction": direction,
        "decision_time": observed_at.isoformat(), "filters": filters,
        "selected_instrument": selection.contract.instrument_key if selection.contract else None,
        "selected_moneyness": selection.moneyness, "selection_reason": selection.reason,
        "candidates": [],
    }
    for contract, moneyness in candidates:
        quote = quotes.get(contract.instrument_key)
        evidence["candidates"].append({
            "instrument_key": contract.instrument_key,
            "trading_symbol": contract.trading_symbol,
            "expiry": contract.expiry.isoformat(), "strike": str(contract.strike),
            "moneyness": moneyness, "lot_size": contract.lot_size,
            "tick_size": str(contract.tick_size), "exchange": contract.exchange,
            "quote": {
                "timestamp": quote.timestamp.isoformat(),
                "bid": str(quote.bid), "ask": str(quote.ask), "last": str(quote.last),
                "volume": quote.volume, "open_interest": quote.open_interest,
                "delta": quote.delta, "iv": quote.iv,
            } if quote else None,
            "rejection_reason": quote_rejection_reason(quote, now=observed_at, **filters) if quote else "quote unavailable",
        })
    session.add(OptionSelectionAudit(
        id=audit_id, provider=provider, mode=mode, underlying_code=underlying_code,
        candle_timestamp=candle_timestamp, observed_at=observed_at,
        model_version=model_version, evidence_json=json.dumps(evidence, default=str),
    ))
