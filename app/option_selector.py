from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Protocol, TypeVar

from .domain import Contract, Quote


@dataclass(frozen=True)
class Selection:
    contract: Contract | None
    quote: Quote | None
    moneyness: str | None
    reason: str


class OptionContract(Protocol):
    instrument_key: str
    expiry: date
    strike: Decimal
    option_type: str
    lot_size: int
    tick_size: Decimal


TOption = TypeVar("TOption", bound=OptionContract)


def option_candidates(
    contracts: list[TOption], *, spot: Decimal, direction: str,
    trading_day: date, policy: str = "ATM_ITM1",
) -> list[tuple[TOption, str]]:
    """Shared expiry/strike policy for runtime, training and replay.

    Expiry-day contracts are excluded. ATM ties choose the lower strike;
    ITM1 is the next lower CE or next higher PE strike. Metadata versions
    are filtered for the decision date before choosing the newest version.
    """
    if policy not in {"ATM_ONLY", "ATM_ITM1"}:
        raise ValueError("Unsupported option-selection policy")
    eligible = [
        item for item in contracts
        if item.option_type == direction and item.expiry > trading_day
        and item.lot_size > 0 and item.tick_size > 0
        and getattr(item, "metadata_verified", True)
        and getattr(item, "valid_from", date.min) <= trading_day
        and (getattr(item, "valid_to", None) is None or item.valid_to >= trading_day)
    ]
    if not eligible:
        return []
    expiry = min(item.expiry for item in eligible)
    chain = sorted(
        (item for item in eligible if item.expiry == expiry),
        key=lambda item: (getattr(item, "valid_from", date.min), item.instrument_key),
    )
    by_strike = {item.strike: item for item in chain}
    strikes = sorted(by_strike)
    atm = min(strikes, key=lambda strike: abs(strike - spot))
    index = strikes.index(atm)
    itm_index = max(0, index - 1) if direction == "CE" else min(len(strikes) - 1, index + 1)
    itm = strikes[itm_index]
    result = [(by_strike[atm], "ATM")]
    if policy == "ATM_ITM1" and itm != atm:
        result.append((by_strike[itm], "ITM1"))
    return result


def historical_option_choice(choices: list[tuple[TOption, str]], signal_candles: dict) -> tuple[TOption, str] | None:
    """Choose using signal-time candle evidence, without looking at exits.

    Quote spread, freshness and delta cannot be reconstructed from OHLCV.
    Their absence remains an explicit limit of historical candle research.
    """
    for contract, moneyness in choices:
        candle = signal_candles.get(contract.instrument_key)
        if candle is not None and candle.close > 0 and candle.volume > 0:
            return contract, moneyness
    return None


def _spread_pct(quote: Quote) -> Decimal:
    midpoint = (quote.bid + quote.ask) / 2
    return Decimal("999") if midpoint <= 0 else (quote.ask - quote.bid) / midpoint * 100


def quote_rejection_reason(
    quote: Quote, *, now: datetime, max_spread_pct: Decimal,
    max_quote_age_seconds: int, delta_min: float, delta_max: float,
) -> str | None:
    if not all(value.is_finite() for value in (quote.bid, quote.ask, quote.last)):
        return "invalid bid/ask"
    stamp = quote.timestamp if quote.timestamp.tzinfo else quote.timestamp.replace(tzinfo=UTC)
    age = (now - stamp.astimezone(UTC)).total_seconds()
    checks = (
        (quote.bid > 0 and quote.ask >= quote.bid, "invalid bid/ask"),
        (_spread_pct(quote) <= max_spread_pct, "spread too wide"),
        (0 <= age <= max_quote_age_seconds, "stale or future quote"),
        (quote.volume > 0 and quote.open_interest > 0, "volume or OI unavailable"),
        (quote.delta is not None and delta_min <= abs(quote.delta) <= delta_max, "delta outside range"),
    )
    return next((reason for passed, reason in checks if not passed), None)


def select_option(
    contracts: list[Contract],
    quotes: dict[str, Quote],
    *,
    spot: Decimal,
    direction: str,
    trading_day: date,
    now: datetime | None = None,
    max_spread_pct: Decimal = Decimal("1.5"),
    max_quote_age_seconds: int = 3,
    delta_min: float = 0.45,
    delta_max: float = 0.60,
) -> Selection:
    now = now or datetime.now(UTC)
    candidates = option_candidates(contracts, spot=spot, direction=direction, trading_day=trading_day)
    if not candidates:
        return Selection(None, None, None, "No eligible non-expiry-day contract")
    reasons: list[str] = []
    for contract, moneyness in candidates:
        quote = quotes.get(contract.instrument_key)
        if not quote:
            reasons.append(f"{moneyness}: quote unavailable")
            continue
        failed = quote_rejection_reason(
            quote, now=now, max_spread_pct=max_spread_pct,
            max_quote_age_seconds=max_quote_age_seconds, delta_min=delta_min, delta_max=delta_max,
        )
        if failed is None:
            return Selection(contract, quote, moneyness, f"{moneyness} passed all filters")
        reasons.append(f"{moneyness}: {failed}")
    return Selection(None, None, None, "; ".join(reasons))

