from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from dataclasses import replace
from types import SimpleNamespace

from app.domain import Contract, Quote
from app.option_selector import historical_option_choice, option_candidates, select_option


def contract(strike: str, option_type: str = "CE") -> Contract:
    return Contract(f"key-{strike}", f"NIFTY-{strike}-{option_type}", "NSE", "NIFTY", date.today() + timedelta(days=7), Decimal(strike), option_type, 50, Decimal("0.05"))


def quote(item: Contract, spread: str = "1", delta: float = 0.52) -> Quote:
    bid = Decimal("100")
    return Quote(item.instrument_key, datetime.now(UTC), bid, bid + Decimal(spread), Decimal("100.5"), 1000, 5000, delta, 14.2)


def test_atm_selected_before_itm():
    items = [contract("24950"), contract("25000"), contract("25050")]
    quotes = {item.instrument_key: quote(item) for item in items}
    selected = select_option(items, quotes, spot=Decimal("25010"), direction="CE", trading_day=date.today())
    assert selected.moneyness == "ATM"
    assert selected.contract.strike == Decimal("25000")


def test_itm1_fallback_when_atm_spread_is_too_wide():
    items = [contract("24950"), contract("25000"), contract("25050")]
    quotes = {item.instrument_key: quote(item) for item in items}
    quotes["key-25000"] = quote(items[1], spread="5")
    selected = select_option(items, quotes, spot=Decimal("25000"), direction="CE", trading_day=date.today())
    assert selected.moneyness == "ITM1"
    assert selected.contract.strike == Decimal("24950")


def test_expiry_day_contract_is_not_eligible():
    item = Contract("today", "NIFTY", "NSE", "NIFTY", date.today(), Decimal("25000"), "CE", 50, Decimal("0.05"))
    selected = select_option([item], {"today": quote(item)}, spot=Decimal("25000"), direction="CE", trading_day=date.today())
    assert selected.contract is None


def test_shared_policy_handles_pe_itm_direction_and_atm_tie():
    items = [contract("24950", "PE"), contract("25000", "PE"), contract("25050", "PE")]
    choices = option_candidates(items[::-1], spot=Decimal("25025"), direction="PE", trading_day=date.today())
    assert [(item.strike, label) for item, label in choices] == [(Decimal("25000"), "ATM"), (Decimal("25050"), "ITM1")]


def test_expiry_from_wrong_direction_does_not_hide_valid_contract():
    ce = contract("25000")
    pe = replace(contract("25000", "PE"), expiry=date.today() + timedelta(days=1))
    choices = option_candidates([pe, ce], spot=Decimal("25000"), direction="CE", trading_day=date.today())
    assert choices == [(ce, "ATM")]


def test_historical_fallback_uses_signal_evidence_only():
    atm, itm = contract("25000"), contract("24950")
    choices = [(atm, "ATM"), (itm, "ITM1")]
    signals = {atm.instrument_key: SimpleNamespace(close=100, volume=0), itm.instrument_key: SimpleNamespace(close=120, volume=10)}
    assert historical_option_choice(choices, signals) == (itm, "ITM1")
    signals[atm.instrument_key].volume = 10
    # Future paths are deliberately not inputs to the selection function.
    assert historical_option_choice(choices, signals) == (atm, "ATM")


def test_future_timestamp_quote_is_rejected():
    item = contract("25000")
    observed = datetime.now(UTC)
    snapshot = replace(quote(item), timestamp=observed + timedelta(seconds=10))
    chosen = select_option([item], {item.instrument_key: snapshot}, spot=Decimal("25000"), direction="CE", trading_day=date.today(), now=observed)
    assert chosen.contract is None
    assert "future quote" in chosen.reason
