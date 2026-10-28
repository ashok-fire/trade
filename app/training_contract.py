"""Versioned, causal opportunity contract for new training runs.

This does not assert quote/fill parity: historical candles cannot supply it.
Legacy model feature lists and their stored validation remain unchanged.
"""
from datetime import UTC, datetime, time
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

from .strategy import QUALITY_FEATURES

VERSION = "decision-aligned-v1"
EXCLUDED_FEATURES = ("option_volume_log", "option_oi_log")
ALIGNED_FEATURES = [name for name in QUALITY_FEATURES if name not in EXCLUDED_FEATURES]


def entry_window_open(boundary: datetime) -> bool:
    stamp = boundary.replace(tzinfo=UTC) if boundary.tzinfo is None else boundary
    local = stamp.astimezone(ZoneInfo("Asia/Kolkata"))
    return local.weekday() < 5 and time(9, 35) <= local.time() < time(15, 10)


def contract_spec(setup_families):
    return {
        "version": VERSION, "sampling_policy": "all_eligible_bars",
        "strike_policy": "ATM_ITM1", "setup_policy": "first_confirmed",
        "setup_priority": list(setup_families), "causal_opportunities": True,
        "retain_unlabelled": True, "excluded_features": list(EXCLUDED_FEATURES),
        "signal": "closed five-minute bar", "entry_window_ist": "09:35 <= time < 15:10; weekdays",
        "underlying_anchor": "signal_close", "atr_stop": 1, "reward_risk": "1.5",
        "premium_stop_pct": "25", "slippage_pct": "0.05", "square_off_ist": "15:20",
        "label_protocol": "entry-bar-stops-exact-next-bar-contiguous-path-v2",
        "label_source": "historical candle proxy, not actual executable fills",
        "incomplete_outcomes": "unlabelled; never substitute a later outcome for selected daily entry",
        "historical_limits": ["bid_ask_spread", "quote_age", "delta", "actual_stop_touch_option_quote",
                              "signal_close_premium_vs_live_quote_last", "portfolio_risk_and_position_sizing"],
        "evaluation_policy": "one first-passing candidate per day; one historical lot; not full runtime portfolio",
    }


def runtime_contract_error(metadata, settings):
    """Fail closed for new bundles when recorded geometry/scope has drifted."""
    contract = metadata.get("training_contract")
    if not contract:
        return None  # Do not silently reinterpret legacy artifacts.
    if contract.get("version") != VERSION:
        return "Unsupported training contract version"
    for model_key, setting_key in (("reward_risk", "reward_risk"),
            ("premium_stop_pct", "premium_emergency_stop_pct"), ("slippage_pct", "paper_slippage_pct")):
        try:
            if Decimal(str(contract[model_key])) != Decimal(settings[setting_key]):
                return f"{setting_key} differs from the frozen training contract"
        except (KeyError, InvalidOperation):
            return f"Invalid or missing contract setting: {setting_key}"
    if settings.get("square_off_time") != contract.get("square_off_ist"):
        return "Square-off time differs from the frozen training contract"
    if settings.get("active_broker") not in metadata.get("compatible_live_providers", []):
        return "Provider is outside the frozen training scope"
    enabled = {item.strip() for item in settings.get("enabled_indices", "").split(",") if item.strip()}
    if not enabled.issubset(metadata.get("indices", [])):
        return "Enabled indices are outside the frozen training scope"
    return None
