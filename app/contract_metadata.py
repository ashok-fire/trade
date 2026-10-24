from __future__ import annotations

from datetime import date


NSE_NIFTY_LOT_SOURCE = {
    "2024_change": "NSE_FAOP61415",
    "2025_change": "NSE_FAOP64672",
    "2026_change": "NSE_FAOP70616",
}


def historical_nifty_lot_size(expiry: date) -> tuple[int, str] | None:
    """Return the exchange-announced NIFTY option lot size by expiry.

    The transition boundaries use the first affected weekly/monthly expiries,
    rather than the circular publication date, so old contracts retain the
    quantity under which they actually traded.
    """
    if date(2023, 1, 1) <= expiry <= date(2024, 4, 25):
        return 50, NSE_NIFTY_LOT_SOURCE["2024_change"]
    if date(2024, 5, 2) <= expiry <= date(2025, 1, 30):
        return 25, NSE_NIFTY_LOT_SOURCE["2024_change"]
    if date(2025, 2, 1) <= expiry <= date(2025, 12, 30):
        return 75, NSE_NIFTY_LOT_SOURCE["2025_change"]
    if expiry >= date(2026, 1, 1):
        return 65, NSE_NIFTY_LOT_SOURCE["2026_change"]
    return None
