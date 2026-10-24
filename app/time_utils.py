from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")


def to_ist(value: datetime) -> datetime:
    """Treat naive database timestamps as UTC and return an aware IST value."""
    utc_value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return utc_value.astimezone(IST)


def format_ist(value: datetime | None, pattern: str = "%d %b %Y, %I:%M:%S %p IST") -> str:
    if value is None:
        return "—"
    return to_ist(value).strftime(pattern)
