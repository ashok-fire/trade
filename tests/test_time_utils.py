from datetime import UTC, datetime

from app.time_utils import format_ist, to_ist


def test_naive_database_timestamp_is_rendered_as_ist():
    stored_utc = datetime(2026, 10, 23, 3, 45)
    assert format_ist(stored_utc) == "23 Oct 2026, 09:15:00 AM IST"


def test_aware_timestamp_keeps_the_same_instant_in_ist():
    stored_utc = datetime(2026, 10, 23, 3, 55, tzinfo=UTC)
    converted = to_ist(stored_utc)
    assert converted.utcoffset().total_seconds() == 19_800
    assert converted.strftime("%H:%M") == "09:25"
