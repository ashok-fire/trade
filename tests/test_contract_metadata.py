from datetime import date

from app.contract_metadata import historical_nifty_lot_size


def test_nifty_lot_size_preserves_expiry_transition_boundaries():
    assert historical_nifty_lot_size(date(2023, 6, 29))[0] == 50
    assert historical_nifty_lot_size(date(2024, 4, 25))[0] == 50
    assert historical_nifty_lot_size(date(2024, 5, 2))[0] == 25
    assert historical_nifty_lot_size(date(2025, 1, 30))[0] == 25
    assert historical_nifty_lot_size(date(2025, 2, 6))[0] == 75
    assert historical_nifty_lot_size(date(2026, 1, 6))[0] == 65
