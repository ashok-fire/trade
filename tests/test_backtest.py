from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

import pandas as pd

from app.backtest import (
    _intrabar_underlying_outcome,
    _utc_datetime,
    current_strategy_fingerprint,
    metrics,
    model_bundle_fingerprint,
    replay_limits,
    replay_spot_candles,
)
from app.risk import can_enter
from app.models import Candle
from app.strategy import indicators


def test_metrics_are_chronological_and_after_costs():
    rows = [
        {"exit_time": datetime(2025, 2, 1, tzinfo=UTC), "underlying_code": "NIFTY", "option_type": "PE", "moneyness": "ITM1", "net_pnl": Decimal("-500")},
        {"exit_time": datetime(2025, 1, 1, tzinfo=UTC), "underlying_code": "NIFTY", "option_type": "CE", "moneyness": "ATM", "net_pnl": Decimal("1000")},
    ]
    result = metrics(rows, Decimal("100000"))
    assert result["net_pnl"] == 500.0
    assert result["trades"] == 2
    assert result["equity_curve"] == [101000.0, 100500.0]


def test_strategy_fingerprint_is_stable_and_auditable():
    fingerprint = current_strategy_fingerprint()
    assert len(fingerprint) == 64
    assert set(fingerprint) <= set("0123456789abcdef")


def test_model_bundle_fingerprint_covers_manifest_metadata_and_components(tmp_path):
    import json
    model = tmp_path / "candidate.json"
    metadata = tmp_path / "candidate.metadata.json"
    ce, pe, ranker = tmp_path / "ce.cbm", tmp_path / "pe.cbm", tmp_path / "ranker.json"
    model.write_text(json.dumps({"quality_models": {"CE": ce.name, "PE": pe.name}, "ranker": ranker.name}))
    metadata.write_text(json.dumps({"version": "test", "feature_schema": "cost-aware-option-v5"}))
    ce.write_bytes(b"ce")
    pe.write_bytes(b"pe")
    ranker.write_bytes(b"rank")
    first = model_bundle_fingerprint(model)
    ranker.write_bytes(b"changed")
    assert len(first) == 64 and model_bundle_fingerprint(model) != first


def test_backtest_timestamp_normalization_handles_sqlite_naive_values():
    naive = datetime(2025, 1, 2, 9, 15)
    aware = datetime(2025, 1, 2, 9, 20, tzinfo=UTC)

    assert _utc_datetime(naive).tzinfo is UTC
    assert _utc_datetime(naive) < _utc_datetime(aware)


def test_replay_uses_saved_portfolio_trade_and_loss_gates():
    limits = replay_limits({"capital":"100000", "risk_pct":"5",
        "daily_loss_limit_pct":"7", "max_trades_per_day":"20",
        "max_consecutive_losses":"5", "premium_emergency_stop_pct":"25",
        "reward_risk":"1.5"}, None)
    def gate(trades, losses, realized):
        return can_enter(limits=limits, open_positions=0, trades_today=trades,
            consecutive_losses=losses, realized_pnl_today=Decimal(realized),
            quote_fresh=True, broker_connected=True, reconciled=True)[0]
    assert limits.risk_per_trade_pct == 5 and limits.reward_risk == Decimal("1.5")
    assert limits.max_trades_per_day == 8
    assert gate(7, 4, "-6999")
    assert not gate(8, 4, "0")
    assert not gate(1, 5, "0")
    assert not gate(1, 0, "-7000")


def test_replay_warms_first_day_indicators_with_verified_prior_spot_only(session):
    prior_start = datetime(2026, 11, 2, 3, 45, tzinfo=UTC)
    test_start = datetime(2026, 11, 3, 3, 45, tzinfo=UTC)
    rows = []
    for first, price in ((prior_start, 100), (test_start, 110)):
        for index in range(60):
            rows.append(Candle(
                provider="GROWW", instrument_key="NSE-NIFTY", underlying_code="NIFTY",
                data_role="SPOT", timeframe="5minute", timestamp=first + timedelta(minutes=5 * index),
                open=price, high=price + 1, low=price - 1, close=price,
                volume=100, open_interest=0, source_verified=True,
            ))
    rows.append(Candle(
        provider="GROWW", instrument_key="NSE-NIFTY", underlying_code="NIFTY",
        data_role="SPOT", timeframe="5minute", timestamp=prior_start - timedelta(minutes=5),
        open=200, high=201, low=199, close=200, volume=100, open_interest=0,
        source_verified=False,
    ))
    session.add_all(rows)
    session.commit()

    start = datetime.combine(date(2026, 11, 3), time.min, tzinfo=UTC)
    end = datetime.combine(date(2026, 11, 3), time.max, tzinfo=UTC)
    warmup, period = replay_spot_candles(session, "GROWW", "NIFTY", start, end)

    assert len(warmup) == 60 and len(period) == 60
    assert _utc_datetime(warmup[0].timestamp) == prior_start
    assert _utc_datetime(period[0].timestamp) == test_start
    frame = indicators(pd.DataFrame([
        {"timestamp": row.timestamp, "open": float(row.open), "high": float(row.high),
         "low": float(row.low), "close": float(row.close), "volume": row.volume}
        for row in [*warmup, *period]
    ]))
    first_test = frame.loc[frame.timestamp == pd.Timestamp(test_start)].iloc[0]
    assert first_test.ema20 < 110  # A test-period-only replay would start at 110.
    assert first_test.previous_day_close == 100


def test_one_minute_data_resolves_stop_before_later_target(session):
    stamp = datetime(2026, 1, 2, 4, 0, tzinfo=UTC)
    five_minute = Candle(
        provider="UPSTOX", instrument_key="spot", underlying_code="NIFTY", timeframe="5minute",
        timestamp=stamp, open=100, high=106, low=98, close=103, volume=0, open_interest=0,
        source_verified=True, data_role="SPOT",
    )
    session.add_all([
        Candle(provider="UPSTOX", instrument_key="spot", underlying_code="NIFTY", timeframe="1minute", timestamp=stamp, open=100, high=102, low=98, close=99, volume=0, open_interest=0, source_verified=True, data_role="SPOT"),
        Candle(provider="UPSTOX", instrument_key="spot", underlying_code="NIFTY", timeframe="1minute", timestamp=stamp + timedelta(minutes=1), open=99, high=106, low=99, close=105, volume=0, open_interest=0, source_verified=True, data_role="SPOT"),
    ])
    session.commit()
    assert _intrabar_underlying_outcome(session, "UPSTOX", "NIFTY", five_minute, "CE", Decimal("99"), Decimal("105")) == "UNDERLYING_STOP"
