from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

from sqlalchemy import select

from .domain import Contract
from .data_quality import futures_alignment_gaps
from .models import Candle, Instrument, ResearchChunk
from .providers import BrokerClient, RetryableBrokerError, provider_name


@dataclass(frozen=True)
class SyncResult:
    underlying_candles: int
    futures_candles: int
    option_candles: int
    futures_contracts: int
    option_contracts: int
    warnings: list[str]
    incomplete_reasons: list[str]


def _parse(raw: list) -> tuple[datetime, Decimal, Decimal, Decimal, Decimal, int, int] | None:
    try:
        if len(raw) < 6:
            return None
        stamp = datetime.fromisoformat(str(raw[0]).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        prices = tuple(Decimal(str(raw[index])) for index in range(1, 5))
        if any(not value.is_finite() or value <= 0 for value in prices):
            return None
        open_, high, low, close = prices
        if high < max(open_, low, close) or low > min(open_, high, close):
            return None
        volume_value = Decimal(str(raw[5] or 0))
        oi_value = Decimal(str(raw[6] or 0)) if len(raw) > 6 else Decimal("0")
        if not volume_value.is_finite() or not oi_value.is_finite():
            return None
        return (
            stamp.astimezone(UTC),
            open_,
            high,
            low,
            close,
            max(int(volume_value), 0),
            max(int(oi_value), 0),
        )
    except (ArithmeticError, IndexError, TypeError, ValueError):
        return None


def _chunks(start: date, end: date, days: int = 28):
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=days - 1))
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def _naive_utc(value: datetime) -> datetime:
    return (value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)).replace(tzinfo=None)


def _has_candles(
    session_factory,
    provider: str,
    underlying_code: str,
    data_role: str,
    start: date,
    end: date,
) -> bool:
    with session_factory() as session:
        return session.scalar(
            select(Candle.id).where(
                Candle.provider == provider,
                Candle.underlying_code == underlying_code,
                Candle.data_role == data_role,
                Candle.timeframe == "5minute",
                Candle.timestamp >= datetime.combine(start, datetime.min.time()),
                Candle.timestamp <= datetime.combine(end, datetime.max.time()),
                Candle.source_verified.is_(True),
            ).limit(1)
        ) is not None


def _futures_gap_reasons(session_factory, provider, code, start, end) -> list[str]:
    with session_factory() as session:
        rows = session.execute(
            select(Candle.data_role, Candle.timestamp).where(
                Candle.provider == provider,
                Candle.underlying_code == code,
                Candle.data_role.in_(["SPOT", "FUTURE"]),
                Candle.timeframe == "5minute",
                Candle.source_verified.is_(True),
                Candle.timestamp >= datetime.combine(start, time.min),
                Candle.timestamp <= datetime.combine(end, time.max),
            )
        ).all()
    gaps = futures_alignment_gaps(
        [stamp for role, stamp in rows if role == "SPOT"],
        [stamp for role, stamp in rows if role == "FUTURE"],
    )
    return [
        f"{code}: {len(gaps)} sessions lack 90% aligned futures coverage "
        f"({gaps[0]} to {gaps[-1]})"
    ] if gaps else []


async def _store_candles(
    client: BrokerClient,
    session_factory,
    instrument_key: str,
    underlying_code: str | None,
    start: date,
    end: date,
    *,
    provider: str,
    contract: Contract | None = None,
    data_role: str = "SPOT",
    minutes: int = 5,
    warnings: list[str] | None = None,
    incomplete_reasons: list[str] | None = None,
    refresh_existing: bool = False,
) -> int:
    inserted = 0
    rejected = 0

    def marker(session, chunk_start: date, chunk_end: date) -> ResearchChunk | None:
        return session.scalar(
            select(ResearchChunk).where(
                ResearchChunk.provider == provider,
                ResearchChunk.instrument_key == instrument_key,
                ResearchChunk.timeframe == f"{minutes}minute",
                ResearchChunk.start_date == chunk_start,
                ResearchChunk.end_date == chunk_end,
            )
        )

    def save_marker(chunk_start: date, chunk_end: date, status: str, error: str | None = None) -> None:
        with session_factory() as session:
            item = marker(session, chunk_start, chunk_end)
            if item is None:
                item = ResearchChunk(
                    provider=provider,
                    instrument_key=instrument_key,
                    underlying_code=underlying_code,
                    data_role=data_role,
                    timeframe=f"{minutes}minute",
                    start_date=chunk_start,
                    end_date=chunk_end,
                )
                session.add(item)
            item.status = status
            item.attempts = (item.attempts or 0) + 1
            item.last_error = error[:1000] if error else None
            session.commit()

    async def sync_chunk(chunk_start: date, chunk_end: date) -> int:
        nonlocal inserted, rejected
        with session_factory() as session:
            checkpoint = marker(session, chunk_start, chunk_end)
            if checkpoint is not None and checkpoint.status == "COMPLETE" and not refresh_existing:
                return 0
            split_known_failure = (
                checkpoint is not None
                and checkpoint.status == "FAILED"
                and chunk_start < chunk_end
            )
            # A candle inside a range does not prove that the entire range was
            # downloaded. Without a completed checkpoint, ask the provider and
            # deduplicate individual timestamps when storing the response.

        if split_known_failure:
            span = (chunk_end - chunk_start).days
            midpoint = chunk_start + timedelta(days=span // 2)
            return await sync_chunk(chunk_start, midpoint) + await sync_chunk(
                midpoint + timedelta(days=1), chunk_end
            )

        try:
            rows = (
                await client.historical_option_candles(contract, chunk_start, chunk_end, minutes)
                if contract
                else await client.historical_candles(instrument_key, chunk_start, chunk_end, minutes)
            )
        except RetryableBrokerError as exc:
            save_marker(chunk_start, chunk_end, "FAILED", str(exc))
            if chunk_start < chunk_end:
                span = (chunk_end - chunk_start).days
                midpoint = chunk_start + timedelta(days=span // 2)
                return await sync_chunk(chunk_start, midpoint) + await sync_chunk(
                    midpoint + timedelta(days=1), chunk_end
                )
            message = (
                f"{provider}: {data_role.lower()} candles unavailable for {instrument_key} "
                f"on {chunk_start}: {exc}"
            )
            if warnings is not None and message not in warnings:
                warnings.append(message)
            if incomplete_reasons is not None and message not in incomplete_reasons:
                incomplete_reasons.append(message)
            return 0

        parsed = []
        for row in rows:
            candle = _parse(row)
            if candle is None:
                rejected += 1
                continue
            parsed.append(candle)
        with session_factory() as session:
            existing = {
                _naive_utc(value)
                for value in session.scalars(
                    select(Candle.timestamp).where(
                        Candle.provider == provider,
                        Candle.instrument_key == instrument_key,
                        Candle.timeframe == f"{minutes}minute",
                        Candle.timestamp >= datetime.combine(chunk_start, datetime.min.time()),
                        Candle.timestamp <= datetime.combine(chunk_end, datetime.max.time()),
                    )
                )
            }
            for stamp, open_, high, low, close, volume, oi in parsed:
                if _naive_utc(stamp) in existing:
                    continue
                session.add(
                    Candle(
                        provider=provider,
                        instrument_key=instrument_key,
                        underlying_code=underlying_code,
                        timeframe=f"{minutes}minute",
                        timestamp=stamp,
                        open=open_,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume,
                        open_interest=oi,
                        source_verified=True,
                        data_role=data_role,
                    )
                )
                inserted += 1
            checkpoint = marker(session, chunk_start, chunk_end)
            if checkpoint is None:
                checkpoint = ResearchChunk(
                    provider=provider,
                    instrument_key=instrument_key,
                    underlying_code=underlying_code,
                    data_role=data_role,
                    timeframe=f"{minutes}minute",
                    start_date=chunk_start,
                    end_date=chunk_end,
                )
                session.add(checkpoint)
            checkpoint.status = "COMPLETE"
            checkpoint.attempts = (checkpoint.attempts or 0) + 1
            checkpoint.last_error = None
            session.commit()
        return len(parsed)

    for chunk_start, chunk_end in _chunks(start, end):
        await sync_chunk(chunk_start, chunk_end)
    if rejected and warnings is not None:
        warnings.append(
            f"{provider}: skipped {rejected} malformed {data_role.lower()} candles for {instrument_key}"
        )
    return inserted


def _monthly_expiries(expiries: list[date]) -> list[date]:
    values: dict[tuple[int, int], date] = {}
    for expiry in expiries:
        key = (expiry.year, expiry.month)
        values[key] = max(values.get(key, expiry), expiry)
    return sorted(values.values())


def _store_contract_version(
    session_factory,
    contract: Contract,
    underlying_code: str,
    first_candle: datetime,
) -> bool:
    valid_from = _naive_utc(first_candle).date()
    with session_factory() as session:
        existing = session.scalar(
            select(Instrument).where(
                Instrument.provider == contract.provider,
                Instrument.instrument_key == contract.instrument_key,
                Instrument.valid_from == valid_from,
            )
        )
        if existing is not None:
            return False
        session.add(
            Instrument(
                provider=contract.provider,
                instrument_key=contract.instrument_key,
                underlying_code=underlying_code,
                exchange=contract.exchange,
                trading_symbol=contract.trading_symbol,
                option_type=contract.option_type if contract.option_type in {"CE", "PE"} else "FU",
                strike=contract.strike,
                expiry=contract.expiry,
                lot_size=contract.lot_size,
                tick_size=contract.tick_size,
                valid_from=valid_from,
                valid_to=contract.expiry,
                active=False,
                instrument_kind="FUTURE" if contract.option_type == "FUT" else "OPTION",
                metadata_verified=contract.metadata_verified,
                metadata_source=contract.metadata_source,
            )
        )
        session.commit()
        return True


def _contract_choices(contracts: list[Contract], spot: Decimal) -> list[Contract]:
    selected: list[Contract] = []
    for option_type in ("CE", "PE"):
        side = [item for item in contracts if item.option_type == option_type]
        strikes = sorted({item.strike for item in side})
        if not strikes:
            continue
        atm = min(strikes, key=lambda strike: abs(strike - spot))
        position = strikes.index(atm)
        itm = strikes[max(0, position - 1)] if option_type == "CE" else strikes[min(len(strikes) - 1, position + 1)]
        selected.extend(item for item in side if item.strike in {atm, itm})
    return selected


async def sync_research_data(
    client: BrokerClient,
    session_factory,
    indices: list[str],
    start: date,
    end: date,
    *,
    include_options: bool,
    include_one_minute: bool = False,
    refresh_existing: bool = False,
    progress: Callable[[float, int, str], Awaitable[None]] | None = None,
) -> SyncResult:
    provider = provider_name(client.provider)
    underlying_count = 0
    futures_count = 0
    option_count = 0
    futures_contract_count = 0
    option_contract_count = 0
    warnings: list[str] = []
    incomplete_reasons: list[str] = []
    completed_steps = 0
    total_steps = len(indices) * (3 if include_options else 2)

    for code in indices:
        underlying_count += await _store_candles(
            client,
            session_factory,
            client.index_key(code),
            code,
            start,
            end,
            provider=provider,
            data_role="SPOT",
            warnings=warnings,
            incomplete_reasons=incomplete_reasons,
            refresh_existing=refresh_existing,
        )
        if include_one_minute:
            underlying_count += await _store_candles(
                client,
                session_factory,
                client.index_key(code),
                code,
                start,
                end,
                provider=provider,
                data_role="SPOT",
                minutes=1,
                warnings=warnings,
                incomplete_reasons=incomplete_reasons,
                refresh_existing=refresh_existing,
            )
        completed_steps += 1
        if progress:
            await progress(completed_steps, total_steps, f"{code}: spot history stored")

    # The front contract for the final requested sessions may expire after
    # both the requested end and today. Discovery is allowed to look ahead;
    # candle requests below remain capped at the requested historical end.
    expiry_end = end + timedelta(days=45)
    for code in indices:
        try:
            expiries = await client.historical_expiries(code, start, expiry_end)
        except Exception as exc:
            warnings.append(
                f"{code}: historical futures discovery unavailable from {provider}: {exc}"
            )
            continue
        futures_found = False
        for expiry in _monthly_expiries(expiries):
            try:
                contracts = await client.historical_future_contracts(code, expiry)
            except Exception as exc:
                warnings.append(f"{code}: futures metadata unavailable for {expiry}: {exc}")
                continue
            if not contracts:
                warnings.append(f"{code}: no futures contract listed by {provider} for {expiry}")
            for contract in contracts:
                candle_end = min(end, contract.expiry)
                if candle_end < start:
                    continue
                futures_count += await _store_candles(
                    client,
                    session_factory,
                    contract.instrument_key,
                    code,
                    start,
                    candle_end,
                    provider=provider,
                    contract=contract,
                    data_role="FUTURE",
                    warnings=warnings,
                    incomplete_reasons=incomplete_reasons,
                    refresh_existing=refresh_existing,
                )
                with session_factory() as session:
                    first_candle = session.scalar(
                        select(Candle.timestamp)
                        .where(
                            Candle.provider == provider,
                            Candle.instrument_key == contract.instrument_key,
                            Candle.timeframe == "5minute",
                            Candle.source_verified.is_(True),
                        )
                        .order_by(Candle.timestamp)
                        .limit(1)
                    )
                if first_candle is not None:
                    futures_found = True
                    futures_contract_count += int(
                        _store_contract_version(session_factory, contract, code, first_candle)
                    )
        if not futures_found:
            warnings.append(
                f"{code}: no verified {provider} futures candles were stored; VWAP/volume training is unavailable"
            )
        completed_steps += 1
        if progress:
            await progress(completed_steps, total_steps, f"{code}: futures history stored")

    if not include_options:
        for code in indices:
            incomplete_reasons.extend(_futures_gap_reasons(session_factory, provider, code, start, end))
        incomplete_reasons.extend(
            reason
            for code in indices
            if not _has_candles(session_factory, provider, code, "FUTURE", start, end)
            for reason in [
                f"{code}: no verified {provider} futures candles are available for the requested period"
            ]
            if reason not in incomplete_reasons
        )
        return SyncResult(
            underlying_count,
            futures_count,
            0,
            futures_contract_count,
            0,
            warnings,
            incomplete_reasons,
        )

    for code in indices:
        with session_factory() as session:
            underlying_rows = session.scalars(
                select(Candle)
                .where(
                    Candle.provider == provider,
                    Candle.underlying_code == code,
                    Candle.data_role == "SPOT",
                    Candle.timeframe == "5minute",
                    Candle.timestamp >= datetime.combine(start, datetime.min.time()),
                    Candle.timestamp <= datetime.combine(end, datetime.max.time()),
                    Candle.source_verified.is_(True),
                )
                .order_by(Candle.timestamp)
            ).all()
        if not underlying_rows:
            warnings.append(f"{code}: no {provider} underlying candles available")
            continue

        try:
            expiries = await client.historical_expiries(code, start, expiry_end)
        except Exception as exc:
            warnings.append(f"{code}: historical option discovery unavailable from {provider}: {exc}")
            continue
        if not expiries:
            warnings.append(f"{code}: {provider} returned no historical expiries for this period")
            continue
        contracts_by_expiry: dict[date, list[Contract]] = {}
        selected: dict[str, Contract] = {}
        daily_spot: dict[date, Decimal] = {}
        for row in underlying_rows:
            stamp = row.timestamp.replace(tzinfo=UTC) if row.timestamp.tzinfo is None else row.timestamp.astimezone(UTC)
            daily_spot[stamp.astimezone(ZoneInfo("Asia/Kolkata")).date()] = row.close

        for trading_day, spot in daily_spot.items():
            eligible = [expiry for expiry in expiries if expiry > trading_day]
            if not eligible:
                continue
            expiry = min(eligible)
            if expiry not in contracts_by_expiry:
                contracts_by_expiry[expiry] = await client.historical_contracts(code, expiry)
            for contract in _contract_choices(contracts_by_expiry[expiry], spot):
                selected[contract.instrument_key] = contract

        if not selected:
            warning = f"{code}: no historical ATM/ITM1 contracts were resolved from {provider}"
            if provider == "GROWW":
                warning += "; Groww contract metadata must include historical lot and tick sizes"
            warnings.append(warning)
            continue

        selected_contracts = list(selected.values())
        for contract_index, contract in enumerate(selected_contracts, start=1):
            if progress:
                await progress(
                    completed_steps + (contract_index - 1) / max(len(selected_contracts), 1),
                    total_steps,
                    f"{code}: options {contract_index}/{len(selected_contracts)} · {contract.trading_symbol}",
                )
            candle_end = min(end, contract.expiry)
            if candle_end < start:
                continue
            count = await _store_candles(
                client,
                session_factory,
                contract.instrument_key,
                code,
                start,
                candle_end,
                provider=provider,
                contract=contract,
                data_role="OPTION",
                warnings=warnings,
                incomplete_reasons=incomplete_reasons,
                refresh_existing=refresh_existing,
            )
            option_count += count
            with session_factory() as session:
                first_candle = session.scalar(
                    select(Candle.timestamp)
                    .where(
                        Candle.provider == provider,
                        Candle.instrument_key == contract.instrument_key,
                        Candle.timeframe == "5minute",
                        Candle.source_verified.is_(True),
                    )
                    .order_by(Candle.timestamp)
                    .limit(1)
                )
                if first_candle is None:
                    continue
            option_contract_count += int(
                _store_contract_version(session_factory, contract, code, first_candle)
            )
        completed_steps += 1
        if progress:
            await progress(completed_steps, total_steps, f"{code}: option history stored")

    if option_count == 0 and not any(
        _has_candles(session_factory, provider, code, "OPTION", start, end)
        for code in indices
    ):
        warnings.append(
            f"No verified {provider} historical option candles were stored. The backtest will fail closed instead of manufacturing premiums."
        )
    for code in indices:
        incomplete_reasons.extend(_futures_gap_reasons(session_factory, provider, code, start, end))
        if not _has_candles(session_factory, provider, code, "FUTURE", start, end):
            incomplete_reasons.append(
                f"{code}: no verified {provider} futures candles are available for the requested period"
            )
        if not _has_candles(session_factory, provider, code, "OPTION", start, end):
            incomplete_reasons.append(
                f"{code}: no verified {provider} option candles are available for the requested period"
            )
    return SyncResult(
        underlying_count,
        futures_count,
        option_count,
        futures_contract_count,
        option_contract_count,
        warnings,
        incomplete_reasons,
    )
