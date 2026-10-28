"""Independent, quote-evidenced counterfactuals. No order/execution dependencies."""
from __future__ import annotations

import asyncio
import json
import math
import uuid
from dataclasses import asdict
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select

from .config import config
from .database import settings_dict
from .domain import Contract
from .market_frames import market_frame_for_boundary
from .models import Instrument, ModelArtifact, RuntimeSetting, Setting, SetupObservation
from .option_selector import option_candidates, select_option
from .providers import provider_name
from .risk import option_charges, paper_fill
from .selection_audit import record_option_selection
from .strategy import FEATURE_SCHEMA_V6, SETUP_FAMILIES, RegimeModel, feature_values, setup_direction
from .v4_training import option_quality_context

IST = ZoneInfo("Asia/Kolkata")
PROTOCOL = "parallel-shadow-v1"


def utc(stamp):
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)


def dumps(value):
    # Missing features are null in exported evidence, not invalid JSON NaN.
    if isinstance(value, dict):
        return {key: dumps(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dumps(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def encode(value):
    return json.dumps(dumps(value), default=str, allow_nan=False)


class ParallelShadowService:
    def __init__(self, market_brokers, session_factory):
        self.market_brokers, self.session_factory = market_brokers, session_factory
        self.model = RegimeModel(config.model_path.parent / ".parallel-unavailable.json")
        self.artifact_id = None
        self.stop_event = asyncio.Event()
        self.last_error = None
        self.model_error = None
        self.last_observation = None
        self.last_monitor = None
        self.waiting_reason = "Waiting for market session"

    def reload_model(self):
        """Freeze the research reference once; never select/promote a trading model."""
        with self.session_factory() as session:
            settings = settings_dict(session)
            frozen = settings.get("parallel_shadow_model_artifact_id", "")
            if frozen and not frozen.isdigit():
                self.model_error = "Invalid frozen research artifact id; outcomes can still be collected"
                return
            artifacts = ([session.get(ModelArtifact, int(frozen))] if frozen else session.scalars(
                select(ModelArtifact).where(ModelArtifact.status == "CANDIDATE")
                .order_by(ModelArtifact.created_at.desc(), ModelArtifact.id.desc())
            ).all())
            for artifact in artifacts:
                if artifact is None:
                    continue
                try:
                    candidate = RegimeModel(Path(artifact.model_path))
                except Exception as exc:
                    self.model_error = f"Research model unavailable: {exc}"
                    continue
                if (candidate.available and candidate.metadata.get("feature_schema") == FEATURE_SCHEMA_V6
                        and set(SETUP_FAMILIES).issubset(candidate.metadata.get("setup_families", []))):
                    self.model, self.artifact_id = candidate, artifact.id
                    self.model_error = None
                    if not frozen:
                        session.get(Setting, "parallel_shadow_model_artifact_id").value = str(artifact.id)
                        session.commit()
                    break

    def health(self):
        return {"artifact_id": self.artifact_id, "model_version": self.model.metadata.get("version"),
                "model_available": self.model.available, "model_error": self.model_error,
                "loop_enabled": config.trading_loop_enabled,
                "research_only": True, "last_observation": self.last_observation,
                "last_monitor": self.last_monitor, "waiting_reason": self.waiting_reason,
                "error": self.last_error}

    def score(self, observation, values):
        if not self.model.available or observation.setup_family not in self.model.metadata.get("setup_families", []):
            observation.score_status = "NO_COMPATIBLE_MODEL"
            return
        providers = self.model.metadata.get("compatible_live_providers", [])
        indices = self.model.metadata.get("indices", [])
        if (providers and observation.provider not in providers) or (indices and observation.underlying_code not in indices):
            observation.score_status = "OUTSIDE_TRAINED_SCOPE"
            return
        try:
            regime, score = self.model.predict(values)
        except Exception as exc:
            observation.score_status = "SCORE_ERROR"
            observation.selection_json = encode({**json.loads(observation.selection_json or "{}"), "scoring_error": str(exc)})
            return
        if regime != "TRADE" or not math.isfinite(score) or not 0 <= score <= 1:
            observation.score_status = "UNSCORABLE_OR_DISABLED"
            return
        observation.quality_score = score
        threshold = self.model.metadata.get("setup_thresholds", {}).get(observation.setup_family, {}).get(observation.direction)
        if threshold is None or not 0 <= float(threshold) <= 1:
            observation.score_status = "RESEARCH_THRESHOLD_DISABLED"
            return
        observation.score_status = "RESEARCH_ONLY"
        observation.research_threshold = float(threshold)
        observation.quality_passed = score >= float(threshold)

    async def observe(self, now=None):
        clock_override = now
        now = utc(now or datetime.now(UTC))
        local = now.astimezone(IST)
        boundary = now.replace(second=0, microsecond=0, minute=now.minute // 5 * 5)
        if local.weekday() >= 5 or not time(9, 35) <= local.time() < time(15, 10):
            self.waiting_reason = "Outside weekday entry window (09:35–15:10 IST)"
            return
        if (now - boundary).total_seconds() < 20:
            return
        with self.session_factory() as session:
            settings = settings_dict(session)
            runtime = session.get(RuntimeSetting, 1)
            if not runtime or runtime.execution_mode != "SHADOW" or settings.get("parallel_shadow_enabled") != "true":
                self.waiting_reason = "New observations require SHADOW mode and enabled observer"
                return
            provider = provider_name(settings.get("active_broker"))
            broker = self.market_brokers[provider]
            for code in filter(None, (item.strip() for item in settings["enabled_indices"].split(","))):
                identities = {family: uuid.uuid5(uuid.NAMESPACE_URL, f"{PROTOCOL}:{provider}:{code}:{boundary.isoformat()}:{family}").hex for family in SETUP_FAMILIES}
                if all(session.get(SetupObservation, key) for key in identities.values()):
                    continue
                result = market_frame_for_boundary(session, provider, code, boundary)
                if result is None:
                    self.waiting_reason = f"Waiting for fresh closed spot/futures candles: {code}"
                    continue
                frame, candle = result
                row, day = frame.iloc[-1], local.date()
                spot = Decimal(candle.close)
                contracts = session.scalars(select(Instrument).where(
                    Instrument.provider == provider, Instrument.underlying_code == code,
                    Instrument.instrument_kind == "OPTION", Instrument.active.is_(True),
                )).all()
                directions = {family: setup_direction(row, family) for family in SETUP_FAMILIES}
                shortlists = {}
                for direction in set(directions.values()) - {None}:
                    shortlists[direction] = [Contract(item.instrument_key, item.trading_symbol, item.exchange, code,
                        item.expiry, item.strike, item.option_type, item.lot_size, item.tick_size, provider=provider)
                        for item, _ in option_candidates(contracts, spot=spot, direction=direction, trading_day=day)]
                keys = sorted({item.instrument_key for items in shortlists.values() for item in items})
                # Include a current underlying quote: no fills at an old candle's spot.
                underlying_key = broker.index_key(code)
                quotes = await broker.quotes(keys + [underlying_key]) if keys else {}
                captured = utc(clock_override or datetime.now(UTC))
                if captured >= boundary + timedelta(minutes=5):
                    self.waiting_reason = "Quote response crossed candle boundary; skipping stale decision"
                    continue
                filters = {"max_spread_pct": Decimal(settings["max_spread_pct"]),
                           "max_quote_age_seconds": int(settings["max_quote_age_seconds"]),
                           "delta_min": float(settings["delta_min"]), "delta_max": float(settings["delta_max"])}
                protocol = {"selection_policy": "ATM_ITM1", "slippage_pct": settings["paper_slippage_pct"],
                            "premium_stop_pct": "25", "atr_stop": 1, "atr_target": 1.5,
                            "quantity_policy": "ONE_LOT", "exit_time_ist": "15:20", "poll_seconds": 15,
                            "max_quote_age_seconds": filters["max_quote_age_seconds"], "quote_filters": filters,
                            "research_artifact_id": self.artifact_id, "feature_schema": self.model.metadata.get("feature_schema"),
                            "fills": "observed ask/bid plus slippage",
                            "limitation": "Polling observes prices, not every intrabar touch; overlapping trials are not portfolio returns"}
                for family, direction in directions.items():
                    if session.get(SetupObservation, identities[family]):
                        continue
                    obs = SetupObservation(id=identities[family], provider=provider, underlying_code=code,
                        setup_family=family, direction=direction or "NONE", signal_time=candle.timestamp,
                        observed_at=captured, protocol_version=PROTOCOL, model_version=self.model.metadata.get("version"),
                        status="NO_SETUP", score_status="NOT_SCORED", reason="Setup conditions not confirmed",
                        protocol_json=encode(protocol))
                    obs.features_json = encode(feature_values(row, {"setup_family": family,
                        "india_vix": candle.india_vix, "pcr": candle.pcr,
                        "futures_basis_pct": candle.futures_basis_pct}))
                    session.add(obs)
                    if not direction:
                        continue
                    selection = select_option(shortlists[direction], quotes, spot=spot, direction=direction,
                        trading_day=day, now=captured, **filters)
                    audit = record_option_selection(session, provider=provider, mode="PARALLEL_SHADOW", underlying_code=code,
                        candle_timestamp=candle.timestamp, model_version=obs.model_version or "none", direction=direction,
                        spot=spot, contracts=shortlists[direction], quotes=quotes, selection=selection,
                        observed_at=captured, filters=filters, scope=family)
                    obs.selection_json = audit.evidence_json
                    obs.status, obs.reason = "QUOTE_REJECTED", selection.reason
                    if not selection.contract:
                        continue
                    contract, quote = selection.contract, selection.quote
                    values = feature_values(row, {"setup_family": family, "india_vix": candle.india_vix,
                        "pcr": candle.pcr, "futures_basis_pct": candle.futures_basis_pct,
                        **option_quality_context(premium=float(quote.last), volume=quote.volume,
                        open_interest=quote.open_interest, expiry=contract.expiry, strike=contract.strike,
                        spot=float(spot), trading_day=day)})
                    obs.features_json = encode(values)
                    self.score(obs, values)
                    underlying = quotes.get(underlying_key)
                    atr = Decimal(str(row.atr))
                    if not fresh(underlying, captured, filters["max_quote_age_seconds"]) or not atr.is_finite() or atr <= 0:
                        obs.reason = "Fresh underlying quote or positive ATR unavailable"
                        continue
                    sign = Decimal(1 if direction == "CE" else -1)
                    obs.instrument_key, obs.trading_symbol, obs.exchange = contract.instrument_key, contract.trading_symbol, contract.exchange
                    obs.quantity, obs.tick_size = contract.lot_size, contract.tick_size
                    obs.entry_time = captured
                    obs.entry_price = paper_fill("BUY", quote.bid, quote.ask, Decimal(protocol["slippage_pct"]), contract.tick_size)
                    obs.premium_stop = obs.entry_price * Decimal("0.75")
                    obs.underlying_stop = underlying.last - sign * atr
                    obs.underlying_target = underlying.last + sign * atr * Decimal("1.5")
                    obs.selection_json = encode({**json.loads(obs.selection_json), "underlying_quote": asdict(underlying)})
                    obs.status, obs.reason = "OPEN", "Counterfactual opened regardless of research score; no order placed"
                session.commit()
                self.last_observation = captured
                self.waiting_reason = "Observing current session"

    async def monitor(self, now=None):
        clock_override = now
        now = utc(now or datetime.now(UTC))
        with self.session_factory() as session:
            opened = session.scalars(select(SetupObservation).where(SetupObservation.status == "OPEN")).all()
            groups = {}
            for obs in opened:
                local, entry_day = now.astimezone(IST), utc(obs.entry_time).astimezone(IST).date()
                if local.date() != entry_day or local.time() >= time(15, 31):
                    obs.status, obs.exit_reason = "INCOMPLETE", "MISSED_SAME_DAY_EXIT"
                    obs.reason = "No executable same-day exit observed; excluded from outcome labels"
                    continue
                groups.setdefault(obs.provider, []).append(obs)
            for provider, observations in groups.items():
                broker = self.market_brokers[provider]
                keys = sorted({key for obs in observations for key in (obs.instrument_key, broker.index_key(obs.underlying_code))})
                quotes = await broker.quotes(keys)
                captured = utc(clock_override or datetime.now(UTC))
                for obs in observations:
                    local = captured.astimezone(IST)
                    if local.date() != utc(obs.entry_time).astimezone(IST).date() or local.time() >= time(15, 31):
                        obs.status, obs.exit_reason = "INCOMPLETE", "MISSED_SAME_DAY_EXIT"
                        continue
                    quote, underlying = quotes.get(obs.instrument_key), quotes.get(broker.index_key(obs.underlying_code))
                    protocol = json.loads(obs.protocol_json)
                    if not fresh(quote, captured, protocol["max_quote_age_seconds"]) or not fresh(underlying, captured, protocol["max_quote_age_seconds"]):
                        continue
                    if utc(quote.timestamp) < utc(obs.entry_time) or utc(underlying.timestamp) < utc(obs.entry_time):
                        continue
                    if not quote.bid.is_finite() or not quote.ask.is_finite() or quote.bid <= 0 or quote.ask < quote.bid:
                        continue
                    reason = None
                    if quote.bid <= obs.premium_stop:
                        reason = "PREMIUM_STOP"
                    elif (obs.direction == "CE" and underlying.last <= obs.underlying_stop) or (obs.direction == "PE" and underlying.last >= obs.underlying_stop):
                        reason = "UNDERLYING_STOP"
                    elif (obs.direction == "CE" and underlying.last >= obs.underlying_target) or (obs.direction == "PE" and underlying.last <= obs.underlying_target):
                        reason = "UNDERLYING_TARGET"
                    elif captured.astimezone(IST).time() >= time(15, 20):
                        reason = "TIME_EXIT"
                    if reason:
                        obs.exit_time, obs.exit_reason = captured, reason
                        obs.exit_price = paper_fill("SELL", quote.bid, quote.ask, Decimal(protocol["slippage_pct"]), obs.tick_size)
                        obs.gross_pnl = (obs.exit_price - obs.entry_price) * obs.quantity
                        obs.charges = option_charges(obs.entry_price, obs.exit_price, obs.quantity,
                            broker=provider, trade_date=captured.astimezone(IST).date(), exchange_name=obs.exchange)
                        obs.net_pnl = obs.gross_pnl - obs.charges
                        obs.exit_evidence_json = encode({"observed_at": captured, "option_quote": asdict(quote), "underlying_quote": asdict(underlying)})
                        obs.status = "CLOSED"
            session.commit()
        self.last_monitor = now

    async def run(self):
        while not self.stop_event.is_set():
            try:
                now = datetime.now(UTC)
                if self.last_monitor is None or (now - self.last_monitor).total_seconds() >= 15:
                    await self.monitor()
                await self.observe()
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc)
            await asyncio.sleep(5)

    def stop(self):
        self.stop_event.set()


def fresh(quote, now, max_age):
    return (quote is not None and quote.last.is_finite() and quote.last > 0
            and 0 <= (utc(now) - utc(quote.timestamp)).total_seconds() <= max_age)
