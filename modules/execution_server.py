"""
modules/execution_server.py
────────────────────────────
FastAPI server that bridges the Python trading bot with MT5 and external consumers.

Endpoints:
  Internal (API key required):
    POST /signals         — Bot pushes a new signal
    POST /bot/heartbeat   — Bot pushes status heartbeat each scan cycle
    POST /position-events — Bot pushes position lifecycle events (e.g. close)

  Public (no API key):
    GET  /signals/current — Returns all current signals in JSON
    GET  /bot/status      — Returns bot frequency, uptime, scan stats
    GET  /analysis/{sym}  — Latest analysis for a specific symbol
    GET  /poll/{symbol}   — MQL5 EA polls for pending trade signals
    GET  /poll/position-events/{symbol} — MQL5 EA polls for pending position events
    GET  /health          — Health check
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional
import uvicorn
import time
import os
from datetime import datetime, timezone
from modules.market_data_store import market_data_store
from analytics import engine as analytics_engine
from db import repository
from db.connection import DatabaseUnavailableError, db_configured, test_connection
from db.schema import init_schema
from utils.logger import get_logger

log = get_logger(__name__)


def _scheduled(fn, name: str):
    """Wrap an analytics job so a failure is logged instead of killing the scheduler thread."""
    def runner() -> None:
        try:
            fn()
            log.info(f"[DB] Scheduled job {name} finished")
        except Exception as exc:
            log.error(f"[DB] Scheduled job {name} failed: {exc}")
    return runner


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialise the DB schema and run the analytics scheduler for the app's lifetime."""
    scheduler = None
    if db_configured():
        if test_connection():
            try:
                init_schema()
            except Exception as exc:
                log.critical(f"[DB] Schema init failed: {exc}")
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler(timezone="UTC")
        scheduler.add_job(_scheduled(analytics_engine.generate_performance_report, "daily_report"),
                          "cron", hour=0, minute=5)
        scheduler.add_job(_scheduled(analytics_engine.compute_factor_effectiveness, "factor_refresh"),
                          "interval", hours=6)
        scheduler.start()
        log.info("[DB] Analytics scheduler started")
    else:
        log.critical("[DB] DB_* environment variables not set — running with file-logging fallback")
    yield
    if scheduler is not None:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="Trading Bot Execution Bridge",
    description="API for trading signal distribution and bot monitoring",
    version="2.0.0",
    lifespan=lifespan,
)


@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable_handler(_request: Request, exc: DatabaseUnavailableError):
    """Return HTTP 503 (instead of crashing) when MySQL is unreachable."""
    return JSONResponse(status_code=503, content={"detail": "Database unavailable"})

# ─── Security ────────────────────────────────────────────────────────────────
API_KEY = os.getenv("EXECUTION_BRIDGE_KEY", "default_secret_key")
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)
REQUIRE_PUSH_API_KEY = os.getenv("REQUIRE_PUSH_API_KEY", "false").lower() in ("1", "true", "yes", "on")


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")


# ─── Models ──────────────────────────────────────────────────────────────────

class TradeSignal(BaseModel):
    symbol: str
    direction: str  # "long" | "short" | "hold" | "BUY" | "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reasoning: str = ""
    timestamp: float = time.time()
    fundamental_rating: Optional[int] = None
    fundamental_conviction: Optional[str] = None
    fundamental_note: Optional[str] = None


class BotHeartbeat(BaseModel):
    last_scan_time: str
    symbols_scanned: List[str]
    scan_interval: int
    timeframes_analyzed: List[str]
    mode: str = "paper"
    total_signals_generated: int = 0


class Candle(BaseModel):
    t: int
    o: float
    h: float
    l: float
    c: float
    v: float


class DataBundle(BaseModel):
    symbol: str
    timeframes: Dict[str, List[Candle]]


class PositionEvent(BaseModel):
    symbol: str
    event_type: str  # e.g. "POSITION_CLOSED"
    reason: str = ""
    exit_price: float
    pnl: float
    timestamp: float = time.time()


# ─── In-Memory State ────────────────────────────────────────────────────────

# Execution signals (consumed by MT5 EA)
pending_signals: Dict[str, TradeSignal] = {}
pending_position_events: Dict[str, List[PositionEvent]] = {}

# Latest analysis per symbol (persistent until replaced)
latest_analysis: Dict[str, TradeSignal] = {}

# Signal history (all signals ever generated this session)
signal_history: List[Dict[str, Any]] = []

# Bot operational status
bot_status: Dict[str, Any] = {
    "status": "waiting",           # "waiting" until first heartbeat, then "running"
    "server_start_time": time.time(),
    "last_scan_time": None,
    "scan_interval_seconds": 60,
    "symbols_tracked": [],
    "timeframes_analyzed": [],
    "mode": "paper",
    "total_signals_generated": 0,
    "signals_per_symbol": {},
}


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL ENDPOINTS (API key required)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/signals")
async def post_signal(signal: TradeSignal, api_key: str = Security(get_api_key)):
    """Python bot posts a signal here."""
    latest_analysis[signal.symbol] = signal

    if signal.direction.upper() in ["LONG", "SHORT", "BUY", "SELL"]:
        pending_signals[signal.symbol] = signal

    # Track in history
    signal_record = {
        "symbol": signal.symbol,
        "direction": signal.direction.upper(),
        "confidence": signal.confidence,
        "entry_price": signal.entry_price,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        "reasoning": signal.reasoning,
        "timestamp": signal.timestamp,
        "received_at": time.time(),
    }
    signal_history.append(signal_record)

    # Update per-symbol stats
    sym = signal.symbol
    if sym not in bot_status["signals_per_symbol"]:
        bot_status["signals_per_symbol"][sym] = {"count": 0, "last_signal_time": None}
    bot_status["signals_per_symbol"][sym]["count"] += 1
    bot_status["signals_per_symbol"][sym]["last_signal_time"] = signal.timestamp
    bot_status["total_signals_generated"] = len(signal_history)

    return {"status": "queued", "symbol": signal.symbol}


@app.post("/bot/heartbeat")
async def bot_heartbeat(heartbeat: BotHeartbeat, api_key: str = Security(get_api_key)):
    """Bot sends a heartbeat after each scan cycle to report its status."""
    bot_status["status"] = "running"
    bot_status["last_scan_time"] = heartbeat.last_scan_time
    bot_status["scan_interval_seconds"] = heartbeat.scan_interval
    bot_status["symbols_tracked"] = heartbeat.symbols_scanned
    bot_status["timeframes_analyzed"] = heartbeat.timeframes_analyzed
    bot_status["mode"] = heartbeat.mode
    bot_status["total_signals_generated"] = heartbeat.total_signals_generated

    return {"status": "ok"}


@app.post("/update-data/{symbol}")
async def update_data(symbol: str, bundle: DataBundle, x_api_key: Optional[str] = Security(api_key_header)):
    """
    EA pushes multi-timeframe OHLCV bundle for low-latency local inference.

    If REQUIRE_PUSH_API_KEY=true, validates X-API-KEY against EXECUTION_BRIDGE_KEY.
    If false, header is optional but still accepted.
    """
    if REQUIRE_PUSH_API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

    if bundle.symbol.upper() != symbol.upper():
        raise HTTPException(status_code=400, detail="Path symbol and payload symbol mismatch")

    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for tf, candles in bundle.timeframes.items():
        normalized[tf] = [c.model_dump() for c in candles]

    summary = market_data_store.update_bundle(bundle.symbol, normalized)
    return {"status": "ok", "ingested": summary}


@app.post("/position-events")
async def post_position_event(event: PositionEvent, api_key: str = Security(get_api_key)):
    """Python bot posts position lifecycle events here (e.g. close events)."""
    symbol = event.symbol.upper()
    if symbol not in pending_position_events:
        pending_position_events[symbol] = []
    pending_position_events[symbol].append(event)
    return {"status": "queued", "symbol": symbol, "event_type": event.event_type}


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE-BACKED ENDPOINTS (API key required)
# ═══════════════════════════════════════════════════════════════════════════════

class InjectedSignal(BaseModel):
    """Payload for POST /debug/signal/inject (extra fields are stored as context)."""
    model_config = ConfigDict(extra="allow")
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    confidence: Optional[float] = None
    open_trade: bool = True


class TradeOutcome(BaseModel):
    """Payload for PUT /trades/{id}/outcome."""
    exit_price: float
    exit_reason: str
    pips_gained: Optional[float] = None
    profit_loss_usd: Optional[float] = None


def _valid_date(value: Optional[str], name: str) -> Optional[str]:
    """Validate a YYYY-MM-DD query parameter."""
    if value is None:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=422, detail=f"{name} must be YYYY-MM-DD")
    return value


@app.post("/debug/signal/inject")
def inject_signal(payload: InjectedSignal, api_key: str = Security(get_api_key)):
    """Inject a test signal (and optionally an OPEN trade) into the DB in one transaction."""
    data = payload.model_dump()
    if not repository.normalize_direction(data["direction"]):
        raise HTTPException(status_code=422, detail="direction must be LONG/SHORT/BUY/SELL")
    open_trade = data.pop("open_trade")
    if open_trade:
        signal_id, trade_id = repository.insert_signal_and_trade(data, {}, source="INJECTED")
    else:
        signal_id, trade_id = repository.insert_signal(data, source="INJECTED"), None
    return {"status": "injected", "signal_id": signal_id, "trade_id": trade_id}


@app.get("/signals")
def list_signals_endpoint(
    page: int = 1, page_size: int = Query(50, le=200), tier: Optional[str] = None,
    symbol: Optional[str] = None, gate_rejected: Optional[bool] = None,
    api_key: str = Security(get_api_key),
):
    """Paginated signal list, filterable by tier/symbol/gate_rejected."""
    return repository.list_signals(page, page_size, tier, symbol, gate_rejected)


@app.get("/trades")
def list_trades_endpoint(
    page: int = 1, page_size: int = Query(50, le=200), symbol: Optional[str] = None,
    status: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None,
    api_key: str = Security(get_api_key),
):
    """Paginated trade list, filterable by symbol/status/opened date range (YYYY-MM-DD)."""
    return repository.list_trades(page, page_size, symbol, status,
                                  _valid_date(date_from, "date_from"), _valid_date(date_to, "date_to"))


@app.put("/trades/{trade_id}/outcome")
def update_trade_outcome_endpoint(trade_id: int, outcome: TradeOutcome, api_key: str = Security(get_api_key)):
    """Record a trade's outcome and refresh analytics for its symbol."""
    updated = repository.update_trade_outcome(
        trade_id, outcome.exit_price, outcome.exit_reason, outcome.pips_gained, outcome.profit_loss_usd)
    if updated is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    repository._refresh_symbol_analytics(updated["symbol"])
    return updated


@app.get("/analytics/performance")
def analytics_performance(api_key: str = Security(get_api_key)):
    """Full performance report across all symbols."""
    return analytics_engine.generate_performance_report()


@app.get("/analytics/performance/{symbol}")
def analytics_symbol(symbol: str, api_key: str = Security(get_api_key)):
    """Performance stats for one symbol."""
    return analytics_engine.compute_symbol_performance(symbol)


@app.get("/analytics/factors")
def analytics_factors(api_key: str = Security(get_api_key)):
    """Confluence factor effectiveness, most impactful first."""
    return {"factors": analytics_engine.compute_factor_effectiveness()}


@app.get("/analytics/thresholds")
def analytics_thresholds(api_key: str = Security(get_api_key)):
    """Adaptive minimum score and enabled flag for every symbol."""
    return {"thresholds": analytics_engine.get_all_thresholds()}


@app.post("/analytics/refresh")
def analytics_refresh(api_key: str = Security(get_api_key)):
    """Re-run all analytics computations."""
    return analytics_engine.refresh_all()


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENDPOINTS (no API key required)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/signals/current")
async def get_current_signals():
    """
    Returns all available trading signals at the current time.
    This is the public-facing endpoint for external consumers.
    """
    now = time.time()
    signals_list = []

    for symbol, signal in latest_analysis.items():
        age = int(now - signal.timestamp)
        signals_list.append({
            "symbol": signal.symbol,
            "direction": signal.direction.upper(),
            "confidence": signal.confidence,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "reasoning": signal.reasoning,
            "age_seconds": age,
            "is_stale": age > 7200,  # Flag signals older than 2 hours
        })

    # Sort by confidence (highest first)
    signals_list.sort(key=lambda s: s["confidence"], reverse=True)

    return {
        "count": len(signals_list),
        "signals": signals_list,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/bot/status")
async def get_bot_status():
    """
    Returns the bot's operational status: frequency, uptime, scan stats.
    """
    now = time.time()
    uptime = int(now - bot_status["server_start_time"])

    # Build per-symbol stats with age info
    per_symbol = {}
    for sym, stats in bot_status["signals_per_symbol"].items():
        last_time = stats.get("last_signal_time")
        per_symbol[sym] = {
            "count": stats["count"],
            "last_signal_age_seconds": int(now - last_time) if last_time else None,
        }

    return {
        "status": bot_status["status"],
        "mode": bot_status["mode"],
        "scan_interval_seconds": bot_status["scan_interval_seconds"],
        "symbols_tracked": bot_status["symbols_tracked"],
        "timeframes_analyzed": bot_status["timeframes_analyzed"],
        "total_signals_generated": bot_status["total_signals_generated"],
        "last_scan_time": bot_status["last_scan_time"],
        "uptime_seconds": uptime,
        "server_start_time": datetime.fromtimestamp(
            bot_status["server_start_time"], tz=timezone.utc
        ).isoformat(),
        "signals_per_symbol": per_symbol,
    }


@app.get("/signals/history")
async def get_signal_history(limit: int = 50):
    """
    Returns recent signal history (most recent first).
    Optional query param: ?limit=20
    """
    recent = signal_history[-limit:][::-1]  # Most recent first
    return {
        "count": len(recent),
        "total": len(signal_history),
        "signals": recent,
    }


# ─── Existing MT5 Bridge Endpoints ──────────────────────────────────────────

@app.get("/analysis/{symbol}")
async def get_analysis(symbol: str):
    """MQL5 EA retrieves the latest analysis for a symbol."""
    analysis = latest_analysis.get(symbol)
    if analysis:
        return {
            "status": "success",
            "symbol": analysis.symbol,
            "direction": analysis.direction.upper(),
            "confidence": analysis.confidence,
            "reasoning": analysis.reasoning,
            "entry": analysis.entry_price,
            "sl": analysis.stop_loss,
            "tp": analysis.take_profit,
            "age_seconds": int(time.time() - analysis.timestamp),
        }
    return {"status": "no_analysis"}


@app.get("/poll/{symbol}")
async def poll_signal(symbol: str):
    """MQL5 EA polls this endpoint for a specific symbol."""
    signal = pending_signals.pop(symbol, None)
    if signal:
        # Check if signal is too old (e.g. > 5 minutes)
        if time.time() - signal.timestamp > 300:
            return {"status": "expired"}
        response = {
            "status": "new_trade",
            "direction": signal.direction.upper(),
            "entry": signal.entry_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
        }
        if signal.fundamental_rating is not None:
            response["fundamental_rating"] = int(signal.fundamental_rating)
            response["fundamental_conviction"] = (
                signal.fundamental_conviction or "weak"
            )
            response["fundamental_note"] = signal.fundamental_note or ""
        return response
    return {"status": "no_signal"}


@app.get("/poll/position-events/{symbol}")
async def poll_position_event(symbol: str):
    """MQL5 EA polls this endpoint for pending position lifecycle events."""
    queue = pending_position_events.get(symbol.upper(), [])
    if not queue:
        return {"status": "no_event"}

    event = queue.pop(0)
    return {
        "status": "event",
        "symbol": event.symbol.upper(),
        "event_type": event.event_type,
        "reason": event.reason,
        "exit_price": event.exit_price,
        "pnl": event.pnl,
        "timestamp": event.timestamp,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - bot_status["server_start_time"]),
        "signals_tracked": len(latest_analysis),
        "pending_position_event_symbols": len(pending_position_events),
        "data_freshness": market_data_store.freshness_report(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
