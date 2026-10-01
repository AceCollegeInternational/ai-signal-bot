"""Raw-SQL repositories for signals and trades, with a JSONL fallback when the DB is down."""

import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from db.connection import DatabaseUnavailableError, get_db
from utils.logger import get_logger

log = get_logger(__name__)

FALLBACK_PATH = os.path.join("logs", "db_fallback.jsonl")

EXIT_REASONS = {"TP1", "TP2", "TP3", "SL", "TRAILING_SL", "MANUAL", "TIMEOUT", "CANCELLED"}
SIGNAL_SOURCES = {"LLM", "INJECTED", "BACKTEST"}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def pip_size(symbol: str) -> float:
    """Return the price value of one pip for a symbol (JPY pairs/gold 0.01, else 0.0001)."""
    s = (symbol or "").upper()
    return 0.01 if ("JPY" in s or s.startswith("XAU") or s.startswith("XAG")) else 0.0001


def tier_from_score(score: Optional[float]) -> str:
    """Map a 0-100 confluence/confidence score to a tier label."""
    if score is None:
        return "NO_SIGNAL"
    if score >= 90:
        return "TIER_1"
    if score >= 80:
        return "TIER_2"
    if score >= 70:
        return "TIER_3"
    return "NO_SIGNAL"


def normalize_direction(value: Any) -> str:
    """Normalise BUY/SELL/long/short variants to LONG or SHORT (empty string if neither)."""
    v = str(value or "").upper()
    if v in ("BUY", "LONG"):
        return "LONG"
    if v in ("SELL", "SHORT"):
        return "SHORT"
    return ""


def map_exit_reason(text: Optional[str]) -> str:
    """Map a free-text close reason onto the trades.exit_reason enum."""
    t = (text or "").upper().replace(" ", "_")
    if t in EXIT_REASONS:
        return t
    if "TRAIL" in t:
        return "TRAILING_SL"
    for key in ("TP3", "TP2", "TP1"):
        if key in t:
            return key
    if t.startswith("SL") or "STOP" in t:
        return "SL"
    if "TIMEOUT" in t or "STALL" in t:
        return "TIMEOUT"
    if "CANCEL" in t:
        return "CANCELLED"
    return "MANUAL"


def _f(value: Any) -> Optional[float]:
    """Coerce to float, returning None for missing/invalid values."""
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _jsonable(obj: Any) -> Any:
    """json.dumps default hook handling Decimal/datetime."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _serialize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Decimal/datetime values in result rows to JSON-friendly types."""
    return json.loads(json.dumps(rows, default=_jsonable))


def write_fallback(kind: str, payload: Dict[str, Any]) -> None:
    """Append a record to the JSONL fallback file (used only when the DB is unreachable)."""
    try:
        os.makedirs(os.path.dirname(FALLBACK_PATH), exist_ok=True)
        with open(FALLBACK_PATH, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"kind": kind, "ts": datetime.now(timezone.utc).isoformat(), "data": payload},
                    default=_jsonable,
                )
                + "\n"
            )
    except Exception as exc:
        log.error(f"[DB] Fallback write failed: {exc}")


# ─── Signals ──────────────────────────────────────────────────────────────────

SIGNAL_COLUMNS = [
    "created_at", "symbol", "timeframe", "direction", "confluence_score", "tier",
    "entry_price", "stop_loss", "tp1", "tp2", "tp3", "sl_pips", "rr_tp1", "rr_tp2",
    "rr_tp3", "blended_rr", "macro_trend", "micro_trend", "last_structure_event",
    "premium_discount", "liquidity_sweep", "sweep_direction", "rsi_value", "rsi_zone",
    "rsi_divergence", "macd_crossover", "atr_pips", "atr_state", "volume_state",
    "best_ob_score", "ob_fvg_overlap", "signal_source", "gate_rejected", "gate_reason",
    "analyst_notes", "raw_json", "source_key",
]


def build_signal_row(
    sig: Dict[str, Any],
    source: str = "LLM",
    gate_rejected: bool = False,
    gate_reason: Optional[str] = None,
    source_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Map a signal dict (bot TradeSignal-style or schema-style) onto a signals-table row."""
    direction = normalize_direction(sig.get("direction") or sig.get("signal"))
    if not direction:
        raise ValueError(f"Signal has no LONG/SHORT direction: {sig.get('direction') or sig.get('signal')!r}")
    symbol = str(sig.get("symbol", "")).upper()
    score = _f(sig.get("confluence_score", sig.get("confidence")))
    entry = _f(sig.get("entry_price"))
    sl = _f(sig.get("stop_loss"))
    tp1 = _f(sig.get("tp1", sig.get("take_profit_1", sig.get("take_profit"))))
    tp2 = _f(sig.get("tp2", sig.get("take_profit_2")))
    tp3 = _f(sig.get("tp3"))
    sl_pips = _f(sig.get("sl_pips"))
    if sl_pips is None and entry is not None and sl is not None:
        sl_pips = round(abs(entry - sl) / pip_size(symbol), 2)

    def rr(tp: Optional[float]) -> Optional[float]:
        if tp is None or entry is None or sl is None or entry == sl:
            return None
        return round(abs(tp - entry) / abs(entry - sl), 2)

    created = sig.get("created_at")
    if isinstance(created, datetime):
        created = created.astimezone(timezone.utc).replace(tzinfo=None) if created.tzinfo else created
    raw = sig.get("raw_json", sig.get("raw_response"))
    row = {
        "created_at": created or datetime.now(timezone.utc).replace(tzinfo=None),
        "symbol": symbol,
        "timeframe": str(sig.get("timeframe") or "H1").upper()[:5],
        "direction": direction,
        "confluence_score": score,
        "tier": sig.get("tier") or tier_from_score(score),
        "entry_price": entry,
        "stop_loss": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl_pips": sl_pips,
        "rr_tp1": _f(sig.get("rr_tp1")) if sig.get("rr_tp1") is not None else rr(tp1),
        "rr_tp2": _f(sig.get("rr_tp2")) if sig.get("rr_tp2") is not None else rr(tp2),
        "rr_tp3": _f(sig.get("rr_tp3")) if sig.get("rr_tp3") is not None else rr(tp3),
        "blended_rr": _f(sig.get("blended_rr", sig.get("risk_reward_ratio"))),
        "macro_trend": sig.get("macro_trend"),
        "micro_trend": sig.get("micro_trend"),
        "last_structure_event": sig.get("last_structure_event"),
        "premium_discount": sig.get("premium_discount"),
        "liquidity_sweep": int(bool(sig.get("liquidity_sweep", 0))),
        "sweep_direction": sig.get("sweep_direction"),
        "rsi_value": _f(sig.get("rsi_value")),
        "rsi_zone": sig.get("rsi_zone"),
        "rsi_divergence": sig.get("rsi_divergence"),
        "macd_crossover": sig.get("macd_crossover"),
        "atr_pips": _f(sig.get("atr_pips")),
        "atr_state": sig.get("atr_state"),
        "volume_state": sig.get("volume_state"),
        "best_ob_score": sig.get("best_ob_score"),
        "ob_fvg_overlap": int(bool(sig.get("ob_fvg_overlap", 0))),
        "signal_source": source if source in SIGNAL_SOURCES else "LLM",
        "gate_rejected": int(bool(gate_rejected)),
        "gate_reason": (gate_reason or None) and gate_reason[:255],
        "analyst_notes": sig.get("analyst_notes", sig.get("reasoning")),
        "raw_json": json.dumps(raw, default=_jsonable) if raw is not None else None,
        "source_key": source_key,
    }
    return row


def _insert_signal_cur(cur, row: Dict[str, Any], ignore: bool = False) -> Optional[int]:
    """Insert a signals row via an open cursor; return id (None if ignored duplicate)."""
    cols = ", ".join(SIGNAL_COLUMNS)
    marks = ", ".join(["%s"] * len(SIGNAL_COLUMNS))
    verb = "INSERT IGNORE" if ignore else "INSERT"
    cur.execute(f"{verb} INTO signals ({cols}) VALUES ({marks})", [row.get(c) for c in SIGNAL_COLUMNS])  # cols are constants
    if cur.rowcount == 0:
        return None
    return int(cur.lastrowid)


def insert_signal(
    sig: Dict[str, Any],
    source: str = "LLM",
    gate_rejected: bool = False,
    gate_reason: Optional[str] = None,
) -> Optional[int]:
    """Insert one signal (DB, or JSONL fallback if the DB is down); return its id or None."""
    try:
        row = build_signal_row(sig, source, gate_rejected, gate_reason)
    except ValueError as exc:
        log.warning(f"[DB] Signal not stored: {exc}")
        return None
    try:
        with get_db() as cur:
            sid = _insert_signal_cur(cur, row)
        log.info(f"[DB] Signal stored id={sid} {row['symbol']} {row['direction']} gate_rejected={row['gate_rejected']}")
        return sid
    except DatabaseUnavailableError:
        log.critical("[DB] Unavailable — signal written to file fallback")
        write_fallback("signal", row)
        return None
    except Exception as exc:
        log.error(f"[DB] Signal insert failed: {exc}")
        write_fallback("signal", row)
        return None


def list_signals(
    page: int = 1,
    page_size: int = 50,
    tier: Optional[str] = None,
    symbol: Optional[str] = None,
    gate_rejected: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return a paginated, filtered list of signals (newest first)."""
    where, params = [], []
    if tier:
        where.append("tier = %s"); params.append(tier.upper())
    if symbol:
        where.append("symbol = %s"); params.append(symbol.upper())
    if gate_rejected is not None:
        where.append("gate_rejected = %s"); params.append(int(gate_rejected))
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    page, page_size = max(1, page), min(max(1, page_size), 200)
    with get_db() as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM signals {clause}", params)  # clause built from constants only
        total = cur.fetchone()["n"]
        cur.execute(
            f"SELECT * FROM signals {clause} ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s",
            params + [page_size, (page - 1) * page_size],
        )
        rows = cur.fetchall()
    return {"page": page, "page_size": page_size, "total": total, "items": _serialize_rows(rows)}


# ─── Trades ───────────────────────────────────────────────────────────────────

TRADE_INSERT_COLUMNS = [
    "signal_id", "opened_at", "closed_at", "symbol", "direction", "entry_actual",
    "sl_actual", "tp_actual", "lot_size", "broker_ticket", "status", "exit_price",
    "exit_reason", "pips_gained", "rr_achieved", "profit_loss_usd", "risk_amount", "notes",
]


def _insert_trade_cur(cur, t: Dict[str, Any], ignore: bool = False) -> Optional[int]:
    """Insert a trades row via an open cursor; return id (None if ignored duplicate)."""
    cols = ", ".join(TRADE_INSERT_COLUMNS)
    marks = ", ".join(["%s"] * len(TRADE_INSERT_COLUMNS))
    verb = "INSERT IGNORE" if ignore else "INSERT"
    cur.execute(f"{verb} INTO trades ({cols}) VALUES ({marks})", [t.get(c) for c in TRADE_INSERT_COLUMNS])
    if cur.rowcount == 0:
        return None
    return int(cur.lastrowid)


def insert_signal_and_trade(
    sig: Dict[str, Any], trade: Dict[str, Any], source: str = "INJECTED"
) -> Tuple[int, int]:
    """Insert a signal and its trade in ONE transaction; return (signal_id, trade_id)."""
    row = build_signal_row(sig, source)
    with get_db() as cur:
        sid = _insert_signal_cur(cur, row)
        t = {
            "symbol": row["symbol"], "direction": row["direction"],
            "entry_actual": row["entry_price"], "sl_actual": row["stop_loss"],
            "tp_actual": row["tp1"], "status": "OPEN", **trade, "signal_id": sid,
        }
        tid = _insert_trade_cur(cur, t)
    log.info(f"[DB] Signal {sid} + trade {tid} stored (transaction)")
    return sid, tid


def record_trade_open(pos: Any, signal_id: Optional[int] = None) -> Optional[int]:
    """Insert an OPEN trade for a RiskManager OpenPosition; raises if the DB write fails."""
    t = {
        "signal_id": signal_id,
        "opened_at": pos.opened_at.astimezone(timezone.utc).replace(tzinfo=None),
        "symbol": pos.symbol.upper(),
        "direction": normalize_direction(pos.direction),
        "entry_actual": pos.entry_price,
        "sl_actual": pos.stop_loss,
        "tp_actual": pos.take_profit_1,
        "lot_size": round(pos.position_size / 100_000, 4),
        "broker_ticket": pos.trade_id,
        "risk_amount": getattr(pos, "initial_risk_amount", None),
        "status": "OPEN",
    }
    with get_db() as cur:
        tid = _insert_trade_cur(cur, t, ignore=True)
    log.info(f"[DB] Trade open stored id={tid} ticket={pos.trade_id}")
    return tid


def _outcome_fields(trade: Dict[str, Any], exit_price: float, pips: Optional[float],
                    pnl: Optional[float]) -> Dict[str, Any]:
    """Derive status/pips/rr from an outcome and the stored trade row."""
    entry = _f(trade.get("entry_actual"))
    sl = _f(trade.get("sl_actual"))
    pip = pip_size(trade["symbol"])
    sign = 1 if trade["direction"] == "LONG" else -1
    if pips is None and entry is not None and exit_price is not None:
        pips = round(sign * (exit_price - entry) / pip, 2)
    rr = None
    if pips is not None and entry is not None and sl is not None and entry != sl:
        rr = round(pips / (abs(entry - sl) / pip), 2)
    ref = pnl if pnl is not None else pips
    if ref is None or abs(ref) < 1e-9:
        status = "CLOSED_BE"
    else:
        status = "CLOSED_WIN" if ref > 0 else "CLOSED_LOSS"
    return {"pips_gained": pips, "rr_achieved": rr, "status": status}


def update_trade_outcome(
    trade_id: int,
    exit_price: float,
    exit_reason: str,
    pips_gained: Optional[float] = None,
    profit_loss_usd: Optional[float] = None,
    closed_at: Optional[datetime] = None,
    by_ticket: bool = False,
) -> Optional[Dict[str, Any]]:
    """Close a trade (by id, or broker_ticket if by_ticket) and return the updated row, or None."""
    reason = map_exit_reason(exit_reason)
    with get_db() as cur:
        if by_ticket:
            cur.execute("SELECT * FROM trades WHERE broker_ticket = %s FOR UPDATE", (str(trade_id),))
        else:
            cur.execute("SELECT * FROM trades WHERE id = %s FOR UPDATE", (trade_id,))
        trade = cur.fetchone()
        if trade is None:
            return None
        d = _outcome_fields(trade, exit_price, pips_gained, profit_loss_usd)
        opened = trade["opened_at"]
        closed = closed_at or datetime.now(timezone.utc).replace(tzinfo=None)
        cur.execute(
            "UPDATE trades SET closed_at=%s, exit_price=%s, exit_reason=%s, pips_gained=%s, "
            "rr_achieved=%s, profit_loss_usd=%s, status=%s, candles_held=%s WHERE id=%s",
            (closed, exit_price, reason, d["pips_gained"], d["rr_achieved"], profit_loss_usd,
             d["status"], int(max(0, (closed - opened).total_seconds() // 3600)) if opened else None,
             trade["id"]),
        )
        cur.execute("SELECT * FROM trades WHERE id = %s", (trade["id"],))
        updated = cur.fetchone()
    log.info(f"[DB] Trade {updated['id']} closed: {updated['status']} ({reason})")
    return _serialize_rows([updated])[0]


def record_trade_close(pos: Any, close_price: float, pnl: float, reason: str) -> None:
    """Close the DB trade matching a RiskManager position; raises if the DB write fails."""
    updated = update_trade_outcome(pos.trade_id, close_price, reason, None, pnl, by_ticket=True)
    if updated is None:  # open row missing (e.g. DB was down at open) — create it, then close it
        record_trade_open(pos, getattr(pos, "signal_id", None))
        update_trade_outcome(pos.trade_id, close_price, reason, None, pnl, by_ticket=True)
    _refresh_symbol_analytics(pos.symbol)


def _refresh_symbol_analytics(symbol: str) -> None:
    """Recompute analytics for one symbol after a trade closes (never raises)."""
    try:
        from analytics.engine import compute_symbol_performance, compute_adaptive_score_threshold
        compute_symbol_performance(symbol)
        compute_adaptive_score_threshold(symbol)
    except Exception as exc:
        log.warning(f"[DB] Analytics refresh for {symbol} failed: {exc}")


def list_trades(
    page: int = 1,
    page_size: int = 50,
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a paginated, filtered list of trades (newest first)."""
    where, params = [], []
    if symbol:
        where.append("symbol = %s"); params.append(symbol.upper())
    if status:
        where.append("status = %s"); params.append(status.upper())
    if date_from:
        where.append("opened_at >= %s"); params.append(date_from)
    if date_to:
        where.append("opened_at < DATE_ADD(%s, INTERVAL 1 DAY)"); params.append(date_to)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    page, page_size = max(1, page), min(max(1, page_size), 200)
    with get_db() as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM trades {clause}", params)  # clause built from constants only
        total = cur.fetchone()["n"]
        cur.execute(
            f"SELECT * FROM trades {clause} ORDER BY opened_at DESC, id DESC LIMIT %s OFFSET %s",
            params + [page_size, (page - 1) * page_size],
        )
        rows = cur.fetchall()
    return {"page": page, "page_size": page_size, "total": total, "items": _serialize_rows(rows)}
