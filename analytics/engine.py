"""
Analytics engine — the feedback loop.

Win rate is wins / (wins + losses); breakeven trades are excluded from the ratio.
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from db.connection import DatabaseUnavailableError, get_db
from db.repository import normalize_direction
from utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_MIN_SCORE = 75.0
MIN_TRADES_FOR_ADAPTATION = 20
PAUSE_WIN_RATE = 40.0
FACTOR_MIN_SAMPLES = 10          # per-side samples needed before a factor can penalise a signal
FACTOR_PENALTY_POINTS = 5.0
MIN_HOUR_SAMPLES = 3

# name -> (SQL boolean over `s` = signals, `t` = trades; python predicate over a signal dict).
# SQL fragments are module constants — never built from user input.
def _present(v: Any) -> bool:
    """True when a categorical indicator field carries a real value."""
    return bool(v) and str(v).upper() not in ("NONE", "NULL", "")

def _aligned(sig: Dict[str, Any], field: str, long_val: str, short_val: str) -> bool:
    """True when `field` supports the signal direction."""
    d = normalize_direction(sig.get("direction") or sig.get("signal"))
    v = str(sig.get(field) or "").upper()
    return (d == "LONG" and v == long_val) or (d == "SHORT" and v == short_val)

FACTORS: Dict[str, Tuple[str, Callable[[Dict[str, Any]], bool]]] = {
    "liquidity_sweep": ("s.liquidity_sweep = 1", lambda g: bool(g.get("liquidity_sweep"))),
    "ob_fvg_overlap": ("s.ob_fvg_overlap = 1", lambda g: bool(g.get("ob_fvg_overlap"))),
    "rsi_divergence": (
        "(s.rsi_divergence IS NOT NULL AND UPPER(s.rsi_divergence) NOT IN ('NONE',''))",
        lambda g: _present(g.get("rsi_divergence")),
    ),
    "macd_crossover": (
        "(s.macd_crossover IS NOT NULL AND UPPER(s.macd_crossover) NOT IN ('NONE',''))",
        lambda g: _present(g.get("macd_crossover")),
    ),
    "volume_above_average": ("UPPER(s.volume_state) = 'ABOVE'", lambda g: str(g.get("volume_state") or "").upper() == "ABOVE"),
    "macro_trend_aligned": (
        "((s.direction = 'LONG' AND UPPER(s.macro_trend) = 'BULLISH') OR (s.direction = 'SHORT' AND UPPER(s.macro_trend) = 'BEARISH'))",
        lambda g: _aligned(g, "macro_trend", "BULLISH", "BEARISH"),
    ),
    "premium_discount_aligned": (
        "((s.direction = 'LONG' AND UPPER(s.premium_discount) = 'DISCOUNT') OR (s.direction = 'SHORT' AND UPPER(s.premium_discount) = 'PREMIUM'))",
        lambda g: _aligned(g, "premium_discount", "DISCOUNT", "PREMIUM"),
    ),
    "high_confluence_score": (
        "s.confluence_score >= 85",
        lambda g: float(g.get("confluence_score", g.get("confidence")) or 0) >= 85,
    ),
}


def _f(v: Any) -> Optional[float]:
    """Decimal/None -> float/None."""
    return None if v is None else float(v)


def _win_rate(wins: int, losses: int) -> Optional[float]:
    """Win rate in percent, or None with no decided trades."""
    n = wins + losses
    return round(100.0 * wins / n, 2) if n else None


# ─── Per-symbol performance ───────────────────────────────────────────────────

def compute_symbol_performance(symbol: str) -> dict:
    """Compute win rate/avg RR/pips/trade count for a symbol and upsert performance_by_symbol."""
    symbol = symbol.upper()
    with get_db() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n, AVG(confluence_score) AS avg_score FROM signals "
            "WHERE symbol = %s AND gate_rejected = 0", (symbol,))
        sig = cur.fetchone()
        cur.execute(
            "SELECT COUNT(*) AS trades, "
            "SUM(status='CLOSED_WIN') AS wins, SUM(status='CLOSED_LOSS') AS losses, "
            "SUM(status='CLOSED_BE') AS be, AVG(rr_achieved) AS avg_rr, SUM(pips_gained) AS pips, "
            "SUM(profit_loss_usd) AS pnl FROM trades "
            "WHERE symbol = %s AND status IN ('CLOSED_WIN','CLOSED_LOSS','CLOSED_BE')", (symbol,))
        t = cur.fetchone()
        wins, losses = int(t["wins"] or 0), int(t["losses"] or 0)
        wr = _win_rate(wins, losses)
        enabled = 0 if (wr is not None and wins + losses >= MIN_TRADES_FOR_ADAPTATION and wr < PAUSE_WIN_RATE) else 1
        cur.execute(
            "INSERT INTO performance_by_symbol (symbol, total_signals, trades_taken, wins, losses, "
            "win_rate_pct, avg_confluence_score, avg_rr_achieved, total_pips, signal_enabled) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE total_signals=VALUES(total_signals), trades_taken=VALUES(trades_taken), "
            "wins=VALUES(wins), losses=VALUES(losses), win_rate_pct=VALUES(win_rate_pct), "
            "avg_confluence_score=VALUES(avg_confluence_score), avg_rr_achieved=VALUES(avg_rr_achieved), "
            "total_pips=VALUES(total_pips), signal_enabled=VALUES(signal_enabled)",
            (symbol, sig["n"], t["trades"], wins, losses, wr, _f(sig["avg_score"]),
             _f(t["avg_rr"]), _f(t["pips"]) or 0, enabled))
    log.info(f"[DB] Symbol performance updated: {symbol} trades={t['trades']} win_rate={wr}")
    return {
        "symbol": symbol, "total_signals": int(sig["n"]), "trades_taken": int(t["trades"]),
        "wins": wins, "losses": losses, "breakevens": int(t["be"] or 0), "win_rate_pct": wr,
        "avg_confluence_score": round(_f(sig["avg_score"]), 2) if sig["avg_score"] is not None else None,
        "avg_rr_achieved": round(_f(t["avg_rr"]), 2) if t["avg_rr"] is not None else None,
        "total_pips": round(_f(t["pips"]) or 0.0, 2), "total_pnl_usd": round(_f(t["pnl"]) or 0.0, 2),
        "signal_enabled": bool(enabled),
    }


# ─── Factor effectiveness ─────────────────────────────────────────────────────

def compute_factor_effectiveness() -> list:
    """Compare win rates with each confluence factor present vs absent; update the table, most impactful first."""
    results: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with get_db() as cur:
        for name, (expr, _) in FACTORS.items():
            cur.execute(
                "SELECT "
                f"COALESCE(SUM(({expr}) AND t.status='CLOSED_WIN'),0) AS pw, "
                f"COALESCE(SUM(({expr}) AND t.status='CLOSED_LOSS'),0) AS pl, "
                f"COALESCE(SUM(NOT COALESCE(({expr}),0) AND t.status='CLOSED_WIN'),0) AS aw, "
                f"COALESCE(SUM(NOT COALESCE(({expr}),0) AND t.status='CLOSED_LOSS'),0) AS al "
                "FROM trades t JOIN signals s ON s.id = t.signal_id "
                "WHERE t.status IN ('CLOSED_WIN','CLOSED_LOSS')")  # expr: module constant
            r = cur.fetchone()
            pw, pl, aw, al = (int(r[k]) for k in ("pw", "pl", "aw", "al"))
            wp, wa = _win_rate(pw, pl), _win_rate(aw, al)
            adj = 0.0
            if wp is not None and wa is not None and pw + pl >= 5 and aw + al >= 5:
                adj = max(-1.0, min(1.0, (wp - wa) / 50.0))
            cur.execute(
                "INSERT INTO factor_effectiveness (factor_name, present_in_wins, present_in_losses, "
                "win_rate_when_present, win_rate_when_absent, weight_adjustment, last_computed_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE present_in_wins=VALUES(present_in_wins), "
                "present_in_losses=VALUES(present_in_losses), win_rate_when_present=VALUES(win_rate_when_present), "
                "win_rate_when_absent=VALUES(win_rate_when_absent), weight_adjustment=VALUES(weight_adjustment), "
                "last_computed_at=VALUES(last_computed_at)",
                (name, pw, pl, wp, wa, round(adj, 2), now))
            results.append({
                "factor_name": name, "present_in_wins": pw, "present_in_losses": pl,
                "win_rate_when_present": wp, "win_rate_when_absent": wa,
                "weight_adjustment": round(adj, 2), "samples": pw + pl + aw + al,
            })
    results.sort(key=lambda d: (abs(d["weight_adjustment"]), d["present_in_wins"] + d["present_in_losses"]), reverse=True)
    log.info(f"[DB] Factor effectiveness recomputed ({len(results)} factors)")
    return results


# ─── Adaptive threshold ───────────────────────────────────────────────────────

def compute_adaptive_score_threshold(symbol: str) -> float:
    """Return (and store) the lowest 5-point score band whose trades win >60%; 75.0 if <20 trades."""
    symbol = symbol.upper()
    with get_db() as cur:
        cur.execute(
            "SELECT s.confluence_score AS score, t.status AS status FROM trades t "
            "JOIN signals s ON s.id = t.signal_id WHERE t.symbol = %s "
            "AND t.status IN ('CLOSED_WIN','CLOSED_LOSS') AND s.confluence_score IS NOT NULL", (symbol,))
        rows = cur.fetchall()
        threshold = DEFAULT_MIN_SCORE
        if len(rows) >= MIN_TRADES_FOR_ADAPTATION:
            scored = [(float(r["score"]), r["status"] == "CLOSED_WIN") for r in rows]
            bands = sorted({int(s // 5) * 5 for s, _ in scored})
            best: Optional[Tuple[float, float]] = None   # (win rate, band) fallback
            chosen: Optional[float] = None
            for band in bands:
                subset = [w for s, w in scored if s >= band]
                if len(subset) < 5:
                    continue
                wr = 100.0 * sum(subset) / len(subset)
                if wr > 60.0 and chosen is None:
                    chosen = float(band)
                if best is None or wr > best[0]:
                    best = (wr, float(band))
            threshold = chosen if chosen is not None else (best[1] if best else DEFAULT_MIN_SCORE)
            threshold = max(50.0, min(95.0, threshold))
        cur.execute(
            "INSERT INTO performance_by_symbol (symbol, recommended_min_score) VALUES (%s,%s) "
            "ON DUPLICATE KEY UPDATE recommended_min_score=VALUES(recommended_min_score)",
            (symbol, threshold))
    log.info(f"[DB] Adaptive threshold {symbol}: {threshold} ({len(rows)} scored trades)")
    return float(threshold)


# ─── Session analysis ─────────────────────────────────────────────────────────

def compute_optimal_session_times(symbol: str) -> dict:
    """Group closed trades by UTC open hour; return best/worst 3 hours by win rate and store them."""
    symbol = symbol.upper()
    with get_db() as cur:
        cur.execute(
            "SELECT HOUR(opened_at) AS h, SUM(status='CLOSED_WIN') AS w, SUM(status='CLOSED_LOSS') AS l "
            "FROM trades WHERE symbol = %s AND status IN ('CLOSED_WIN','CLOSED_LOSS') GROUP BY HOUR(opened_at)",
            (symbol,))
        hours = [(int(r["h"]), int(r["w"]), int(r["l"])) for r in cur.fetchall() if int(r["w"]) + int(r["l"]) >= MIN_HOUR_SAMPLES]
        ranked = sorted(hours, key=lambda x: (x[1] / (x[1] + x[2]), x[1] + x[2]), reverse=True)
        best = [h for h, _, _ in ranked[:3]]
        worst = [h for h, w, l in sorted(ranked[3:], key=lambda x: (x[1] / (x[1] + x[2]), -(x[1] + x[2])))[:3]]
        best_s, worst_s = ",".join(map(str, best)), ",".join(map(str, worst))
        cur.execute(
            "INSERT INTO performance_by_symbol (symbol, best_hours_utc, worst_hours_utc) VALUES (%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE best_hours_utc=VALUES(best_hours_utc), worst_hours_utc=VALUES(worst_hours_utc)",
            (symbol, best_s, worst_s))
    return {"symbol": symbol, "best_hours_utc": best, "worst_hours_utc": worst,
            "hours_analysed": len(hours)}


# ─── Report ───────────────────────────────────────────────────────────────────

def _symbols_with_activity() -> List[str]:
    """Distinct symbols that appear in signals or trades."""
    with get_db() as cur:
        cur.execute("SELECT symbol FROM signals UNION SELECT symbol FROM trades")
        return sorted(r["symbol"] for r in cur.fetchall())


def refresh_all() -> dict:
    """Re-run every analytics computation (symbols, thresholds, sessions, factors, report)."""
    for sym in _symbols_with_activity():
        compute_symbol_performance(sym)
        compute_adaptive_score_threshold(sym)
        compute_optimal_session_times(sym)
    return generate_performance_report()


def generate_performance_report() -> dict:
    """Build the full cross-symbol report and upsert today's performance_daily row."""
    factors = compute_factor_effectiveness()
    per_symbol = [compute_symbol_performance(s) for s in _symbols_with_activity()]
    with get_db() as cur:
        cur.execute("SELECT symbol, recommended_min_score, signal_enabled FROM performance_by_symbol")
        cfg = {r["symbol"]: r for r in cur.fetchall()}
    for p in per_symbol:
        c = cfg.get(p["symbol"], {})
        p["recommended_min_score"] = _f(c.get("recommended_min_score")) or DEFAULT_MIN_SCORE

    wins = sum(p["wins"] for p in per_symbol)
    losses = sum(p["losses"] for p in per_symbol)
    be = sum(p["breakevens"] for p in per_symbol)
    ranked = [p for p in per_symbol if p["win_rate_pct"] is not None]
    ranked.sort(key=lambda p: (p["win_rate_pct"], p["total_pips"]), reverse=True)
    best_sym = ranked[0]["symbol"] if ranked else None
    worst_sym = ranked[-1]["symbol"] if ranked else None
    reliable = [f for f in factors if f["win_rate_when_present"] is not None and f["present_in_wins"] + f["present_in_losses"] >= 5]
    reliable.sort(key=lambda f: f["win_rate_when_present"], reverse=True)

    today = datetime.now(timezone.utc).date()
    with get_db() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n, SUM(tier='TIER_1') AS t1, SUM(tier='TIER_2') AS t2 FROM signals "
            "WHERE DATE(created_at) = %s", (today,))
        sg = cur.fetchone()
        cur.execute(
            "SELECT COUNT(*) AS n, SUM(status='CLOSED_WIN') AS w, SUM(status='CLOSED_LOSS') AS l, "
            "SUM(status='CLOSED_BE') AS b, AVG(rr_achieved) AS rr, SUM(pips_gained) AS pips, "
            "SUM(profit_loss_usd) AS pnl FROM trades WHERE DATE(opened_at) = %s", (today,))
        td = cur.fetchone()
        dw, dl = int(td["w"] or 0), int(td["l"] or 0)
        cur.execute(
            "INSERT INTO performance_daily (snapshot_date, total_signals, tier1_signals, tier2_signals, "
            "trades_taken, wins, losses, breakevens, win_rate_pct, avg_rr_achieved, total_pips, "
            "total_pnl_usd, best_symbol, worst_symbol) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE total_signals=VALUES(total_signals), tier1_signals=VALUES(tier1_signals), "
            "tier2_signals=VALUES(tier2_signals), trades_taken=VALUES(trades_taken), wins=VALUES(wins), "
            "losses=VALUES(losses), breakevens=VALUES(breakevens), win_rate_pct=VALUES(win_rate_pct), "
            "avg_rr_achieved=VALUES(avg_rr_achieved), total_pips=VALUES(total_pips), "
            "total_pnl_usd=VALUES(total_pnl_usd), best_symbol=VALUES(best_symbol), worst_symbol=VALUES(worst_symbol)",
            (today, sg["n"], int(sg["t1"] or 0), int(sg["t2"] or 0), td["n"], dw, dl, int(td["b"] or 0),
             _win_rate(dw, dl), _f(td["rr"]), _f(td["pips"]) or 0, _f(td["pnl"]) or 0, best_sym, worst_sym))
    log.info(f"[DB] performance_daily written for {today}")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": {"trades_closed": wins + losses + be, "wins": wins, "losses": losses,
                    "breakevens": be, "win_rate_pct": _win_rate(wins, losses),
                    "total_pips": round(sum(p["total_pips"] for p in per_symbol), 2),
                    "total_pnl_usd": round(sum(p["total_pnl_usd"] for p in per_symbol), 2)},
        "best_symbol": best_sym, "worst_symbol": worst_sym,
        "most_reliable_factor": reliable[0] if reliable else None,
        "least_reliable_factor": reliable[-1] if reliable else None,
        "recommended_thresholds": {p["symbol"]: p["recommended_min_score"] for p in per_symbol},
        "suggested_pauses": [p["symbol"] for p in per_symbol
                             if p["win_rate_pct"] is not None and p["wins"] + p["losses"] >= MIN_TRADES_FOR_ADAPTATION
                             and p["win_rate_pct"] < PAUSE_WIN_RATE],
        "symbols": per_symbol,
        "factors": factors,
    }


def get_all_thresholds() -> List[Dict[str, Any]]:
    """Return every symbol's adaptive minimum score and enabled flag."""
    with get_db() as cur:
        cur.execute("SELECT symbol, recommended_min_score, signal_enabled, trades_taken, win_rate_pct, "
                    "best_hours_utc, worst_hours_utc FROM performance_by_symbol ORDER BY symbol")
        rows = cur.fetchall()
    return [{"symbol": r["symbol"], "recommended_min_score": _f(r["recommended_min_score"]),
             "signal_enabled": bool(r["signal_enabled"]), "trades_taken": r["trades_taken"],
             "win_rate_pct": _f(r["win_rate_pct"]), "best_hours_utc": r["best_hours_utc"],
             "worst_hours_utc": r["worst_hours_utc"]} for r in rows]


# ─── Feedback gate ────────────────────────────────────────────────────────────

def should_take_signal(signal: dict) -> tuple:
    """Feedback gate: return (approved, reason) from symbol status, adaptive score, factor stats and session hour."""
    symbol = str(signal.get("symbol", "")).upper()
    score = float(signal.get("confluence_score", signal.get("confidence")) or 0)
    try:
        with get_db() as cur:
            cur.execute("SELECT * FROM performance_by_symbol WHERE symbol = %s", (symbol,))
            perf = cur.fetchone()
            cur.execute("SELECT factor_name, present_in_wins, present_in_losses, win_rate_when_present "
                        "FROM factor_effectiveness")
            factor_rows = {r["factor_name"]: r for r in cur.fetchall()}
    except DatabaseUnavailableError as exc:
        log.critical(f"[ANALYTICS GATE] DB unavailable, gate bypassed: {exc}")
        return True, "Analytics unavailable — gate bypassed"

    threshold = DEFAULT_MIN_SCORE
    worst_hours: List[int] = []
    if perf:
        if not perf["signal_enabled"]:
            return False, "Symbol paused by analytics"
        threshold = _f(perf["recommended_min_score"]) or DEFAULT_MIN_SCORE
        worst_hours = [int(h) for h in (perf.get("worst_hours_utc") or "").split(",") if h.strip().isdigit()]
    if score < threshold:
        return False, f"Score below adaptive threshold for {symbol} ({score:.1f} < {threshold:.1f})"

    weak = []
    for name, (_, present) in FACTORS.items():
        row = factor_rows.get(name)
        if not row or not present(signal):
            continue
        n = int(row["present_in_wins"]) + int(row["present_in_losses"])
        wr = _f(row["win_rate_when_present"])
        if n >= FACTOR_MIN_SAMPLES and wr is not None and wr < PAUSE_WIN_RATE:
            weak.append(name)
    if weak:
        adjusted = score - FACTOR_PENALTY_POINTS * len(weak)
        if adjusted < threshold:
            return False, (f"Score below adaptive threshold for {symbol} after penalising weak factors "
                           f"{weak} ({adjusted:.1f} < {threshold:.1f})")

    hour = signal.get("hour_utc")
    hour = int(hour) if hour is not None else datetime.now(timezone.utc).hour
    if hour in worst_hours:
        return False, f"Unfavourable session for {symbol} (hour {hour:02d} UTC)"
    return True, "Passed all analytics gates"
