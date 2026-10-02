"""
Idempotent migration of legacy file records into MySQL (importable; CLI in scripts/migrate_files_to_db.py).

Sources (all optional):
  logs/trade_lifecycle.txt   JSONL OPEN/CLOSE events      -> trades (broker_ticket = trade_id)
  logs/trade_journal.csv     closed-trade CSV              -> trades (only rows not already in lifecycle)
  logs/open_positions.json   currently open positions      -> trades (status OPEN)
  logs/db_fallback.jsonl     records buffered during outage -> signals

Re-running is safe: trades dedupe on the unique broker_ticket, signals on source_key.
"""

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from db.connection import get_db
from db.repository import (
    _insert_signal_cur, _insert_trade_cur, _outcome_fields,
    map_exit_reason, normalize_direction,
)
from utils.logger import get_logger

log = get_logger(__name__)

DEBUG_MARKERS = ("DEBUG", "TEST", "INJECT")


class Report:
    """Counts found/inserted/skipped records per source."""

    def __init__(self) -> None:
        self.rows: Dict[str, Dict[str, int]] = {}

    def add(self, source: str, found: int = 0, inserted: int = 0, skipped: int = 0) -> None:
        """Accumulate counters for a source."""
        r = self.rows.setdefault(source, {"found": 0, "inserted": 0, "skipped": 0})
        r["found"] += found; r["inserted"] += inserted; r["skipped"] += skipped

    def print(self) -> None:
        """Print the migration report."""
        print("\n=== Migration report ===")
        tf = ti = ts = 0
        for src, r in self.rows.items():
            print(f"{src:<24} {r['found']} found, {r['inserted']} inserted, {r['skipped']} skipped")
            tf += r["found"]; ti += r["inserted"]; ts += r["skipped"]
        print(f"{'TOTAL':<24} {tf} found, {ti} inserted, {ts} skipped")


def _dt(value: Any) -> Optional[datetime]:
    """Parse an ISO/CSV timestamp to naive UTC."""
    if not value:
        return None
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        try:
            d = datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S%z")
        except ValueError:
            return None
    if d.tzinfo is not None:
        d = d.astimezone(timezone.utc).replace(tzinfo=None)
    return d


def _is_debug(ticket: str) -> bool:
    """True for trade ids that look like injected/test records."""
    return any(m in ticket.upper() for m in DEBUG_MARKERS)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file, skipping blank/comment/corrupt lines."""
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  ! skipping corrupt line in {path}")
    return out


def _store_trade(cur, t: Dict[str, Any]) -> bool:
    """INSERT IGNORE a trade; True if a new row was inserted."""
    return _insert_trade_cur(cur, t, ignore=True) is not None


def migrate_lifecycle(cur, path: str, rep: Report) -> set:
    """Migrate lifecycle OPEN/CLOSE events; return the set of (symbol,dir,entry,exit) closes seen."""
    events = read_jsonl(path)
    opens: Dict[str, Dict[str, Any]] = {}
    closes: Dict[str, Dict[str, Any]] = {}
    for e in events:
        tid = e.get("trade_id")
        if not tid:
            continue
        (opens if e.get("event") == "OPEN" else closes)[tid] = e
    seen = set()
    for tid in sorted(set(opens) | set(closes)):
        o, c = opens.get(tid), closes.get(tid)
        base = o or c
        direction = normalize_direction(base.get("direction"))
        if not direction:
            rep.add("lifecycle", found=1, skipped=1)
            continue
        t: Dict[str, Any] = {
            "symbol": base["symbol"].upper(), "direction": direction, "broker_ticket": tid,
            "entry_actual": base.get("entry_price"),
            "opened_at": _dt(o.get("timestamp_utc")) if o else None,
            "sl_actual": o.get("stop_loss") if o else None,
            "tp_actual": o.get("take_profit_1") if o else None,
            "lot_size": round(o["size"] / 100_000, 4) if o and o.get("size") else None,
            "risk_amount": base.get("risk_amount"), "status": "OPEN",
            "notes": "migrated from trade_lifecycle.txt" + (" [debug]" if _is_debug(tid) else ""),
        }
        if t["opened_at"] is None:
            t["opened_at"] = _dt(c.get("timestamp_utc")) if c else datetime.now(timezone.utc).replace(tzinfo=None)
        if c:
            exit_price = float(c["close_price"])
            d = _outcome_fields(t, exit_price, None, c.get("pnl"))
            t.update(d, exit_price=exit_price, exit_reason=map_exit_reason(c.get("reason")),
                     profit_loss_usd=c.get("pnl"), closed_at=_dt(c.get("timestamp_utc")))
            seen.add((t["symbol"], direction, f"{float(t['entry_actual']):.5f}", f"{exit_price:.5f}"))
        ok = _store_trade(cur, t)
        rep.add("lifecycle", found=1, inserted=int(ok), skipped=int(not ok))
    return seen


def migrate_journal(cur, path: str, lifecycle_closes: set, rep: Report) -> None:
    """Migrate trade_journal.csv rows that the lifecycle log does not already cover."""
    if not os.path.exists(path):
        return
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            direction = normalize_direction(row.get("Direction"))
            try:
                entry, exit_p = float(row["Entry"]), float(row["Exit"])
            except (KeyError, TypeError, ValueError):
                rep.add("journal_csv", found=1, skipped=1)
                continue
            sym = (row.get("Symbol") or "").upper()
            if not direction or not sym:
                rep.add("journal_csv", found=1, skipped=1)
                continue
            if (sym, direction, f"{entry:.5f}", f"{exit_p:.5f}") in lifecycle_closes:
                rep.add("journal_csv", found=1, skipped=1)  # already migrated via lifecycle
                continue
            key = hashlib.sha1(f"{row.get('Timestamp')}|{sym}|{direction}|{entry}|{exit_p}".encode()).hexdigest()[:16]
            closed = _dt(row.get("Timestamp"))
            pnl = float(row.get("PnL") or 0)
            dur = float(row.get("Duration_Mins") or 0)
            t = {"symbol": sym, "direction": direction, "broker_ticket": f"csv-{key}",
                 "entry_actual": entry, "lot_size": round(float(row.get("Size") or 0) / 100_000, 4),
                 "risk_amount": float(row["Risk_Amount"]) if row.get("Risk_Amount") else None,
                 "exit_price": exit_p, "exit_reason": map_exit_reason(row.get("Reason")),
                 "profit_loss_usd": pnl, "closed_at": closed,
                 "opened_at": datetime.fromtimestamp(closed.replace(tzinfo=timezone.utc).timestamp() - dur * 60, timezone.utc).replace(tzinfo=None) if closed else None,
                 "notes": "migrated from trade_journal.csv"}
            t.update(_outcome_fields(t, exit_p, None, pnl))
            ok = _store_trade(cur, t)
            rep.add("journal_csv", found=1, inserted=int(ok), skipped=int(not ok))


def migrate_open_positions(cur, path: str, rep: Report) -> None:
    """Migrate open_positions.json entries as OPEN trades (deduped on trade_id)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    for _sym, p in data.items():
        t = {"symbol": p["symbol"].upper(), "direction": normalize_direction(p["direction"]),
             "broker_ticket": p.get("trade_id") or f"open-{p['symbol']}", "entry_actual": p["entry_price"],
             "sl_actual": p["stop_loss"], "tp_actual": p["take_profit_1"],
             "lot_size": round(p["position_size"] / 100_000, 4), "opened_at": _dt(p["opened_at"]),
             "risk_amount": p.get("initial_risk_amount"), "status": "OPEN",
             "notes": "migrated from open_positions.json"}
        ok = _store_trade(cur, t)
        rep.add("open_positions", found=1, inserted=int(ok), skipped=int(not ok))


def migrate_fallback(cur, path: str, rep: Report) -> None:
    """Replay signals buffered in db_fallback.jsonl during a DB outage."""
    for rec in read_jsonl(path):
        if rec.get("kind") != "signal":
            continue
        row = rec["data"]
        key = hashlib.sha1(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:40]
        if row.get("created_at"):
            row["created_at"] = _dt(row["created_at"])
        row["source_key"] = key
        sid = _insert_signal_cur(cur, row, ignore=True)
        rep.add(f"fallback_signals[{row.get('signal_source', 'LLM')}]", found=1,
                inserted=int(sid is not None), skipped=int(sid is None))


class _DryRun(Exception):
    """Raised inside a transaction to force a rollback."""


def run_migration(logs_dir: str = "logs", dry_run: bool = False) -> Dict[str, Any]:
    """Migrate all legacy file records; return {records_found, records_inserted, records_skipped, errors, by_source}."""
    rep = Report()
    errors: List[str] = []
    try:
        with get_db() as cur:  # one transaction: all-or-nothing
            seen = migrate_lifecycle(cur, os.path.join(logs_dir, "trade_lifecycle.txt"), rep)
            migrate_journal(cur, os.path.join(logs_dir, "trade_journal.csv"), seen, rep)
            migrate_open_positions(cur, os.path.join(logs_dir, "open_positions.json"), rep)
            migrate_fallback(cur, os.path.join(logs_dir, "db_fallback.jsonl"), rep)
            if dry_run:
                raise _DryRun()
    except _DryRun:
        pass
    except Exception as exc:  # transaction rolled back by get_db
        log.error(f"[DB] Migration failed and was rolled back: {exc}")
        errors.append(str(exc))
        rep = Report()
    total = lambda k: sum(r[k] for r in rep.rows.values())  # noqa: E731
    return {
        "records_found": total("found"), "records_inserted": total("inserted"),
        "records_skipped": total("skipped"), "errors": errors, "by_source": rep.rows,
    }
