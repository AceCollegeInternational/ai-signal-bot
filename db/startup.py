"""Startup/maintenance helpers: schema creation, one-time migration, status for /health."""

import os
from typing import Any, Dict

from db.connection import DatabaseUnavailableError, create_tables, db_execute, db_get_one, test_connection
from db.migration import run_migration
from utils.logger import get_logger

log = get_logger(__name__)

MIGRATION_KEY = "migration_completed"


def migration_done() -> bool:
    """True if the one-time file migration has been recorded in app_settings."""
    row = db_get_one("SELECT value FROM app_settings WHERE `key` = %s", (MIGRATION_KEY,))
    return bool(row and row["value"] == "1")


def run_and_record_migration() -> Dict[str, Any]:
    """Run the file migration and set the done-flag only if it finished without errors."""
    report = run_migration()
    if not report["errors"]:
        db_execute(
            "INSERT INTO app_settings (`key`, value) VALUES (%s, '1') "
            "ON DUPLICATE KEY UPDATE value = '1', updated_at = NOW()", (MIGRATION_KEY,))
    return report


def startup_db() -> bool:
    """Create tables and run the one-time migration; never raises. Returns True if the DB is usable."""
    log.info(f"[DB] Connecting to {os.getenv('DB_HOST', '<DB_HOST unset>')}...")
    try:
        if not test_connection():
            raise DatabaseUnavailableError("connection test failed")
        create_tables()
        log.info("[DB] Tables verified OK.")
    except Exception as exc:
        log.critical(f"[DB] CRITICAL — could not reach database on startup. Falling back to file persistence. ({exc})")
        return False
    try:
        if migration_done():
            log.info("[DB] Migration already completed — skipping.")
        else:
            log.info("[DB] Running one-time file migration...")
            r = run_and_record_migration()
            log.info(f"[DB] Migration complete: {r['records_inserted']} inserted, "
                     f"{r['records_skipped']} skipped, {len(r['errors'])} errors.")
            if r["errors"]:
                log.error(f"[DB] Migration errors (flag NOT set, will retry next start): {r['errors']}")
    except Exception as exc:
        log.error(f"[DB] Migration step failed (app continues): {exc}")
    return True
