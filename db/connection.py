"""
Centralised DB connection for FxGuru.
Uses a connection pool — never open/close per-query.
"""

import os
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

import pymysql
from dbutils.pooled_db import PooledDB
from dotenv import load_dotenv
from pymysql.cursors import DictCursor

from utils.logger import get_logger

load_dotenv()
log = get_logger(__name__)

POOL_MIN = 2
POOL_MAX = 10


class DatabaseUnavailableError(Exception):
    """Raised when the MySQL server cannot be reached (API maps this to HTTP 503)."""


_pool: Optional[PooledDB] = None
_pool_lock = threading.Lock()


def db_configured() -> bool:
    """Return True if the minimum DB environment variables are present."""
    return all(os.getenv(k) for k in ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"))


def _get_pool() -> PooledDB:
    """Lazily build the shared connection pool from environment variables."""
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        if not db_configured():
            raise DatabaseUnavailableError(
                "DB_HOST/DB_NAME/DB_USER/DB_PASSWORD are not all set"
            )
        try:
            _pool = PooledDB(
                creator=pymysql,
                mincached=POOL_MIN,
                maxcached=POOL_MAX,
                maxconnections=POOL_MAX,
                blocking=True,
                ping=1,  # ping on checkout -> transparently reconnect stale connections
                host=os.getenv("DB_HOST"),
                port=int(os.getenv("DB_PORT", "3306")),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME"),
                charset="utf8mb4",
                cursorclass=DictCursor,
                autocommit=False,
                connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "10")),
                read_timeout=30,
                write_timeout=30,
            )
        except pymysql.err.OperationalError as exc:
            log.error(f"[DB] Cannot create connection pool: {exc}")
            raise DatabaseUnavailableError(str(exc)) from exc
        log.info(f"[DB] Pool created (min={POOL_MIN}, max={POOL_MAX})")
        return _pool


def reset_pool() -> None:
    """Close and discard the shared pool (used by tests and after config changes)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.close()
            except Exception:  # pragma: no cover - best effort
                pass
        _pool = None


@contextmanager
def get_db() -> Iterator[DictCursor]:
    """Yield a dict cursor from the pool; commit on success, roll back on error."""
    pool = _get_pool()
    try:
        conn = pool.connection()
    except pymysql.err.OperationalError as exc:
        log.error(f"[DB] Connection unavailable: {exc}")
        raise DatabaseUnavailableError(str(exc)) from exc
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except pymysql.err.OperationalError as exc:
        _safe_rollback(conn)
        log.error(f"[DB] Operational error, rolled back: {exc}")
        raise DatabaseUnavailableError(str(exc)) from exc
    except Exception:
        _safe_rollback(conn)
        raise
    finally:
        try:
            cur.close()
        finally:
            conn.close()  # returns the connection to the pool


def _safe_rollback(conn) -> None:
    """Roll back, ignoring errors from an already-dead connection."""
    try:
        conn.rollback()
    except Exception:
        pass


def test_connection() -> bool:
    """Print and log DB connection status; return True if reachable (never raises)."""
    try:
        with get_db() as cur:
            cur.execute("SELECT VERSION() AS v, DATABASE() AS d")
            row = cur.fetchone()
        msg = (f"[DB] Connection OK — host: {os.getenv('DB_HOST')}, db: {row['d']} "
               f"(server {row['v']})")
        log.info(msg)
        print(msg)
        return True
    except DatabaseUnavailableError as exc:
        log.critical(f"[DB] UNAVAILABLE — falling back to file logging: {exc}")
        print(f"[DB] UNAVAILABLE: {exc}")
        return False
    except Exception as exc:
        log.critical(f"[DB] Connection test failed: {exc}")
        print(f"[DB] Connection test failed: {exc}")
        return False


def create_tables() -> None:
    """Create all tables on the configured database (alias for db.schema.init_schema)."""
    from db.schema import init_schema
    init_schema()
