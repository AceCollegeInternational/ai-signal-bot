"""Schema definition and idempotent initialisation for the FxGuru MySQL database."""

from typing import List

from db.connection import get_db
from utils.logger import get_logger

log = get_logger(__name__)

DDL: List[str] = [
    """
    CREATE TABLE IF NOT EXISTS signals (
        id                      INT AUTO_INCREMENT PRIMARY KEY,
        created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
        symbol                  VARCHAR(10) NOT NULL,
        timeframe               VARCHAR(5)  DEFAULT 'H1',
        direction               ENUM('LONG','SHORT') NOT NULL,
        confluence_score        DECIMAL(5,2),
        tier                    VARCHAR(10),
        entry_price             DECIMAL(12,5),
        stop_loss               DECIMAL(12,5),
        tp1                     DECIMAL(12,5),
        tp2                     DECIMAL(12,5),
        tp3                     DECIMAL(12,5),
        sl_pips                 DECIMAL(8,2),
        rr_tp1                  DECIMAL(6,2),
        rr_tp2                  DECIMAL(6,2),
        rr_tp3                  DECIMAL(6,2),
        blended_rr              DECIMAL(6,2),
        macro_trend             VARCHAR(10),
        micro_trend             VARCHAR(10),
        last_structure_event    VARCHAR(10),
        premium_discount        VARCHAR(12),
        liquidity_sweep         TINYINT(1)  DEFAULT 0,
        sweep_direction         VARCHAR(12),
        rsi_value               DECIMAL(6,2),
        rsi_zone                VARCHAR(12),
        rsi_divergence          VARCHAR(10),
        macd_crossover          VARCHAR(12),
        atr_pips                DECIMAL(8,2),
        atr_state               VARCHAR(8),
        volume_state            VARCHAR(8),
        best_ob_score           TINYINT,
        ob_fvg_overlap          TINYINT(1)  DEFAULT 0,
        signal_source           ENUM('LLM','INJECTED','BACKTEST') DEFAULT 'LLM',
        gate_rejected           TINYINT(1)  DEFAULT 0,
        gate_reason             VARCHAR(255),
        analyst_notes           TEXT,
        raw_json                JSON,
        source_key              VARCHAR(64) NULL,
        UNIQUE KEY uk_signal_source_key (source_key),
        INDEX idx_symbol  (symbol),
        INDEX idx_tier    (tier),
        INDEX idx_created (created_at)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        id                      INT AUTO_INCREMENT PRIMARY KEY,
        signal_id               INT,
        opened_at               DATETIME DEFAULT CURRENT_TIMESTAMP,
        closed_at               DATETIME,
        symbol                  VARCHAR(10) NOT NULL,
        direction               ENUM('LONG','SHORT') NOT NULL,
        entry_actual            DECIMAL(12,5),
        sl_actual               DECIMAL(12,5),
        tp_actual               DECIMAL(12,5),
        lot_size                DECIMAL(8,4),
        broker_ticket           VARCHAR(64),
        status                  ENUM('OPEN','CLOSED_WIN','CLOSED_LOSS','CLOSED_BE','CANCELLED') DEFAULT 'OPEN',
        exit_price              DECIMAL(12,5),
        exit_reason             ENUM('TP1','TP2','TP3','SL','TRAILING_SL','MANUAL','TIMEOUT','CANCELLED'),
        pips_gained             DECIMAL(8,2),
        rr_achieved             DECIMAL(6,2),
        profit_loss_usd         DECIMAL(10,2),
        max_adverse_excursion   DECIMAL(8,2),
        max_favourable_excursion DECIMAL(8,2),
        candles_held            INT,
        risk_amount             DECIMAL(10,2),
        notes                   TEXT,
        FOREIGN KEY (signal_id) REFERENCES signals(id) ON DELETE SET NULL,
        UNIQUE KEY uk_broker_ticket (broker_ticket),
        INDEX idx_symbol (symbol),
        INDEX idx_status (status),
        INDEX idx_opened (opened_at)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS performance_daily (
        id                      INT AUTO_INCREMENT PRIMARY KEY,
        snapshot_date           DATE NOT NULL UNIQUE,
        total_signals           INT DEFAULT 0,
        tier1_signals           INT DEFAULT 0,
        tier2_signals           INT DEFAULT 0,
        trades_taken            INT DEFAULT 0,
        wins                    INT DEFAULT 0,
        losses                  INT DEFAULT 0,
        breakevens              INT DEFAULT 0,
        win_rate_pct            DECIMAL(6,2),
        avg_rr_achieved         DECIMAL(6,2),
        total_pips              DECIMAL(10,2),
        total_pnl_usd           DECIMAL(12,2),
        best_symbol             VARCHAR(10),
        worst_symbol            VARCHAR(10),
        created_at              DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS performance_by_symbol (
        id                      INT AUTO_INCREMENT PRIMARY KEY,
        symbol                  VARCHAR(10) NOT NULL,
        updated_at              DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        total_signals           INT DEFAULT 0,
        trades_taken            INT DEFAULT 0,
        wins                    INT DEFAULT 0,
        losses                  INT DEFAULT 0,
        win_rate_pct            DECIMAL(6,2),
        avg_confluence_score    DECIMAL(6,2),
        avg_rr_achieved         DECIMAL(6,2),
        total_pips              DECIMAL(10,2),
        recommended_min_score   DECIMAL(5,2) DEFAULT 75.0,
        signal_enabled          TINYINT(1)   DEFAULT 1,
        paused_at               DATETIME DEFAULT NULL,
        pause_reason            VARCHAR(255) DEFAULT NULL,
        recovered_at            DATETIME DEFAULT NULL,
        best_hours_utc          VARCHAR(64),
        worst_hours_utc         VARCHAR(64),
        UNIQUE KEY uk_symbol (symbol)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS factor_effectiveness (
        id                      INT AUTO_INCREMENT PRIMARY KEY,
        updated_at              DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        factor_name             VARCHAR(64) NOT NULL UNIQUE,
        present_in_wins         INT DEFAULT 0,
        present_in_losses       INT DEFAULT 0,
        win_rate_when_present   DECIMAL(6,2),
        win_rate_when_absent    DECIMAL(6,2),
        weight_adjustment       DECIMAL(4,2) DEFAULT 0.00,
        last_computed_at        DATETIME
    )
    """,
]

# (table, column, definition) — added when an older deployment lacks the column.
MIGRATIONS = [
    ("performance_by_symbol", "paused_at", "DATETIME DEFAULT NULL"),
    ("performance_by_symbol", "pause_reason", "VARCHAR(255) DEFAULT NULL"),
    ("performance_by_symbol", "recovered_at", "DATETIME DEFAULT NULL"),
    ("trades", "risk_amount", "DECIMAL(10,2)"),
    ("signals", "gate_rejected", "TINYINT(1) DEFAULT 0"),
    ("signals", "gate_reason", "VARCHAR(255)"),
    ("signals", "source_key", "VARCHAR(64) NULL"),
    ("performance_by_symbol", "best_hours_utc", "VARCHAR(64)"),
    ("performance_by_symbol", "worst_hours_utc", "VARCHAR(64)"),
]


def init_schema() -> None:
    """Create all tables (IF NOT EXISTS) and add any missing columns."""
    with get_db() as cur:
        for stmt in DDL:
            cur.execute(stmt)
        for table, column, definition in MIGRATIONS:
            cur.execute(
                "SELECT 1 FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s",
                (table, column),
            )
            if cur.fetchone() is None:
                # table/column/definition are module constants, never user input
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                log.info(f"[DB] Added column {table}.{column}")
    log.info("[DB] Schema ready")
