"""
utils/logger.py
───────────────
Centralized logging configuration for the trading bot.

Provides a factory function `get_logger(name)` that returns a Python
`logging.Logger` with two handlers:
  - RotatingFileHandler writing to logs/<log_file>
  - StreamHandler (console) with coloured output via rich

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Bot started")
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from rich.logging import RichHandler


def get_logger(
    name: str,
    log_dir: str = "logs",
    log_file: str = "trading_bot.log",
    max_bytes: int = 10_485_760,   # 10 MB
    backup_count: int = 5,
    level: str = "INFO",
) -> logging.Logger:
    """
    Create and return a named logger with rotating file + rich console handlers.

    Args:
        name:         Logger name (typically __name__ of the calling module).
        log_dir:      Directory to write log files into.
        log_file:     Base log filename.
        max_bytes:    Maximum bytes per log file before rotation.
        backup_count: Number of backup files to retain.
        level:        Logging level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # ── File Handler (rotating) ───────────────────────────────────────────────
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    # ── Console Handler (rich) ────────────────────────────────────────────────
    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False,
    )
    rich_handler.setLevel(log_level)
    logger.addHandler(rich_handler)

    # Prevent propagation to root logger (avoid double output)
    logger.propagate = False

    return logger


def configure_from_config(config: dict) -> None:
    """
    Apply logging settings from the bot's config.yaml dictionary.

    Args:
        config: Parsed config.yaml as a dict (the 'logging' sub-key).
    """
    log_cfg = config.get("logging", {})
    # Re-configure the root "trading_bot" logger with config values
    get_logger(
        name="trading_bot",
        log_dir=log_cfg.get("log_dir", "logs"),
        log_file=log_cfg.get("log_file", "trading_bot.log"),
        max_bytes=log_cfg.get("max_bytes", 10_485_760),
        backup_count=log_cfg.get("backup_count", 5),
        level=log_cfg.get("level", "INFO"),
    )
