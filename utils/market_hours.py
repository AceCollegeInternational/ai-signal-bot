"""
utils/market_hours.py
─────────────────────────────────────────────────────────────────────────────
Forex Market Hours Guard

The global forex market closes Friday at 5:00 PM EST (New York time)
and reopens Sunday at 5:00 PM EST. This module determines whether
the market is currently in this closed window.
"""

from datetime import datetime, timedelta
from typing import Tuple
from zoneinfo import ZoneInfo

from utils.logger import get_logger

log = get_logger(__name__)

# Forex market closes Friday 17:00 and reopens Sunday 17:00 (New York time)
MARKET_CLOSE_DAY = 4      # Friday (Monday=0 in weekday())
MARKET_CLOSE_HOUR = 17    # 5:00 PM
MARKET_OPEN_DAY = 6       # Sunday
MARKET_OPEN_HOUR = 17     # 5:00 PM


def is_market_closed(timezone_str: str = "America/New_York") -> Tuple[bool, str]:
    """
    Check if the forex market is currently closed.

    The forex market is closed from Friday 5:00 PM to Sunday 5:00 PM
    in the specified timezone (default: America/New_York / EST).

    Args:
        timezone_str: IANA timezone string, e.g. "America/New_York".

    Returns:
        A tuple of (is_closed: bool, message: str).
        - is_closed is True during the weekend closure.
        - message is a human-readable status string.
    """
    try:
        tz = ZoneInfo(timezone_str)
    except Exception:
        log.warning(f"Invalid timezone '{timezone_str}', falling back to America/New_York")
        tz = ZoneInfo("America/New_York")

    now = datetime.now(tz)
    weekday = now.weekday()  # Monday=0 … Sunday=6
    hour = now.hour

    # ── Determine if we are in the closed window ──────────────────────
    #
    # Closed window:
    #   Friday   (weekday=4) from 17:00 onward
    #   Saturday (weekday=5) all day
    #   Sunday   (weekday=6) before 17:00
    #
    is_closed = False

    if weekday == MARKET_CLOSE_DAY and hour >= MARKET_CLOSE_HOUR:
        # Friday 5 PM or later
        is_closed = True
    elif weekday == 5:
        # All of Saturday
        is_closed = True
    elif weekday == MARKET_OPEN_DAY and hour < MARKET_OPEN_HOUR:
        # Sunday before 5 PM
        is_closed = True

    if is_closed:
        msg = (
            f"Forex market is CLOSED (weekend). "
            f"Reopens Sunday 5:00 PM EST. "
            f"Current time: {now.strftime('%A %I:%M %p %Z')}"
        )
        return True, msg

    msg = (
        f"Forex market is OPEN. "
        f"Current time: {now.strftime('%A %I:%M %p %Z')}"
    )
    return False, msg


def get_market_close_window_id(timezone_str: str = "America/New_York") -> str:
    """
    Return a stable identifier for the active/nearest weekend close window.
    """
    try:
        tz = ZoneInfo(timezone_str)
    except Exception:
        tz = ZoneInfo("America/New_York")

    now = datetime.now(tz)
    weekday = now.weekday()  # Monday=0 ... Sunday=6

    # Anchor each window to the Friday date that starts the closure.
    days_since_friday = (weekday - 4) % 7
    anchor = (now - timedelta(days=days_since_friday)).date()

    return anchor.strftime("%Y-%m-%d")


def is_friday_close_dispatch_time(now: datetime, grace_minutes: int = 10) -> bool:
    """
    Return True only during the Friday close dispatch window.

    This is used for notifications that must fire strictly at (or very near)
    Friday 5:00 PM in local market time, while still allowing small scheduler
    delays.
    """
    if grace_minutes <= 0:
        grace_minutes = 1

    return (
        now.weekday() == MARKET_CLOSE_DAY
        and now.hour == MARKET_CLOSE_HOUR
        and 0 <= now.minute < grace_minutes
    )
