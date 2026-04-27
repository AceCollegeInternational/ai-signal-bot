"""
tests/test_market_hours.py
─────────────────────────────────────────────────────────────────
Unit tests for the forex market hours guard.
Tests various time scenarios to ensure the market closure detection
correctly identifies the Friday 5 PM -> Sunday 5 PM EST window.
"""

import os
import sys
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

# Add project root so we can import utils
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from utils.market_hours import is_market_closed


def _mock_now(year, month, day, hour, minute=0, tz_str="America/New_York"):
    """Helper to create a timezone-aware datetime."""
    tz = ZoneInfo(tz_str)
    return datetime(year, month, day, hour, minute, tzinfo=tz)


def test_friday_before_close():
    """Friday 4:59 PM EST -> market should be OPEN."""
    fake_now = _mock_now(2026, 3, 13, 16, 59)  # Friday 4:59 PM
    with patch("utils.market_hours.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        closed, msg = is_market_closed()
        print(f"  Friday 4:59 PM: closed={closed} -> {'PASS' if not closed else 'FAIL'}")
        assert not closed, f"Expected OPEN at Friday 4:59 PM, got: {msg}"


def test_friday_after_close():
    """Friday 5:01 PM EST -> market should be CLOSED."""
    fake_now = _mock_now(2026, 3, 13, 17, 1)  # Friday 5:01 PM
    with patch("utils.market_hours.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        closed, msg = is_market_closed()
        print(f"  Friday 5:01 PM: closed={closed} -> {'PASS' if closed else 'FAIL'}")
        assert closed, f"Expected CLOSED at Friday 5:01 PM, got: {msg}"


def test_saturday_midday():
    """Saturday 12:00 PM EST -> market should be CLOSED."""
    fake_now = _mock_now(2026, 3, 14, 12, 0)  # Saturday noon
    with patch("utils.market_hours.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        closed, msg = is_market_closed()
        print(f"  Saturday noon:  closed={closed} -> {'PASS' if closed else 'FAIL'}")
        assert closed, f"Expected CLOSED on Saturday noon, got: {msg}"


def test_sunday_before_open():
    """Sunday 4:00 PM EST -> market should be CLOSED."""
    fake_now = _mock_now(2026, 3, 15, 16, 0)  # Sunday 4 PM
    with patch("utils.market_hours.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        closed, msg = is_market_closed()
        print(f"  Sunday 4:00 PM: closed={closed} -> {'PASS' if closed else 'FAIL'}")
        assert closed, f"Expected CLOSED on Sunday 4 PM, got: {msg}"


def test_sunday_after_open():
    """Sunday 5:01 PM EST -> market should be OPEN."""
    fake_now = _mock_now(2026, 3, 15, 17, 1)  # Sunday 5:01 PM
    with patch("utils.market_hours.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        closed, msg = is_market_closed()
        print(f"  Sunday 5:01 PM: closed={closed} -> {'PASS' if not closed else 'FAIL'}")
        assert not closed, f"Expected OPEN at Sunday 5:01 PM, got: {msg}"


def test_tuesday_midday():
    """Tuesday 12:00 PM EST -> market should be OPEN."""
    fake_now = _mock_now(2026, 3, 10, 12, 0)  # Tuesday noon
    with patch("utils.market_hours.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        closed, msg = is_market_closed()
        print(f"  Tuesday noon:   closed={closed} -> {'PASS' if not closed else 'FAIL'}")
        assert not closed, f"Expected OPEN on Tuesday noon, got: {msg}"


if __name__ == "__main__":
    print("\n=== Market Hours Guard Tests ===\n")
    tests = [
        test_friday_before_close,
        test_friday_after_close,
        test_saturday_midday,
        test_sunday_before_open,
        test_sunday_after_open,
        test_tuesday_midday,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")

    print(f"\n{'PASS' if passed == len(tests) else 'FAIL'} {passed}/{len(tests)} tests passed.\n")
