import requests
import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from utils.logger import get_logger

log = get_logger(__name__)


class NewsFilter:
    """
    Advanced filter to detect high-impact economic news events.
    Caches results to avoid frequent API calls.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("news", {})
        self.enabled = self.config.get("finnhub_enabled", False)
        self.api_key = os.getenv("FINNHUB_API_KEY", "")
        self.buffer_minutes = self.config.get("news_buffer_minutes", 30)
        self.skip_high_impact = self.config.get("skip_high_impact", True)

        self.cache: List[Dict[str, Any]] = []
        self.last_fetch_time = 0
        self.cache_ttl = 3600  # 1 hour

    def is_trading_suspended(self) -> bool:
        """
        Check if trading should be suspended due to an imminent high-impact news event.
        """
        if not self.enabled or not self.api_key:
            return False

        events = self._get_events()
        now = datetime.now(timezone.utc)

        for event in events:
            if event.get("impact", "").lower() != "high":
                continue

            event_time_str = event.get("time", "")
            if not event_time_str:
                continue

            try:
                # Finnhub time format is "YYYY-MM-DD HH:MM:SS"
                event_time = datetime.strptime(
                    event_time_str, "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc)

                time_diff = (event_time - now).total_seconds() / 60

                # Suspend if we are within the buffer before or after the event
                if abs(time_diff) <= self.buffer_minutes:
                    log.warning(
                        f"HIGH IMPACT NEWS: {event.get('event')} at {event_time_str}. Suspending trading."
                    )
                    return True
            except Exception as e:
                log.error(f"Error parsing news event time: {e}")

        return False

    def _get_events(self) -> List[Dict[str, Any]]:
        """
        Fetch economic calendar events with caching.
        """
        now_ts = time.time()
        if now_ts - self.last_fetch_time < self.cache_ttl and self.cache:
            return self.cache

        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            url = "https://finnhub.io/api/v1/calendar/economic"
            params = {
                "token": self.api_key,
                "from": today,
                "to": today,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                self.cache = resp.json().get("economicCalendar", [])
                self.last_fetch_time = now_ts
                return self.cache
            else:
                log.error(f"Finnhub API error: {resp.status_code} - {resp.text}")
        except Exception as e:
            log.error(f"Failed to fetch news events: {e}")

        return []
