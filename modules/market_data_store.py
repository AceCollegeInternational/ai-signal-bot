"""
In-memory market data store for EA-pushed candle bundles.

Stores the latest candles per symbol/timeframe and pre-computes indicators
and S/R snapshots on ingestion so inference can use fresh broker data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, List, Optional

import pandas as pd

from modules.indicator_engine import IndicatorEngine
from modules.pattern_detector import PatternDetector
from utils.logger import get_logger

log = get_logger(__name__)


TF_ALIAS_MAP: Dict[str, str] = {
    "H1": "1h",
    "H4": "4h",
    "1H": "1h",
    "4H": "4h",
    "1h": "1h",
    "4h": "4h",
}


@dataclass
class TimeframeSnapshot:
    raw_df: pd.DataFrame
    indicator_df: pd.DataFrame
    updated_at: datetime
    support_resistance: List[float]


class MarketDataStore:
    """Thread-safe in-memory store for pushed OHLCV bundles."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._data: Dict[str, Dict[str, TimeframeSnapshot]] = {}
        self._indicator_engine = IndicatorEngine({})
        self._pattern_detector = PatternDetector({})

    def _to_df(self, candles: List[Dict[str, Any]]) -> pd.DataFrame:
        rows: List[List[float]] = []
        for idx, candle in enumerate(candles):
            if "v" not in candle:
                raise ValueError(f"Missing 'v' (tick volume) in candle index {idx}")

            rows.append(
                [
                    int(candle["t"]),
                    float(candle["o"]),
                    float(candle["h"]),
                    float(candle["l"]),
                    float(candle["c"]),
                    float(candle.get("v", 0.0) or 0.0),
                ]
            )

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df.tail(100)

    def update_bundle(self, symbol: str, timeframes: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        norm_symbol = symbol.upper()
        summary: Dict[str, Any] = {"symbol": norm_symbol, "timeframes": {}}

        with self._lock:
            if norm_symbol not in self._data:
                self._data[norm_symbol] = {}

            for tf_in, candles in timeframes.items():
                tf = TF_ALIAS_MAP.get(tf_in, tf_in.lower())
                df = self._to_df(candles)
                indicator_df = self._indicator_engine.compute_all(df.copy())
                _, sr_levels = self._pattern_detector.detect_all(indicator_df.copy())

                self._data[norm_symbol][tf] = TimeframeSnapshot(
                    raw_df=df,
                    indicator_df=indicator_df,
                    updated_at=now,
                    support_resistance=sr_levels,
                )

                summary["timeframes"][tf] = {
                    "rows": len(df),
                    "updated_at": now.isoformat(),
                    "support_resistance_levels": len(sr_levels),
                }

        return summary

    def get_df(
        self,
        symbol: str,
        timeframe: str,
        max_age_seconds: Optional[int] = None,
        with_indicators: bool = True,
    ) -> Optional[pd.DataFrame]:
        norm_symbol = symbol.upper()
        tf = TF_ALIAS_MAP.get(timeframe, timeframe.lower())
        now = datetime.now(timezone.utc)

        with self._lock:
            symbol_data = self._data.get(norm_symbol, {})
            snap = symbol_data.get(tf)
            if not snap:
                return None

            age_seconds = (now - snap.updated_at).total_seconds()
            if max_age_seconds is not None and age_seconds > max_age_seconds:
                return None

            return (snap.indicator_df if with_indicators else snap.raw_df).copy()

    def freshness_report(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        report: Dict[str, Any] = {}
        with self._lock:
            for symbol, tf_map in self._data.items():
                report[symbol] = {}
                for tf, snap in tf_map.items():
                    report[symbol][tf] = {
                        "last_update": snap.updated_at.isoformat(),
                        "age_seconds": int((now - snap.updated_at).total_seconds()),
                        "rows": len(snap.raw_df),
                    }
        return report


market_data_store = MarketDataStore()

