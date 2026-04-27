"""
modules/pattern_detector.py
─────────────────────────────────────────────────────────────────────────────
Module 3 — Chart Pattern Detector

Detects candlestick patterns, chart patterns, and support/resistance levels
from OHLCV DataFrames.

Candlestick Patterns (last 1–5 candles):
  Doji, Hammer, Inverted Hammer, Bullish/Bearish Engulfing,
  Morning Star, Evening Star, Shooting Star, Hanging Man,
  Three White Soldiers, Three Black Crows, Piercing Line, Dark Cloud Cover

Chart Patterns (last 20–100 candles):
  Head & Shoulders (+ Inverse), Double Top/Bottom, Triple Top/Bottom,
  Ascending/Descending/Symmetrical Triangles, Bull/Bear Flag,
  Rising/Falling Wedge, Cup and Handle

Support & Resistance:
  Swing high/low identification over last 100 candles, proximity check.

Each detected pattern returns a PatternResult dataclass.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema  # type: ignore[import]

from scipy.stats import linregress  # type: ignore[import]
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PatternResult:
    """
    Represents a detected chart or candlestick pattern.

    Attributes:
        pattern_name: Human-readable pattern name.
        direction:    'bullish', 'bearish', or 'neutral'.
        confidence:   Detection confidence from 0.0 to 1.0.
        candle_range: [start_index, end_index] of the pattern in the DataFrame.
        metadata:     Extra data (pattern-specific measurements, levels, etc.)
    """

    pattern_name: str
    direction: str  # "bullish" | "bearish" | "neutral"
    confidence: float  # 0.0 – 1.0
    candle_range: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "candle_range": self.candle_range,
            "metadata": self.metadata,
        }


class PatternDetector:
    """
    Detects candlestick patterns, chart patterns, and S/R levels in OHLCV data.

    Args:
        config: Optional config dict (for tolerance parameters).
    """

    # Minimum body size relative to candle range to be considered non-doji
    REAL_BODY_THRESHOLD = 0.05
    # S/R proximity: price within this fraction of a level is "near" it
    SR_PROXIMITY_PCT = 0.005  # 0.5%

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    # ─── Main Entry Point ─────────────────────────────────────────────────────

    def detect_all(self, df: pd.DataFrame) -> Tuple[List[PatternResult], List[float]]:
        """
        Run all pattern detectors on the DataFrame.
        """
        # Ensure no duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        patterns: List[PatternResult] = []

        if df is None or len(df) < 5:
            log.warning("DataFrame too short for pattern detection")
            return patterns, []

        # Candlestick patterns (require only last 5 candles)
        patterns.extend(self._detect_candlestick_patterns(df))

        # Chart patterns (require 20–100 candles)
        if len(df) >= 20:
            patterns.extend(self._detect_chart_patterns(df))
            patterns.extend(self._detect_market_structure(df))

        # Support & Resistance
        sr_levels = self._detect_support_resistance(df)

        log.info(
            f"Pattern detection complete: {len(patterns)} patterns, "
            f"{len(sr_levels)} S/R levels"
        )
        return patterns, sr_levels

    # ─── Candlestick Pattern Detection ────────────────────────────────────────

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detect all candlestick patterns from the last 5 candles."""
        results: List[PatternResult] = []
        n = len(df)

        detectors = [
            self._detect_doji,
            self._detect_hammer,
            self._detect_inverted_hammer,
            self._detect_engulfing,
            self._detect_morning_evening_star,
            self._detect_shooting_star,
            self._detect_hanging_man,
            self._detect_three_soldiers_crows,
            self._detect_piercing_dark_cloud,
        ]

        for detector in detectors:
            try:
                found = detector(df)
                results.extend(found)
            except Exception as exc:
                log.debug(f"Candlestick detector error ({detector.__name__}): {exc}")

        return results

    def _body_size(self, row: pd.Series) -> float:
        return abs(row["close"] - row["open"])

    def _candle_range(self, row: pd.Series) -> float:
        return row["high"] - row["low"]

    def _is_bullish(self, row: pd.Series) -> bool:
        return row["close"] > row["open"]

    def _is_bearish(self, row: pd.Series) -> bool:
        return row["close"] < row["open"]

    def _body_ratio(self, row: pd.Series) -> float:
        cr = self._candle_range(row)
        if cr == 0:
            return 0.0
        return self._body_size(row) / cr

    def _detect_doji(self, df: pd.DataFrame) -> List[PatternResult]:
        results = []
        last_idx = len(df) - 1
        row = df.iloc[last_idx]
        body = self._body_size(row)
        cr = self._candle_range(row)
        if cr == 0:
            return results
        if body / cr < 0.1:  # Very small body = doji
            results.append(
                PatternResult(
                    pattern_name="Doji",
                    direction="neutral",
                    confidence=0.65,
                    candle_range=[last_idx, last_idx],
                )
            )
        return results

    def _detect_hammer(self, df: pd.DataFrame) -> List[PatternResult]:
        results = []
        last_idx = len(df) - 1
        if last_idx < 1:
            return results
        row = df.iloc[last_idx]
        body = self._body_size(row)
        cr = self._candle_range(row)
        if cr == 0:
            return results
        lower_shadow = min(row["open"], row["close"]) - row["low"]
        upper_shadow = row["high"] - max(row["open"], row["close"])
        # Hammer: small body, lower shadow ≥ 2× body, small upper shadow
        if (
            body / cr >= self.REAL_BODY_THRESHOLD
            and lower_shadow >= 2 * body
            and upper_shadow <= 0.3 * body
        ):
            # Confirm downtrend before hammer
            prev_closes = df["close"].iloc[max(0, last_idx - 5) : last_idx]
            in_downtrend = prev_closes.iloc[-1] < prev_closes.iloc[0]
            confidence = 0.72 if in_downtrend else 0.55
            results.append(
                PatternResult(
                    pattern_name="Hammer",
                    direction="bullish",
                    confidence=confidence,
                    candle_range=[last_idx, last_idx],
                )
            )
        return results

    def _detect_inverted_hammer(self, df: pd.DataFrame) -> List[PatternResult]:
        results = []
        last_idx = len(df) - 1
        if last_idx < 1:
            return results
        row = df.iloc[last_idx]
        body = self._body_size(row)
        cr = self._candle_range(row)
        if cr == 0:
            return results
        upper_shadow = row["high"] - max(row["open"], row["close"])
        lower_shadow = min(row["open"], row["close"]) - row["low"]
        if (
            body / cr >= self.REAL_BODY_THRESHOLD
            and upper_shadow >= 2 * body
            and lower_shadow <= 0.3 * body
        ):
            results.append(
                PatternResult(
                    pattern_name="Inverted Hammer",
                    direction="bullish",
                    confidence=0.60,
                    candle_range=[last_idx, last_idx],
                )
            )
        return results

    def _detect_engulfing(self, df: pd.DataFrame) -> List[PatternResult]:
        results = []
        last_idx = len(df) - 1
        if last_idx < 1:
            return results
        curr = df.iloc[last_idx]
        prev = df.iloc[last_idx - 1]
        curr_body = self._body_size(curr)
        prev_body = self._body_size(prev)
        # Bullish engulfing: prev bearish, curr bullish, engulfs prev body
        if (
            self._is_bearish(prev)
            and self._is_bullish(curr)
            and curr["open"] <= prev["close"]
            and curr["close"] >= prev["open"]
            and curr_body > prev_body
        ):
            results.append(
                PatternResult(
                    pattern_name="Bullish Engulfing",
                    direction="bullish",
                    confidence=0.78,
                    candle_range=[last_idx - 1, last_idx],
                )
            )
        # Bearish engulfing: prev bullish, curr bearish, engulfs prev body
        elif (
            self._is_bullish(prev)
            and self._is_bearish(curr)
            and curr["open"] >= prev["close"]
            and curr["close"] <= prev["open"]
            and curr_body > prev_body
        ):
            results.append(
                PatternResult(
                    pattern_name="Bearish Engulfing",
                    direction="bearish",
                    confidence=0.78,
                    candle_range=[last_idx - 1, last_idx],
                )
            )
        return results

    def _detect_morning_evening_star(self, df: pd.DataFrame) -> List[PatternResult]:
        results = []
        last_idx = len(df) - 1
        if last_idx < 2:
            return results
        c0, c1, c2 = df.iloc[last_idx - 2], df.iloc[last_idx - 1], df.iloc[last_idx]
        b0, b1, b2 = self._body_size(c0), self._body_size(c1), self._body_size(c2)
        # Morning Star: bearish → small body (gap down) → bullish
        if (
            self._is_bearish(c0) and b0 > 0 and b1 / self._candle_range(c1) < 0.3
            if self._candle_range(c1) > 0
            else False
            and self._is_bullish(c2)
            and c2["close"] > (c0["open"] + c0["close"]) / 2
        ):
            results.append(
                PatternResult(
                    pattern_name="Morning Star",
                    direction="bullish",
                    confidence=0.80,
                    candle_range=[last_idx - 2, last_idx],
                )
            )
        # Evening Star: bullish → small body (gap up) → bearish
        elif (
            self._is_bullish(c0) and b0 > 0 and b1 / self._candle_range(c1) < 0.3
            if self._candle_range(c1) > 0
            else False
            and self._is_bearish(c2)
            and c2["close"] < (c0["open"] + c0["close"]) / 2
        ):
            results.append(
                PatternResult(
                    pattern_name="Evening Star",
                    direction="bearish",
                    confidence=0.80,
                    candle_range=[last_idx - 2, last_idx],
                )
            )
        return results

    def _detect_shooting_star(self, df: pd.DataFrame) -> List[PatternResult]:
        results = []
        last_idx = len(df) - 1
        if last_idx < 1:
            return results
        row = df.iloc[last_idx]
        body = self._body_size(row)
        cr = self._candle_range(row)
        if cr == 0:
            return results
        upper_shadow = row["high"] - max(row["open"], row["close"])
        lower_shadow = min(row["open"], row["close"]) - row["low"]
        if (
            upper_shadow >= 2 * body
            and lower_shadow <= 0.2 * cr
            and self._is_bearish(row)
        ):
            results.append(
                PatternResult(
                    pattern_name="Shooting Star",
                    direction="bearish",
                    confidence=0.70,
                    candle_range=[last_idx, last_idx],
                )
            )
        return results

    def _detect_hanging_man(self, df: pd.DataFrame) -> List[PatternResult]:
        results = []
        last_idx = len(df) - 1
        if last_idx < 1:
            return results
        row = df.iloc[last_idx]
        body = self._body_size(row)
        cr = self._candle_range(row)
        if cr == 0:
            return results
        lower_shadow = min(row["open"], row["close"]) - row["low"]
        upper_shadow = row["high"] - max(row["open"], row["close"])
        prev_closes = df["close"].iloc[max(0, last_idx - 5) : last_idx]
        in_uptrend = len(prev_closes) > 1 and prev_closes.iloc[-1] > prev_closes.iloc[0]
        if lower_shadow >= 2 * body and upper_shadow <= 0.2 * body and in_uptrend:
            results.append(
                PatternResult(
                    pattern_name="Hanging Man",
                    direction="bearish",
                    confidence=0.68,
                    candle_range=[last_idx, last_idx],
                )
            )
        return results

    def _detect_three_soldiers_crows(self, df: pd.DataFrame) -> List[PatternResult]:
        results = []
        last_idx = len(df) - 1
        if last_idx < 2:
            return results
        c0, c1, c2 = df.iloc[last_idx - 2], df.iloc[last_idx - 1], df.iloc[last_idx]
        # Three White Soldiers: 3 consecutive bullish candles, each opening within prior body
        if (
            self._is_bullish(c0)
            and self._is_bullish(c1)
            and self._is_bullish(c2)
            and c1["open"] > c0["open"]
            and c1["open"] < c0["close"]
            and c2["open"] > c1["open"]
            and c2["open"] < c1["close"]
            and c2["close"] > c1["close"] > c0["close"]
        ):
            results.append(
                PatternResult(
                    pattern_name="Three White Soldiers",
                    direction="bullish",
                    confidence=0.82,
                    candle_range=[last_idx - 2, last_idx],
                )
            )
        # Three Black Crows: 3 consecutive bearish candles
        elif (
            self._is_bearish(c0)
            and self._is_bearish(c1)
            and self._is_bearish(c2)
            and c1["open"] < c0["open"]
            and c1["open"] > c0["close"]
            and c2["open"] < c1["open"]
            and c2["open"] > c1["close"]
            and c2["close"] < c1["close"] < c0["close"]
        ):
            results.append(
                PatternResult(
                    pattern_name="Three Black Crows",
                    direction="bearish",
                    confidence=0.82,
                    candle_range=[last_idx - 2, last_idx],
                )
            )
        return results

    def _detect_piercing_dark_cloud(self, df: pd.DataFrame) -> List[PatternResult]:
        results = []
        last_idx = len(df) - 1
        if last_idx < 1:
            return results
        prev = df.iloc[last_idx - 1]
        curr = df.iloc[last_idx]
        prev_mid = (prev["open"] + prev["close"]) / 2
        # Piercing Line: prev bearish, curr opens below prev low, closes above prev midpoint
        if (
            self._is_bearish(prev)
            and self._is_bullish(curr)
            and curr["open"] < prev["low"]
            and curr["close"] > prev_mid
            and curr["close"] < prev["open"]
        ):
            results.append(
                PatternResult(
                    pattern_name="Piercing Line",
                    direction="bullish",
                    confidence=0.72,
                    candle_range=[last_idx - 1, last_idx],
                )
            )
        # Dark Cloud Cover: prev bullish, curr opens above prev high, closes below midpoint
        elif (
            self._is_bullish(prev)
            and self._is_bearish(curr)
            and curr["open"] > prev["high"]
            and curr["close"] < prev_mid
            and curr["close"] > prev["open"]
        ):
            results.append(
                PatternResult(
                    pattern_name="Dark Cloud Cover",
                    direction="bearish",
                    confidence=0.72,
                    candle_range=[last_idx - 1, last_idx],
                )
            )
        return results

    # ─── Chart Pattern Detection ──────────────────────────────────────────────

    def _detect_chart_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detect all multi-candle chart patterns."""
        results: List[PatternResult] = []
        window = df.iloc[-100:] if len(df) > 100 else df
        n = len(window)

        detectors = [
            self._detect_head_and_shoulders,
            self._detect_double_top_bottom,
            self._detect_triple_top_bottom,
            self._detect_triangles,
            self._detect_flags,
            self._detect_wedges,
            self._detect_cup_and_handle,
        ]

        for detector in detectors:
            try:
                found = detector(window)
                results.extend(found)
            except Exception as exc:
                log.debug(f"Chart pattern detector error ({detector.__name__}): {exc}")

        return results

    def _find_swing_highs(self, df: pd.DataFrame, order: int = 5) -> np.ndarray:
        """Find indices of local swing highs in the close column."""
        highs = df["high"].values
        idx = argrelextrema(highs, np.greater_equal, order=order)[0]
        return idx

    def _find_swing_lows(self, df: pd.DataFrame, order: int = 5) -> np.ndarray:
        """Find indices of local swing lows in the close column."""
        lows = df["low"].values
        idx = argrelextrema(lows, np.less_equal, order=order)[0]
        return idx

    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> List[PatternResult]:
        results = []
        if len(df) < 30:
            return results
        peaks = self._find_swing_highs(df, order=4)
        if len(peaks) < 3:
            return results

        for i in range(len(peaks) - 2):
            l_idx, h_idx, r_idx = peaks[i], peaks[i + 1], peaks[i + 2]
            l_h = df["high"].iloc[l_idx]
            h_h = df["high"].iloc[h_idx]
            r_h = df["high"].iloc[r_idx]

            # Head must be tallest; shoulders roughly equal (within 3%)
            if h_h > l_h and h_h > r_h:
                shoulder_diff = abs(l_h - r_h) / max(l_h, r_h)
                if shoulder_diff < 0.03:
                    # Neckline = average of troughs between peaks
                    left_trough = df["low"].iloc[l_idx:h_idx].min()
                    right_trough = df["low"].iloc[h_idx:r_idx].min()
                    neckline = (left_trough + right_trough) / 2
                    current_price = df["close"].iloc[-1]
                    confidence = max(0.55, 0.85 - shoulder_diff * 10)
                    results.append(
                        PatternResult(
                            pattern_name="Head and Shoulders",
                            direction="bearish",
                            confidence=confidence,
                            candle_range=[int(l_idx), int(r_idx)],
                            metadata={"neckline": neckline, "head_high": h_h},
                        )
                    )
                    break  # Only report the most recent instance

        # Inverse H&S in troughs
        troughs = self._find_swing_lows(df, order=4)
        if len(troughs) >= 3:
            for i in range(len(troughs) - 2):
                l_idx, h_idx, r_idx = troughs[i], troughs[i + 1], troughs[i + 2]
                l_l = df["low"].iloc[l_idx]
                h_l = df["low"].iloc[h_idx]
                r_l = df["low"].iloc[r_idx]
                if h_l < l_l and h_l < r_l:
                    shoulder_diff = abs(l_l - r_l) / max(l_l, r_l)
                    if shoulder_diff < 0.03:
                        neckline = (
                            df["high"].iloc[l_idx:h_idx].max()
                            + df["high"].iloc[h_idx:r_idx].max()
                        ) / 2
                        confidence = max(0.55, 0.85 - shoulder_diff * 10)
                        results.append(
                            PatternResult(
                                pattern_name="Inverse Head and Shoulders",
                                direction="bullish",
                                confidence=confidence,
                                candle_range=[int(l_idx), int(r_idx)],
                                metadata={"neckline": neckline, "head_low": h_l},
                            )
                        )
                        break
        return results

    def _detect_double_top_bottom(self, df: pd.DataFrame) -> List[PatternResult]:
        results = []
        if len(df) < 20:
            return results
        peaks = self._find_swing_highs(df, order=5)
        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            h1, h2 = df["high"].iloc[p1], df["high"].iloc[p2]
            if abs(h1 - h2) / max(h1, h2) < 0.02:  # within 2%
                trough = df["low"].iloc[p1:p2].min()
                results.append(
                    PatternResult(
                        pattern_name="Double Top",
                        direction="bearish",
                        confidence=0.75,
                        candle_range=[int(p1), int(p2)],
                        metadata={"resistance": (h1 + h2) / 2, "trough": trough},
                    )
                )

        troughs = self._find_swing_lows(df, order=5)
        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            l1, l2 = df["low"].iloc[t1], df["low"].iloc[t2]
            if abs(l1 - l2) / max(l1, l2) < 0.02:
                peak = df["high"].iloc[t1:t2].max()
                results.append(
                    PatternResult(
                        pattern_name="Double Bottom",
                        direction="bullish",
                        confidence=0.75,
                        candle_range=[int(t1), int(t2)],
                        metadata={"support": (l1 + l2) / 2, "peak": peak},
                    )
                )
        return results

    def _detect_triple_top_bottom(self, df: pd.DataFrame) -> List[PatternResult]:
        results = []
        if len(df) < 30:
            return results
        peaks = self._find_swing_highs(df, order=4)
        if len(peaks) >= 3:
            p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
            h1, h2, h3 = df["high"].iloc[p1], df["high"].iloc[p2], df["high"].iloc[p3]
            avg_h = (h1 + h2 + h3) / 3
            if all(abs(h - avg_h) / avg_h < 0.02 for h in [h1, h2, h3]):
                results.append(
                    PatternResult(
                        pattern_name="Triple Top",
                        direction="bearish",
                        confidence=0.78,
                        candle_range=[int(p1), int(p3)],
                        metadata={"resistance": avg_h},
                    )
                )
        troughs = self._find_swing_lows(df, order=4)
        if len(troughs) >= 3:
            t1, t2, t3 = troughs[-3], troughs[-2], troughs[-1]
            l1, l2, l3 = df["low"].iloc[t1], df["low"].iloc[t2], df["low"].iloc[t3]
            avg_l = (l1 + l2 + l3) / 3
            if all(abs(l - avg_l) / avg_l < 0.02 for l in [l1, l2, l3]):
                results.append(
                    PatternResult(
                        pattern_name="Triple Bottom",
                        direction="bullish",
                        confidence=0.78,
                        candle_range=[int(t1), int(t3)],
                        metadata={"support": avg_l},
                    )
                )
        return results

    def _detect_triangles(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detect ascending, descending, and symmetrical triangles via trendline slope."""
        results = []
        if len(df) < 20:
            return results

        highs = df["high"].values
        lows = df["low"].values
        x = np.arange(len(highs))

        # Linear regression on highs and lows
        high_coef = np.polyfit(x, highs, 1)
        low_coef = np.polyfit(x, lows, 1)
        high_slope = high_coef[0]
        low_slope = low_coef[0]

        price_scale = df["close"].mean()
        norm_high = high_slope / price_scale
        norm_low = low_slope / price_scale

        FLAT = 0.0002  # slope threshold to consider "flat"

        if norm_high < FLAT and norm_low > FLAT:
            results.append(
                PatternResult(
                    pattern_name="Ascending Triangle",
                    direction="bullish",
                    confidence=0.70,
                    candle_range=[0, len(df) - 1],
                    metadata={
                        "high_slope": round(high_slope, 6),
                        "low_slope": round(low_slope, 6),
                    },
                )
            )
        elif norm_high < -FLAT and norm_low < -FLAT and abs(norm_high) > abs(norm_low):
            results.append(
                PatternResult(
                    pattern_name="Descending Triangle",
                    direction="bearish",
                    confidence=0.70,
                    candle_range=[0, len(df) - 1],
                    metadata={
                        "high_slope": round(high_slope, 6),
                        "low_slope": round(low_slope, 6),
                    },
                )
            )
        elif norm_high < -FLAT and norm_low > FLAT:
            results.append(
                PatternResult(
                    pattern_name="Symmetrical Triangle",
                    direction="neutral",
                    confidence=0.65,
                    candle_range=[0, len(df) - 1],
                    metadata={
                        "high_slope": round(high_slope, 6),
                        "low_slope": round(low_slope, 6),
                    },
                )
            )
        return results

    def _detect_flags(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detect bull and bear flag patterns (strong move → consolidation channel)."""
        results = []
        if len(df) < 20:
            return results

        # Use last 20 candles as the flag pole + flag
        pole_end = max(0, len(df) - 20)
        flag_section = df.iloc[pole_end:]
        if len(flag_section) < 10:
            return results

        # Pole: check for strong directional move in the first half of the section
        half = len(flag_section) // 2
        pole = flag_section.iloc[:half]
        flag_body = flag_section.iloc[half:]

        pole_change = (pole["close"].iloc[-1] - pole["close"].iloc[0]) / pole[
            "close"
        ].iloc[0]
        flag_range = flag_body["high"].max() - flag_body["low"].min()
        pole_range = pole["high"].max() - pole["low"].min()

        if pole_range == 0:
            return results

        # Flag channel should be ≤ 50% of pole range (tight consolidation)
        if flag_range / pole_range > 0.5:
            return results

        if pole_change > 0.03:  # 3%+ bullish pole
            results.append(
                PatternResult(
                    pattern_name="Bull Flag",
                    direction="bullish",
                    confidence=0.73,
                    candle_range=[pole_end, len(df) - 1],
                    metadata={"pole_change_pct": round(pole_change * 100, 2)},
                )
            )
        elif pole_change < -0.03:  # 3%+ bearish pole
            results.append(
                PatternResult(
                    pattern_name="Bear Flag",
                    direction="bearish",
                    confidence=0.73,
                    candle_range=[pole_end, len(df) - 1],
                    metadata={"pole_change_pct": round(pole_change * 100, 2)},
                )
            )
        return results

    def _detect_wedges(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detect rising and falling wedges (converging trendlines in same direction)."""
        results = []
        if len(df) < 20:
            return results

        highs = df["high"].values
        lows = df["low"].values
        x = np.arange(len(highs))

        high_coef = np.polyfit(x, highs, 1)
        low_coef = np.polyfit(x, lows, 1)
        high_slope = high_coef[0] / df["close"].mean()
        low_slope = low_coef[0] / df["close"].mean()

        THRESHOLD = 0.0002

        # Rising wedge: both trendlines slope up, but highs slope less → converging
        if high_slope > THRESHOLD and low_slope > THRESHOLD and low_slope > high_slope:
            results.append(
                PatternResult(
                    pattern_name="Rising Wedge",
                    direction="bearish",
                    confidence=0.68,
                    candle_range=[0, len(df) - 1],
                    metadata={
                        "high_slope": round(high_slope, 6),
                        "low_slope": round(low_slope, 6),
                    },
                )
            )
        # Falling wedge: both slope down, highs slope more steeply → converging
        elif (
            high_slope < -THRESHOLD
            and low_slope < -THRESHOLD
            and high_slope < low_slope
        ):
            results.append(
                PatternResult(
                    pattern_name="Falling Wedge",
                    direction="bullish",
                    confidence=0.68,
                    candle_range=[0, len(df) - 1],
                    metadata={
                        "high_slope": round(high_slope, 6),
                        "low_slope": round(low_slope, 6),
                    },
                )
            )
        return results

    def _detect_cup_and_handle(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detect Cup and Handle: rounded bottom followed by small pullback."""
        results = []
        if len(df) < 40:
            return results

        cup = df.iloc[:-10]  # left portion = the cup
        handle = df.iloc[-10:]  # right portion = the handle

        cup_left_high = cup["close"].iloc[:5].mean()
        cup_right_high = cup["close"].iloc[-5:].mean()
        cup_bottom = cup["close"].min()

        # Cup edges roughly equal; handle retraces ≤ 50% of cup
        cup_depth = cup_left_high - cup_bottom
        handle_retrace = handle["close"].iloc[0] - handle["close"].min()

        if cup_depth == 0:
            return results

        edges_similar = abs(cup_left_high - cup_right_high) / cup_left_high < 0.05
        handle_shallow = handle_retrace / cup_depth <= 0.5

        if edges_similar and handle_shallow and cup_depth > 0:
            results.append(
                PatternResult(
                    pattern_name="Cup and Handle",
                    direction="bullish",
                    confidence=0.70,
                    candle_range=[0, len(df) - 1],
                    metadata={
                        "cup_depth_pct": round(cup_depth / cup_left_high * 100, 2),
                        "cup_left_high": round(cup_left_high, 4),
                        "cup_bottom": round(cup_bottom, 4),
                    },
                )
            )
        return results

    # ─── Support & Resistance ─────────────────────────────────────────────────

    def _detect_support_resistance(
        self, df: pd.DataFrame, lookback: int = 100
    ) -> List[float]:
        """
        Identify key support and resistance levels from swing highs/lows.

        Args:
            df:       OHLCV DataFrame.
            lookback: Number of candles to look back.

        Returns:
            Sorted list of unique price levels (deduplicated within 0.5%).
        """
        window = df.iloc[-lookback:] if len(df) > lookback else df

        peak_indices = self._find_swing_highs(window, order=3)
        trough_indices = self._find_swing_lows(window, order=3)

        levels: List[float] = []
        for idx in peak_indices:
            levels.append(float(window["high"].iloc[idx]))
        for idx in trough_indices:
            levels.append(float(window["low"].iloc[idx]))

        # Deduplicate levels within SR_PROXIMITY_PCT
        levels = sorted(set(levels))
        deduplicated: List[float] = []
        for level in levels:
            if not deduplicated:
                deduplicated.append(level)
                continue
            if abs(level - deduplicated[-1]) / deduplicated[-1] > self.SR_PROXIMITY_PCT:
                deduplicated.append(level)

        return deduplicated

    def get_nearest_levels(
        self,
        current_price: float,
        sr_levels: List[float],
        num_levels: int = 3,
    ) -> Dict[str, List[float]]:
        """
        Return the nearest support and resistance levels to the current price.

        Args:
            current_price: Current market price.
            sr_levels:     List of all S/R levels.
            num_levels:    How many levels to return on each side.

        Returns:
            Dict with 'support' and 'resistance' keys, each a list of prices.
        """
        supports = sorted([l for l in sr_levels if l < current_price], reverse=True)
        resistances = sorted([l for l in sr_levels if l > current_price])
        return {
            "support": supports[:num_levels],
            "resistance": resistances[:num_levels],
        }

    def calculate_slope(self, df: pd.DataFrame, window_size: int = 50) -> float:
        """
        Calculate the slope of the price trend over the last N candles.
        Returns a normalized slope (percentage change per candle).
        """
        if len(df) < window_size:
            window_size = len(df)

        y = df["close"].iloc[-window_size:].values
        x = np.arange(len(y))

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Normalize slope to % change per bar based on the average price
        avg_price = np.mean(y)
        if avg_price > 0:
            return (slope / avg_price) * 100
        return 0.0

    def is_near_level(
        self, price: float, level: float, tolerance_pct: Optional[float] = None
    ) -> bool:
        """
        Check whether `price` is within `tolerance_pct` of `level`.

        Args:
            price:         Current price.
            level:         S/R level to check proximity to.
            tolerance_pct: Override proximity threshold (default SR_PROXIMITY_PCT).

        Returns:
            True if price is within tolerance of the level.
        """
        tol = tolerance_pct if tolerance_pct is not None else self.SR_PROXIMITY_PCT
        return abs(price - level) / level <= tol

    def _detect_market_structure(self, df: pd.DataFrame) -> List[PatternResult]:
        """
        Detect Market Structure Breaks (BOS) by tracking swing highs and lows.
        """
        results: List[PatternResult] = []

        # Find recent swings
        swing_highs = self._find_swing_highs(df)
        swing_lows = self._find_swing_lows(df)

        if len(swing_highs) == 0 or len(swing_lows) == 0:
            return results

        current_close = float(df["close"].iloc[-1])

        # 1. Bearish BOS: Close below recent swing low
        last_low_idx = swing_lows[-1]
        last_low_price = float(df["close"].iloc[last_low_idx])

        if current_close < last_low_price:
            log.info(
                f"SMC: Bearish Break of Structure (BOS) detected at {current_close:.5f}"
            )
            results.append(
                PatternResult(
                    pattern_name="Break of Structure (Bearish)",
                    direction="bearish",
                    confidence=80.0,
                    candle_range=[int(last_low_idx), len(df) - 1],
                    metadata={"level": last_low_price},
                )
            )

        # 2. Bullish BOS: Close above recent swing high
        last_high_idx = swing_highs[-1]
        last_high_price = float(df["close"].iloc[last_high_idx])

        if current_close > last_high_price:
            log.info(
                f"SMC: Bullish Break of Structure (BOS) detected at {current_close:.5f}"
            )
            results.append(
                PatternResult(
                    pattern_name="Break of Structure (Bullish)",
                    direction="bullish",
                    confidence=80.0,
                    candle_range=[int(last_high_idx), len(df) - 1],
                    metadata={"level": last_high_price},
                )
            )

        return results
