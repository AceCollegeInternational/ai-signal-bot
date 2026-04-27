"""
modules/indicator_engine.py
─────────────────────────────────────────────────────────────────────────────
Module 2 — Technical Indicator Engine

Computes 50+ technical indicators and attaches them to an OHLCV DataFrame
using the pandas_ta library.

Indicator categories:
  Trend:      EMA(9,21,50,200), SMA(50,200), ADX(14), Parabolic SAR
  Momentum:   RSI(14), MACD(12,26,9), Stochastic(14,3,3), CCI(20)
  Volatility: Bollinger Bands(20,2), ATR(14), Keltner Channels
  Volume:     OBV, VWAP, Volume MA(20)

Usage:
    from modules.indicator_engine import IndicatorEngine
    engine = IndicatorEngine()
    df = engine.compute_all(df)
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import sys

if "numba" not in sys.modules:

    class MockNumba:
        def njit(self, *args, **kwargs):
            def decorator(func):
                return func

            if len(args) == 1 and callable(args[0]):
                return args[0]
            return decorator

        def jit(self, *args, **kwargs):
            return self.njit(*args, **kwargs)

    sys.modules["numba"] = MockNumba()

import pandas_ta as ta  # type: ignore[import]

from utils.logger import get_logger

log = get_logger(__name__)


class IndicatorEngine:
    """
    Computes all technical indicators for an OHLCV DataFrame.

    All indicator values are appended as new columns to the DataFrame.
    Column names follow the pattern: <INDICATOR>_<PERIOD> (e.g. EMA_9, RSI_14).

    Args:
        config: Optional config dict (unused currently, reserved for future
                per-strategy indicator parameters).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    # ─── Main Entry Point ─────────────────────────────────────────────────────

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and attach all indicator groups to the DataFrame.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
                and a DatetimeIndex.

        Returns:
            The same DataFrame with all indicator columns appended.
            Returns input DataFrame unchanged on error.
        """
        if df is None or df.empty or len(df) < 30:
            log.warning("DataFrame too short to compute indicators (need ≥ 30 rows)")
            return df

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            log.error(
                f"DataFrame missing required columns: {required - set(df.columns)}"
            )
            return df

        # Ensure no duplicate columns before computing indicators
        df = df.loc[:, ~df.columns.duplicated()].copy()

        try:
            df = self._compute_trend(df)
            df = self._compute_momentum(df)
            df = self._compute_volatility(df)
            df = self._compute_volume(df)
            df = self._compute_fibonacci(df)
            df = self._compute_divergence(df)
            df = self._compute_smc_filters(df)

            # Global Safety Fix: Ensure all columns are numeric and handle NaNs/None
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            df = self._add_signal_flags(df)
        except Exception as exc:
            log.error(f"Indicator computation failed: {exc}", exc_info=True)

        return df

    # ─── Trend Indicators ─────────────────────────────────────────────────────

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trend-following indicators.

        Indicators added:
          EMA_9, EMA_21, EMA_50, EMA_200
          SMA_50, SMA_200
          ADX_14, DMP_14 (+DI), DMN_14 (-DI)
          PSARs (Parabolic SAR stop price), PSARaf (acceleration factor),
          PSARr (reversal signal)
        """
        # ── EMAs ─────────────────────────────────────────────────────────────
        for period in (9, 21, 50, 200):
            col = f"EMA_{period}"
            result = ta.ema(df["close"], length=period)
            df[col] = result

        # ── SMAs ─────────────────────────────────────────────────────────────
        for period in (50, 200):
            col = f"SMA_{period}"
            result = ta.sma(df["close"], length=period)
            df[col] = result

        # ── ADX + DI lines ───────────────────────────────────────────────────
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is not None and not adx_df.empty:
            df["ADX_14"] = adx_df.get("ADX_14", np.nan)
            df["DMP_14"] = adx_df.get("DMP_14", np.nan)  # +DI
            df["DMN_14"] = adx_df.get("DMN_14", np.nan)  # -DI

        # ── Parabolic SAR ────────────────────────────────────────────────────
        psar_df = ta.psar(df["high"], df["low"], df["close"])
        if psar_df is not None and not psar_df.empty:
            # pandas_ta returns multiple columns; grab the main stop column
            psar_cols = [c for c in psar_df.columns if c.startswith("PSARs")]
            if psar_cols:
                df["PSAR"] = psar_df[psar_cols[0]]
            rev_cols = [c for c in psar_df.columns if c.startswith("PSARr")]
            if rev_cols:
                df["PSAR_Rev"] = psar_df[rev_cols[0]]

        log.debug("Trend indicators computed")
        return df

    # ─── Momentum Indicators ─────────────────────────────────────────────────

    def _compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum indicators.

        Indicators added:
          RSI_14
          MACD_12_26_9, MACDs_12_26_9 (signal), MACDh_12_26_9 (histogram)
          STOCHk_14_3_3 (%K), STOCHd_14_3_3 (%D)
          CCI_20_0.015
        """
        # ── RSI ──────────────────────────────────────────────────────────────
        df["RSI_14"] = ta.rsi(df["close"], length=14)

        # ── MACD ─────────────────────────────────────────────────────────────
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            df["MACD"] = macd_df.get("MACD_12_26_9", np.nan)
            df["MACD_Signal"] = macd_df.get("MACDs_12_26_9", np.nan)
            df["MACD_Hist"] = macd_df.get("MACDh_12_26_9", np.nan)

        # ── Stochastic Oscillator ─────────────────────────────────────────────
        stoch_df = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
        if stoch_df is not None and not stoch_df.empty:
            stoch_cols = stoch_df.columns.tolist()
            k_cols = [c for c in stoch_cols if c.startswith("STOCHk")]
            d_cols = [c for c in stoch_cols if c.startswith("STOCHd")]
            if k_cols:
                df["STOCH_K"] = stoch_df[k_cols[0]]
            if d_cols:
                df["STOCH_D"] = stoch_df[d_cols[0]]

        # ── CCI ──────────────────────────────────────────────────────────────
        cci_result = ta.cci(df["high"], df["low"], df["close"], length=20)
        if cci_result is not None:
            df["CCI_20"] = cci_result

        log.debug("Momentum indicators computed")
        return df

    # ─── Volatility Indicators ────────────────────────────────────────────────

    def _compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volatility indicators.

        Indicators added:
          BBU_20_2 (upper), BBM_20_2 (middle), BBL_20_2 (lower),
          BBB_20_2 (bandwidth), BBP_20_2 (percent)
          ATR_14
          KCe_20_2 (upper/lower Keltner Channel)
        """
        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb_df = ta.bbands(df["close"], length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            df["BB_Upper"] = bb_df.get("BBU_20_2.0", np.nan)
            df["BB_Mid"] = bb_df.get("BBM_20_2.0", np.nan)
            df["BB_Lower"] = bb_df.get("BBL_20_2.0", np.nan)
            df["BB_Width"] = bb_df.get("BBB_20_2.0", np.nan)
            df["BB_Pct"] = bb_df.get("BBP_20_2.0", np.nan)
            # Fallback: key names vary slightly between pandas_ta versions
            if df["BB_Upper"].isna().all():
                bb_upper_cols = [c for c in bb_df.columns if "BBU" in c]
                bb_lower_cols = [c for c in bb_df.columns if "BBL" in c]
                bb_mid_cols = [c for c in bb_df.columns if "BBM" in c]
                if bb_upper_cols:
                    df["BB_Upper"] = bb_df[bb_upper_cols[0]]
                if bb_lower_cols:
                    df["BB_Lower"] = bb_df[bb_lower_cols[0]]
                if bb_mid_cols:
                    df["BB_Mid"] = bb_df[bb_mid_cols[0]]

        # ── ATR ──────────────────────────────────────────────────────────────
        atr_result = ta.atr(df["high"], df["low"], df["close"], length=14)
        if atr_result is not None:
            df["ATR_14"] = atr_result

        # ── Keltner Channels ─────────────────────────────────────────────────
        kc_df = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=2)
        if kc_df is not None and not kc_df.empty:
            kcu_cols = [c for c in kc_df.columns if "KCU" in c]
            kcl_cols = [c for c in kc_df.columns if "KCL" in c]
            kcb_cols = [c for c in kc_df.columns if "KCB" in c]
            if kcu_cols:
                df["KC_Upper"] = kc_df[kcu_cols[0]]
            if kcl_cols:
                df["KC_Lower"] = kc_df[kcl_cols[0]]
            if kcb_cols:
                df["KC_Mid"] = kc_df[kcb_cols[0]]

        log.debug("Volatility indicators computed")
        return df

    # ─── Volume Indicators ────────────────────────────────────────────────────

    def _compute_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volume-based indicators.

        Indicators added:
          OBV (On-Balance Volume)
          VWAP (Volume-Weighted Average Price)
          Volume_MA_20 (20-period volume moving average)
          Volume_Ratio (current volume / Volume_MA_20)
        """
        # ── OBV ──────────────────────────────────────────────────────────────
        obv_result = ta.obv(df["close"], df["volume"])
        if obv_result is not None:
            df["OBV"] = obv_result

        # ── VWAP ─────────────────────────────────────────────────────────────
        # pandas_ta VWAP requires an anchor; use daily reset by default
        try:
            vwap_result = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            if vwap_result is not None:
                df["VWAP"] = vwap_result
        except Exception as exc:
            log.debug(f"VWAP computation skipped: {exc}")

        # ── Volume MA ─────────────────────────────────────────────────────────
        vol_ma = ta.sma(df["volume"], length=20)
        if vol_ma is not None:
            df["Volume_MA_20"] = vol_ma
            # Volume ratio: >1 means above-average volume (confirms signals)
            df["Volume_Ratio"] = df["volume"] / df["Volume_MA_20"].replace(0, np.nan)

        log.debug("Volume indicators computed")
        return df

    # ─── Fibonacci Levels ─────────────────────────────────────────────────────
    def _compute_fibonacci(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Fibonacci retracement levels based on recent swing high/low.
        Uses a 100-period lookback to identify the structural range.
        Kojo Forex style confluence.
        """
        # Ensure we have enough data
        lookback = min(100, len(df))
        window = df.iloc[-lookback:]

        highest = window["high"].max()
        lowest = window["low"].min()
        diff = highest - lowest

        if diff > 0:
            # Traditional retracement levels
            df["Fib_0"] = lowest
            df["Fib_236"] = highest - (diff * 0.236)
            df["Fib_382"] = highest - (diff * 0.382)
            df["Fib_500"] = highest - (diff * 0.500)
            df["Fib_618"] = highest - (diff * 0.618)
            df["Fib_786"] = highest - (diff * 0.786)
            df["Fib_100"] = highest
        else:
            for lvl in ["0", "236", "382", "500", "618", "786", "100"]:
                df[f"Fib_{lvl}"] = df["close"]

        log.debug("Fibonacci levels computed")
        return df

    # ─── Divergence Detection ─────────────────────────────────────────────────

    def _compute_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect robust bullish/bearish divergences between price and RSI.
        Uses local extrema (peaks/troughs) to find non-alignments.
        """
        if "RSI_14" not in df.columns or len(df) < 50:
            return df

        df["is_bullish_div"] = False
        df["is_bearish_div"] = False

        # Find local troughs for bullish divergence
        # Lowest in 5-bar window
        df["price_low_window"] = df["low"].rolling(5, center=True).min()
        df["rsi_low_window"] = df["RSI_14"].rolling(5, center=True).min()

        # Identify local troughs where price makes LL but RSI makes HL
        # Previous trough (shifted 5 to 20 bars)
        df["prev_price_low"] = df["low"].shift(10).rolling(15).min()
        df["prev_rsi_low"] = df["RSI_14"].shift(10).rolling(15).min()

        df["is_bullish_div"] = (
            (df["low"] < df["prev_price_low"])
            & (df["RSI_14"] > df["prev_rsi_low"])
            & (df["RSI_14"] < 35)
        )

        # Identify local peaks where price makes HH but RSI makes LH
        df["prev_price_high"] = df["high"].shift(10).rolling(15).max()
        df["prev_rsi_high"] = df["RSI_14"].shift(10).rolling(15).max()

        df["is_bearish_div"] = (
            (df["high"] > df["prev_price_high"])
            & (df["RSI_14"] < df["prev_rsi_high"])
            & (df["RSI_14"] > 65)
        )

        # Cleanup temp columns
        df.drop(
            columns=[
                "price_low_window",
                "rsi_low_window",
                "prev_price_low",
                "prev_rsi_low",
                "prev_price_high",
                "prev_rsi_high",
            ],
            inplace=True,
        )

        return df

    # ─── Signal Flags ────────────────────────────────────────────────────────

    def _compute_smc_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Smart Money Concepts (SMC) filters:
        - Liquidity Sweep (Stop Hunt)
        - Consolidation (Tight Range)
        """
        win = 20
        df["prev_high_max"] = df["high"].rolling(window=win).max().shift(1)
        df["prev_low_min"] = df["low"].rolling(window=win).min().shift(1)

        # 1. Bearish Liquidity Sweep (High breaks above level but fails to hold)
        df["is_liquidity_sweep_bearish"] = (df["high"] > df["prev_high_max"]) & (
            df["close"] < df["prev_high_max"]
        )

        # 2. Bullish Liquidity Sweep (Low breaks below level but fails to hold)
        df["is_liquidity_sweep_bullish"] = (df["low"] < df["prev_low_min"]) & (
            df["close"] > df["prev_low_min"]
        )

        # 3. Consolidation Detector (Tight Range relative to ATR)
        if "ATR_14" in df.columns:
            # Range of last 10 candles
            rolling_max = df["close"].rolling(10).max()
            rolling_min = df["close"].rolling(10).min()
            rolling_range = rolling_max - rolling_min

            # If rolling range < 0.8 * ATR, mark as consolidating
            df["is_consolidating"] = rolling_range < (df["ATR_14"] * 1.5)
        else:
            df["is_consolidating"] = False

        last_idx = df.index[-1]

        # Robust scalar check to avoid 'truth value of array is ambiguous' errors
        def get_last_val(col_name: str) -> bool:
            if col_name not in df.columns:
                return False
            col = df[col_name]
            # If duplicated, take the first one
            val = (
                col.iloc[-1]
                if not isinstance(col.iloc[-1], pd.Series)
                else col.iloc[-1].iloc[0]
            )
            return bool(val)

        if get_last_val("is_liquidity_sweep_bearish"):
            log.info(f"SMC: Bearish Liquidity Sweep detected on {last_idx}")

        if get_last_val("is_liquidity_sweep_bullish"):
            log.info(f"SMC: Bullish Liquidity Sweep detected on {last_idx}")

        return df

    def _add_signal_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive binary signal helper columns for quick condition checking.
        """
        close = df["close"]

        # RSI flags
        if "RSI_14" in df.columns:
            df["is_rsi_overbought"] = df["RSI_14"] > 70
            df["is_rsi_oversold"] = df["RSI_14"] < 30

        # MACD flags
        if "MACD" in df.columns and "MACD_Signal" in df.columns:
            df["is_macd_bullish"] = (df["MACD_Hist"] > 0) & (
                df["MACD"] > df["MACD_Signal"]
            )
            df["is_macd_bearish"] = (df["MACD_Hist"] < 0) & (
                df["MACD"] < df["MACD_Signal"]
            )

        # EMA position flags
        if "EMA_50" in df.columns:
            df["is_above_ema50"] = close > df["EMA_50"]
        if "EMA_200" in df.columns:
            df["is_above_ema200"] = close > df["EMA_200"]

        # Golden / death cross (EMA_50 vs EMA_200)
        if "EMA_50" in df.columns and "EMA_200" in df.columns:
            ema_diff = df["EMA_50"] - df["EMA_200"]
            df["is_golden_cross"] = (ema_diff > 0) & (ema_diff.shift(1) <= 0)
            df["is_death_cross"] = (ema_diff < 0) & (ema_diff.shift(1) >= 0)

        # ADX trend strength
        if "ADX_14" in df.columns:
            df["is_adx_trending"] = df["ADX_14"] > 25

        # Volume spike
        if "Volume_Ratio" in df.columns:
            df["is_high_volume"] = df["Volume_Ratio"] > 1.5

        # Bollinger Band position (-1, 0, 1)
        if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
            conditions = [
                close < df["BB_Lower"],
                close > df["BB_Upper"],
            ]
            choices = [-1, 1]
            df["price_vs_bb"] = np.select(conditions, choices, default=0)

        log.debug("Signal flags computed")
        return df

    # ─── Per-Indicator Helpers ────────────────────────────────────────────────

    def get_current_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Return the most recent value for every indicator column.

        Args:
            df: DataFrame with indicators already computed.

        Returns:
            Dict mapping column name → last (non-NaN) value as float.
        """
        if df.empty:
            return {}

        ohlcv_cols = {"open", "high", "low", "close", "volume"}
        result: Dict[str, float] = {}
        last_row = df.iloc[-1]

        for col in df.columns:
            if col in ohlcv_cols:
                continue
            val = last_row[col]
            try:
                result[col] = float(val) if not pd.isna(val) else 0.0
            except (TypeError, ValueError):
                result[col] = 0.0

        return result

    def count_bullish_signals(self, df: pd.DataFrame) -> int:
        """
        Count how many indicator-based bullish conditions are active in the
        last candle (for confluence scoring).

        Args:
            df: DataFrame with indicators + signal flags.

        Returns:
            Count of active bullish conditions (0–N).
        """
        if df.empty:
            return 0

        last = df.iloc[-1]
        conditions = [
            last.get("is_rsi_oversold", False),
            last.get("is_macd_bullish", False),
            last.get("is_above_ema50", False),
            last.get("is_above_ema200", False),
            last.get("is_golden_cross", False),
            last.get("is_adx_trending", False)
            and (last.get("DMP_14") or 0) > (last.get("DMN_14") or 0),
            last.get("price_vs_bb", 0) == -1,  # price below lower band (oversold setup)
            last.get("STOCH_K", 50) < 20,  # stochastic oversold
            last.get("CCI_20", 0) < -100,  # CCI oversold
        ]
        return sum(1 for c in conditions if c)

    def count_bearish_signals(self, df: pd.DataFrame) -> int:
        """
        Count how many indicator-based bearish conditions are active in the
        last candle (for confluence scoring).

        Args:
            df: DataFrame with indicators + signal flags.

        Returns:
            Count of active bearish conditions (0–N).
        """
        if df.empty:
            return 0

        last = df.iloc[-1]
        conditions = [
            last.get("is_rsi_overbought", False),
            last.get("is_macd_bearish", False),
            not last.get("is_above_ema50", True),
            not last.get("is_above_ema200", True),
            last.get("is_death_cross", False),
            last.get("is_adx_trending", False)
            and (last.get("DMN_14") or 0) > (last.get("DMP_14") or 0),
            last.get("price_vs_bb", 0)
            == 1,  # price above upper band (overbought setup)
            last.get("STOCH_K", 50) > 80,  # stochastic overbought
            last.get("CCI_20", 0) > 100,  # CCI overbought
        ]
        return sum(1 for c in conditions if c)
