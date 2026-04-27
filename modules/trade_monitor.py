import pandas as pd
from typing import Dict, Any, Optional
from utils.logger import get_logger
from modules.risk_manager import OpenPosition

log = get_logger(__name__)


class TradeMonitor:
    """
    Intelligent monitoring for active positions.
    Provides logic for re-validating trades in drawdown and
    locking in profits based on momentum stalling.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Momentum threshold: if ADX drops or RSI flattens, we might close
        self.momentum_threshold = config.get("risk", {}).get(
            "momentum_stall_threshold", 0.5
        )

    def revalidate_thesis(self, pos: OpenPosition, df: pd.DataFrame) -> bool:
        """
        Check if the technical thesis for a trade is still valid while in drawdown.
        Returns True if thesis holds, False if thesis is broken and we should exit.
        """
        if df.empty:
            return True

        # Example logic: if we are LONG but RSI is now below 40 and
        # price is below EMA50, the bullish thesis might be dead.
        last_rsi = df["RSI_14"].iloc[-1] if "RSI_14" in df.columns else 50
        is_below_ema50 = (
            (df["close"].iloc[-1] < df["EMA_50"].iloc[-1])
            if "EMA_50" in df.columns
            else False
        )

        if pos.direction == "long":
            if last_rsi < 35 and is_below_ema50:
                log.info(
                    f"Thesis broken for {pos.symbol} (Long): RSI={last_rsi:.1f}, Below EMA50"
                )
                return False
        else:  # short
            if last_rsi > 65 and not is_below_ema50:
                log.info(
                    f"Thesis broken for {pos.symbol} (Short): RSI={last_rsi:.1f}, Above EMA50"
                )
                return False

        return True

    def check_momentum_stall(self, pos: OpenPosition, df: pd.DataFrame) -> bool:
        """
        Check if a profitable trade should be closed early due to trend exhaustion.
        Returns True if momentum has stalled.
        """
        if len(df) < 5 or "ADX_14" not in df.columns:
            return False

        # If trade is in profit (rough check)
        # Note: real PnL is in RiskManager, but we can check price vs entry here
        curr_price = df["close"].iloc[-1]
        is_in_profit = (
            (curr_price > pos.entry_price)
            if pos.direction == "long"
            else (curr_price < pos.entry_price)
        )

        if not is_in_profit:
            return False

        adx = df["ADX_14"].iloc[-5:]
        # If ADX is decreasing (trend weakening) while we are in profit
        if adx.iloc[-1] < adx.iloc[-3] and adx.iloc[-1] < 25:
            log.info(f"Momentum stalling for {pos.symbol}: ADX={adx.iloc[-1]:.1f}")
            return True

        return False
