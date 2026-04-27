"""
modules/risk_manager.py
─────────────────────────────────────────────────────────────────────────────
Module 5 — Risk Management Engine

Enforces all risk constraints before a trade is allowed to execute:
  - Position sizing: (account × risk%) / (entry - stop_loss)
  - ATR-based dynamic stop-loss
  - Take-profit levels: TP1 = 1.5× risk, TP2 = 3× risk
  - Max open trades (default 3)
  - Daily loss limit (default 5%): pauses bot if breached
  - Minimum R/R ratio (default 1.5:1)
  - Optional high-impact news event filter (Finnhub API)

Usage:
    rm = RiskManager(config, account_balance=10000.0)
    approved, sizing = rm.evaluate_signal(signal, df)
"""

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from modules.news_filter import NewsFilter
from modules.ml_validator import MLValidator
from modules.ai_signal_engine import TradeSignal
from utils.helpers import (
    calc_position_size,
    calc_stop_loss,
    calc_take_profits,
    risk_reward_ratio,
)
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PositionSizing:
    """
    Contains all computed values for a prospective trade.

    Attributes:
        symbol:          Trading pair.
        direction:       'long' or 'short'.
        entry_price:     Trade entry price.
        stop_loss:       Stop-loss price (ATR-adjusted).
        take_profit_1:   First take-profit target.
        take_profit_2:   Second take-profit target.
        position_size:   Units of base asset to buy/sell.
        risk_amount:     Dollar amount at risk.
        risk_reward:     Computed R/R ratio.
        atr:             ATR value used for stop calculation.
        approved:        Whether risk manager approved this trade.
        rejection_reason: If not approved, the reason.
    """

    symbol: str = ""
    direction: str = "long"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    position_size: float = 0.0
    risk_amount: float = 0.0
    risk_reward: float = 0.0
    atr: float = 0.0
    approved: bool = False
    rejection_reason: str = ""
    ml_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "position_size": self.position_size,
            "risk_amount": self.risk_amount,
            "risk_reward": self.risk_reward,
            "approved": self.approved,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class OpenPosition:
    """Tracks a live or paper-trading open position."""

    symbol: str
    direction: str
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trailing_stop: Optional[float] = None
    tp1_hit: bool = False
    ml_snapshot: Dict[str, Any] = field(default_factory=dict)

    def unrealised_pnl(self, current_price: float) -> float:
        """Calculate unrealised P&L in quote currency."""
        if self.direction == "long":
            return (current_price - self.entry_price) * self.position_size
        return (self.entry_price - current_price) * self.position_size


class RiskManager:
    """
    Evaluates trade signals against risk parameters and sizes positions.

    Args:
        config:          Parsed config.yaml dictionary.
        account_balance: Starting account balance in quote currency (e.g. USDT).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        account_balance: float = 10_000.0,
    ) -> None:
        risk_cfg = config.get("risk", {})
        self.max_risk_per_trade: float = risk_cfg.get("max_risk_per_trade", 0.015)
        self.max_open_trades: int = risk_cfg.get("max_open_trades", 3)
        self.max_daily_loss: float = risk_cfg.get("max_daily_loss", 0.05)
        self.min_rr_ratio: float = risk_cfg.get("min_rr_ratio", 1.5)
        self.atr_multiplier: float = risk_cfg.get("atr_multiplier", 1.5)
        self.tp1_multiplier: float = risk_cfg.get("tp1_multiplier", 1.5)
        self.tp2_multiplier: float = risk_cfg.get("tp2_multiplier", 3.0)
        self.max_trades_per_day: int = risk_cfg.get("max_trades_per_day", 5)

        # Journaling setup
        self.journal_path = os.path.join("logs", "trade_journal.csv")
        self.ml_dataset_path = os.path.join("logs", "ml_dataset.jsonl")
        self._init_journal()

        news_cfg = config.get("news", {})
        self.skip_high_impact_news: bool = news_cfg.get("skip_high_impact", True)
        self.news_buffer_minutes: int = news_cfg.get("news_buffer_minutes", 30)
        self.finnhub_enabled: bool = news_cfg.get("finnhub_enabled", False)
        self.finnhub_key: str = os.getenv("FINNHUB_API_KEY", "")

        self.news_filter = NewsFilter(config)
        self.ml_validator = MLValidator()

        # Session windows (UTC)
        self.sessions = {
            "london": {"start": 8, "end": 16},
            "new_york": {"start": 13, "end": 21},
        }
        self.session_filter_enabled = config.get("trading", {}).get(
            "session_filter", True
        )

        # Correlation groups to prevent over-exposure
        self.correlation_groups = [
            {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"},
            {"USDJPY", "EURJPY", "GBPJPY"},
            {"EURGBP", "EURUSD"},  # Secondary correlations
        ]

        # Live state tracking
        self.account_balance: float = account_balance
        self.initial_balance: float = account_balance
        self.open_positions: Dict[str, OpenPosition] = {}  # symbol → position
        self.daily_pnl: float = 0.0
        self.trades_today: int = 0
        self.daily_pnl_date: date = date.today()
        self.bot_paused: bool = False

        log.info(
            f"RiskManager initialised | balance={account_balance:.2f} "
            f"max_risk={self.max_risk_per_trade*100:.1f}% "
            f"max_open={self.max_open_trades}"
        )

    # ─── Main Evaluation ──────────────────────────────────────────────────────

    def evaluate_signal(
        self,
        signal: TradeSignal,
        df: "pd.DataFrame",  # noqa: F821 (avoid circular import in type hint)
        symbol: str = "",
    ) -> Tuple[bool, PositionSizing]:
        """
        Full risk evaluation pipeline for an incoming trade signal.

        Checks in order:
          1. Bot pause status (daily loss limit)
          2. Max open trades
          3. Existing position in same symbol
          4. High-impact news window
          5. Signal confidence threshold
          6. Compute ATR-based levels
          7. Minimum R/R ratio
          8. Position sizing

        Args:
            signal: TradeSignal from AI engine.
            df:     OHLCV + indicator DataFrame (for ATR).
            symbol: Trading pair (often same as signal.symbol).

        Returns:
            Tuple of (approved: bool, PositionSizing with all levels).
        """
        sym = symbol or signal.symbol
        sizing = PositionSizing(symbol=sym)

        # Pre-calculate indicator snapshot for ML training (even if rejected, we might want to know why)
        indicators = {}
        if df is not None and not df.empty:
            last_row = df.iloc[-1]
            for col in df.columns:
                if col not in ["open", "high", "low", "close", "volume"]:
                    val = last_row[col]
                    indicators[col] = 0.0 if pd.isna(val) else float(val)
        sizing.ml_snapshot = indicators

        # 1. Bot paused?
        self._reset_daily_pnl_if_new_day()
        if self.bot_paused:
            sizing.rejection_reason = "Bot paused — daily loss limit reached"
            log.warning(f"Signal rejected (bot paused): {sym}")
            return False, sizing

        # 2. Max trades per day
        self._reset_daily_pnl_if_new_day()
        if self.trades_today >= self.max_trades_per_day:
            sizing.rejection_reason = (
                f"Daily trade limit reached ({self.max_trades_per_day})"
            )
            log.info(f"Signal rejected (daily limit): {sym}")
            return False, sizing

        # 3. Max open trades
        if len(self.open_positions) >= self.max_open_trades:
            sizing.rejection_reason = (
                f"Max open trades reached ({self.max_open_trades})"
            )
            log.info(f"Signal rejected (max trades): {sym}")
            return False, sizing

        # 3. Already in a position for this symbol
        if sym in self.open_positions:
            sizing.rejection_reason = f"Already in an open position for {sym}"
            return False, sizing

        # 4. Correlation check (avoid multiple positions in highly correlated pairs)
        if self._is_highly_correlated_open(sym):
            sizing.rejection_reason = (
                f"Highly correlated pair already has an open position"
            )
            log.info(f"Signal rejected (correlation): {sym}")
            return False, sizing

        # 4. Session Filter
        if self.session_filter_enabled and not self._is_in_trading_session():
            sizing.rejection_reason = "Outside of volatile trading sessions (London/NY)"
            log.info(f"Signal rejected (session filter): {sym}")
            return False, sizing

        # 5. High-impact news event check
        if self.news_filter.is_trading_suspended():
            sizing.rejection_reason = "High-impact news event — trading suspended"
            log.info(f"Signal rejected (news event): {sym}")
            return False, sizing

        # 6. ML Signature Check (The Gatekeeper)
        if not self.ml_validator.is_approved(sizing.ml_snapshot):
            sizing.rejection_reason = (
                "ML Rejection: Setup history shows low win probability"
            )
            log.info(f"Signal rejected (ML validation): {sym}")
            return False, sizing
        if not signal.is_actionable(self.min_rr_ratio):
            sizing.rejection_reason = (
                f"Signal not actionable: confidence={signal.confidence:.1f} "
                f"R/R={signal.risk_reward_ratio:.2f}"
            )
            return False, sizing

        # 6. Compute ATR-based levels
        atr = self._get_atr(df)
        direction = "long" if signal.signal == "BUY" else "short"

        # Use Claude-suggested entry, override stops with ATR-based levels
        entry = signal.entry_price if signal.entry_price > 0 else self._last_close(df)
        if entry <= 0:
            sizing.rejection_reason = "Cannot determine entry price"
            return False, sizing

        # ATR stop takes precedence over Claude's stop if ATR is available
        if atr > 0:
            stop = calc_stop_loss(entry, atr, self.atr_multiplier, direction)
        else:
            stop = (
                signal.stop_loss
                if signal.stop_loss > 0
                else entry * (0.98 if direction == "long" else 1.02)
            )

        tp1, tp2 = calc_take_profits(
            entry, stop, self.tp1_multiplier, self.tp2_multiplier, direction
        )

        # 7. R/R ratio check
        rr = risk_reward_ratio(entry, stop, tp1)
        if rr < self.min_rr_ratio:
            sizing.rejection_reason = (
                f"R/R ratio too low: {rr:.2f} < {self.min_rr_ratio}"
            )
            log.info(f"Signal rejected (R/R={rr:.2f}): {sym}")
            return False, sizing

        # 8. Position sizing
        pos_size = calc_position_size(
            self.account_balance,
            self.max_risk_per_trade,
            entry,
            stop,
        )
        if pos_size <= 0:
            sizing.rejection_reason = "Position size calculated as zero"
            return False, sizing

        risk_amt = abs(entry - stop) * pos_size

        sizing = PositionSizing(
            symbol=sym,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            position_size=pos_size,
            risk_amount=risk_amt,
            risk_reward=rr,
            atr=atr,
            approved=True,
        )
        log.info(
            f"Signal APPROVED: {sym} {direction.upper()} @ {entry:.4f} "
            f"SL={stop:.4f} TP1={tp1:.4f} TP2={tp2:.4f} "
            f"size={pos_size:.4f} R/R={rr:.2f}"
        )
        return True, sizing

    # ─── Position Tracking ────────────────────────────────────────────────────

    def open_position(self, sizing: PositionSizing) -> OpenPosition:
        """
        Register a new open position after order execution.

        Args:
            sizing: Approved PositionSizing from evaluate_signal().

        Returns:
            The created OpenPosition.
        """
        pos = OpenPosition(
            symbol=sizing.symbol,
            direction=sizing.direction,
            entry_price=sizing.entry_price,
            position_size=sizing.position_size,
            stop_loss=sizing.stop_loss,
            take_profit_1=sizing.take_profit_1,
            take_profit_2=sizing.take_profit_2,
            ml_snapshot=sizing.ml_snapshot,
        )
        self.open_positions[sizing.symbol] = pos
        self.trades_today += 1
        log.info(
            f"Position opened: {sizing.symbol} {sizing.direction} @ {sizing.entry_price:.4f} (Trade #{self.trades_today} today)"
        )
        return pos

    def close_position(
        self, symbol: str, close_price: float, reason: str = ""
    ) -> float:
        """
        Close an open position and update daily P&L.

        Args:
            symbol:       Trading pair.
            close_price:  Price at which the position was closed.
            reason:       Why the position was closed (SL, TP1, TP2, manual).

        Returns:
            Realised P&L in quote currency.
        """
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            log.warning(f"No open position found for {symbol}")
            return 0.0

        pnl = pos.unrealised_pnl(close_price)
        self.daily_pnl += pnl
        self.account_balance += pnl

        log.info(
            f"Position closed: {symbol} @ {close_price:.4f} | "
            f"P&L = {pnl:+.2f} | reason={reason}"
        )

        # Record to trade journal
        self._log_trade(pos, close_price, pnl, reason)

        # Check daily loss limit
        daily_loss_pct = -self.daily_pnl / self.initial_balance
        if daily_loss_pct >= self.max_daily_loss:
            self.bot_paused = True
            log.warning(
                f"Daily loss limit reached ({daily_loss_pct*100:.1f}%). "
                f"Bot paused until next trading day."
            )

        return pnl

    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        atr: float,
    ) -> Optional[float]:
        """
        Update trailing stop for an open position as price moves favorably.

        Trailing stop is moved to lock in profits when price is ≥ 1 ATR
        above the current trailing stop (for longs).

        Args:
            symbol:        Trading pair.
            current_price: Current market price.
            atr:           Current ATR value.

        Returns:
            New trailing stop price, or None if not updated.
        """
        pos = self.open_positions.get(symbol)
        if pos is None:
            return None

        if pos.direction == "long":
            candidate = current_price - atr * self.atr_multiplier
            if pos.trailing_stop is None or candidate > pos.trailing_stop:
                pos.trailing_stop = candidate
                pos.stop_loss = candidate
                log.debug(f"Trailing stop updated: {symbol} → {candidate:.4f}")
                return candidate
        else:  # short
            candidate = current_price + atr * self.atr_multiplier
            if pos.trailing_stop is None or candidate < pos.trailing_stop:
                pos.trailing_stop = candidate
                pos.stop_loss = candidate
                log.debug(f"Trailing stop updated: {symbol} → {candidate:.4f}")
                return candidate

        return None

    def check_exit_conditions(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if an open position has hit SL, TP1, or TP2.

        Args:
            symbol:        Trading pair.
            current_price: Current market price.

        Returns:
            'SL', 'TP1', 'TP2', or None if no exit triggered.
        """
        pos = self.open_positions.get(symbol)
        if pos is None:
            return None

        if pos.direction == "long":
            if current_price <= pos.stop_loss:
                return "SL"
            if not pos.tp1_hit and current_price >= pos.take_profit_1:
                pos.tp1_hit = True
                # Scale out: reduce size by 50% at TP1
                pos.position_size *= 0.5
                return "TP1"
            if pos.tp1_hit and current_price >= pos.take_profit_2:
                return "TP2"
        else:  # short
            if current_price >= pos.stop_loss:
                return "SL"
            if not pos.tp1_hit and current_price <= pos.take_profit_1:
                pos.tp1_hit = True
                pos.position_size *= 0.5
                return "TP1"
            if pos.tp1_hit and current_price <= pos.take_profit_2:
                return "TP2"

        return None

    # ─── Daily P&L Management ─────────────────────────────────────────────────

    def _reset_daily_pnl_if_new_day(self) -> None:
        """Reset daily P&L counter on a new calendar day."""
        today = date.today()
        if today != self.daily_pnl_date:
            log.info(
                f"New trading day — resetting daily P&L (was {self.daily_pnl:+.2f})"
            )
            self.daily_pnl = 0.0
            self.daily_pnl_date = today
            self.bot_paused = False

    def get_daily_pnl_pct(self) -> float:
        """Return today's P&L as a fraction of initial balance."""
        return (
            self.daily_pnl / self.initial_balance if self.initial_balance > 0 else 0.0
        )

    def _is_in_trading_session(self) -> bool:
        """
        Check if current UTC time falls within London or New York sessions.
        """
        now_utc = datetime.now(timezone.utc).hour

        for session, window in self.sessions.items():
            if window["start"] <= now_utc < window["end"]:
                return True
        return False

    # ─── News Event Filter ────────────────────────────────────────────────────
    # High-impact news filtering is now handled by modules/news_filter.py
    # integrated via the evaluate_signal method.

    def _is_highly_correlated_open(self, symbol: str) -> bool:
        """Check if any currently open position is in a symbol highly correlated with `symbol`."""
        for group in self.correlation_groups:
            if symbol in group:
                # Check if any OTHER member of this group is open
                other_members = group - {symbol}
                for open_sym in self.open_positions:
                    if open_sym in other_members:
                        return True
        return False

    # ─── Journaling ───────────────────────────────────────────────────────────

    def _init_journal(self) -> None:
        """Initialize the CSV trade journal with headers if it doesn't exist."""
        os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)
        if not os.path.exists(self.journal_path):
            with open(self.journal_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Timestamp",
                        "Symbol",
                        "Direction",
                        "Entry",
                        "Exit",
                        "Size",
                        "PnL",
                        "PnL_Pct",
                        "Reason",
                        "Duration_Mins",
                    ]
                )

    def _log_trade(
        self, pos: OpenPosition, close_price: float, pnl: float, reason: str
    ) -> None:
        """Log trade details to the CSV journal."""
        try:
            duration = (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 60
            # Rough PnL% calculation
            pnl_pct = (
                (pnl / (pos.entry_price * pos.position_size)) * 100
                if pos.entry_price > 0
                else 0
            )

            with open(self.journal_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        pos.symbol,
                        pos.direction,
                        f"{pos.entry_price:.5f}",
                        f"{close_price:.5f}",
                        f"{pos.position_size:.4f}",
                        f"{pnl:.2f}",
                        f"{pnl_pct:.2f}%",
                        reason,
                        f"{duration:.1f}",
                    ]
                )

            # Persist ML snapshot for training
            if pos.ml_snapshot:
                ml_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": pos.symbol,
                    "direction": pos.direction,
                    "pnl_pct": pnl_pct,
                    "win": 1 if pnl > 0 else 0,
                    "features": pos.ml_snapshot,
                }
                with open(self.ml_dataset_path, mode="a") as f:
                    f.write(json.dumps(ml_entry) + "\n")

            log.debug(f"Trade logged to journal and ML dataset: {pos.symbol}")
        except Exception as e:
            log.error(f"Failed to log trade to journal: {e}")

    # ─── Daily P&L Management ─────────────────────────────────────────────────

    def _reset_daily_pnl_if_new_day(self) -> None:
        """Reset daily counters on a new calendar day."""
        today = date.today()
        if today != self.daily_pnl_date:
            log.info(
                f"New trading day — resetting daily stats (PnL: {self.daily_pnl:+.2f}, Trades: {self.trades_today})"
            )
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.daily_pnl_date = today
            self.bot_paused = False

    @staticmethod
    def _get_atr(df: "pd.DataFrame") -> float:  # noqa: F821
        """Extract the last ATR_14 value from the DataFrame."""
        if df is None or df.empty or "ATR_14" not in df.columns:
            return 0.0
        try:
            val = df["ATR_14"].dropna().iloc[-1]
            return float(val)
        except (IndexError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def _last_close(df: "pd.DataFrame") -> float:  # noqa: F821
        """Return the most recent closing price from the DataFrame."""
        if df is None or df.empty or "close" not in df.columns:
            return 0.0
        return float(df["close"].iloc[-1])

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Return a summary of current portfolio state.

        Returns:
            Dict with balance, open positions, daily P&L, and bot status.
        """
        return {
            "account_balance": self.account_balance,
            "initial_balance": self.initial_balance,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.get_daily_pnl_pct() * 100, 2),
            "open_positions": len(self.open_positions),
            "open_symbols": list(self.open_positions.keys()),
            "bot_paused": self.bot_paused,
        }
