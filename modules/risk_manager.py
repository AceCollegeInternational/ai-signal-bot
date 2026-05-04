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
from zoneinfo import ZoneInfo

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
    is_reversal: bool = False
    reversal_from: str = ""

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
    initial_risk_amount: float = 0.0
    planned_risk_reward: float = 0.0
    trade_id: str = ""

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
        self.opposite_signal_confidence: float = risk_cfg.get(
            "opposite_signal_confidence", 85.0
        )

        # Journaling setup
        self.journal_path = os.path.join("logs", "trade_journal.csv")
        self.ml_dataset_path = os.path.join("logs", "ml_dataset.jsonl")
        self.trade_lifecycle_path = os.path.join("logs", "trade_lifecycle.txt")
        self._init_journal()
        self._init_trade_lifecycle_log()

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
        signal_direction = "long" if signal.signal == "BUY" else "short"
        open_pos = self.open_positions.get(sym)
        is_symbol_reversal = False

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

        # 3. Position-aware gate for same symbol (supports explicit reversal checks)
        if open_pos is not None:
            if signal_direction == open_pos.direction:
                sizing.rejection_reason = (
                    f"Same-direction signal rejected while {sym} position is open"
                )
                log.info(f"Signal rejected (same-direction follow-up): {sym}")
                return False, sizing

            if signal.confidence < self.opposite_signal_confidence:
                sizing.rejection_reason = (
                    "Opposite signal rejected: confidence "
                    f"{signal.confidence:.1f} < {self.opposite_signal_confidence:.1f}"
                )
                log.info(f"Signal rejected (reversal confidence): {sym}")
                return False, sizing

            is_symbol_reversal = True

        # 4. Max open trades (allow same-symbol reversal to continue through checks)
        if len(self.open_positions) >= self.max_open_trades and not is_symbol_reversal:
            sizing.rejection_reason = (
                f"Max open trades reached ({self.max_open_trades})"
            )
            log.info(f"Signal rejected (max trades): {sym}")
            return False, sizing

        # 5. Correlation check (avoid multiple positions in highly correlated pairs)
        if not is_symbol_reversal and self._is_highly_correlated_open(sym):
            sizing.rejection_reason = (
                f"Highly correlated pair already has an open position"
            )
            log.info(f"Signal rejected (correlation): {sym}")
            return False, sizing

        # 6. Session Filter
        if self.session_filter_enabled and not self._is_in_trading_session():
            sizing.rejection_reason = "Outside of volatile trading sessions (London/NY)"
            log.info(f"Signal rejected (session filter): {sym}")
            return False, sizing

        # 7. High-impact news event check
        if self.news_filter.is_trading_suspended():
            sizing.rejection_reason = "High-impact news event — trading suspended"
            log.info(f"Signal rejected (news event): {sym}")
            return False, sizing

        # 8. ML Signature Check (The Gatekeeper)
        if not self.ml_validator.is_approved(sizing.ml_snapshot):
            sizing.rejection_reason = (
                "ML Rejection: Setup history shows low win probability"
            )
            log.info(f"Signal rejected (ML validation): {sym}")
            return False, sizing
        if not signal.is_actionable(min_confidence=0.0, min_rr=self.min_rr_ratio):
            sizing.rejection_reason = (
                f"Signal not actionable: confidence={signal.confidence:.1f} "
                f"R/R={signal.risk_reward_ratio:.2f}"
            )
            return False, sizing

        # 9. Compute ATR-based levels
        atr = self._get_atr(df)
        direction = signal_direction

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

        # 10. R/R ratio check
        rr = risk_reward_ratio(entry, stop, tp1)
        if rr < self.min_rr_ratio:
            sizing.rejection_reason = (
                f"R/R ratio too low: {rr:.2f} < {self.min_rr_ratio}"
            )
            log.info(f"Signal rejected (R/R={rr:.2f}): {sym}")
            return False, sizing

        # 11. Position sizing
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
            is_reversal=is_symbol_reversal,
            reversal_from=open_pos.direction if open_pos else "",
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
            initial_risk_amount=sizing.risk_amount,
            planned_risk_reward=sizing.risk_reward,
            trade_id=f"{sizing.symbol}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
        )
        self.open_positions[sizing.symbol] = pos
        self.trades_today += 1
        self._log_trade_open(pos)
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
        self._log_trade_close(pos, close_price, pnl, reason)

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
                        "Risk_Amount",
                        "Planned_RR",
                    ]
                )

    def _init_trade_lifecycle_log(self) -> None:
        """Initialize lifecycle log for open/close events."""
        os.makedirs(os.path.dirname(self.trade_lifecycle_path), exist_ok=True)
        if not os.path.exists(self.trade_lifecycle_path):
            with open(self.trade_lifecycle_path, mode="w", encoding="utf-8") as f:
                f.write("# trade lifecycle log (json lines)\n")

    def _append_lifecycle_event(self, payload: Dict[str, Any]) -> None:
        try:
            with open(self.trade_lifecycle_path, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as exc:
            log.error(f"Failed writing trade lifecycle event: {exc}")

    def _log_trade_open(self, pos: OpenPosition) -> None:
        self._append_lifecycle_event(
            {
                "event": "OPEN",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "trade_id": pos.trade_id,
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry_price": float(pos.entry_price),
                "size": float(pos.position_size),
                "stop_loss": float(pos.stop_loss),
                "take_profit_1": float(pos.take_profit_1),
                "take_profit_2": float(pos.take_profit_2),
                "risk_amount": float(pos.initial_risk_amount),
                "planned_rr": float(pos.planned_risk_reward),
            }
        )

    def _log_trade_close(
        self, pos: OpenPosition, close_price: float, pnl: float, reason: str
    ) -> None:
        self._append_lifecycle_event(
            {
                "event": "CLOSE",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "trade_id": pos.trade_id,
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry_price": float(pos.entry_price),
                "close_price": float(close_price),
                "pnl": float(pnl),
                "result": "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN"),
                "reason": reason,
                "risk_amount": float(pos.initial_risk_amount),
            }
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
                        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
                        pos.symbol,
                        pos.direction,
                        f"{pos.entry_price:.5f}",
                        f"{close_price:.5f}",
                        f"{pos.position_size:.4f}",
                        f"{pnl:.2f}",
                        f"{pnl_pct:.2f}%",
                        reason,
                        f"{duration:.1f}",
                        f"{pos.initial_risk_amount:.2f}",
                        f"{pos.planned_risk_reward:.2f}",
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

    def get_previous_week_summary(
        self, timezone_str: str = "America/New_York"
    ) -> Dict[str, Any]:
        """
        Aggregate closed-trade stats for the previous calendar week (Mon-Sun).
        """
        try:
            tz = ZoneInfo(timezone_str)
        except Exception:
            tz = ZoneInfo("America/New_York")

        now = datetime.now(tz)
        start_of_this_week = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start_prev_week = start_of_this_week - timedelta(days=7)
        end_prev_week = start_of_this_week
        week_label = f"{start_prev_week.date()} to {(end_prev_week - timedelta(seconds=1)).date()}"

        if not os.path.exists(self.journal_path):
            return {
                "week_label": week_label,
                "total_orders": 0,
                "wins": 0,
                "losses": 0,
                "total_profit": 0.0,
                "total_risked": 0.0,
                "cumulative_rr": 0.0,
            }

        try:
            df = pd.read_csv(self.journal_path)
        except Exception as exc:
            log.error(f"Failed reading trade journal for weekly summary: {exc}")
            return {
                "week_label": week_label,
                "total_orders": 0,
                "wins": 0,
                "losses": 0,
                "total_profit": 0.0,
                "total_risked": 0.0,
                "cumulative_rr": 0.0,
            }

        if df.empty or "Timestamp" not in df.columns:
            return {
                "week_label": week_label,
                "total_orders": 0,
                "wins": 0,
                "losses": 0,
                "total_profit": 0.0,
                "total_risked": 0.0,
                "cumulative_rr": 0.0,
            }

        timestamps = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
        timestamps_local = timestamps.dt.tz_convert(tz)
        week_df = df[(timestamps_local >= start_prev_week) & (timestamps_local < end_prev_week)].copy()

        if week_df.empty:
            return {
                "week_label": week_label,
                "total_orders": 0,
                "wins": 0,
                "losses": 0,
                "total_profit": 0.0,
                "total_risked": 0.0,
                "cumulative_rr": 0.0,
            }

        pnl = pd.to_numeric(week_df.get("PnL"), errors="coerce").fillna(0.0)
        risk = pd.to_numeric(week_df.get("Risk_Amount"), errors="coerce").fillna(0.0)
        total_profit = float(pnl.sum())
        total_risked = float(risk.sum())

        return {
            "week_label": week_label,
            "total_orders": int(len(week_df)),
            "wins": int((pnl > 0).sum()),
            "losses": int((pnl < 0).sum()),
            "total_profit": total_profit,
            "total_risked": total_risked,
            "cumulative_rr": (total_profit / total_risked) if total_risked > 0 else 0.0,
        }

    def get_week_summary(
        self,
        start_dt: datetime,
        end_dt: datetime,
        timezone_str: str = "America/New_York",
    ) -> Dict[str, Any]:
        """
        Aggregate closed-trade stats for an arbitrary [start_dt, end_dt] period.
        """
        try:
            tz = ZoneInfo(timezone_str)
        except Exception:
            tz = ZoneInfo("America/New_York")

        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=tz)
        else:
            start_dt = start_dt.astimezone(tz)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=tz)
        else:
            end_dt = end_dt.astimezone(tz)

        week_label = f"{start_dt.date()} to {end_dt.date()}"
        default_summary = {
            "week_label": week_label,
            "total_orders": 0,
            "wins": 0,
            "losses": 0,
            "total_profit": 0.0,
            "total_risked": 0.0,
            "cumulative_rr": 0.0,
        }

        if not os.path.exists(self.trade_lifecycle_path):
            return default_summary

        try:
            events: List[Dict[str, Any]] = []
            with open(self.trade_lifecycle_path, mode="r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw or raw.startswith("#"):
                        continue
                    try:
                        evt = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("event") == "CLOSE":
                        events.append(evt)
        except Exception as exc:
            log.error(f"Failed reading lifecycle log for weekly summary: {exc}")
            return default_summary

        if not events:
            return default_summary

        df = pd.DataFrame(events)
        if "timestamp_utc" not in df.columns:
            return default_summary
        timestamps = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        timestamps_local = timestamps.dt.tz_convert(tz)
        period_df = df[
            (timestamps_local >= start_dt) & (timestamps_local <= end_dt)
        ].copy()
        if period_df.empty:
            return default_summary

        pnl = pd.to_numeric(period_df.get("pnl"), errors="coerce").fillna(0.0)
        risk = pd.to_numeric(period_df.get("risk_amount"), errors="coerce").fillna(0.0)
        total_profit = float(pnl.sum())
        total_risked = float(risk.sum())
        return {
            "week_label": week_label,
            "total_orders": int(len(period_df)),
            "wins": int((pnl > 0).sum()),
            "losses": int((pnl < 0).sum()),
            "total_profit": total_profit,
            "total_risked": total_risked,
            "cumulative_rr": (total_profit / total_risked) if total_risked > 0 else 0.0,
        }

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
