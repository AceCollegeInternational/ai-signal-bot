"""
main.py
───────
Production-ready AI-powered chart pattern trading bot.

This is the main entry point that initializes all modules:
  - DataEngine (OHLCV fetching)
  - IndicatorEngine (50+ indicators)
  - PatternDetector (Chart + Candlestick patterns)
  - AISignalEngine (Claude AI confirmation)
  - RiskManager (Position sizing & risk controls)
  - ExecutionEngine (Order placement)
  - AlertingEngine (Telegram + CLI Dashboard)

Usage:
  python main.py --mode paper --symbol BTC/USDT
  python main.py --backtest --symbol BTC/USDT --timeframe 1h
"""

import argparse
import asyncio
from collections import deque
import signal
import os
import sys
import time
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from analysis.fundamental_analyst import FundamentalAnalyst
from modules.ai_signal_engine import AISignalEngine
from modules.alerting import AlertingEngine, Dashboard
from modules.backtester import Backtester
from modules.data_engine import DataEngine
from modules.execution_engine import ExecutionEngine
from modules.market_data_store import market_data_store
from modules.trade_monitor import TradeMonitor
from modules.indicator_engine import IndicatorEngine
from modules.pattern_detector import PatternDetector
from modules.risk_manager import RiskManager
from utils.helpers import load_config
from utils.logger import get_logger, configure_from_config
from utils.market_hours import is_market_closed

# Load environment variables from .env
load_dotenv()

log = get_logger("trading_bot")


def start_health_server(port: int) -> None:
    """
    Start the full execution bridge API server on the Render-assigned PORT.

    This exposes real `/poll/{symbol}`, `/analysis/{symbol}`, `/signals/current`,
    and `/health` endpoints required by the Expert Advisor integration.
    """
    from modules.execution_server import app
    import uvicorn

    config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    log.info(f"Execution bridge server started on 0.0.0.0:{port}")


class TradingBot:
    """
    The orchestrator that runs the main trading logic loop.
    """

    def __init__(self, config: Dict[str, Any], args: argparse.Namespace) -> None:
        self.config = config
        self.args = args

        # Override config mode if provided in args
        if args.mode:
            self.config["trading"]["mode"] = args.mode

        # Initialize Modules
        exchange_name = self.config["trading"].get("exchange", "binance")
        if exchange_name == "mt5":
            try:
                from modules.mt5_connector import MT5Connector

                self.data_engine = MT5Connector(self.config)
                self.execution_engine = self.data_engine
            except ModuleNotFoundError as exc:
                fallback_exchange = self.config["trading"].get("fallback_exchange", "twelvedata")
                fallback_symbols = self.config["trading"].get("fallback_symbols")
                log.warning(
                    "MetaTrader5 is not available in this environment. "
                    f"Falling back to '{fallback_exchange}'. Original error: {exc}"
                )
                self.config["trading"]["exchange"] = fallback_exchange
                if fallback_symbols:
                    self.config["trading"]["symbols"] = fallback_symbols
                self.data_engine = DataEngine(self.config)
                self.execution_engine = ExecutionEngine(
                    self.config, exchange=getattr(self.data_engine, "exchange", None)
                )
        else:
            self.data_engine = DataEngine(self.config)
            self.execution_engine = ExecutionEngine(
                self.config, exchange=getattr(self.data_engine, "exchange", None)
            )

        self.indicator_engine = IndicatorEngine(self.config)
        self.pattern_detector = PatternDetector(self.config)
        self.ai_signal_engine = AISignalEngine(self.config)
        self.fundamental_analyst = FundamentalAnalyst(self.config)
        self.fundamental_enabled = bool(
            self.config.get("fundamental_analysis", {}).get("enabled", False)
        )
        self.risk_manager = RiskManager(self.config)
        self.trade_monitor = TradeMonitor(config)
        self.alerting_engine = AlertingEngine(config)
        self.dashboard = Dashboard()
        self.bridge_cfg = self.config.get("execution_bridge", {})
        self.bridge_enabled = self.bridge_cfg.get("enabled", False)
        self.bridge_url = self.bridge_cfg.get("url", "http://localhost:8000")
        scan_batch_cfg = self.config.get("trading", {}).get("scan_batch", {})
        self.batch_interval_seconds = int(
            scan_batch_cfg.get(
                "interval_seconds", self.config.get("trading", {}).get("scan_interval", 60)
            )
        )
        self.symbols_per_batch = int(
            scan_batch_cfg.get("symbols_per_batch", len(self.config.get("trading", {}).get("symbols", [])) or 1)
        )
        self.max_calls_per_minute = int(scan_batch_cfg.get("max_calls_per_minute", 8))
        self.estimated_calls_per_symbol = max(
            1, int(scan_batch_cfg.get("estimated_calls_per_symbol", 3))
        )
        self._batch_cursor = 0
        self._td_call_timestamps: deque[float] = deque()
        self._rate_guard_enabled = (
            self.config.get("trading", {}).get("exchange", "").lower() == "twelvedata"
            and self.max_calls_per_minute > 0
        )
        ea_data_cfg = self.config.get("ea_data", {})
        self.ea_data_enabled = ea_data_cfg.get("enabled", True)
        self.ea_data_stale_after_seconds = int(ea_data_cfg.get("stale_after_seconds", 600))
        self.ea_data_fallback_to_api = ea_data_cfg.get("fallback_to_api", True)

        # Historical tracking for dashboard
        self.signal_history: List[Dict[str, Any]] = []
        self.last_bar_time: Dict[str, datetime] = {}
        self.last_signal_time: Dict[str, datetime] = {}  # Per-symbol cooldown
        self._market_closed_notified = False  # Avoid spamming "market closed"
        self.running = False

    async def start(self) -> None:
        """Start the main bot loop."""
        self.running = True
        log.info(f"Bot starting... Mode: {self.execution_engine.mode.upper()}")

        # Send startup alert
        self.alerting_engine.send_message(
            f"🚀 *AI Trading Bot Started*\nMode: `{self.execution_engine.mode.upper()}`\nSymbols: `{self.config['trading']['symbols']}`",
            silent=True,
        )

        # Run Dashboard in parallel if not in backtest mode
        dashboard_task = asyncio.create_task(self.dashboard.run_live(self))

        try:
            while self.running:
                # ── Market Hours Guard ─────────────────────────────
                mh_cfg = self.config.get("market_hours", {})
                if mh_cfg.get("enforce_forex_close", True):
                    tz_str = mh_cfg.get("timezone", "America/New_York")
                    closed, msg = is_market_closed(tz_str)
                    if closed:
                        if not self._market_closed_notified:
                            log.info(msg)
                            self.alerting_engine.send_message(
                                f"🌙 *Market Closed*\n{msg}\n\nBot is paused until market reopens.",
                                silent=True,
                            )
                            self._market_closed_notified = True
                        await asyncio.sleep(300)  # Check again in 5 minutes
                        continue
                    else:
                        if self._market_closed_notified:
                            log.info("Forex market is now OPEN. Resuming scanning.")
                            self.alerting_engine.send_message(
                                "☀️ *Market Open*\nForex market has reopened. Bot is resuming.",
                                silent=True,
                            )
                            self._market_closed_notified = False

                symbols_this_cycle = self._next_symbols_batch()
                if not symbols_this_cycle:
                    log.warning("No symbols scheduled for this cycle. Sleeping until next interval.")
                for symbol in symbols_this_cycle:
                    try:
                        await self.process_symbol(symbol)
                    except Exception as exc:
                        log.error(f"Error processing {symbol}: {exc}", exc_info=True)

                # Send heartbeat to execution server after each scan cycle
                if self.bridge_enabled:
                    try:
                        self._send_heartbeat()
                    except Exception as e:
                        log.debug(f"Heartbeat send failed (server may not be running): {e}")

                # Wait for next scan interval
                await asyncio.sleep(self.batch_interval_seconds)

        except asyncio.CancelledError:
            self.running = False
        finally:
            log.info("Bot shutting down...")
            dashboard_task.cancel()

    async def process_symbol(self, symbol: str) -> None:
        """Process a single symbol: Data -> Indicators -> Patterns -> Signal -> Execute."""
        timeframe = self.config["trading"]["timeframe"]
        htf = self.config["trading"].get("higher_timeframe", "4h")

        # 1. Fetch Data (prefers EA push store, optional Twelve Data fallback)
        df, df_htf = self._get_symbol_data(symbol, timeframe, htf)
        if df.empty:
            return

        htf_context = {}
        if not df_htf.empty:
            df_htf = self.indicator_engine.compute_all(df_htf)
            last_htf = df_htf.iloc[-1]
            htf_context = {
                "timeframe": htf,
                "trend": (
                    "bullish" if last_htf.get("is_above_ema50", False) else "bearish"
                ),
                "rsi": last_htf.get("RSI_14", 50),
                "adx": last_htf.get("ADX_14", 0),
            }

        # 3. Compute Indicators (Trading Timeframe)
        df = self.indicator_engine.compute_all(df)

        # 4. Detect Patterns + S/R
        patterns, sr_levels = self.pattern_detector.detect_all(df)
        slope = self.pattern_detector.calculate_slope(df)

        if htf_context:
            htf_context["slope"] = slope

        # 5. Check for active position
        if symbol in self.risk_manager.open_positions:
            await self.manage_open_position(symbol, df)
            return

        # 5.5 Bar Close Logic (Prevents Scalping)
        last_ts = df.index[-1]
        if self.config["trading"].get("wait_for_bar_close", True):
            if symbol in self.last_bar_time and last_ts <= self.last_bar_time[symbol]:
                # Still in the same candle, skip analysis effectively throttling the bot
                return
            self.last_bar_time[symbol] = last_ts

        # 5.6 Signal Cooldown (Prevents excessive signals)
        cooldown_min = self.config.get("signals", {}).get("signal_cooldown_minutes", 15)
        if symbol in self.last_signal_time:
            elapsed = (datetime.now(timezone.utc) - self.last_signal_time[symbol]).total_seconds() / 60
            if elapsed < cooldown_min:
                log.debug(f"Signal cooldown active for {symbol}: {cooldown_min - elapsed:.1f} min remaining")
                return

        # 6. Generate AI Signal (with MTA Context)
        signal = self.ai_signal_engine.analyze(
            df, patterns, sr_levels, symbol, timeframe, htf_context=htf_context
        )
        if self.fundamental_enabled:
            signal_direction = signal.signal if signal.signal in ("BUY", "SELL") else "BUY"
            fundamental_context = self.fundamental_analyst.analyse_live(
                symbol=symbol,
                signal_direction=signal_direction,
            )
            setattr(signal, "fundamental_context", fundamental_context)

        # Track signal history
        self.signal_history.append(
            {
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "symbol": symbol,
                "signal": signal.signal,
                "confidence": signal.confidence,
                "risk_reward_ratio": signal.risk_reward_ratio,
            }
        )

        # Sync all signals to Execution Server for MT5 Dashboard
        try:
            self._sync_signal_to_server(signal)
        except Exception as e:
            log.warning(f"Failed to sync signal to server: {e}")

        if signal.signal == "HOLD":
            return

        # Record signal time for cooldown
        self.last_signal_time[symbol] = datetime.now(timezone.utc)

        # Notify on potentially actionable signal (high confidence)
        if signal.confidence >= 50:
            self.alerting_engine.notify_signal(signal)

        # 6. Risk Management Evaluation
        approved, sizing = self.risk_manager.evaluate_signal(signal, df, symbol)

        if not approved:
            log.info(
                f"Signal rejected by RiskManager for {symbol}: {sizing.rejection_reason}"
            )
            return

        # 7. Execution
        current_price = self.data_engine.get_live_price(symbol)
        order = self.execution_engine.place_order(sizing, current_price)

        if order.status == "filled":
            self.risk_manager.open_position(sizing)
            self.alerting_engine.notify_order_filled(order)

    def _get_symbol_data(self, symbol: str, timeframe: str, higher_timeframe: str) -> Any:
        """Return (df, df_htf) from EA push store when fresh; fallback to API if enabled."""
        if self.ea_data_enabled:
            df = market_data_store.get_df(
                symbol=symbol,
                timeframe=timeframe,
                max_age_seconds=self.ea_data_stale_after_seconds,
                with_indicators=True,
            )
            df_htf = market_data_store.get_df(
                symbol=symbol,
                timeframe=higher_timeframe,
                max_age_seconds=self.ea_data_stale_after_seconds,
                with_indicators=True,
            )
            if df is not None and df_htf is not None and not df.empty and not df_htf.empty:
                return df.copy(), df_htf.copy()

            freshness = market_data_store.freshness_report().get(symbol.upper(), {})
            missing_tfs = [tf for tf in (timeframe, higher_timeframe) if tf not in freshness]
            stale_tfs: List[str] = []
            fresh_tfs: List[str] = []
            for tf in (timeframe, higher_timeframe):
                tf_state = freshness.get(tf)
                if not tf_state:
                    continue
                if tf_state.get("age_seconds", self.ea_data_stale_after_seconds + 1) > self.ea_data_stale_after_seconds:
                    stale_tfs.append(f"{tf}:{tf_state.get('age_seconds')}s")
                else:
                    fresh_tfs.append(f"{tf}:{tf_state.get('age_seconds')}s")

            if not self.ea_data_fallback_to_api:
                log.info(
                    "EA data unavailable for %s (missing_tfs=%s stale_tfs=%s fresh_tfs=%s, max_age=%ss), and fallback_to_api is disabled.",
                    symbol,
                    missing_tfs,
                    stale_tfs,
                    fresh_tfs,
                    self.ea_data_stale_after_seconds,
                )
                return self._empty_df(), self._empty_df()

            log.info(
                "EA data unavailable for %s (missing_tfs=%s stale_tfs=%s fresh_tfs=%s, max_age=%ss). Falling back to API data fetch.",
                symbol,
                missing_tfs,
                stale_tfs,
                fresh_tfs,
                self.ea_data_stale_after_seconds,
            )

        df = self.data_engine.fetch_ohlcv(symbol, timeframe, limit=100)
        df_htf = self.data_engine.fetch_ohlcv(symbol, higher_timeframe, limit=100)
        return df, df_htf

    @staticmethod
    def _empty_df() -> Any:
        import pandas as pd

        return pd.DataFrame()

    def _prune_call_timestamps(self) -> None:
        """Drop Twelve Data call estimates older than 60 seconds."""
        now = time.time()
        while self._td_call_timestamps and now - self._td_call_timestamps[0] > 60:
            self._td_call_timestamps.popleft()

    def _reserve_call_budget(self, call_count: int) -> None:
        """Reserve estimated Twelve Data calls for this cycle."""
        now = time.time()
        for _ in range(call_count):
            self._td_call_timestamps.append(now)

    def _next_symbols_batch(self) -> List[str]:
        """Return the next symbol batch in round-robin order."""
        symbols = list(self.config.get("trading", {}).get("symbols", []))
        if not symbols:
            return []

        total_symbols = len(symbols)
        batch_size = max(1, min(self.symbols_per_batch, total_symbols))
        start = self._batch_cursor
        selected: List[str] = []
        for offset in range(batch_size):
            selected.append(symbols[(start + offset) % total_symbols])

        if self._rate_guard_enabled:
            self._prune_call_timestamps()
            calls_used_last_min = len(self._td_call_timestamps)
            calls_remaining = max(0, self.max_calls_per_minute - calls_used_last_min)
            max_symbols_allowed = calls_remaining // self.estimated_calls_per_symbol
            if max_symbols_allowed <= 0:
                log.warning(
                    "Twelve Data call budget exhausted: used=%s, limit=%s. Deferring this cycle.",
                    calls_used_last_min,
                    self.max_calls_per_minute,
                )
                return []
            if max_symbols_allowed < len(selected):
                selected = selected[:max_symbols_allowed]

            estimated_calls = len(selected) * self.estimated_calls_per_symbol
            self._reserve_call_budget(estimated_calls)
            log.info(
                "Batch scan (rate-guarded): symbols=%s/%s selected=%s used_calls=%s remaining_calls=%s est_calls=%s",
                len(selected),
                total_symbols,
                selected,
                calls_used_last_min,
                calls_remaining,
                estimated_calls,
            )
        else:
            log.info(
                "Batch scan: symbols=%s/%s selected=%s",
                len(selected),
                total_symbols,
                selected,
            )

        self._batch_cursor = (start + len(selected)) % total_symbols
        return selected

    async def manage_open_position(self, symbol: str, df: Any) -> None:
        """Monitor and exit open positions."""
        pos = self.risk_manager.open_positions[symbol]
        curr_price = self.data_engine.get_live_price(symbol)
        if curr_price <= 0 and df is not None and not df.empty:
            curr_price = float(df["close"].iloc[-1])
            log.warning(
                "Live price for %s is invalid (<=0). Using last candle close fallback: %.5f",
                symbol,
                curr_price,
            )
        if curr_price <= 0:
            log.warning(
                "Skipping position management for %s due to invalid current price: %s",
                symbol,
                curr_price,
            )
            return
        atr = self.risk_manager._get_atr(df)

        # Update trailing stop
        self.risk_manager.update_trailing_stop(symbol, curr_price, atr)

        # Check exit triggers (SL, TP1, TP2)
        exit_trigger = self.risk_manager.check_exit_conditions(symbol, curr_price)

        # Intelligent Monitoring (Thesis Re-validation & Momentum Stalling)
        if not exit_trigger:
            if not self.trade_monitor.revalidate_thesis(pos, df):
                exit_trigger = "Thesis Broken"
            elif self.trade_monitor.check_momentum_stall(pos, df):
                exit_trigger = "Momentum Stall"

        if exit_trigger:
            reason = f"{exit_trigger} hit"
            order = self.execution_engine.close_position(
                symbol, pos.position_size, curr_price, pos.direction, reason=reason
            )

            if order.status == "filled":
                pnl = self.risk_manager.close_position(
                    symbol, order.price, reason=reason
                )
                self.alerting_engine.notify_position_closed(
                    symbol, pnl, reason, order.price
                )

    def _sync_signal_to_server(self, signal: Any) -> None:
        """Helper to push the latest AI analysis to the execution server."""
        if not self.bridge_enabled:
            return
        import requests
        
        url = f"{self._bridge_base_url()}/signals"
        api_key = os.getenv("EXECUTION_BRIDGE_KEY", "default_secret_key")
        
        payload = {
            "symbol": signal.symbol,
            "direction": signal.signal,
            "entry_price": float(signal.entry_price),
            "stop_loss": float(signal.stop_loss),
            "take_profit": float(signal.take_profit_1),
            "confidence": float(signal.confidence),
            "reasoning": signal.reasoning,
            "timestamp": time.time()
        }
        if self.fundamental_enabled:
            ctx = getattr(signal, "fundamental_context", None)
            if ctx is not None:
                payload.update(
                    {
                        "fundamental_rating": int(ctx.fundamental_rating),
                        "fundamental_conviction": ctx.fundamental_conviction,
                        "fundamental_note": ctx.fundamental_note,
                    }
                )
        
        headers = {"X-API-KEY": api_key}
        requests.post(url, json=payload, headers=headers, timeout=5)

    def _send_heartbeat(self) -> None:
        """Send a status heartbeat to the execution server after each scan cycle."""
        if not self.bridge_enabled:
            return
        import requests

        url = f"{self._bridge_base_url()}/bot/heartbeat"
        api_key = os.getenv("EXECUTION_BRIDGE_KEY", "default_secret_key")

        timeframes = [
            self.config["trading"].get("timeframe", "1h"),
            self.config["trading"].get("higher_timeframe", "4h"),
        ]
        # Include backtesting timeframes if configured (shows all analyzed TFs)
        bt_tfs = self.config.get("backtesting", {}).get("test_timeframes", [])
        for tf in bt_tfs:
            if tf not in timeframes:
                timeframes.append(tf)

        payload = {
            "last_scan_time": datetime.now(timezone.utc).isoformat(),
            "symbols_scanned": self.config["trading"]["symbols"],
            "scan_interval": self.config["trading"].get("scan_interval", 60),
            "timeframes_analyzed": timeframes,
            "mode": self.config["trading"].get("mode", "paper"),
            "total_signals_generated": len(self.signal_history),
        }

        headers = {"X-API-KEY": api_key}
        requests.post(url, json=payload, headers=headers, timeout=5)

    def _bridge_base_url(self) -> str:
        """Normalize bridge URL for loopback calls inside the same process."""
        base = self.bridge_url.rstrip("/")
        return base.replace("://0.0.0.0", "://127.0.0.1")

    async def run_backtest(self) -> None:
        """Run backtesting mode for multiple timeframes and exit."""
        bt = Backtester(self.config, self.data_engine)
        
        # Use config timeframes if none provided in args
        timeframes = [self.args.timeframe] if self.args.timeframe else self.config.get("backtesting", {}).get("test_timeframes", ["1h"])
        
        for symbol in self.config["trading"]["symbols"]:
            for tf in timeframes:
                log.info(f"Starting backtest for {symbol} on {tf}...")
                result = bt.run(symbol, tf)
                report_path = bt.generate_html_report(result)
                print(f"[{tf}] Backtest Report for {symbol}: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument(
        "--mode", choices=["paper", "live", "bridge"], help="Override trading mode"
    )
    parser.add_argument("--symbol", help="Single symbol to trade")
    parser.add_argument("--backtest", action="store_true", help="Run backtest mode")
    parser.add_argument("--timeframe", help="Override timeframe")
    args = parser.parse_args()

    # Render web services expect a bound port; keep a tiny health endpoint open.
    port = os.getenv("PORT")
    if port and not args.backtest:
        try:
            start_health_server(int(port))
        except Exception as exc:
            log.warning(f"Failed to start health server on PORT={port}: {exc}")

    # Load Config
    config = load_config("config.yaml")

    # Configure Logging
    configure_from_config(config)

    # Render web services expect a bound port; keep a tiny health endpoint open.
    # Do this after logging setup so bind failures are visible in logs.
    port = os.getenv("PORT")
    if port and not args.backtest:
        try:
            start_health_server(int(port))
        except Exception:
            log.exception(f"Failed to start health server on PORT={port}")

    # Initialize Bot
    bot = TradingBot(config, args)

    if args.backtest:
        asyncio.run(bot.run_backtest())
    else:
        try:
            asyncio.run(bot.start())
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
