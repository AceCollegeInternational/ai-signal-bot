"""
modules/alerting.py
─────────────────────────────────────────────────────────────────────────────
Module 8 — Monitoring & Alerting

Handles real-time notifications via Telegram and provides a live CLI
dashboard using the rich library.

Alerts:
  - New signal generation
  - Order execution (Entry)
  - Stop-loss / Take-profit hits
  - Daily P&L summary
  - Bot errors or connection issues

Dashboard:
  - Current open positions
  - Today's P&L
  - Recent signal history
  - Bot status and uptime
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.logger import get_logger

log = get_logger(__name__)


class AlertingEngine:
    """
    Sends alerts to Telegram and logs to a rotating file.

    Args:
        config: Parsed config.yaml dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        alert_cfg = config.get("alerts", {})
        self.telegram_enabled: bool = alert_cfg.get("telegram_enabled", False)
        self.bot_token: str = alert_cfg.get("bot_token", "") or os.getenv(
            "TELEGRAM_BOT_TOKEN", ""
        )
        self.chat_id: str = alert_cfg.get("chat_id", "") or os.getenv(
            "TELEGRAM_CHAT_ID", ""
        )

        if self.telegram_enabled and (not self.bot_token or not self.chat_id):
            log.warning("Telegram alerting enabled but token or chat_id is missing.")
            self.telegram_enabled = False

        log.info(
            f"AlertingEngine initialised — Telegram enabled: {self.telegram_enabled}"
        )

    def send_message(self, message: str, silent: bool = False) -> bool:
        """
        Send a raw text message to the configured Telegram chat.

        Args:
            message: Text to send (supports basic Markdown).
            silent:  If True, send without notification sound.

        Returns:
            True if successful, False otherwise.
        """
        if not self.telegram_enabled:
            log.debug(f"Alert (Local-only): {message}")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_notification": silent,
        }

        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return True
            else:
                log.error(f"Telegram error {resp.status_code}: {resp.text}")
                return False
        except Exception as exc:
            log.error(f"Failed to send Telegram alert: {exc}")
            return False

    def notify_signal(self, signal: Any) -> None:
        """Alert on new signal generation."""
        msg = self.format_signal_message(signal)
        self.send_message(msg)

    def format_signal_message(self, signal: Any) -> str:
        """Build Telegram-ready signal text."""
        icon = "🚀" if signal.signal == "BUY" else "🔻"

        # Hold duration display
        hold_icons = {"SCALP": "⚡", "INTRADAY": "☀️", "SWING": "🌊"}
        hold_icon = hold_icons.get(getattr(signal, "hold_duration", ""), "⏱")
        hold_duration = getattr(signal, "hold_duration", "N/A") or "N/A"
        hold_reasoning = getattr(signal, "hold_reasoning", "") or ""

        msg = (
            f"{icon} *NEW SIGNAL: {signal.symbol}* {icon}\n"
            f"Direction: `{signal.signal}`\n"
            f"Confidence: `{signal.confidence}%`\n"
            f"Entry: `{signal.entry_price:.4f}`\n"
            f"SL: `{signal.stop_loss:.4f}`\n"
            f"TP1: `{signal.take_profit_1:.4f}`\n"
            f"TP2: `{signal.take_profit_2:.4f}`\n"
            f"R/R: `{signal.risk_reward_ratio:.2f}`\n"
        )
        fa_cfg = self.config.get("fundamental_analysis", {})
        if fa_cfg.get("enabled", False):
            fundamental_context = getattr(signal, "fundamental_context", None)
            if fundamental_context is not None:
                rating_map = {
                    1: ("🟢", "SUPPORTS"),
                    0: ("⚪", "NEUTRAL"),
                    -1: ("🔴", "OPPOSES"),
                }
                conviction_map = {
                    "strong": "★★★ Strong",
                    "moderate": "★★☆ Moderate",
                    "weak": "★☆☆ Weak",
                }
                emoji, label = rating_map.get(
                    int(fundamental_context.fundamental_rating), ("⚪", "NEUTRAL")
                )
                conviction_label = conviction_map.get(
                    str(fundamental_context.fundamental_conviction), "★☆☆ Weak"
                )
                msg += (
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "📊 FUNDAMENTAL ANALYSIS\n"
                    f"Rating  : {emoji} {label} ({int(fundamental_context.fundamental_rating):+d})\n"
                    f"Conviction: {conviction_label}\n"
                    f"Note    : {fundamental_context.fundamental_note}\n"
                )
                if getattr(fundamental_context, "caution_flag", False):
                    lookahead_hours = int(fa_cfg.get("lookahead_hours", 4))
                    msg += (
                        f"⚠️ High-impact event within {lookahead_hours}h — consider reducing size.\n"
                    )
                msg += "━━━━━━━━━━━━━━━━━━━━\n"

        msg += f"\n{hold_icon} *Hold:* `{hold_duration}`\n"
        if hold_reasoning:
            msg += f"_{hold_reasoning}_\n\n"
        msg += f"*Reasoning:* {signal.reasoning}"
        return msg

    def notify_order_filled(self, order: Any) -> None:
        """Alert on order execution."""
        msg = (
            f"✅ *ORDER FILLED ({order.symbol})*\n"
            f"ID: `{order.order_id}`\n"
            f"Side: `{order.side.upper()}`\n"
            f"Price: `{order.price:.4f}`\n"
            f"Amount: `{order.amount:.6f}`\n"
            f"Mode: `{'Paper' if order.is_paper else 'Live'}`"
        )
        self.send_message(msg)

    def notify_position_closed(
        self, symbol: str, pnl: float, reason: str, price: float
    ) -> None:
        """Alert on position closure."""
        icon = "💰" if pnl >= 0 else "❌"
        msg = (
            f"{icon} *POSITION CLOSED ({symbol})* {icon}\n"
            f"Reason: `{reason}`\n"
            f"Exit Price: `{price:.4f}`\n"
            f"P&L: *{pnl:+.2f} USDT*"
        )
        self.send_message(msg)

    def notify_error(self, error_msg: str) -> None:
        """Alert on critical bot errors."""
        msg = f"⚠️ *BOT ERROR* ⚠️\n`{error_msg}`"
        self.send_message(msg)

    def send_daily_summary(self, summary: Dict[str, Any]) -> None:
        """Send daily P&L and activity summary."""
        pnl = summary.get("daily_pnl", 0.0)
        pnl_pct = summary.get("daily_pnl_pct", 0.0)
        icon = "📈" if pnl >= 0 else "📉"
        msg = (
            f"{icon} *DAILY SUMMARY* {icon}\n"
            f"Total P&L: *{pnl:+.2f} USDT* (`{pnl_pct:+.2f}%`)\n"
            f"Open Positions: `{summary.get('open_positions', 0)}`\n"
            f"Account Balance: `{summary.get('account_balance', 0.0):.2f} USDT`"
        )
        self.send_message(msg)


class Dashboard:
    """
    Rich-based CLI dashboard for real-time bot monitoring.
    """

    def __init__(self, bot_name: str = "AI Trading Bot") -> None:
        self.bot_name = bot_name
        self.console = Console()
        self.layout = Layout()
        self.start_time = datetime.now(timezone.utc)
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Define the dashboard grid structure."""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )
        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )
        self.layout["left"].split_column(
            Layout(name="positions"),
            Layout(name="signals"),
        )

    def _get_header(self, status: str = "RUNNING") -> Panel:
        """Generate the top header panel."""
        uptime = str(datetime.now(timezone.utc) - self.start_time).split(".")[0]
        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(justify="center")
        grid.add_column(justify="right")
        grid.add_row(
            f"[bold cyan]{self.bot_name}[/]",
            f"Status: [{'bold green' if status == 'RUNNING' else 'bold yellow'}]{status}[/]",
            f"Uptime: [bold white]{uptime}[/]",
        )
        return Panel(grid, style="white on blue")

    def _get_positions_table(self, positions: List[Dict[str, Any]]) -> Panel:
        """Generate the active positions table."""
        table = Table(box=None, expand=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Side", justify="center")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L (%)", justify="right")

        for pos in positions:
            pnl_pct = pos.get("pnl_pct", 0.0)
            pnl_style = "green" if pnl_pct >= 0 else "red"
            side_style = "bold green" if pos["direction"] == "long" else "bold red"

            table.add_row(
                pos["symbol"],
                Text(pos["direction"].upper(), style=side_style),
                f"{pos['entry_price']:.4f}",
                f"{pos.get('current_price', 0.0):.4f}",
                Text(f"{pnl_pct:+.2f}%", style=pnl_style),
            )

        return Panel(
            table, title="[bold white]Active Positions[/]", border_style="cyan"
        )

    def _get_signals_table(self, signals: List[Dict[str, Any]]) -> Panel:
        """Generate the recent signals table."""
        table = Table(box=None, expand=True)
        table.add_column("Time", style="dim")
        table.add_column("Symbol")
        table.add_column("Action")
        table.add_column("Conf.", justify="right")
        table.add_column("R/R", justify="right")

        for sig in signals[-5:]:  # Last 5
            style = (
                "green"
                if sig["signal"] == "BUY"
                else "red" if sig["signal"] == "SELL" else "white"
            )
            table.add_row(
                sig.get("time", ""),
                sig["symbol"],
                Text(sig["signal"], style=style),
                f"{sig['confidence']}%",
                f"{sig['risk_reward_ratio']:.2f}",
            )

        return Panel(
            table, title="[bold white]Recent Signals[/]", border_style="magenta"
        )

    def _get_stats_panel(self, stats: Dict[str, Any]) -> Panel:
        """Generate the right-side stats panel."""
        pnl = stats.get("daily_pnl", 0.0)
        pnl_style = "bold green" if pnl >= 0 else "bold red"

        summary = Text.assemble(
            ("Daily P&L: ", "white"),
            (f"{pnl:+.2f} USDT", pnl_style),
            ("\nBalance:   ", "white"),
            (f"{stats.get('account_balance', 0.0):.2f}", "cyan"),
            ("\nTrades:    ", "white"),
            (f"{stats.get('trades_today', 0)}", "yellow"),
            ("\nMode:      ", "white"),
            (f"{stats.get('mode', 'PAPER').upper()}", "bold magenta"),
        )
        return Panel(summary, title="[bold white]Daily Stats[/]", border_style="green")

    async def run_live(self, bot_instance: Any) -> None:
        """
        Start the live-refreshing dashboard.

        Args:
            bot_instance: The main Bot instance to pull data from.
        """
        with Live(self.layout, refresh_per_second=1, screen=True):
            while True:
                # Update header
                status = (
                    "PAUSED"
                    if getattr(bot_instance.risk_manager, "bot_paused", False)
                    else "RUNNING"
                )
                self.layout["header"].update(self._get_header(status))

                # Update positions
                pos_data = []
                for sym, pos in bot_instance.risk_manager.open_positions.items():
                    curr_price = bot_instance.data_engine.get_live_price(sym)
                    pnl_pct = (
                        (
                            pos.unrealised_pnl(curr_price)
                            / (pos.entry_price * pos.position_size)
                        )
                        * 100
                        if pos.entry_price > 0
                        else 0
                    )
                    pos_data.append(
                        {
                            "symbol": sym,
                            "direction": pos.direction,
                            "entry_price": pos.entry_price,
                            "current_price": curr_price,
                            "pnl_pct": pnl_pct,
                        }
                    )
                self.layout["positions"].update(self._get_positions_table(pos_data))

                # Update signals
                self.layout["signals"].update(
                    self._get_signals_table(bot_instance.signal_history)
                )

                # Update stats
                rm_summary = bot_instance.risk_manager.get_portfolio_summary()
                stats = {
                    "daily_pnl": rm_summary["daily_pnl"],
                    "account_balance": rm_summary["account_balance"],
                    "trades_today": len(bot_instance.execution_engine.order_history),
                    "mode": bot_instance.execution_engine.mode,
                }
                self.layout["right"].update(self._get_stats_panel(stats))

                await asyncio.sleep(1)
