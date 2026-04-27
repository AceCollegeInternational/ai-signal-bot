"""
modules/mt5_connector.py
────────────────────────
MetaTrader 5 connector that acts as both a DataEngine and ExecutionEngine.
"""

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
import MetaTrader5 as mt5

from modules.execution_engine import OrderResult
from modules.risk_manager import PositionSizing
from utils.logger import get_logger

log = get_logger(__name__)

# Map string timeframes to MT5 timeframe constants
TF_MAP = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
}


class MT5Connector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = config["trading"].get("mode", "paper")
        self.active = False

        self.mt5_cfg = config.get("mt5", {})
        self.magic = self.mt5_cfg.get("magic_number", 234000)
        self.deviation = self.mt5_cfg.get("deviation", 20)
        self.default_lot = float(self.mt5_cfg.get("lot_size", 0.01))

        self.connect()

    def connect(self) -> bool:
        """Initialize connection to MetaTrader 5 terminal."""
        path = os.getenv("MT5_TERMINAL_PATH")
        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")

        log.info(f"Attempting to connect to MT5...")

        if path and login and password and server:
            init_res = mt5.initialize(
                path=path,
                login=int(login),
                password=password,
                server=server,
            )
        else:
            # Try default initialization
            log.warning("MT5 credentials missing in .env. Attempting default connect.")
            init_res = mt5.initialize()

        if not init_res:
            log.error(f"MT5 initialization failed: {mt5.last_error()}")
            self.active = False
            return False

        log.info("MT5 connection successful.")
        self.active = True
        return True

    def shutdown(self):
        """Shutdown MT5 connection."""
        if self.active:
            mt5.shutdown()
            self.active = False

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 500, **kwargs
    ) -> pd.DataFrame:
        """Fetch OHLCV historical data for a symbol."""
        if not self.active and not self.connect():
            return pd.DataFrame()

        mt5_tf = TF_MAP.get(timeframe, mt5.TIMEFRAME_H1)

        # Ensure symbol is visible in Market Watch
        if not mt5.symbol_select(symbol, True):
            log.warning(f"Failed to select {symbol} in MT5 Market Watch.")
            return pd.DataFrame()

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, limit)
        if rates is None or len(rates) == 0:
            log.warning(
                f"No rates found for {symbol} ({timeframe}) - {mt5.last_error()}"
            )
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("datetime", inplace=True)

        # Rename MT5 columns to match crypto format
        df.rename(
            columns={"tick_volume": "volume", "real_volume": "real_vol"}, inplace=True
        )
        return df

    def get_live_price(self, symbol: str) -> float:
        """Fetch the current market price (using bid for simplicity)."""
        if not self.active and not self.connect():
            return 0.0

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return 0.0
        return tick.bid

    def place_order(self, sizing: PositionSizing, current_price: float) -> OrderResult:
        """Place a market order based on PositionSizing."""
        symbol = sizing.symbol
        side = "buy" if sizing.direction == "long" else "sell"

        # Parse amount (lot size). For forex, use MT5 config lot size if sizing.position_size is raw USDT.
        # In a real system, compute lots from risk. Here we use the default lot size from config.
        lot = self.default_lot

        if self.mode == "paper":
            log.info(f"PAPER TRADE: {side.upper()} {lot} {symbol} at {current_price}")
            return OrderResult(
                order_id=f"PAPER-{int(time.time())}",
                symbol=symbol,
                side=side,
                amount=lot,
                price=current_price,
                status="filled",
                is_paper=True,
            )

        if not self.active and not self.connect():
            return OrderResult(symbol=symbol, side=side, status="failed")

        order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
        price = (
            mt5.symbol_info_tick(symbol).ask
            if side == "buy"
            else mt5.symbol_info_tick(symbol).bid
        )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": float(sizing.stop_loss) if sizing.stop_loss else 0.0,
            "tp": float(sizing.take_profit_1) if sizing.take_profit_1 else 0.0,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "ai_trading_bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = mt5.last_error()
            retcode = result.retcode if result else error
            log.error(f"MT5 Order Failed. Retcode: {retcode}, Error: {error}")
            return OrderResult(symbol=symbol, side=side, status="failed")

        return OrderResult(
            order_id=str(result.order),
            symbol=symbol,
            side=side,
            amount=lot,
            price=result.price,
            status="filled",
            is_paper=False,
        )

    def close_position(
        self,
        symbol: str,
        amount: float,
        current_price: float,
        direction: str = "long",
        reason: str = "",
    ) -> OrderResult:
        """Close an open position."""
        if self.mode == "paper":
            side = "sell" if direction == "long" else "buy"
            log.info(
                f"PAPER CLOSE: {side.upper()} {symbol} at {current_price} ({reason})"
            )
            return OrderResult(
                order_id=f"PAPER-{int(time.time())}",
                symbol=symbol,
                side=side,
                amount=amount,
                price=current_price,
                status="filled",
                is_paper=True,
            )

        # Basic live MT5 position close (assumes no hedging)
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return OrderResult(symbol=symbol, side="", status="failed")

        position = positions[0]  # Close first position
        tick = mt5.symbol_info_tick(symbol)

        close_type = (
            mt5.ORDER_TYPE_SELL
            if position.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": f"Close {reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        side = "buy" if close_type == mt5.ORDER_TYPE_BUY else "sell"

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"MT5 Close Failed: {mt5.last_error()}")
            return OrderResult(symbol=symbol, side=side, status="failed")

        return OrderResult(
            order_id=str(result.order),
            symbol=symbol,
            side=side,
            amount=position.volume,
            price=result.price,
            status="filled",
            is_paper=False,
        )
