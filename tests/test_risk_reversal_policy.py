import os
import sys

import pandas as pd

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from modules.ai_signal_engine import TradeSignal
from modules.risk_manager import OpenPosition, RiskManager


def _base_config():
    return {
        "risk": {
            "max_open_trades": 2,
            "min_rr_ratio": 1.5,
            "opposite_signal_confidence": 85.0,
        },
        "trading": {"session_filter": False},
        "news": {"skip_high_impact": False, "finnhub_enabled": False},
    }


def _df():
    return pd.DataFrame(
        {
            "close": [1.1000, 1.1010, 1.1020],
            "ATR_14": [0.0010, 0.0011, 0.0012],
        }
    )


def _open_long(symbol: str) -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        direction="long",
        entry_price=1.1000,
        position_size=1000,
        stop_loss=1.0950,
        take_profit_1=1.1050,
        take_profit_2=1.1100,
    )


def test_same_direction_signal_rejected_while_open():
    rm = RiskManager(_base_config())
    rm.open_positions["EURUSD"] = _open_long("EURUSD")
    rm.ml_validator.is_approved = lambda _: True
    rm.news_filter.is_trading_suspended = lambda: False

    signal = TradeSignal(
        signal="BUY",
        confidence=98,
        entry_price=1.1020,
        risk_reward_ratio=2.0,
        symbol="EURUSD",
    )

    approved, sizing = rm.evaluate_signal(signal, _df(), "EURUSD")
    assert approved is False
    assert "Same-direction signal rejected" in sizing.rejection_reason


def test_opposite_signal_requires_high_confidence():
    rm = RiskManager(_base_config())
    rm.open_positions["EURUSD"] = _open_long("EURUSD")
    rm.ml_validator.is_approved = lambda _: True
    rm.news_filter.is_trading_suspended = lambda: False

    signal = TradeSignal(
        signal="SELL",
        confidence=80,
        entry_price=1.1020,
        stop_loss=1.1050,
        take_profit_1=1.0970,
        risk_reward_ratio=1.6,
        symbol="EURUSD",
    )

    approved, sizing = rm.evaluate_signal(signal, _df(), "EURUSD")
    assert approved is False
    assert "Opposite signal rejected" in sizing.rejection_reason


def test_reversal_allowed_when_trade_cap_reached_for_same_symbol():
    rm = RiskManager(_base_config())
    rm.open_positions["EURUSD"] = _open_long("EURUSD")
    rm.open_positions["GBPUSD"] = _open_long("GBPUSD")
    rm.ml_validator.is_approved = lambda _: True
    rm.news_filter.is_trading_suspended = lambda: False

    signal = TradeSignal(
        signal="SELL",
        confidence=92,
        entry_price=1.1020,
        stop_loss=1.1050,
        take_profit_1=1.0970,
        risk_reward_ratio=1.7,
        symbol="EURUSD",
    )

    approved, sizing = rm.evaluate_signal(signal, _df(), "EURUSD")
    assert approved is True
    assert sizing.is_reversal is True
    assert sizing.reversal_from == "long"
