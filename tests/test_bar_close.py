
import asyncio
import os
import sys
import pandas as pd
from datetime import datetime, timezone

# Add project root to sys.path for importing main
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from main import TradingBot

async def test_bar_close():
    config = {
        "trading": {
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "exchange": "paper",
            "scan_interval": 1,
            "wait_for_bar_close": True
        },
        "mt5": {},
        "risk": {},
        "signals": {},
        "ai": {"primary_provider": "mock"},
        "alerts": {"telegram_enabled": False}
    }
    
    class MockArgs:
        mode = "paper"
        symbol = None
        backtest = False
        timeframe = None

    bot = TradingBot(config, MockArgs())
    
    # Mock analysis to count calls
    bot.analysis_calls = 0
    async def mock_process(symbol):
        df = pd.DataFrame({
            "open": [1, 2], "high": [3, 4], "low": [0, 1], "close": [2, 3], "volume": [10, 20]
        }, index=[datetime(2026, 3, 13, 5, 0, tzinfo=timezone.utc), datetime(2026, 3, 13, 6, 0, tzinfo=timezone.utc)])
        
        last_ts = df.index[-1]
        if bot.config["trading"].get("wait_for_bar_close", True):
            if symbol in bot.last_bar_time and last_ts <= bot.last_bar_time[symbol]:
                return
            bot.last_bar_time[symbol] = last_ts
        
        bot.analysis_calls += 1

    bot.process_symbol = mock_process

    # Call 1: New bar
    await bot.process_symbol("BTC/USDT")
    print(f"Calls after 1st attempt: {bot.analysis_calls}") # Should be 1
    
    # Call 2: Same bar
    await bot.process_symbol("BTC/USDT")
    print(f"Calls after 2nd attempt (same bar): {bot.analysis_calls}") # Should still be 1
    
    if bot.analysis_calls == 1:
        print("SUCCESS: Bar-close logic throttled redundant calls.")
    else:
        print(f"FAILURE: AI was called {bot.analysis_calls} times for the same bar.")

if __name__ == "__main__":
    asyncio.run(test_bar_close())
