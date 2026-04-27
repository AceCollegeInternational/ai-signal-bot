"""
scripts/test_ai_providers.py
────────────────────────────
Quick test script to verify AI providers (Gemini, Groq, Claude)
independently of the full trading bot.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.ai_signal_engine import AISignalEngine
from utils.helpers import load_config


def generate_mock_data():
    """Generate dummy OHLCV + indicator data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq="h")
    df = pd.DataFrame(
        {
            "open": np.random.uniform(50000, 60000, 100),
            "high": np.random.uniform(60000, 65000, 100),
            "low": np.random.uniform(45000, 50000, 100),
            "close": np.random.uniform(50000, 60000, 100),
            "volume": np.random.uniform(10, 100, 100),
            "rsi": np.random.uniform(30, 70, 100),
            "ema_20": np.random.uniform(50000, 60000, 100),
        },
        index=dates,
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Test AI Providers")
    parser.add_argument(
        "--provider", choices=["gemini", "groq", "claude"], help="Provider to test"
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config("config.yaml")

    # Force primary provider if specified
    if args.provider:
        config["ai"]["primary_provider"] = args.provider
        config["ai"]["fallback_enabled"] = False

    engine = AISignalEngine(config)

    print(f"\n🚀 Testing AI Signal Engine...")
    print(f"Primary Provider: {engine.primary}")
    print(f"Available Providers: {list(engine.providers.keys())}")

    if not engine.providers:
        print("❌ Error: No providers initialized. Check your .env for API keys.")
        return

    df = generate_mock_data()
    patterns = [{"name": "Bullish Engulfing", "confidence": 0.8}]
    sr_levels = [55000, 58000]

    print(f"--- Sending request to {engine.primary} ---")
    try:
        signal = engine.analyze(
            df, patterns, sr_levels, symbol="BTC/USDT", timeframe="1h"
        )

        print("\n✅ Signal Received:")
        print(f"Provider:   {signal.provider}")
        print(f"Signal:     {signal.signal}")
        print(f"Confidence: {signal.confidence}%")
        print(f"Entry:      {signal.entry_price}")
        print(f"SL:         {signal.stop_loss}")
        print(f"TP1:        {signal.take_profit_1}")
        print(f"Reasoning:  {signal.reasoning[:150]}...")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")


if __name__ == "__main__":
    main()
