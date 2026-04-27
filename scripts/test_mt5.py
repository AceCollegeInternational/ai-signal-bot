"""
scripts/test_mt5.py
───────────────────
Test script to independently verify the MetaTrader 5 connection,
data fetching, and paper trade execution.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import load_config
from modules.mt5_connector import MT5Connector
from modules.risk_manager import PositionSizing


def main():
    parser = argparse.ArgumentParser(description="Test MT5 Connection")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol to test")
    args = parser.parse_args()

    load_dotenv()
    config = load_config("config.yaml")

    print(f"\n🚀 Testing MetaTrader 5 Connector...")
    print(f"Target Symbol: {args.symbol}")

    connector = MT5Connector(config)

    if not connector.active:
        print(
            "❌ Could not connect to MetaTrader 5. Make sure the terminal is running and credentials in .env are correct."
        )
        return

    print("✅ Connected to MetaTrader 5 successfully.")

    print(f"\n--- Testing Data Fetch ({args.symbol}) ---")
    df = connector.fetch_ohlcv(args.symbol, timeframe="1h", limit=5)

    if df.empty:
        print("❌ Data fetch failed or returned empty DataFrame.")
    else:
        print(f"✅ Data fetched successfully! Shape: {df.shape}")
        print(df.tail(2))

    print("\n--- Testing Live Price ---")
    price = connector.get_live_price(args.symbol)
    if price > 0:
        print(f"✅ Live Price (Bid): {price}")
    else:
        print("❌ Failed to get live price.")

    print("\n--- Testing Paper Order ---")

    # Switch to paper mode for safety
    connector.mode = "paper"

    sizing = PositionSizing(
        symbol=args.symbol,
        direction="long",
        position_size=100.0,  # doesn't matter for MT5 paper mock
        entry_price=price,
        stop_loss=price * 0.99,
        take_profit_1=price * 1.01,
        take_profit_2=price * 1.02,
    )

    order = connector.place_order(sizing, price)
    print(f"✅ Paper Order logic check:")
    print(f"Status: {order.status}")
    print(f"ID:     {order.order_id}")
    print(f"Side:   {order.side}")
    print(f"Price:  {order.price}")


if __name__ == "__main__":
    main()
