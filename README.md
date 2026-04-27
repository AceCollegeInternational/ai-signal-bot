# 🤖 AI-Powered Chart Pattern Trading Bot

A production-ready algorithmic trading bot powered by **Claude AI**, featuring real-time market data, technical indicator analysis, chart pattern detection, automated risk management, and multi-exchange execution.

## Features

- **Live Data**: WebSocket streaming + historical OHLCV from Binance or Bybit
- **50+ Technical Indicators**: EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, and more
- **Pattern Detection**: 12 candlestick patterns + 9 chart patterns + S/R levels
- **AI Signal Engine**: Claude AI confirms every signal with structured JSON response
- **Risk Management**: ATR-based dynamic stops, 1.5% max risk per trade, daily loss limits
- **Paper & Live Trading**: Full paper trading simulation before going live
- **Backtesting**: Vectorbt-based engine with HTML performance reports
- **Telegram Alerts**: Real-time notifications for signals, fills, SL/TP hits
- **Rich CLI Dashboard**: Live updating terminal UI with positions, P&L, and signals

---

## Quick Start

### 1. Clone / Navigate to the Project

```bash
cd "trading bot"
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required keys:
| Key | Source |
|-----|--------|
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | [Binance API Management](https://www.binance.com/en/my/settings/api-management) |
| `TWELVEDATA_API_KEY` | [Twelve Data API Dashboard](https://twelvedata.com/) |
| `ANTHROPIC_API_KEY` | [Anthropic Console](https://console.anthropic.com/) |
| `TELEGRAM_BOT_TOKEN` | Create a bot via [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Get from [@userinfobot](https://t.me/userinfobot) |

### 5. Edit Config

Open `config.yaml` and adjust:
- Trading symbols, timeframe
- Risk parameters
- Mode: `paper` (default) or `live`

### 6. Run the Bot

```bash
# Paper trading (safe default)
python main.py

# With specific symbol override
python main.py --symbol BTC/USDT --timeframe 4h

# Live trading (⚠️ real money)
python main.py --mode live

# Run backtesting only
python main.py --backtest
```

---

## Project Structure

```
trading_bot/
├── main.py                  # Entry point, main bot loop
├── config.yaml              # All configuration settings
├── .env                     # API keys (never commit!)
├── .env.example             # Template for .env
├── requirements.txt         # Pinned dependencies
├── modules/
│   ├── data_engine.py       # OHLCV data fetching & caching
│   ├── indicator_engine.py  # Technical indicators (pandas_ta)
│   ├── pattern_detector.py  # Candlestick & chart pattern detection
│   ├── ai_signal_engine.py  # Claude AI signal generation
│   ├── risk_manager.py      # Position sizing & risk controls
│   ├── execution_engine.py  # Order placement (paper & live)
│   ├── backtester.py        # Historical backtesting engine
│   └── alerting.py          # Telegram alerts & Rich dashboard
├── utils/
│   ├── logger.py            # Rotating file + console logger
│   └── helpers.py           # Shared utility functions
├── logs/                    # Rotating log files
├── data/cache/              # Local OHLCV cache (parquet)
└── backtest_reports/        # Generated HTML reports
```

---

## Modules Overview

| Module | Purpose |
|--------|---------|
| `data_engine.py` | Fetches and caches OHLCV data; WebSocket live feeds |
| `indicator_engine.py` | Computes 50+ technical indicators via pandas_ta |
| `pattern_detector.py` | Detects candlestick + chart patterns + S/R levels |
| `ai_signal_engine.py` | Sends data to Claude AI for high-confidence signals |
| `risk_manager.py` | ATR stops, position sizing, daily loss limits, news filter |
| `execution_engine.py` | Paper/live order execution with retry logic |
| `backtester.py` | Vectorbt backtest with full HTML report generation |
| `alerting.py` | Telegram notifications + Rich terminal dashboard |

---

## Configuration Reference

All settings in `config.yaml`:

```yaml
trading:
  symbols: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
  timeframe: "1h"        # 1m, 5m, 15m, 1h, 4h, 1d
  mode: "paper"          # "paper" | "live"
  exchange: "binance"    # "binance" | "bybit"

risk:
  max_risk_per_trade: 0.015   # 1.5% per trade
  max_open_trades: 3
  max_daily_loss: 0.05        # 5% → bot pauses
  min_rr_ratio: 1.5

signals:
  min_confidence: 75          # Claude AI confidence threshold
  require_confluence: true
  min_confluence_count: 3
```

---

## Safety Warnings

> ⚠️ **Paper trade first**: Always validate with `mode: paper` before going live.

> ⚠️ **API key security**: Use read+trade permissions only. Never enable withdrawals.

> ⚠️ **Not financial advice**: This bot is for educational purposes only. Trading involves significant risk of loss.

---

## How Signals Work

```
Market Data → Indicators → Pattern Detection → Claude AI Analysis
     ↓               ↓              ↓                    ↓
  Live OHLCV    50 indicators   12 candlestick    Structured JSON
                               +9 chart patterns   (BUY/SELL/HOLD)
                                                        ↓
                                             Risk Check → Execute
```

1. **Data Engine** fetches and caches OHLCV data
2. **Indicator Engine** computes all technical indicators
3. **Pattern Detector** identifies patterns and S/R levels
4. **AI Signal Engine** serializes context → Claude AI → parses JSON response
5. **Risk Manager** validates signal, sizes position, sets stops
6. **Execution Engine** places order (paper or live)
7. **Alerting** sends Telegram message + updates dashboard

---

## Backtesting

```bash
python main.py --backtest
```

Generates an HTML report in `backtest_reports/` with:
- Equity curve chart
- Win rate, Sharpe ratio, max drawdown, profit factor
- Per-trade breakdown

---

## License

MIT License — use at your own risk.
