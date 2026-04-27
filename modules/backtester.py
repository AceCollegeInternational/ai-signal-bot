"""
modules/backtester.py
─────────────────────────────────────────────────────────────────────────────
Module 7 — Backtesting Engine

Runs the full signal pipeline over historical data without lookahead bias
and generates a comprehensive HTML performance report.

Uses vectorbt for fast vectorised backtesting. Simulates the same indicator
→ pattern → signal pipeline the live bot uses, applied column-by-column
across the historical DataFrame.

Performance metrics reported:
  Total return %, Annualized return %, Max drawdown % and duration,
  Sharpe ratio, Sortino ratio, Win rate %, Profit factor,
  Average win vs loss, Total trades, Average holding time.

Usage:
    bt = Backtester(config, data_engine)
    report = bt.run("BTC/USDT", "1h")
    bt.generate_html_report(report, "BTC_USDT_1h_backtest.html")
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys

if "numba" not in sys.modules:

    class MockNumba:
        def njit(self, *args, **kwargs):
            def decorator(func):
                return func

            if len(args) == 1 and callable(args[0]):
                return args[0]
            return decorator

        def jit(self, *args, **kwargs):
            return self.njit(*args, **kwargs)

    sys.modules["numba"] = MockNumba()

try:
    import vectorbt as vbt  # type: ignore[import]
except ImportError:
    vbt = None  # type: ignore[assignment]

try:
    import plotly.graph_objects as go  # type: ignore[import]
    from plotly.subplots import make_subplots  # type: ignore[import]

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from modules.indicator_engine import IndicatorEngine
from modules.pattern_detector import PatternDetector
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    """
    Holds all performance metrics and trade data for a completed backtest.
    """

    symbol: str = ""
    timeframe: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 10_000.0
    commission: float = 0.001
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_hold_bars: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "commission": self.commission,
            "total_trades": self.total_trades,
            "win_rate_pct": round(self.win_rate * 100, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "annualized_return_pct": round(self.annualized_return_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "avg_hold_bars": round(self.avg_hold_bars, 2),
        }


class Backtester:
    """
    Historical backtesting engine.

    Applies the indicator + pattern pipeline to historical data row-by-row
    (no lookahead bias) and calculates comprehensive performance metrics.

    Args:
        config:      Parsed config.yaml dictionary.
        data_engine: Initialised DataEngine for fetching historical data.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_engine: Any = None,
    ) -> None:
        self.config = config
        self.data_engine = data_engine

        bt_cfg = config.get("backtesting", {})
        risk_cfg = config.get("risk", {})
        signal_cfg = config.get("signals", {})

        self.initial_capital: float = bt_cfg.get("initial_capital", 10_000.0)
        self.commission: float = bt_cfg.get("commission", 0.001)
        self.report_dir: Path = Path(bt_cfg.get("report_dir", "backtest_reports"))
        self.start_date: str = bt_cfg.get("start_date", "2024-01-01")
        self.end_date: str = bt_cfg.get("end_date", "2025-12-31")

        self.max_risk: float = risk_cfg.get("max_risk_per_trade", 0.015)
        self.atr_multiplier: float = risk_cfg.get("atr_multiplier", 1.5)
        self.tp1_multiplier: float = risk_cfg.get("tp1_multiplier", 1.5)
        self.tp2_multiplier: float = risk_cfg.get("tp2_multiplier", 3.0)
        self.min_rr: float = risk_cfg.get("min_rr_ratio", 1.5)

        self.indicator_engine = IndicatorEngine(config)
        self.pattern_detector = PatternDetector(config)

        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ─── Main Backtest Runner ─────────────────────────────────────────────────

    def run(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Execute a full backtest for a given symbol and timeframe.

        Args:
            symbol:    Trading pair (e.g. 'BTC/USDT').
            timeframe: Timeframe string.
            df:        Pre-loaded DataFrame (optional; fetches from DataEngine if None).

        Returns:
            BacktestResult with all metrics and equity curve.
        """
        tf = timeframe or self.config.get("trading", {}).get("timeframe", "1h")
        data_cfg = self.config.get("data", {})
        hist_days = data_cfg.get("historical_days", 365)

        # ── Load data ─────────────────────────────────────────────────────────
        if df is None:
            if self.data_engine is None:
                log.error("No data engine provided for backtesting")
                return BacktestResult(symbol=symbol, timeframe=tf)
            log.info(f"Fetching {hist_days} days of {symbol} / {tf} for backtest…")
            df = self.data_engine.fetch_historical(symbol, tf, days=hist_days)

        if df is None or len(df) < 100:
            log.error(
                f"Insufficient data for backtest: {len(df) if df is not None else 0} rows"
            )
            return BacktestResult(symbol=symbol, timeframe=tf)

        # ── Apply date range filter ───────────────────────────────────────────
        df = self._filter_date_range(df)
        log.info(
            f"Backtesting {symbol} / {tf}: {len(df)} candles from {df.index[0]} to {df.index[-1]}"
        )

        # ── Compute indicators once on full DataFrame ─────────────────────────
        df = self.indicator_engine.compute_all(df)

        # ── Generate signals without lookahead bias ───────────────────────────
        entries, exits = self._generate_signals_vectorised(df)

        # ── Run vectorbt portfolio simulation ─────────────────────────────────
        if vbt is not None:
            result = self._run_vectorbt(df, entries, exits, symbol, tf)
        else:
            log.warning("vectorbt not installed — using simple custom backtester")
            result = self._run_simple(df, entries, exits, symbol, tf)

        return result

    def _filter_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to the configured date range."""
        try:
            start = pd.Timestamp(self.start_date, tz="UTC")
            end = pd.Timestamp(self.end_date, tz="UTC")
            mask = (df.index >= start) & (df.index <= end)
            return df[mask]
        except Exception as exc:
            log.warning(f"Date range filter failed: {exc} — using full dataset")
            return df

    # ─── Vectorised Signal Generation ────────────────────────────────────────

    def _generate_signals_vectorised(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry/exit signals across the full DataFrame using a
        rolling window approach without lookahead bias.

        Each row's signal is determined using only data up to (but not
        including) that row's indicators — since indicators are computed
        on the fully shifted rolling window, we simply lag by 1 bar.

        Args:
            df: OHLCV + indicator DataFrame.

        Returns:
            Tuple of (entry_signals, exit_signals) as boolean pd.Series.
        """
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        # ── Minimum warmup for indicators ─────────────────────────────────────
        warmup = 50

        for i in range(warmup, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1] if i > 0 else row

            # Entry conditions (bullish confluence):
            bullish = self._check_bullish_entry(row, prev)
            bearish = self._check_bearish_entry(row, prev)

            # Apply at the NEXT bar (shift by 1 to avoid lookahead)
            if i + 1 < len(df):
                if bullish:
                    entries.iloc[i + 1] = True
                if bearish:
                    exits.iloc[i + 1] = True

        return entries, exits

    def _check_bullish_entry(self, row: pd.Series, prev: pd.Series) -> bool:
        """
        Check if multiple bullish indicator conditions align on this bar.

        Uses at least 3 of: RSI < 35, MACD cross, price > EMA50,
        ADX trending, price < lower BB.
        """
        conditions = []
        try:
            if not pd.isna(row.get("RSI_14", np.nan)):
                conditions.append(row["RSI_14"] < 35)
            if not pd.isna(row.get("MACD_Hist", np.nan)):
                hist = row["MACD_Hist"]
                prev_hist = prev.get("MACD_Hist", 0)
                conditions.append(hist > 0 and prev_hist <= 0)  # histogram crossover
            if not pd.isna(row.get("EMA_50", np.nan)):
                conditions.append(row["close"] > row["EMA_50"])
            if not pd.isna(row.get("ADX_14", np.nan)):
                dmp = row.get("DMP_14", 0) or 0
                dmn = row.get("DMN_14", 0) or 0
                conditions.append(row["ADX_14"] > 25 and dmp > dmn)
            if not pd.isna(row.get("BB_Lower", np.nan)):
                conditions.append(row["close"] < row["BB_Lower"])
            if not pd.isna(row.get("STOCH_K", np.nan)):
                conditions.append(row["STOCH_K"] < 25)
        except Exception:
            pass
        return sum(conditions) >= 3

    def _check_bearish_entry(self, row: pd.Series, prev: pd.Series) -> bool:
        """
        Check if multiple bearish indicator conditions align (for exit / short).
        """
        conditions = []
        try:
            if not pd.isna(row.get("RSI_14", np.nan)):
                conditions.append(row["RSI_14"] > 65)
            if not pd.isna(row.get("MACD_Hist", np.nan)):
                hist = row["MACD_Hist"]
                prev_hist = prev.get("MACD_Hist", 0)
                conditions.append(hist < 0 and prev_hist >= 0)
            if not pd.isna(row.get("EMA_50", np.nan)):
                conditions.append(row["close"] < row["EMA_50"])
            if not pd.isna(row.get("BB_Upper", np.nan)):
                conditions.append(row["close"] > row["BB_Upper"])
            if not pd.isna(row.get("STOCH_K", np.nan)):
                conditions.append(row["STOCH_K"] > 75)
        except Exception:
            pass
        return sum(conditions) >= 3

    # ─── Vectorbt Simulation ──────────────────────────────────────────────────

    def _run_vectorbt(
        self,
        df: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
        symbol: str,
        timeframe: str,
    ) -> BacktestResult:
        """Run backtest using vectorbt Portfolio.from_signals()."""
        try:
            portfolio = vbt.Portfolio.from_signals(
                df["close"],
                entries,
                exits,
                init_cash=self.initial_capital,
                fees=self.commission,
                freq=timeframe,
            )

            stats = portfolio.stats()
            equity = portfolio.value()
            trades = portfolio.trades.records_readable

            total_return = float(stats.get("Total Return [%]", 0) or 0)
            ann_return = float(stats.get("Annualized Return [%]", 0) or 0)
            max_dd = float(stats.get("Max Drawdown [%]", 0) or 0)
            sharpe = float(stats.get("Sharpe Ratio", 0) or 0)
            sortino = float(stats.get("Sortino Ratio", 0) or 0)
            win_rate = float(stats.get("Win Rate [%]", 0) or 0) / 100.0
            profit_factor = float(stats.get("Profit Factor", 0) or 0)
            total_trades = int(stats.get("Total Trades", 0) or 0)

            # Compute avg win/loss from trade records
            avg_win, avg_loss = self._compute_avg_win_loss(trades)
            avg_hold = (
                float(trades["Duration"].mean().total_seconds() / 3600)
                if len(trades) > 0
                else 0.0
            )

            equity_vals = equity.tolist()
            ts_vals = [str(t) for t in equity.index]

            result = BacktestResult(
                symbol=symbol,
                timeframe=timeframe,
                start_date=str(df.index[0]),
                end_date=str(df.index[-1]),
                initial_capital=self.initial_capital,
                commission=self.commission,
                total_trades=total_trades,
                winning_trades=int(total_trades * win_rate),
                losing_trades=int(total_trades * (1 - win_rate)),
                win_rate=win_rate,
                total_return_pct=total_return,
                annualized_return_pct=ann_return,
                max_drawdown_pct=max_dd,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                avg_hold_bars=avg_hold,
                equity_curve=equity_vals,
                timestamps=ts_vals,
            )
            log.info(
                f"Backtest complete: {symbol} / {timeframe} | return={total_return:.2f}% sharpe={sharpe:.2f}"
            )
            return result

        except Exception as exc:
            log.error(f"vectorbt backtest error: {exc}", exc_info=True)
            return self._run_simple(df, entries, exits, symbol, timeframe)

    def _compute_avg_win_loss(self, trades: Any) -> Tuple[float, float]:
        """Extract average win and loss from vectorbt trade records."""
        try:
            if hasattr(trades, "PnL"):
                wins = trades[trades["PnL"] > 0]["PnL"]
                losses = trades[trades["PnL"] < 0]["PnL"]
                avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
                avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
                return avg_win, avg_loss
        except Exception:
            pass
        return 0.0, 0.0

    # ─── Simple fallback simulation ───────────────────────────────────────────

    def _run_simple(
        self,
        df: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
        symbol: str,
        timeframe: str,
    ) -> BacktestResult:
        """
        Minimal custom backtester used when vectorbt is unavailable.

        Simulates long-only trading with ATR-based stops.
        """
        capital = self.initial_capital
        equity = [capital]
        timestamps = [str(df.index[0])]
        trades: List[Dict[str, Any]] = []

        in_trade = False
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        entry_idx = 0
        atr = 0.0

        for i in range(len(df)):
            row = df.iloc[i]
            price = float(row["close"])
            try:
                atr = float(row.get("ATR_14", 0) or 0)
            except (TypeError, ValueError):
                atr = 0.0

            if in_trade:
                # Check exit conditions
                exit_reason = ""
                if price <= stop_loss:
                    exit_reason = "SL"
                elif price >= take_profit:
                    exit_reason = "TP"
                elif exits.iloc[i]:
                    exit_reason = "Signal"

                if exit_reason:
                    pnl_pct = (price - entry_price) / entry_price
                    risk_amount = capital * self.max_risk
                    position_size = (
                        risk_amount / abs(entry_price - stop_loss)
                        if abs(entry_price - stop_loss) > 0
                        else 0
                    )
                    pnl = position_size * (price - entry_price) - (
                        position_size * price * self.commission
                    )
                    capital += pnl
                    trades.append(
                        {
                            "entry": entry_price,
                            "exit": price,
                            "pnl": pnl,
                            "reason": exit_reason,
                            "hold_bars": i - entry_idx,
                        }
                    )
                    in_trade = False

            elif entries.iloc[i] and not in_trade:
                entry_price = price
                if atr > 0:
                    stop_loss = entry_price - atr * self.atr_multiplier
                    take_profit = (
                        entry_price + atr * self.atr_multiplier * self.tp1_multiplier
                    )
                else:
                    stop_loss = entry_price * 0.98
                    take_profit = entry_price * 1.03
                entry_idx = i
                in_trade = True

            equity.append(capital)
            timestamps.append(str(df.index[i]))

        return self._compute_metrics_from_trades(
            trades, equity, timestamps, capital, symbol, timeframe, df
        )

    def _compute_metrics_from_trades(
        self,
        trades: List[Dict],
        equity: List[float],
        timestamps: List[str],
        final_capital: float,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> BacktestResult:
        """Compute metrics from raw trade list (simple backtester)."""
        if not trades:
            return BacktestResult(
                symbol=symbol,
                timeframe=timeframe,
                initial_capital=self.initial_capital,
                equity_curve=equity,
                timestamps=timestamps,
                start_date=str(df.index[0]),
                end_date=str(df.index[-1]),
            )

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_return = (
            (final_capital - self.initial_capital) / self.initial_capital * 100
        )

        eq_series = pd.Series(equity)
        running_max = eq_series.cummax()
        drawdown = (eq_series - running_max) / running_max * 100
        max_dd = float(drawdown.min())

        # Annualized return
        n_days = (df.index[-1] - df.index[0]).days or 1
        ann_return = ((1 + total_return / 100) ** (365 / n_days) - 1) * 100

        # Sharpe (simplified: daily returns)
        eq_s = pd.Series(equity)
        daily_returns = eq_s.pct_change().dropna()
        sharpe = (
            float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
            if daily_returns.std() > 0
            else 0.0
        )

        # Sortino
        neg_returns = daily_returns[daily_returns < 0]
        sortino = (
            float(daily_returns.mean() / neg_returns.std() * np.sqrt(252))
            if len(neg_returns) > 0 and neg_returns.std() > 0
            else 0.0
        )

        profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")
        avg_hold = float(np.mean([t["hold_bars"] for t in trades]))

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date=str(df.index[0]),
            end_date=str(df.index[-1]),
            initial_capital=self.initial_capital,
            commission=self.commission,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / len(trades) if trades else 0,
            total_return_pct=total_return,
            annualized_return_pct=ann_return,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            profit_factor=profit_factor,
            avg_win=float(np.mean(wins)) if wins else 0.0,
            avg_loss=float(np.mean(losses)) if losses else 0.0,
            avg_hold_bars=avg_hold,
            equity_curve=equity,
            timestamps=timestamps,
            trades=trades,
        )

    # ─── HTML Report Generation ───────────────────────────────────────────────

    def generate_html_report(
        self,
        result: BacktestResult,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate a full HTML backtest report with equity curve chart.

        Args:
            result:   Completed BacktestResult.
            filename: Output filename (auto-generated if None).

        Returns:
            Absolute path to the saved HTML file.
        """
        if filename is None:
            safe_sym = result.symbol.replace("/", "_")
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_sym}_{result.timeframe}_{ts}.html"

        output_path = self.report_dir / filename

        # ── Build equity curve chart ──────────────────────────────────────────
        equity_chart_html = self._build_equity_chart(result)

        metrics = result.to_dict()
        metrics_html = "".join(
            f"<tr><td>{k.replace('_', ' ').title()}</td><td><strong>{v}</strong></td></tr>"
            for k, v in metrics.items()
            if k not in ("symbol", "timeframe", "equity_curve", "timestamps", "trades")
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Report — {result.symbol} {result.timeframe}</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f1117; color: #e0e0e0; margin: 0; padding: 20px; }}
  h1 {{ color: #00d4aa; border-bottom: 2px solid #00d4aa; padding-bottom: 10px; }}
  h2 {{ color: #7eb8f7; margin-top: 30px; }}
  .metrics-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
  .metrics-table th {{ background: #1e2130; color: #00d4aa; padding: 10px; text-align: left; }}
  .metrics-table td {{ padding: 8px 10px; border-bottom: 1px solid #2a2d3e; }}
  .metrics-table tr:nth-child(even) {{ background: #161820; }}
  .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
  .card {{ background: #1e2130; border-radius: 8px; padding: 15px; text-align: center; border: 1px solid #2a2d3e; }}
  .card .value {{ font-size: 1.8em; font-weight: bold; color: #00d4aa; }}
  .card .label {{ font-size: 0.85em; color: #888; margin-top: 5px; }}
  .positive {{ color: #00d4aa !important; }}
  .negative {{ color: #ff6b6b !important; }}
  .chart-container {{ background: #1e2130; border-radius: 8px; padding: 10px; margin: 20px 0; }}
  footer {{ text-align: center; color: #555; margin-top: 40px; font-size: 0.8em; }}
</style>
</head>
<body>
<h1>📊 Backtest Report</h1>
<p><strong>Symbol:</strong> {result.symbol} &nbsp;|&nbsp;
   <strong>Timeframe:</strong> {result.timeframe} &nbsp;|&nbsp;
   <strong>Period:</strong> {result.start_date[:10]} → {result.end_date[:10]}</p>

<div class="summary">
  <div class="card">
    <div class="value {'positive' if result.total_return_pct >= 0 else 'negative'}">{result.total_return_pct:+.2f}%</div>
    <div class="label">Total Return</div>
  </div>
  <div class="card">
    <div class="value">{result.max_drawdown_pct:.2f}%</div>
    <div class="label">Max Drawdown</div>
  </div>
  <div class="card">
    <div class="value">{result.sharpe_ratio:.2f}</div>
    <div class="label">Sharpe Ratio</div>
  </div>
  <div class="card">
    <div class="value">{result.win_rate*100:.1f}%</div>
    <div class="label">Win Rate ({result.total_trades} trades)</div>
  </div>
</div>

<div class="chart-container">
{equity_chart_html}
</div>

<h2>Full Performance Metrics</h2>
<table class="metrics-table">
  <tr><th>Metric</th><th>Value</th></tr>
  {metrics_html}
</table>

<footer>Generated by AI-Powered Trading Bot · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</footer>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        log.info(f"Backtest report saved: {output_path}")
        return str(output_path)

    def _build_equity_chart(self, result: BacktestResult) -> str:
        """Build an interactive Plotly equity curve chart (or static fallback)."""
        if not PLOTLY_AVAILABLE or not result.equity_curve:
            return "<p>Install plotly for interactive charts: pip install plotly</p>"

        try:
            ts = result.timestamps[: len(result.equity_curve)]
            eq = result.equity_curve

            baseline = [result.initial_capital] * len(eq)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=eq,
                    mode="lines",
                    name="Portfolio Value",
                    line=dict(color="#00d4aa", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0, 212, 170, 0.05)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=baseline,
                    mode="lines",
                    name="Initial Capital",
                    line=dict(color="#555", width=1, dash="dash"),
                )
            )
            fig.update_layout(
                title="Equity Curve",
                paper_bgcolor="#1e2130",
                plot_bgcolor="#161820",
                font=dict(color="#e0e0e0"),
                xaxis=dict(gridcolor="#2a2d3e"),
                yaxis=dict(gridcolor="#2a2d3e", title="Portfolio Value (USDT)"),
                legend=dict(bgcolor="#1e2130"),
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            return fig.to_html(full_html=False, include_plotlyjs="cdn")
        except Exception as exc:
            log.warning(f"Chart generation failed: {exc}")
            return f"<p>Chart unavailable: {exc}</p>"
