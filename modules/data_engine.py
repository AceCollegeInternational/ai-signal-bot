"""
modules/data_engine.py
─────────────────────────────────────────────────────────────────────────────
Module 1 — Data Engine

Responsibilities:
  - Connect to Binance or Bybit via the ccxt library
  - Fetch historical OHLCV data for configurable symbols and timeframes
  - Store data as pandas DataFrames with UTC DatetimeIndex
  - Cache data locally as Parquet files to avoid redundant API calls
  - Provide async WebSocket streaming for live price updates (ccxt.pro)
  - Expose a synchronous `get_live_price()` for quick current-price lookups

Public API:
    engine = DataEngine(config)
    df = engine.fetch_ohlcv("BTC/USDT", "1h", limit=500)
    price = engine.get_live_price("BTC/USDT")
    await engine.stream_ohlcv("BTC/USDT", "1m", callback)
"""

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import ccxt
import pandas as pd
import requests

from utils.helpers import ensure_datetime_index, load_config, retry, validate_ohlcv
from utils.logger import get_logger

log = get_logger(__name__)

# Mapping from ccxt exchange id to constructor
EXCHANGE_MAP: Dict[str, Any] = {
    "binance": ccxt.binance,
    "bybit": ccxt.bybit,
}

# MT5-like symbol to Bybit symbol aliases (expand as needed)
MT5_TO_BYBIT: Dict[str, str] = {
    "EURUSD": "EURUSDT",
    "XAUUSD": "XAUUSDT",  # Gold
    "BTCUSD": "BTCUSDT",
    "US500": "SPX500USDT",  # S&P 500
}

MT5_TO_TWELVEDATA: Dict[str, str] = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "XAUUSD": "XAU/USD",  # Gold
    "BTCUSD": "BTC/USD",
    "US500": "SPX",
}

# Supported timeframes and their millisecond equivalents
TF_MS: Dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

TWELVEDATA_INTERVAL_MAP: Dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1day",
}


class DataEngine:
    """
    Fetches, caches, and streams market OHLCV data from a crypto exchange.

    Args:
        config: Parsed config.yaml dictionary.
        api_key: Exchange API key (overrides env-loaded value if provided).
        api_secret: Exchange API secret.
        testnet: If True, connect to exchange testnet.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
    ) -> None:
        self.config = config
        trading_cfg = config.get("trading", {})
        data_cfg = config.get("data", {})

        self.exchange_id: str = trading_cfg.get("exchange", "binance").lower()
        self.symbols: List[str] = trading_cfg.get("symbols", ["BTC/USDT"])
        self.timeframe: str = trading_cfg.get("timeframe", "1h")
        self.cache_dir: Path = Path(data_cfg.get("cache_dir", "data/cache"))
        self.cache_expiry_hours: int = data_cfg.get("cache_expiry_hours", 4)
        self.exchange: Optional[ccxt.Exchange] = None

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._td_session: Optional[requests.Session] = None
        self._td_api_key: str = ""
        self._td_last_request_at: float = 0.0
        self._td_min_interval_seconds: float = 8.0  # free tier safety (~8 req/min)

        # ── Exchange credentials ──────────────────────────────────────────────
        exchange_upper = self.exchange_id.upper()
        resolved_key = api_key or os.getenv(f"{exchange_upper}_API_KEY", "")
        resolved_secret = api_secret or os.getenv(f"{exchange_upper}_API_SECRET", "")

        if self.exchange_id == "twelvedata":
            td_env_candidates = [
                "TWELVEDATA_API_KEY",
                "TWELVE_DATA_API_KEY",
                "TWELVEDATA_APIKEY",
                "TWELVEDATA_KEY",
            ]
            env_key = ""
            env_key_name = ""
            for name in td_env_candidates:
                value = os.getenv(name, "").strip()
                if value:
                    env_key = value
                    env_key_name = name
                    break

            self._td_api_key = (api_key or env_key).strip()
            if not self._td_api_key:
                log.warning(
                    "Twelve Data API key is not set. Checked: "
                    f"{', '.join(td_env_candidates)}. Requests will fail."
                )
            else:
                source = "constructor argument" if api_key else env_key_name
                log.info(f"Twelve Data API key loaded from {source}.")
            self._td_session = requests.Session()
            log.info(
                f"DataEngine initialised — exchange={self.exchange_id} "
                f"symbols={self.symbols} timeframe={self.timeframe}"
            )
            return

        exchange_cls = EXCHANGE_MAP.get(self.exchange_id)
        if exchange_cls is None:
            raise ValueError(
                f"Unsupported exchange: {self.exchange_id!r}. "
                f"Supported: {list(EXCHANGE_MAP.keys())}"
            )

        default_type = "linear" if self.exchange_id == "bybit" else "spot"
        exchange_opts: Dict[str, Any] = {
            "apiKey": resolved_key,
            "secret": resolved_secret,
            "enableRateLimit": True,
            "options": {"defaultType": default_type},
        }
        if testnet:
            exchange_opts["options"]["testnet"] = True

        self.exchange: ccxt.Exchange = exchange_cls(exchange_opts)
        log.info(
            f"DataEngine initialised — exchange={self.exchange_id} "
            f"symbols={self.symbols} timeframe={self.timeframe}"
        )

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize incoming symbol to exchange-specific format.

        For Bybit fallback, this maps MT5-style symbols (e.g. EURUSD) to
        Bybit-style symbols expected by ccxt (e.g. EURUSDT).
        """
        if self.exchange_id != "bybit":
            if self.exchange_id == "twelvedata":
                raw = symbol.replace("/", "").upper()
                return MT5_TO_TWELVEDATA.get(raw, symbol)
            return symbol

        raw = symbol.replace("/", "").upper()
        return MT5_TO_BYBIT.get(raw, symbol)

    def _throttle_twelvedata(self) -> None:
        """Throttle requests to respect Twelve Data free-tier pacing."""
        now = time.time()
        wait_for = self._td_min_interval_seconds - (now - self._td_last_request_at)
        if wait_for > 0:
            time.sleep(wait_for)
        self._td_last_request_at = time.time()

    def _fetch_raw_twelvedata(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> List[List[Any]]:
        if not self._td_session:
            raise RuntimeError("Twelve Data session is not initialised.")

        self._throttle_twelvedata()
        interval = TWELVEDATA_INTERVAL_MAP.get(timeframe, "1h")
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": min(limit, 5000),
            "timezone": "UTC",
            "apikey": self._td_api_key,
        }
        resp = self._td_session.get(
            "https://api.twelvedata.com/time_series", params=params, timeout=30
        )
        resp.raise_for_status()
        payload = resp.json()

        values = payload.get("values", [])
        if not values:
            return []

        rows: List[List[Any]] = []
        # Twelve Data returns newest first; reverse to oldest→newest
        for row in reversed(values):
            ts = int(
                datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=timezone.utc)
                .timestamp()
                * 1000
            )
            rows.append(
                [
                    ts,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row.get("volume", 0.0) or 0.0),
                ]
            )
        return rows

    # ─── Cache Helpers ────────────────────────────────────────────────────────

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        """Return the Parquet cache file path for a symbol/timeframe pair."""
        safe_symbol = symbol.replace("/", "_")
        return self.cache_dir / f"{safe_symbol}_{timeframe}.parquet"

    def _cache_is_fresh(self, path: Path) -> bool:
        """Return True if the cache file exists and was written recently."""
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age = datetime.now(timezone.utc) - mtime
        return age < timedelta(hours=self.cache_expiry_hours)

    def _load_cache(self, path: Path) -> Optional[pd.DataFrame]:
        """Load a cached DataFrame from Parquet."""
        try:
            df = pd.read_parquet(path)
            df = ensure_datetime_index(df)
            log.debug(f"Cache loaded: {path.name} ({len(df)} rows)")
            return df
        except Exception as exc:
            log.warning(f"Failed to load cache {path}: {exc}")
            return None

    def _save_cache(self, df: pd.DataFrame, path: Path) -> None:
        """Persist a DataFrame to Parquet cache."""
        try:
            df.to_parquet(path, engine="pyarrow", index=True)
            log.debug(f"Cache saved: {path.name} ({len(df)} rows)")
        except Exception as exc:
            log.warning(f"Failed to save cache {path}: {exc}")

    # ─── OHLCV Fetching ───────────────────────────────────────────────────────

    @retry(attempts=3, initial_delay=1.0, backoff=2.0, exceptions=(ccxt.NetworkError, ccxt.RequestTimeout))
    def _fetch_raw(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: int = 500,
    ) -> List[List[Any]]:
        """
        Fetch raw OHLCV list from the exchange with retry logic.

        Args:
            symbol:    Trading pair, e.g. 'BTC/USDT'.
            timeframe: Timeframe string, e.g. '1h'.
            since:     Start timestamp in milliseconds (optional).
            limit:     Maximum number of candles to fetch per request.

        Returns:
            List of [timestamp, open, high, low, close, volume] rows.
        """
        if self.exchange_id == "twelvedata":
            return self._fetch_raw_twelvedata(symbol, timeframe, limit=limit)
        return self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        limit: int = 500,
        since: Optional[datetime] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol/timeframe, using local cache when available.

        Args:
            symbol:        Trading pair (e.g. 'BTC/USDT').
            timeframe:     Timeframe string (default: from config).
            limit:         Number of candles to fetch if cache miss.
            since:         Start datetime for historical data (UTC).
            use_cache:     Check local cache before fetching from API.
            force_refresh: Ignore cache and always fetch from API.

        Returns:
            pandas DataFrame with DatetimeIndex and columns:
            [open, high, low, close, volume]
        """
        resolved_symbol = self._normalize_symbol(symbol)
        tf = timeframe or self.timeframe
        cache_path = self._cache_path(symbol, tf)

        # Try to return cached data
        if use_cache and not force_refresh and self._cache_is_fresh(cache_path):
            cached = self._load_cache(cache_path)
            if cached is not None and validate_ohlcv(cached):
                return cached

        # Fetch paginated data to build full history
        log.info(f"Fetching OHLCV: {symbol} -> {resolved_symbol} / {tf} (limit={limit})")
        since_ms: Optional[int] = None
        if since is not None:
            since_ms = int(since.timestamp() * 1000)

        all_candles: List[List[Any]] = []
        while True:
            try:
                candles = self._fetch_raw(resolved_symbol, tf, since=since_ms, limit=limit)
            except ccxt.BadSymbol:
                log.error(f"Symbol not found on {self.exchange_id}: {symbol} ({resolved_symbol})")
                return pd.DataFrame()
            except Exception as exc:
                log.error(f"Unexpected error fetching {symbol} ({resolved_symbol}): {exc}")
                return pd.DataFrame()

            if not candles:
                break

            all_candles.extend(candles)

            # If we got a full page, fetch the next batch
            if len(candles) < limit or since is None:
                break

            # Advance since to after last fetched candle
            last_ts = candles[-1][0]
            tf_ms = TF_MS.get(tf, 3_600_000)
            since_ms = last_ts + tf_ms

            # Safety: avoid infinite loop if exchange returns same timestamps
            if since_ms and since_ms >= int(datetime.now(timezone.utc).timestamp() * 1000):
                break

        if not all_candles:
            log.warning(f"No candles returned for {symbol} / {tf}")
            return pd.DataFrame()

        df = self._candles_to_df(all_candles)

        # Cache the result
        if use_cache:
            self._save_cache(df, cache_path)

        log.info(f"Fetched {len(df)} candles for {symbol} / {tf}")
        return df

    def fetch_historical(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        days: int = 365,
    ) -> pd.DataFrame:
        """
        Fetch up to `days` days of historical OHLCV data.

        Uses pagination to retrieve the full history for backtesting.

        Args:
            symbol:    Trading pair.
            timeframe: Timeframe string.
            days:      Number of calendar days to look back.

        Returns:
            pandas DataFrame with full history.
        """
        resolved_symbol = self._normalize_symbol(symbol)
        tf = timeframe or self.timeframe
        since = datetime.now(timezone.utc) - timedelta(days=days)
        cache_path = self._cache_path(symbol, f"{tf}_hist")

        if self._cache_is_fresh(cache_path):
            cached = self._load_cache(cache_path)
            if cached is not None:
                return cached

        log.info(f"Fetching {days} days of history for {symbol} / {tf}")
        all_candles: List[List[Any]] = []
        since_ms = int(since.timestamp() * 1000)
        limit = 1000

        while True:
            try:
                candles = self._fetch_raw(resolved_symbol, tf, since=since_ms, limit=limit)
            except Exception as exc:
                log.error(f"Error fetching historical data: {exc}")
                break

            if not candles:
                break

            all_candles.extend(candles)
            last_ts = candles[-1][0]
            tf_ms = TF_MS.get(tf, 3_600_000)
            since_ms = last_ts + tf_ms

            if since_ms >= int(datetime.now(timezone.utc).timestamp() * 1000):
                break

            if len(candles) < limit:
                break

        if not all_candles:
            return pd.DataFrame()

        df = self._candles_to_df(all_candles)
        self._save_cache(df, cache_path)
        log.info(f"Historical fetch complete: {len(df)} candles ({symbol} / {tf})")
        return df

    def fetch_all_symbols(
        self,
        timeframe: Optional[str] = None,
        limit: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for all symbols defined in config.

        Returns:
            Dict mapping symbol → DataFrame.
        """
        result: Dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            df = self.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not df.empty:
                result[symbol] = df
        return result

    # ─── Live Price ───────────────────────────────────────────────────────────

    @retry(attempts=3, initial_delay=0.5, backoff=2.0)
    def get_live_price(self, symbol: str) -> float:
        """
        Fetch the current last traded price for a symbol.

        Args:
            symbol: Trading pair (e.g. 'BTC/USDT').

        Returns:
            Last price as float, or 0.0 on error.
        """
        try:
            resolved_symbol = self._normalize_symbol(symbol)
            if self.exchange_id == "twelvedata":
                if not self._td_session:
                    return 0.0
                self._throttle_twelvedata()
                resp = self._td_session.get(
                    "https://api.twelvedata.com/price",
                    params={"symbol": resolved_symbol, "apikey": self._td_api_key},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                return float(data.get("price", 0.0))

            ticker = self.exchange.fetch_ticker(resolved_symbol)
            return float(ticker.get("last", 0.0))
        except Exception as exc:
            log.error(f"Failed to get live price for {symbol}: {exc}")
            return 0.0

    # ─── WebSocket Streaming ──────────────────────────────────────────────────

    async def stream_ohlcv(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        callback: Optional[Callable[[str, pd.DataFrame], None]] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """
        Stream live OHLCV candles via WebSocket using ccxt.pro.

        This coroutine runs indefinitely (or until stop_event is set).
        Each time a new/updated candle is received, `callback` is invoked
        with (symbol, df_of_latest_candles).

        Args:
            symbol:     Trading pair to stream.
            timeframe:  Timeframe for candles.
            callback:   Coroutine or function called on each update.
                        Signature: callback(symbol: str, df: pd.DataFrame)
            stop_event: asyncio.Event; streaming stops when set.

        Note:
            ccxt.pro must be installed:  pip install ccxt[pro]
        """
        if self.exchange_id == "twelvedata":
            log.error(
                "WebSocket streaming is not implemented for Twelve Data in this bot. "
                "Use polling via fetch_ohlcv/get_live_price."
            )
            return

        try:
            import ccxt.pro as ccxtpro  # type: ignore[import]
        except ImportError:
            log.error(
                "ccxt.pro is not installed. "
                "Run: pip install ccxt[pro]  or  pip install ccxt.pro"
            )
            return

        resolved_symbol = self._normalize_symbol(symbol)
        tf = timeframe or self.timeframe
        exchange_upper = self.exchange_id.upper()
        exchange_pro_cls = getattr(ccxtpro, self.exchange_id, None)
        if exchange_pro_cls is None:
            log.error(f"ccxt.pro does not support exchange: {self.exchange_id}")
            return

        pro_exchange = exchange_pro_cls({
            "apiKey": os.getenv(f"{exchange_upper}_API_KEY", ""),
            "secret": os.getenv(f"{exchange_upper}_API_SECRET", ""),
            "enableRateLimit": True,
        })

        log.info(f"Starting WebSocket stream: {symbol} -> {resolved_symbol} / {tf}")
        try:
            while True:
                if stop_event and stop_event.is_set():
                    log.info("WebSocket stream stopped by event.")
                    break
                try:
                    candles = await pro_exchange.watch_ohlcv(resolved_symbol, tf, limit=100)
                    df = self._candles_to_df(candles)
                    if callback is not None:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(symbol, df)
                        else:
                            callback(symbol, df)
                except ccxt.NetworkError as exc:
                    log.warning(f"WebSocket network error ({symbol}): {exc}. Reconnecting…")
                    await asyncio.sleep(5)
                except Exception as exc:
                    log.error(f"WebSocket error ({symbol}): {exc}")
                    await asyncio.sleep(10)
        finally:
            await pro_exchange.close()
            log.info(f"WebSocket connection closed for {symbol}")

    # ─── DataFrame Construction ───────────────────────────────────────────────

    @staticmethod
    def _candles_to_df(candles: List[List[Any]]) -> pd.DataFrame:
        """
        Convert a raw ccxt OHLCV list to a clean pandas DataFrame.

        Args:
            candles: List of [timestamp_ms, open, high, low, close, volume].

        Returns:
            DataFrame with DatetimeIndex (UTC) and float columns.
        """
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df.astype(float)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        return df

    # ─── Market Info ─────────────────────────────────────────────────────────

    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieve market metadata for a symbol (tick size, lot size, etc.).

        Args:
            symbol: Trading pair.

        Returns:
            Market info dict from ccxt.
        """
        try:
            self.exchange.load_markets()
            return self.exchange.markets.get(symbol, {})
        except Exception as exc:
            log.error(f"Failed to get market info for {symbol}: {exc}")
            return {}

    def get_account_balance(self, currency: str = "USDT") -> float:
        """
        Fetch the free balance for a given currency.

        Args:
            currency: Currency code (default 'USDT').

        Returns:
            Free balance as float, or 0.0 on failure.
        """
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get("free", {}).get(currency, 0.0))
        except Exception as exc:
            log.error(f"Failed to fetch balance: {exc}")
            return 0.0

    def get_supported_timeframes(self) -> List[str]:
        """
        Return list of timeframes supported by the connected exchange.

        Returns:
            List of timeframe strings (e.g. ['1m', '5m', '1h', ...]).
        """
        try:
            self.exchange.load_markets()
            return list(self.exchange.timeframes.keys())
        except Exception as exc:
            log.warning(f"Could not load exchange timeframes: {exc}")
            return list(TF_MS.keys())
