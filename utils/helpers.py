"""
utils/helpers.py
────────────────
Shared utility functions used across all trading bot modules.

Includes:
  - Config loading with environment variable substitution
  - Percentage and price calculations
  - DataFrame validation utilities
  - Generic retry decorator with exponential backoff
  - Timestamp helpers
"""

import functools
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import pandas as pd
import yaml
from dotenv import load_dotenv

F = TypeVar("F", bound=Callable[..., Any])

# Load .env file at import time
load_dotenv()


# ─── Configuration ───────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load config.yaml and substitute ${ENV_VAR} placeholders with environment values.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary with env vars resolved.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If YAML is malformed.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = fh.read()

    # Guard against unresolved git merge markers before YAML parsing so
    # startup errors are deterministic and easier to diagnose.
    conflict_patterns = ("<<<<<<<", "=======", ">>>>>>>")
    for idx, line in enumerate(raw.splitlines(), start=1):
        if any(line.startswith(pattern) for pattern in conflict_patterns):
            raise ValueError(
                "Unresolved merge conflict marker found in "
                f"{config_path} at line {idx}: {line!r}. "
                "Resolve conflict markers (<<<<<<<, =======, >>>>>>>) "
                "before starting the bot."
            )

    # Substitute ${VAR_NAME} with environment variable values
    def _replace_env(match: re.Match) -> str:
        var_name = match.group(1)
        value = os.getenv(var_name, "")
        return value

    raw = re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", _replace_env, raw)
    config: Dict[str, Any] = yaml.safe_load(raw)
    return config


def get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Retrieve an environment variable value.

    Args:
        key:      Environment variable name.
        default:  Default value if variable is not set.
        required: Raise ValueError if True and variable is missing.

    Returns:
        Environment variable value as string.
    """
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(
            f"Required environment variable '{key}' is not set. "
            f"Check your .env file."
        )
    return value or ""


# ─── Retry Decorator ─────────────────────────────────────────────────────────

def retry(
    attempts: int = 3,
    initial_delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator that retries the wrapped function on specified exceptions using
    exponential backoff.

    Args:
        attempts:      Maximum number of attempts (including first try).
        initial_delay: Seconds to wait before the first retry.
        backoff:       Multiplier applied to delay after each retry.
        exceptions:    Exception types to catch and retry on.

    Returns:
        Decorated function with retry logic.

    Example:
        @retry(attempts=3, initial_delay=1.0, backoff=2.0)
        def fetch_data():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exc: Optional[Exception] = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < attempts:
                        time.sleep(delay)
                        delay *= backoff
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator


# ─── DataFrame Utilities ──────────────────────────────────────────────────────

def validate_ohlcv(df: pd.DataFrame) -> bool:
    """
    Verify that a DataFrame contains the required OHLCV columns and has data.

    Args:
        df: DataFrame to validate.

    Returns:
        True if valid, False otherwise.
    """
    required = {"open", "high", "low", "close", "volume"}
    if df is None or df.empty:
        return False
    return required.issubset(set(df.columns))


def ensure_datetime_index(df: pd.DataFrame, utc: bool = True) -> pd.DataFrame:
    """
    Ensure that the DataFrame has a proper DatetimeIndex.

    Args:
        df:  OHLCV DataFrame.
        utc: Localise to UTC if True.

    Returns:
        DataFrame with DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, unit="ms", utc=utc)
    elif utc and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


# ─── Price / Risk Calculations ────────────────────────────────────────────────

def pct_change(old_price: float, new_price: float) -> float:
    """
    Calculate percentage change between two prices.

    Args:
        old_price: Starting price.
        new_price: Ending price.

    Returns:
        Percentage change as float (e.g. 0.05 = 5%).
    """
    if old_price == 0:
        return 0.0
    return (new_price - old_price) / old_price


def calc_position_size(
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float,
) -> float:
    """
    Calculate position size based on fixed-fractional risk.

    Formula:
        position_size = (account_balance × risk_pct) / |entry - stop_loss|

    Args:
        account_balance: Total account equity in quote currency.
        risk_pct:        Risk fraction per trade (e.g. 0.015 = 1.5%).
        entry_price:     Planned entry price.
        stop_loss_price: Stop-loss price.

    Returns:
        Position size in base currency units.
    """
    risk_amount = account_balance * risk_pct
    price_risk = abs(entry_price - stop_loss_price)
    if price_risk == 0:
        return 0.0
    return risk_amount / price_risk


def calc_stop_loss(
    entry_price: float,
    atr: float,
    atr_multiplier: float,
    direction: str,
) -> float:
    """
    Calculate ATR-based dynamic stop-loss price.

    Args:
        entry_price:    Trade entry price.
        atr:            Current ATR value.
        atr_multiplier: Multiplier for ATR distance (e.g. 1.5).
        direction:      'long' or 'short'.

    Returns:
        Stop-loss price.
    """
    distance = atr * atr_multiplier
    if direction.lower() == "long":
        return entry_price - distance
    return entry_price + distance


def calc_take_profits(
    entry_price: float,
    stop_loss_price: float,
    tp1_multiplier: float = 1.5,
    tp2_multiplier: float = 3.0,
    direction: str = "long",
) -> Tuple[float, float]:
    """
    Calculate TP1 and TP2 levels from risk distance.

    Args:
        entry_price:    Entry price.
        stop_loss_price: Stop-loss price.
        tp1_multiplier: Risk multiple for TP1 (default 1.5×).
        tp2_multiplier: Risk multiple for TP2 (default 3.0×).
        direction:      'long' or 'short'.

    Returns:
        Tuple of (tp1_price, tp2_price).
    """
    risk = abs(entry_price - stop_loss_price)
    if direction.lower() == "long":
        return (entry_price + risk * tp1_multiplier,
                entry_price + risk * tp2_multiplier)
    return (entry_price - risk * tp1_multiplier,
            entry_price - risk * tp2_multiplier)


def risk_reward_ratio(
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: float,
) -> float:
    """
    Calculate risk/reward ratio for a trade.

    Args:
        entry_price:       Entry price.
        stop_loss_price:   Stop-loss price.
        take_profit_price: Take-profit price.

    Returns:
        Risk/reward ratio (e.g. 2.0 means 2:1 reward to risk).
    """
    risk = abs(entry_price - stop_loss_price)
    reward = abs(take_profit_price - entry_price)
    if risk == 0:
        return 0.0
    return reward / risk


# ─── Timestamp Helpers ────────────────────────────────────────────────────────

def now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def ts_to_str(ts: datetime) -> str:
    """Format a datetime to a human-readable string."""
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def ms_to_datetime(ms: int) -> datetime:
    """Convert millisecond Unix timestamp to UTC datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


# ─── Formatting ──────────────────────────────────────────────────────────────

def fmt_pct(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string (e.g. 0.0512 → '5.12%')."""
    return f"{value * 100:.{decimals}f}%"


def fmt_price(price: float, decimals: int = 4) -> str:
    """Format a price to a fixed number of decimal places."""
    return f"{price:.{decimals}f}"


def truncate_list(items: List[Any], max_items: int = 5) -> List[Any]:
    """Return the last `max_items` elements of a list."""
    return items[-max_items:] if len(items) > max_items else items
