"""Builds the DB signal dict (market context included) from a bot TradeSignal and indicator frame."""

from typing import Any, Dict, Optional

from db.repository import pip_size


def _last(df: Any, col: str, offset: int = 1) -> Optional[float]:
    """Return a column value `offset` rows from the end, or None if unavailable/NaN."""
    try:
        v = df[col].iloc[-offset]
        return None if v != v else float(v)
    except Exception:
        return None


def build_signal_dict(signal: Any, df: Any, htf_context: Optional[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
    """Merge a TradeSignal with RSI/ATR/MACD/volume/sweep/trend context taken from `df`."""
    d: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": signal.timeframe,
        "direction": signal.signal,
        "confidence": signal.confidence,
        "entry_price": signal.entry_price,
        "stop_loss": signal.stop_loss,
        "take_profit_1": signal.take_profit_1,
        "take_profit_2": signal.take_profit_2,
        "risk_reward_ratio": signal.risk_reward_ratio,
        "reasoning": signal.reasoning,
        "raw_response": getattr(signal, "raw_response", None),
    }
    if htf_context and htf_context.get("trend"):
        d["macro_trend"] = str(htf_context["trend"]).upper()
    if df is None or getattr(df, "empty", True):
        return d
    rsi = _last(df, "RSI_14")
    if rsi is not None:
        d["rsi_value"] = rsi
        d["rsi_zone"] = "OVERBOUGHT" if rsi >= 70 else ("OVERSOLD" if rsi <= 30 else "NEUTRAL")
    atr = _last(df, "ATR_14")
    if atr is not None:
        d["atr_pips"] = round(atr / pip_size(symbol), 2)
    m, ms, pm, pms = _last(df, "MACD"), _last(df, "MACD_Signal"), _last(df, "MACD", 2), _last(df, "MACD_Signal", 2)
    if None not in (m, ms, pm, pms):
        if pm <= pms and m > ms:
            d["macd_crossover"] = "BULLISH"
        elif pm >= pms and m < ms:
            d["macd_crossover"] = "BEARISH"
        else:
            d["macd_crossover"] = "NONE"
    vr = _last(df, "Volume_Ratio")
    if vr is not None:
        d["volume_state"] = "ABOVE" if vr >= 1.2 else ("BELOW" if vr <= 0.8 else "NORMAL")
    bull, bear = _last(df, "is_liquidity_sweep_bullish"), _last(df, "is_liquidity_sweep_bearish")
    if bull or bear:
        d["liquidity_sweep"] = 1
        d["sweep_direction"] = "BULLISH" if bull else "BEARISH"
    return d
