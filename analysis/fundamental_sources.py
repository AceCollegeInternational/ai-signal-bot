"""
analysis/fundamental_sources.py
───────────────────────────────
Live data-source adapters for the FundamentalAnalyst layer.
"""

from __future__ import annotations

import io
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml

from modules.market_data_store import market_data_store
from utils.logger import get_logger

log = get_logger(__name__)

_CACHE: Dict[str, Dict[str, Any]] = {}


def _now_ts() -> float:
    return time.time()


def _cache_get(key: str) -> Optional[dict]:
    item = _CACHE.get(key)
    if not item:
        return None
    if _now_ts() >= float(item.get("expires_at", 0)):
        _CACHE.pop(key, None)
        return None
    return item


def _cache_set(key: str, payload: dict, ttl_seconds: int) -> None:
    _CACHE[key] = {
        "fetched_at": _now_ts(),
        "expires_at": _now_ts() + max(1, int(ttl_seconds)),
        "data": payload,
    }


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _fx_base_quote(symbol: str) -> Tuple[str, str]:
    s = symbol.upper().replace("/", "")
    if len(s) >= 6:
        return s[:3], s[3:6]
    return s[:3], s[-3:]


def _map_usd_strength_to_symbol(symbol: str, usd_strength: float) -> float:
    base, quote = _fx_base_quote(symbol)
    if base == "USD":
        return _clamp(usd_strength)   # USD strength supports BUY for USDxxx
    if quote == "USD":
        return _clamp(-usd_strength)  # USD strength hurts BUY for xxxUSD
    return 0.0


def _calendar_score_for_currency(events: List[Dict[str, Any]], currency: str, now: datetime) -> float:
    score = 0.0
    for ev in events:
        curr = str(ev.get("country", "") or ev.get("currency", "")).upper()
        if currency not in curr:
            continue

        impact = str(ev.get("impact", "")).lower()
        impact_w = 1.0 if impact == "high" else (0.6 if impact == "medium" else 0.3)

        t_raw = str(ev.get("time", "") or ev.get("date", ""))
        if not t_raw:
            continue
        try:
            t = datetime.strptime(t_raw[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            continue

        hours = abs((t - now).total_seconds()) / 3600.0
        proximity = max(0.1, 1.0 - min(hours / 24.0, 1.0))

        actual = ev.get("actual")
        forecast = ev.get("forecast")
        delta = 0.0
        try:
            if actual is not None and forecast is not None:
                fv = float(str(forecast).replace("%", "").replace(",", ""))
                av = float(str(actual).replace("%", "").replace(",", ""))
                if abs(fv) > 1e-9:
                    delta = (av - fv) / abs(fv)
        except Exception:
            delta = 0.0

        signed = max(-1.0, min(1.0, delta))
        score += signed * impact_w * proximity

    return _clamp(score)


def _parse_investing_calendar(html: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    rows = re.findall(r"<tr[^>]*eventRowId[^>]*>(.*?)</tr>", html, flags=re.I | re.S)
    for row in rows:
        cur_match = re.search(r"flagCur[^>]*>\s*([A-Z]{3})\s*<", row, flags=re.I | re.S)
        imp_match = re.search(r"title=\"([^\"]*(?:Low|Medium|High)[^\"]*)\"", row, flags=re.I)
        time_match = re.search(r"data-event-datetime=\"([^\"]+)\"", row, flags=re.I)
        act_match = re.search(r"id=\"eventActual[^>]*>\s*([^<]*)<", row, flags=re.I | re.S)
        fct_match = re.search(r"id=\"eventForecast[^>]*>\s*([^<]*)<", row, flags=re.I | re.S)
        if not cur_match or not time_match:
            continue
        events.append(
            {
                "currency": cur_match.group(1).upper(),
                "impact": (imp_match.group(1).lower() if imp_match else "medium"),
                "time": time_match.group(1).replace("T", " ")[:19],
                "actual": (act_match.group(1).strip() if act_match else None),
                "forecast": (fct_match.group(1).strip() if fct_match else None),
            }
        )
    return events


class FundamentalSources:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.fa_cfg = self.config.get("fundamental_analysis", {})
        self.src_cfg = self.fa_cfg.get("sources", {})
        self.timeout = 10

    def get_cb_bias(self, symbol: str) -> Tuple[float, List[str], List[str], Dict[str, float]]:
        cfg = self.src_cfg.get("cb_bias", {})
        cache_key = f"fa:cb_bias:{symbol.upper()}"
        ttl = int(cfg.get("cache_ttl_seconds", 21600))
        hit = _cache_get(cache_key)
        if hit:
            d = hit["data"]
            return d["score"], d["sources_used"], d["sources_failed"], d["ages"]

        sources_used: List[str] = []
        sources_failed: List[str] = []
        ages: Dict[str, float] = {}
        score = 0.0
        ok = False

        if cfg.get("scrape_enabled", True):
            try:
                url = cfg.get("scrape_url", "https://www.centralbanknews.info/")
                resp = requests.get(url, timeout=self.timeout)
                resp.raise_for_status()
                html = resp.text.lower()
                base, quote = _fx_base_quote(symbol)
                hawkish = len(re.findall(rf"{base.lower()}[^\\n]{{0,80}}hawkish", html))
                dovish = len(re.findall(rf"{base.lower()}[^\\n]{{0,80}}dovish", html))
                bias = 0.0
                if hawkish + dovish > 0:
                    bias = (hawkish - dovish) / max(1, hawkish + dovish)
                score = _clamp(bias)
                sources_used.append("cb_bias")
                ages["cb_bias"] = 0.0
                ok = True
            except Exception as exc:
                sources_failed.append(f"cb_bias_scrape:{exc}")

        if not ok:
            try:
                p = Path(cfg.get("yaml_fallback_path", "data/fundamentals/cb_bias.yaml"))
                raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                by_symbol = raw.get("symbols", {})
                by_currency = raw.get("currencies", {})
                entry = by_symbol.get(symbol.upper(), {})
                if not entry:
                    base, quote = _fx_base_quote(symbol)
                    base_bias = float(by_currency.get(base, 0.0))
                    quote_bias = float(by_currency.get(quote, 0.0))
                    score = _clamp(base_bias - quote_bias)
                else:
                    score = _clamp(float(entry.get("score", 0.0)))
                mtime = p.stat().st_mtime
                ages["cb_bias"] = max(0.0, _now_ts() - mtime)
                sources_used.append("cb_bias")
                ok = True
            except Exception as exc:
                sources_failed.append(f"cb_bias_yaml:{exc}")

        payload = {
            "score": score if ok else 0.0,
            "sources_used": sources_used,
            "sources_failed": sources_failed,
            "ages": ages,
        }
        _cache_set(cache_key, payload, ttl)
        return payload["score"], sources_used, sources_failed, ages

    def get_calendar_score(
        self, symbol: str
    ) -> Tuple[float, List[str], List[str], Dict[str, float], bool]:
        cfg = self.src_cfg.get("calendar", {})
        window_back = int(cfg.get("window_back_hours", 4))
        window_fwd = int(cfg.get("window_forward_hours", 24))
        now = datetime.now(timezone.utc)
        frm = (now - timedelta(hours=window_back)).strftime("%Y-%m-%d")
        to = (now + timedelta(hours=window_fwd)).strftime("%Y-%m-%d")

        events: List[Dict[str, Any]] = []
        sources_used: List[str] = []
        sources_failed: List[str] = []
        ages: Dict[str, float] = {}

        finnhub_key = str(self.config.get("secrets", {}).get("FINNHUB_API_KEY", "")) or ""
        if not finnhub_key:
            import os
            finnhub_key = os.getenv("FINNHUB_API_KEY", "")

        if cfg.get("finnhub_enabled", True) and finnhub_key:
            try:
                url = "https://finnhub.io/api/v1/calendar/economic"
                resp = requests.get(
                    url,
                    params={"token": finnhub_key, "from": frm, "to": to},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                events = resp.json().get("economicCalendar", [])
                sources_used.append("calendar")
                ages["calendar"] = 0.0
            except Exception as exc:
                sources_failed.append(f"calendar_finnhub:{exc}")

        if not events and cfg.get("scraping_enabled", True):
            try:
                resp = requests.get("https://www.investing.com/economic-calendar/", timeout=self.timeout)
                resp.raise_for_status()
                events = _parse_investing_calendar(resp.text)
                sources_used.append("calendar")
                ages["calendar"] = 0.0
            except Exception as exc:
                sources_failed.append(f"calendar_investing:{exc}")

        if not events:
            return 0.0, sources_used, sources_failed, ages, False

        base, quote = _fx_base_quote(symbol)
        base_score = _calendar_score_for_currency(events, base, now)
        quote_score = _calendar_score_for_currency(events, quote, now)
        score = _clamp(base_score - quote_score)
        caution = any(
            str(ev.get("impact", "")).lower() == "high"
            for ev in events
        )
        return score, sources_used, sources_failed, ages, caution

    def get_dxy_score(self, symbol: str) -> Tuple[float, List[str], List[str], Dict[str, float]]:
        cfg = self.src_cfg.get("dxy", {})
        cache_key = "fa:dxy"
        ttl = int(cfg.get("cache_ttl_seconds", 3600))
        hit = _cache_get(cache_key)
        if hit:
            d = hit["data"]
            mapped = _map_usd_strength_to_symbol(symbol, d["usd_strength"])
            return mapped, d["sources_used"], d["sources_failed"], d["ages"]

        sources_used: List[str] = []
        sources_failed: List[str] = []
        ages: Dict[str, float] = {}
        usd_strength = 0.0
        ok = False

        try:
            url = cfg.get("stooq_url", "https://stooq.com/q/d/l/?s=dxy.f&i=d")
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            if len(df) >= 6:
                close = pd.to_numeric(df["Close"], errors="coerce").dropna()
                roc5 = float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6])
                usd_strength = _clamp(roc5 * 10.0)
                sources_used.append("dxy")
                ages["dxy"] = 0.0
                ok = True
        except Exception as exc:
            sources_failed.append(f"dxy_stooq:{exc}")

        if not ok:
            try:
                proxy_symbol = cfg.get("proxy_symbol", "XAUUSD")
                df = market_data_store.get_df(proxy_symbol, "1h", max_age_seconds=7200, with_indicators=False)
                if df is None or df.empty or len(df) < 20:
                    raise ValueError("proxy market data unavailable")
                pct = float((df["close"].iloc[-1] - df["close"].iloc[-20]) / df["close"].iloc[-20])
                usd_strength = _clamp(-pct * 8.0)  # inverse of gold
                sources_used.append("dxy")
                ages["dxy"] = 0.0
                ok = True
            except Exception as exc:
                sources_failed.append(f"dxy_proxy:{exc}")

        payload = {
            "usd_strength": usd_strength if ok else 0.0,
            "sources_used": sources_used,
            "sources_failed": sources_failed,
            "ages": ages,
        }
        _cache_set(cache_key, payload, ttl)
        mapped = _map_usd_strength_to_symbol(symbol, payload["usd_strength"])
        return mapped, sources_used, sources_failed, ages

    def get_sentiment_score(self, symbol: str) -> Tuple[float, List[str], List[str], Dict[str, float]]:
        cfg = self.src_cfg.get("sentiment", {})
        cache_key = "fa:sentiment"
        ttl = int(cfg.get("cache_ttl_seconds", 1800))
        hit = _cache_get(cache_key)
        if hit:
            d = hit["data"]
            score = float(d["score"])
        else:
            score = 0.0
            sources_used: List[str] = []
            sources_failed: List[str] = []
            ages: Dict[str, float] = {}
            try:
                url = cfg.get(
                    "cnn_url",
                    "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                )
                resp = requests.get(url, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json() or {}
                now_section = data.get("fear_and_greed", {})
                val = now_section.get("score")
                if val is None:
                    val = now_section.get("value")
                score = _clamp((float(val) - 50.0) / 50.0)
                sources_used.append("sentiment")
                ages["sentiment"] = 0.0
            except Exception as exc:
                sources_used = []
                sources_failed = [f"sentiment_cnn:{exc}"]
                ages = {}
            payload = {
                "score": score,
                "sources_used": sources_used,
                "sources_failed": sources_failed,
                "ages": ages,
            }
            _cache_set(cache_key, payload, ttl)
            d = payload

        safe_havens = {s.upper() for s in cfg.get("safe_haven_symbols", ["XAUUSD", "USDJPY", "USDCHF"])}
        final_score = -score if symbol.upper() in safe_havens else score
        return final_score, d.get("sources_used", []), d.get("sources_failed", []), d.get("ages", {})
