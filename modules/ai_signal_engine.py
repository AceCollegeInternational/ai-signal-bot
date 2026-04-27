"""
modules/ai_signal_engine.py
─────────────────────────────────────────────────────────────────────────────
Module 4 — AI Signal Engine (Multi-Provider Support)

Integrates with multiple AI providers (Anthropic, Google Gemini, Groq)
to generate high-confidence trading signals from market data.

Workflow:
  1. build_payload() — serialize market context into a structured JSON.
  2. get_signal() — send context to the configured provider (with fallbacks).
  3. Parse and validate the structured JSON response.
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# ─── System prompt for all providers ───────────────────
SYSTEM_PROMPT = """You are an expert quantitative technical analyst and algorithmic trader with 20+ years of experience. 
You analyze cryptocurrency market data, technical indicators, and chart patterns to generate precise trading signals.

You will receive a JSON object containing:
- The last 50 OHLCV candles
- Computed technical indicators
- Detected candlestick and chart patterns
- Key support and resistance levels
- SMC Filters: Liquidity Sweeps, Break of Structure (BOS), and Consolidation/Range markers

Your task:
1. Perform a thorough multi-factor technical analysis
2. Pay special attention to institutional flow:
   - LIQUIDITY SWEEPS often mark reversals; avoid buying at the top of a sweep.
   - BREAK OF STRUCTURE (BOS) confirms trend shifts.
   - CONSOLIDATION means high risk of fakeouts; wait for clear expansion.
3. Weigh the evidence from indicators, patterns, and structure
4. Generate a high-precision trading signal

⚠️ CRITICAL: You MUST respond with ONLY a valid JSON object — no markdown, no explanation, no backticks.
The JSON must exactly follow this schema:
{
  "signal": "BUY" or "SELL" or "HOLD",
  "confidence": integer between 0 and 100,
  "entry_price": float,
  "stop_loss": float,
  "take_profit_1": float,
  "take_profit_2": float,
  "reasoning": "concise explanation under 200 words",
  "key_patterns": ["list", "of", "key", "pattern"],
  "risk_reward_ratio": float,
  "hold_duration": "SCALP" or "INTRADAY" or "SWING",
  "hold_reasoning": "brief explanation of expected hold time, e.g. '15-30 min scalp on momentum' or '2-3 day swing on strong trend'"
}

Hold Duration Guidelines:
- SCALP: Hold for minutes to ~1 hour. Use for quick momentum plays, news spikes, or range-bound scalps.
- INTRADAY: Hold for hours within the same trading day. Use when trend is clear but not strong enough for multi-day.
- SWING: Hold for 1-5+ days. Use when there is a strong trend (high ADX), clear breakout, or strong support/resistance setup.
"""


@dataclass
class TradeSignal:
    """Structured trading signal returned by the AI Signal Engine."""

    signal: str = "HOLD"
    confidence: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    reasoning: str = ""
    key_patterns: List[str] = field(default_factory=list)
    risk_reward_ratio: float = 0.0
    hold_duration: str = ""        # "SCALP" | "INTRADAY" | "SWING"
    hold_reasoning: str = ""       # Why this hold duration
    symbol: str = ""
    timeframe: str = ""
    provider: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)

    def is_actionable(self, min_confidence: float = 75.0, min_rr: float = 1.5) -> bool:
        return (
            self.signal in ("BUY", "SELL")
            and self.confidence >= min_confidence
            and self.risk_reward_ratio >= min_rr
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "raw_response"}


# ─── Provider Implementations ──────────────────────────────────────────────────


class AIProvider(ABC):
    @abstractmethod
    def generate_content(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        pass


class AnthropicProvider(AIProvider):
    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate_content(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            temperature=self.temperature,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


class GeminiProvider(AIProvider):
    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
            system_instruction=(
                SYSTEM_PROMPT
                if "system_instruction" in dir(genai.GenerativeModel)
                else None
            ),
        )
        # Note: If system_instruction isn't supported in old versions,
        # we append it to the user prompt in generate_content.
        self.has_sys_prompt = hasattr(self.model, "system_instruction")
        self.raw_sys_prompt = SYSTEM_PROMPT

    def generate_content(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.has_sys_prompt:
            combined_prompt = f"{system_prompt}\n\nUser context:\n{user_prompt}"
            response = self.model.generate_content(combined_prompt)
        else:
            response = self.model.generate_content(user_prompt)
        return response.text


class GroqProvider(AIProvider):
    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float):
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate_content(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return chat_completion.choices[0].message.content


# ─── Main Engine ──────────────────────────────────────────────────────────────


class AISignalEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ai_cfg = config.get("ai", {})

        # Load Providers
        self.providers: Dict[str, AIProvider] = {}
        self._init_providers(ai_cfg)

        self.primary = ai_cfg.get("primary_provider", "gemini")
        self.fallback_enabled = ai_cfg.get("fallback_enabled", True)
        self._no_provider_logged = False

        # Signal thresholds
        sig_cfg = config.get("signals", {})
        self.min_confidence = sig_cfg.get("min_confidence", 75.0)

        log.info(f"AISignalEngine init with primary={self.primary}")

    def _init_providers(self, ai_cfg: Dict[str, Any]):
        # Gemini
        gem_key = os.getenv("GEMINI_API_KEY")
        if gem_key and gem_key != "your_gemini_api_key_here":
            cfg = ai_cfg.get("gemini", {})
            self.providers["gemini"] = GeminiProvider(
                gem_key,
                cfg.get("model", "gemini-2.0-flash"),
                cfg.get("max_tokens", 1000),
                cfg.get("temperature", 0.1),
            )

        # Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and groq_key != "your_groq_api_key_here":
            cfg = ai_cfg.get("groq", {})
            self.providers["groq"] = GroqProvider(
                groq_key,
                cfg.get("model", "llama-3.3-70b-versatile"),
                cfg.get("max_tokens", 1000),
                cfg.get("temperature", 0.1),
            )

        # Claude (Legacy support / Optional)
        ant_key = os.getenv("ANTHROPIC_API_KEY")
        if ant_key and ant_key != "your_anthropic_api_key_here":
            cfg = ai_cfg.get("claude", {})
            self.providers["claude"] = AnthropicProvider(
                ant_key,
                cfg.get("model", "claude-3-5-sonnet-20240620"),
                cfg.get("max_tokens", 1000),
                cfg.get("temperature", 0.1),
            )

    def build_payload(
        self,
        df: pd.DataFrame,
        patterns: List[Any],
        sr_levels: List[float],
        symbol: str = "",
        timeframe: str = "",
        htf_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        window = df.iloc[-50:].copy()
        ohlcv_records = []
        for ts, row in window.iterrows():
            ohlcv_records.append(
                {
                    "t": str(ts),
                    "o": round(row.open, 6),
                    "h": round(row.high, 6),
                    "l": round(row.low, 6),
                    "c": round(row.close, 6),
                    "v": round(row.volume, 2),
                }
            )

        indicators = {}
        last_row = window.iloc[-1]
        for col in window.columns:
            if col not in ["open", "high", "low", "close", "volume"]:
                val = last_row[col]
                indicators[col] = None if pd.isna(val) else round(float(val), 6)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": round(float(last_row.close), 6),
            "candles": ohlcv_records,
            "indicators": indicators,
            "patterns": [p.to_dict() if hasattr(p, "to_dict") else p for p in patterns],
            "sr": [round(l, 6) for l in sr_levels],
            "higher_timeframe_context": htf_context or {},
        }

    def get_signal(
        self, payload: Dict[str, Any], symbol: str = "", timeframe: str = ""
    ) -> TradeSignal:
        ordered_providers = []
        if self.primary in self.providers:
            ordered_providers.append(self.primary)

        if self.fallback_enabled:
            for p in ["gemini", "groq", "claude"]:
                if p in self.providers and p not in ordered_providers:
                    ordered_providers.append(p)

        if not ordered_providers:
            if not self._no_provider_logged:
                log.error("No AI providers configured or available!")
                self._no_provider_logged = True
            return self._hold_signal(symbol, timeframe, "No providers available")

        payload_json = json.dumps(payload, indent=2)

        for provider_name in ordered_providers:
            try:
                log.info(f"Requesting signal via {provider_name}...")
                provider = self.providers[provider_name]
                raw_text = provider.generate_content(SYSTEM_PROMPT, payload_json)

                if not raw_text:
                    continue

                signal = self._parse_response(raw_text, symbol, timeframe)
                signal.provider = provider_name
                return signal

            except Exception as e:
                log.warning(f"{provider_name} failed: {e}")
                continue

        return self._hold_signal(symbol, timeframe, "All providers failed")

    def analyze(
        self,
        df: pd.DataFrame,
        patterns: List[Any],
        sr_levels: List[float],
        symbol: str = "",
        timeframe: str = "",
        htf_context: Optional[Dict[str, Any]] = None,
    ) -> TradeSignal:
        payload = self.build_payload(
            df, patterns, sr_levels, symbol, timeframe, htf_context
        )
        return self.get_signal(payload, symbol, timeframe)

    def _parse_response(self, text: str, symbol: str, timeframe: str) -> TradeSignal:
        # Simple extraction for JSON in markdown fences
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]

        try:
            data = json.loads(text)
            return TradeSignal(
                signal=data.get("signal", "HOLD").upper(),
                confidence=float(data.get("confidence", 0)),
                entry_price=float(data.get("entry_price", 0)),
                stop_loss=float(data.get("stop_loss", 0)),
                take_profit_1=float(data.get("take_profit_1", 0)),
                take_profit_2=float(data.get("take_profit_2", 0)),
                reasoning=data.get("reasoning", ""),
                key_patterns=data.get("key_patterns", []),
                risk_reward_ratio=float(data.get("risk_reward_ratio", 0)),
                hold_duration=data.get("hold_duration", "INTRADAY").upper(),
                hold_reasoning=data.get("hold_reasoning", ""),
                symbol=symbol,
                timeframe=timeframe,
                raw_response=data,
            )
        except Exception as e:
            log.error(f"Parse error: {e}")
            return self._hold_signal(symbol, timeframe, f"Parse failed: {e}")

    def _hold_signal(self, symbol, timeframe, reason) -> TradeSignal:
        return TradeSignal(
            signal="HOLD", symbol=symbol, timeframe=timeframe, reasoning=reason
        )
