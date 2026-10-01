"""Provider chain tests: DeepSeek support and switchable primary (no network, no real SDKs)."""

import json

import pytest

from modules import ai_signal_engine as ase

GOOD = json.dumps({"signal": "BUY", "confidence": 80, "entry_price": 1.1, "stop_loss": 1.09,
                   "take_profit_1": 1.12, "take_profit_2": 1.13, "risk_reward_ratio": 2.0})


def make_fake(name, calls, fail=False):
    """Build a stand-in provider class that records calls and optionally raises."""
    class Fake(ase.AIProvider):
        def __init__(self, api_key, model, max_tokens, temperature):
            self.model, self.temperature = model, temperature

        def generate_content(self, system_prompt, user_prompt):
            calls.append(name)
            if fail:
                raise RuntimeError(f"{name} boom")
            return GOOD
    return Fake


@pytest.fixture
def env(monkeypatch):
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "AI_PRIMARY",
              "DEEPSEEK_MODEL"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def build(monkeypatch, primary, keys=("GROQ_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"), failing=(), calls=None):
    calls = calls if calls is not None else []
    for k in keys:
        monkeypatch.setenv(k, "k-" + k.lower())
    monkeypatch.setattr(ase, "GroqProvider", make_fake("groq", calls, "groq" in failing))
    monkeypatch.setattr(ase, "GeminiProvider", make_fake("gemini", calls, "gemini" in failing))
    monkeypatch.setattr(ase, "DeepSeekProvider", make_fake("deepseek", calls, "deepseek" in failing))
    return ase.AISignalEngine({"ai": {"primary_provider": primary, "fallback_enabled": True}}), calls


def test_deepseek_skipped_when_key_missing(env):
    warnings = []
    env.setattr(ase.log, "warning", lambda msg, *a, **k: warnings.append(str(msg)))
    engine, _ = build(env, "groq", keys=("GROQ_API_KEY", "GEMINI_API_KEY"))
    assert "deepseek" not in engine.providers
    assert engine.provider_order() == ["groq", "gemini"]
    assert any("DeepSeek provider not configured — DEEPSEEK_API_KEY missing." in w for w in warnings)


def test_order_when_primary_deepseek(env):
    engine, _ = build(env, "deepseek")
    assert engine.provider_order() == ["deepseek", "groq", "gemini"]


def test_order_when_primary_groq(env):
    engine, _ = build(env, "groq")
    assert engine.provider_order() == ["groq", "deepseek", "gemini"]


def test_order_when_primary_gemini(env):
    engine, _ = build(env, "gemini")
    assert engine.provider_order() == ["gemini", "deepseek", "groq"]


def test_ai_primary_env_overrides_config(env):
    env.setenv("AI_PRIMARY", "deepseek")
    engine, _ = build(env, "groq")
    assert engine.primary == "deepseek" and engine.provider_order()[0] == "deepseek"


def test_deepseek_failure_falls_through_to_groq(env):
    engine, calls = build(env, "deepseek", failing=("deepseek",))
    sig = engine.get_signal({}, "EURUSD", "1h")
    assert calls == ["deepseek", "groq"]
    assert sig.signal == "BUY" and sig.provider == "groq"


def test_all_fail_returns_hold(env):
    engine, calls = build(env, "deepseek", failing=("deepseek", "groq", "gemini"))
    assert engine.get_signal({}, "EURUSD", "1h").signal == "HOLD"
    assert calls == ["deepseek", "groq", "gemini"]


def test_real_deepseek_provider_wiring(env):
    """The real provider targets DeepSeek's base_url and honours DEEPSEEK_MODEL/temperature."""
    pytest.importorskip("openai")
    env.setenv("DEEPSEEK_API_KEY", "sk-test")
    env.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    engine = ase.AISignalEngine({"ai": {"primary_provider": "deepseek"}})
    p = engine.providers["deepseek"]
    assert str(p.client.base_url).startswith("https://api.deepseek.com")
    assert p.model == "deepseek-reasoner" and p.temperature == 0.1
