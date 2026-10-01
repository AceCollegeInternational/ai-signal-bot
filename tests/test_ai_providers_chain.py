"""Provider chain tests: DeepSeek support and switchable primary (no network, no real SDKs)."""

import json
import time

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


def _hanging_deepseek(seconds, calls):
    """Provider class whose API call blocks for `seconds` (simulates a stalled network call)."""
    class Hang(ase.AIProvider):
        def __init__(self, *a, **k):
            pass

        def generate_content(self, system_prompt, user_prompt):
            calls.append("deepseek")
            time.sleep(seconds)
            return GOOD
    return Hang


def _failure_logger(env):
    warnings = []
    env.setattr(ase.log, "warning", lambda msg, *a, **k: warnings.append(str(msg)))
    return warnings


def test_deepseek_hang_times_out_within_32s_and_falls_through(env):
    """API hangs 35s: the engine must give up in <32s, log 'deepseek failed:' and use Groq."""
    calls = []
    warnings = _failure_logger(env)
    engine, _ = build(env, "deepseek", calls=calls)
    env.setattr(ase, "PROVIDER_TIMEOUT_S", 30.0)
    env.setattr(ase, "HARD_DEADLINE_S", 31.0)
    engine.providers["deepseek"] = _hanging_deepseek(35, calls)()
    start = time.monotonic()
    sig = engine.get_signal({}, "EURUSD", "1h")
    elapsed = time.monotonic() - start
    assert elapsed < 32, f"engine blocked for {elapsed:.1f}s"
    assert any(w.startswith("deepseek failed:") for w in warnings)
    assert calls == ["deepseek", "groq"] and sig.provider == "groq" and sig.signal == "BUY"


def test_deepseek_timeout_exception_falls_through(env):
    """An SDK Timeout error is logged as 'deepseek failed:' and the chain continues."""
    class TimeoutErr(Exception):
        pass

    calls = []
    warnings = _failure_logger(env)
    engine, _ = build(env, "deepseek", calls=calls)

    class Raises(ase.AIProvider):
        def generate_content(self, system_prompt, user_prompt):
            calls.append("deepseek")
            raise TimeoutErr("Request timed out.")

    engine.providers["deepseek"] = Raises()
    assert engine.get_signal({}, "EURUSD", "1h").provider == "groq"
    assert any(w.startswith("deepseek failed:") and "timed out" in w for w in warnings)


def test_sdk_clients_are_configured_with_timeouts(env):
    """Real Groq/DeepSeek/Gemini wrappers pass a 30s timeout and disable retries."""
    pytest.importorskip("openai")
    env.setenv("DEEPSEEK_API_KEY", "sk-test")
    p = ase.DeepSeekProvider("sk-test", "deepseek-chat", 1000, 0.1)
    assert p.client.timeout == ase.PROVIDER_TIMEOUT_S == 30.0 and p.client.max_retries == 0
    groq = pytest.importorskip("groq")
    g = ase.GroqProvider("gsk-test", "llama3-70b-8192", 1000, 0.1)
    assert g.client.timeout == 30.0 and g.client.max_retries == 0


def test_gemini_call_passes_request_timeout(env):
    seen = {}

    class FakeModel:
        def generate_content(self, prompt, **kw):
            seen.update(kw)
            class R: text = GOOD
            return R()

    g = ase.GeminiProvider.__new__(ase.GeminiProvider)
    g.model, g.has_sys_prompt = FakeModel(), True
    assert g.generate_content("sys", "user") == GOOD
    assert seen["request_options"]["timeout"] == 30.0


# ─── extract_json ─────────────────────────────────────────────────────────────

def test_extract_json_fenced():
    assert ase.extract_json('```json\n{"signal": "BUY", "confidence": 80}\n```') == {"signal": "BUY", "confidence": 80}
    assert ase.extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_raw():
    assert ase.extract_json('{"signal": "HOLD"}') == {"signal": "HOLD"}


def test_extract_json_prose_before_and_after():
    text = 'Sure! Here is my analysis:\n```json\n{"signal": "SELL", "n": {"x": 1}}\n```\nHope that helps.'
    assert ase.extract_json(text) == {"signal": "SELL", "n": {"x": 1}}


def test_extract_json_empty_raises():
    for bad in ("", None):
        with pytest.raises(ValueError, match="Empty response"):
            ase.extract_json(bad)


def test_extract_json_no_object_raises():
    with pytest.raises(ValueError, match="No JSON object found"):
        ase.extract_json("I cannot provide a signal right now.")


def test_parse_response_handles_fenced_reply_and_logs_raw_on_failure(env):
    debug = []
    env.setattr(ase.log, "debug", lambda msg, *a, **k: debug.append(str(msg)))
    engine, _ = build(env, "groq")
    sig = engine._parse_response("```json\n" + GOOD + "\n```", "EURUSD", "1h")
    assert sig.signal == "BUY" and sig.confidence == 80
    bad = engine._parse_response('```json\n{"signal": "BUY", "confidence": ', "EURUSD", "1h")  # truncated reply
    assert bad.signal == "HOLD"
    assert any("[AI] Raw response that failed parsing:" in d for d in debug)
