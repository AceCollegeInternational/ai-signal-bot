from types import SimpleNamespace

from fastapi.testclient import TestClient

from analysis.fundamental_analyst import compute_fundamental_rating
from modules.alerting import AlertingEngine
from modules.execution_server import app, pending_signals


def _base_rating_config() -> dict:
    return {
        "rating": {
            "approval_threshold": 0.30,
            "strong_threshold": 0.60,
            "moderate_threshold": 0.30,
        }
    }


def test_rating_buy_alignment():
    rating, conviction, _, _ = compute_fundamental_rating(
        composite_score=0.65,
        signal_direction="BUY",
        sources_used=["cb_bias", "calendar"],
        data_age_seconds={"cb_bias": 60.0},
        config=_base_rating_config(),
    )
    assert rating == 1
    assert conviction == "strong"


def test_rating_sell_conflict():
    rating, conviction, _, _ = compute_fundamental_rating(
        composite_score=0.65,
        signal_direction="SELL",
        sources_used=["cb_bias", "calendar"],
        data_age_seconds={"cb_bias": 60.0},
        config=_base_rating_config(),
    )
    assert rating == -1
    assert conviction == "strong"


def test_rating_weak_neutral():
    rating, conviction, _, _ = compute_fundamental_rating(
        composite_score=0.15,
        signal_direction="BUY",
        sources_used=[],
        data_age_seconds={},
        config=_base_rating_config(),
    )
    assert rating == 0
    assert conviction == "weak"


def test_note_contains_age():
    _, _, note, _ = compute_fundamental_rating(
        composite_score=0.65,
        signal_direction="BUY",
        sources_used=["cb_bias"],
        data_age_seconds={"cb_bias": 3720.0},
        config=_base_rating_config(),
        components={"cb_bias": 0.65},
    )
    assert "1h 2m" in note


def test_note_unavailable_when_no_sources():
    _, _, note, _ = compute_fundamental_rating(
        composite_score=0.0,
        signal_direction="BUY",
        sources_used=[],
        data_age_seconds={},
        config=_base_rating_config(),
    )
    assert note == "Fundamental data unavailable. Technical signal unaffected."


def test_telegram_block_skipped_when_disabled():
    engine = AlertingEngine(
        {
            "alerts": {"telegram_enabled": False},
            "fundamental_analysis": {"enabled": False},
        }
    )
    signal = SimpleNamespace(
        symbol="EURUSD",
        signal="BUY",
        confidence=80,
        entry_price=1.1,
        stop_loss=1.0,
        take_profit_1=1.2,
        take_profit_2=1.3,
        risk_reward_ratio=2.0,
        hold_duration="INTRADAY",
        hold_reasoning="trend continuation",
        reasoning="test reason",
    )
    text = engine.format_signal_message(signal)
    assert "FUNDAMENTAL ANALYSIS" not in text


def test_poll_response_omits_fields_when_disabled():
    client = TestClient(app)
    pending_signals.clear()

    pending_signals["EURUSD"] = SimpleNamespace(
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.0845,
        stop_loss=1.0810,
        take_profit=1.0910,
        confidence=80.0,
        reasoning="test",
        timestamp=0.0 + __import__("time").time(),
        fundamental_rating=None,
        fundamental_conviction=None,
        fundamental_note=None,
    )

    data = client.get("/poll/EURUSD").json()
    assert data["status"] == "new_trade"
    assert "fundamental_rating" not in data


def _apply_fundamental_filter(
    rating: int,
    conviction: str,
    mode: str,
    block_threshold: int,
    volume: float,
    volume_min: float = 0.01,
) -> tuple[bool, float]:
    """Python simulation of EA filter logic for unit testing."""
    should_place = True
    final_volume = volume
    if rating == -1:
        conviction_level = 0
        if conviction == "moderate":
            conviction_level = 1
        elif conviction == "strong":
            conviction_level = 2

        if mode == "HARD" and conviction_level >= block_threshold:
            should_place = False
        elif mode == "SOFT":
            final_volume = max(volume_min, round(volume * 0.5, 2))

    return should_place, final_volume


def test_ea_hard_block_strong_conviction():
    should_place, _ = _apply_fundamental_filter(
        rating=-1,
        conviction="strong",
        mode="HARD",
        block_threshold=0,
        volume=0.10,
    )
    assert should_place is False


def test_ea_soft_mode_halves_lot():
    should_place, final_volume = _apply_fundamental_filter(
        rating=-1,
        conviction="moderate",
        mode="SOFT",
        block_threshold=0,
        volume=0.10,
    )
    assert should_place is True
    assert final_volume == 0.05


def test_ea_hard_block_bypassed_on_weak_conviction():
    should_place, final_volume = _apply_fundamental_filter(
        rating=-1,
        conviction="weak",
        mode="HARD",
        block_threshold=1,
        volume=0.10,
    )
    assert should_place is True
    assert final_volume == 0.10

