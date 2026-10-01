"""Integration tests for the MySQL layer, analytics engine, gate and API.

Run only against a disposable database: set FXGURU_TEST_DB=1 plus DB_* env vars.
The tests TRUNCATE the signal/trade/analytics tables.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("FXGURU_TEST_DB") != "1", reason="set FXGURU_TEST_DB=1 with a disposable DB_* to run"
)

from fastapi.testclient import TestClient  # noqa: E402

from analytics import engine  # noqa: E402
from db import repository  # noqa: E402
from db.connection import get_db  # noqa: E402
from db.schema import init_schema  # noqa: E402
from modules.execution_server import app  # noqa: E402

H = {"X-API-KEY": os.getenv("EXECUTION_BRIDGE_KEY", "default_secret_key")}


@pytest.fixture(autouse=True)
def clean_db():
    """Start every test from empty tables."""
    init_schema()
    with get_db() as cur:
        cur.execute("SET FOREIGN_KEY_CHECKS=0")
        for t in ("trades", "signals", "performance_daily", "performance_by_symbol", "factor_effectiveness"):
            cur.execute(f"TRUNCATE TABLE {t}")
        cur.execute("SET FOREIGN_KEY_CHECKS=1")


def _seed(symbol, n, win_when_sweep=True, score=85.0, hour=9):
    """Insert n closed trades: sweep trades win, non-sweep trades lose."""
    for i in range(n):
        sweep = i % 2 == 0
        sid, tid = repository.insert_signal_and_trade(
            {"symbol": symbol, "direction": "LONG", "entry_price": 1.1, "stop_loss": 1.095,
             "take_profit": 1.11, "confidence": score, "liquidity_sweep": int(sweep)}, {}, source="LLM")
        win = sweep if win_when_sweep else not sweep
        with get_db() as cur:
            cur.execute("UPDATE trades SET opened_at = CONCAT(DATE(opened_at), ' ', %s, ':00:00') WHERE id=%s", (f"{hour:02d}", tid))
        repository.update_trade_outcome(tid, 1.105 if win else 1.095, "TP1" if win else "SL", None, 50 if win else -50)


def test_requires_auth():
    c = TestClient(app)
    assert c.get("/trades").status_code == 403
    assert c.get("/analytics/performance").status_code == 403


def test_inject_list_and_outcome():
    with TestClient(app) as c:
        r = c.post("/debug/signal/inject", headers=H, json={
            "symbol": "EURUSD", "direction": "BUY", "entry_price": 1.1, "stop_loss": 1.095,
            "take_profit": 1.11, "confidence": 88})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["signal_id"] and body["trade_id"]
        sigs = c.get("/signals?symbol=EURUSD&tier=TIER_2", headers=H).json()
        assert sigs["total"] == 1 and sigs["items"][0]["signal_source"] == "INJECTED"
        out = c.put(f"/trades/{body['trade_id']}/outcome", headers=H,
                    json={"exit_price": 1.11, "exit_reason": "TP1", "profit_loss_usd": 100})
        assert out.status_code == 200 and out.json()["status"] == "CLOSED_WIN"
        assert out.json()["pips_gained"] == 100.0 and out.json()["rr_achieved"] == 2.0
        assert c.put("/trades/99999/outcome", headers=H, json={"exit_price": 1, "exit_reason": "SL"}).status_code == 404
        tr = c.get("/trades?status=closed_win&date_from=2000-01-01", headers=H).json()
        assert tr["total"] == 1
        assert c.get("/trades?date_from=bogus", headers=H).status_code == 422
        # outcome triggered an analytics refresh for the symbol
        assert c.get("/analytics/performance/EURUSD", headers=H).json()["wins"] == 1


def test_rollback_on_multi_table_failure():
    with pytest.raises(Exception):
        repository.insert_signal_and_trade(
            {"symbol": "EURUSD", "direction": "LONG", "entry_price": 1.1, "stop_loss": 1.0}, {"status": "BOGUS"})
    assert repository.list_signals()["total"] == 0  # signal insert rolled back with the failed trade


def test_factor_effectiveness_and_report():
    _seed("EURUSD", 20)
    with TestClient(app) as c:
        factors = c.get("/analytics/factors", headers=H).json()["factors"]
        sweep = next(f for f in factors if f["factor_name"] == "liquidity_sweep")
        assert sweep["win_rate_when_present"] == 100.0 and sweep["win_rate_when_absent"] == 0.0
        assert factors[0]["weight_adjustment"] == 1.0
        rep = c.get("/analytics/performance", headers=H).json()
        assert rep["overall"]["win_rate_pct"] == 50.0 and rep["best_symbol"] == "EURUSD"
        assert rep["most_reliable_factor"]["factor_name"] == "liquidity_sweep"
        with get_db() as cur:
            cur.execute("SELECT trades_taken FROM performance_daily")
            assert cur.fetchone()["trades_taken"] == 20


def test_adaptive_threshold_default_and_learned():
    _seed("GBPUSD", 5)
    assert engine.compute_adaptive_score_threshold("GBPUSD") == 75.0  # <20 trades
    _seed("USDJPY", 10, score=70.0)
    _seed("USDJPY", 10, score=90.0, win_when_sweep=True)
    with get_db() as cur:  # make all low-score trades lose, high-score trades win
        cur.execute("UPDATE trades t JOIN signals s ON s.id=t.signal_id SET t.status="
                    "IF(s.confluence_score>=90,'CLOSED_WIN','CLOSED_LOSS') WHERE t.symbol='USDJPY'")
    assert engine.compute_adaptive_score_threshold("USDJPY") == 90.0


def test_gate():
    _seed("EURUSD", 20, win_when_sweep=False)  # 50% overall -> enabled
    assert engine.should_take_signal({"symbol": "EURUSD", "confidence": 60})[0] is False
    ok, why = engine.should_take_signal({"symbol": "EURUSD", "confidence": 80, "hour_utc": 3})
    assert ok and "Passed" in why
    # unknown symbol -> defaults
    assert engine.should_take_signal({"symbol": "AUDUSD", "confidence": 74.9})[0] is False
    assert engine.should_take_signal({"symbol": "AUDUSD", "confidence": 75})[0] is True
    # pause: >=20 trades, win rate < 40
    with get_db() as cur:
        cur.execute("UPDATE trades SET status='CLOSED_LOSS' WHERE symbol='EURUSD'")
    engine.compute_symbol_performance("EURUSD")
    ok, why = engine.should_take_signal({"symbol": "EURUSD", "confidence": 99})
    assert (ok, why) == (False, "Symbol paused by analytics")


def test_gate_session_and_factor_penalty():
    _seed("EURUSD", 20, hour=3)
    _seed("EURUSD", 20, hour=9, win_when_sweep=False)
    engine.compute_optimal_session_times("EURUSD")
    times = engine.compute_optimal_session_times("EURUSD")
    assert set(times["best_hours_utc"]) | set(times["worst_hours_utc"]) <= {3, 9}
    with get_db() as cur:
        cur.execute("UPDATE performance_by_symbol SET worst_hours_utc='4', recommended_min_score=75 WHERE symbol='EURUSD'")
    ok, why = engine.should_take_signal({"symbol": "EURUSD", "confidence": 90, "hour_utc": 4})
    assert not ok and "Unfavourable session" in why
    # weak factor (sweep wins 50% overall here, so force a weak row) penalises a borderline score
    with get_db() as cur:
        cur.execute("INSERT INTO factor_effectiveness (factor_name, present_in_wins, present_in_losses, win_rate_when_present) "
                    "VALUES ('liquidity_sweep', 2, 18, 10) ON DUPLICATE KEY UPDATE win_rate_when_present=10, present_in_losses=18")
    ok, why = engine.should_take_signal({"symbol": "EURUSD", "confidence": 77, "liquidity_sweep": 1, "hour_utc": 5})
    assert not ok and "weak factors" in why


def test_gate_rejected_signal_is_logged_not_executed():
    sid = repository.insert_signal({"symbol": "EURUSD", "signal": "BUY", "confidence": 60,
                                    "entry_price": 1.1, "stop_loss": 1.09}, "LLM", True, "Score below adaptive threshold")
    row = repository.list_signals(gate_rejected=True)["items"][0]
    assert row["id"] == sid and row["gate_rejected"] == 1


def test_db_unavailable_returns_503(monkeypatch):
    from db import connection
    connection.reset_pool()
    monkeypatch.setenv("DB_PORT", "1")  # nothing listens here
    monkeypatch.setenv("DB_CONNECT_TIMEOUT", "1")
    try:
        c = TestClient(app, raise_server_exceptions=False)
        assert c.get("/trades", headers=H).status_code == 503
        assert engine.should_take_signal({"symbol": "EURUSD", "confidence": 80})[0] is True  # fail-open
        # signal insert degrades to the file fallback
        repository.FALLBACK_PATH, old = os.path.join(os.getenv("TMPDIR", "/tmp"), "fb_test.jsonl"), repository.FALLBACK_PATH
        try:
            assert repository.insert_signal({"symbol": "EURUSD", "signal": "BUY", "entry_price": 1}) is None
            assert os.path.exists(repository.FALLBACK_PATH)
        finally:
            os.path.exists(repository.FALLBACK_PATH) and os.remove(repository.FALLBACK_PATH)
            repository.FALLBACK_PATH = old
    finally:
        monkeypatch.undo()
        connection.reset_pool()
