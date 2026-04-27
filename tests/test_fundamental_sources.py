from analysis.fundamental_analyst import FundamentalAnalyst
from analysis.fundamental_sources import FundamentalSources


def _cfg():
    return {
        "fundamental_analysis": {
            "enabled": True,
            "weights": {"cb_bias": 0.35, "calendar": 0.25, "dxy": 0.2, "sentiment": 0.2},
            "sources": {
                "cb_bias": {"yaml_fallback_path": "data/fundamentals/cb_bias.yaml"},
                "calendar": {"finnhub_enabled": False, "scraping_enabled": False},
                "dxy": {"cache_ttl_seconds": 1},
                "sentiment": {"safe_haven_symbols": ["XAUUSD", "USDJPY", "USDCHF"]},
            },
        }
    }


def test_cb_bias_yaml_fallback_returns_source():
    fs = FundamentalSources(_cfg())
    score, used, failed, ages = fs.get_cb_bias("EURUSD")
    assert isinstance(score, float)
    assert "cb_bias" in used
    assert "cb_bias" in ages
    assert not failed or failed[0].startswith("cb_bias_scrape")


def test_analyse_live_populates_components_and_sources(monkeypatch):
    fa = FundamentalAnalyst(_cfg())
    monkeypatch.setattr(fa.sources, "get_cb_bias", lambda s: (0.4, ["cb_bias"], [], {"cb_bias": 120.0}))
    monkeypatch.setattr(
        fa.sources,
        "get_calendar_score",
        lambda s: (0.2, ["calendar"], [], {"calendar": 60.0}, False),
    )
    monkeypatch.setattr(fa.sources, "get_dxy_score", lambda s: (0.1, ["dxy"], [], {"dxy": 300.0}))
    monkeypatch.setattr(
        fa.sources, "get_sentiment_score", lambda s: (-0.3, ["sentiment"], [], {"sentiment": 30.0})
    )

    ctx = fa.analyse_live("EURUSD", "BUY")
    assert ctx.sources_used == ["calendar", "cb_bias", "dxy", "sentiment"]
    assert "Composite score" in ctx.narrative
    assert set(ctx.components.keys()) == {"cb_bias", "calendar", "dxy", "sentiment"}


def test_sentiment_safe_haven_inversion(monkeypatch):
    fs = FundamentalSources(_cfg())

    class DummyResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"fear_and_greed": {"score": 20}}

    monkeypatch.setattr("analysis.fundamental_sources.requests.get", lambda *a, **k: DummyResp())
    xau, *_ = fs.get_sentiment_score("XAUUSD")
    eur, *_ = fs.get_sentiment_score("EURUSD")
    assert xau > 0
    assert eur < 0
