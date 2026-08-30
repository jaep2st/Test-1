"""Covers mlb_props_main.run_live_odds_scan - the --live-odds-scan
diagnostic that looks for real cross-book price value in already-started
games' odds, without ever comparing them to this model's pregame-only
score (see odds_monitor/providers/theoddsapi.py's module docstring for
why that comparison would be invalid).
"""

import datetime as dt

from mlb_props_main import run_live_odds_scan
from odds_monitor.providers.theoddsapi import TheOddsApiProvider


class _FakeResponse:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self.responses = responses

    def get(self, url, params=None, timeout=None):
        for path_suffix, payload in self.responses.items():
            if url.endswith(path_suffix):
                return _FakeResponse(payload)
        raise AssertionError(f"Unexpected URL requested: {url}")


def _past_iso(hours=1):
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _patch_provider(monkeypatch, responses):
    # run_live_odds_scan constructs its own TheOddsApiProvider internally -
    # patch the class so it uses our fake session instead of a real one.
    real_init = TheOddsApiProvider.__init__

    def fake_init(self, api_key=None, **kwargs):
        real_init(self, api_key=api_key, session=_FakeSession(responses))

    monkeypatch.setattr(TheOddsApiProvider, "__init__", fake_init)


def test_no_live_games_reports_zero_lines(monkeypatch):
    _patch_provider(monkeypatch, {"/events": []})

    text = run_live_odds_scan("test-key", books=None)

    assert "0 live-game prop line(s)" in text
    assert "No live-game odds available right now." in text


def test_multi_book_live_prop_shows_cross_book_spread(monkeypatch):
    events_payload = [{"id": "evt1", "home_team": "A", "away_team": "B", "commence_time": _past_iso()}]
    odds_payload = {
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [{"key": "batter_total_bases", "outcomes": [
                    {"name": "Over", "description": "Player X", "point": 1.5, "price": 140},
                ]}],
            },
            {
                "key": "fanduel",
                "markets": [{"key": "batter_total_bases", "outcomes": [
                    {"name": "Over", "description": "Player X", "point": 1.5, "price": 120},
                ]}],
            },
        ]
    }
    _patch_provider(monkeypatch, {"/events": events_payload, "/evt1/odds": odds_payload})

    text = run_live_odds_scan("test-key", books=None)

    assert "1 live-game prop line(s)" not in text  # 2 lines, not 1
    assert "1 distinct live prop(s) total, 1 quoted by 2+ books" in text
    assert "best draftkings=+140, worst fanduel=+120" in text
    assert "spread +20" in text


def test_single_book_live_prop_is_listed_but_not_compared(monkeypatch):
    events_payload = [{"id": "evt1", "home_team": "A", "away_team": "B", "commence_time": _past_iso()}]
    odds_payload = {
        "bookmakers": [
            {
                "key": "betrivers",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Yes", "description": "Player X", "point": 0.5, "price": 1900},
                ]}],
            },
        ]
    }
    _patch_provider(monkeypatch, {"/events": events_payload, "/evt1/odds": odds_payload})

    text = run_live_odds_scan("test-key", books=None)

    assert "0 quoted by 2+ books" in text
    assert "No live prop is currently quoted by more than one book" in text
    assert "betrivers=+1900" in text


def test_pregame_lines_are_excluded_from_the_scan(monkeypatch):
    # A live-odds scan should only report on live lines - a pregame line
    # belongs in the normal report, not here.
    future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events_payload = [{"id": "evt1", "home_team": "A", "away_team": "B", "commence_time": future}]
    odds_payload = {
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Yes", "description": "Player X", "price": 300},
                ]}],
            },
        ]
    }
    _patch_provider(monkeypatch, {"/events": events_payload, "/evt1/odds": odds_payload})

    text = run_live_odds_scan("test-key", books=None)

    assert "0 live-game prop line(s)" in text
