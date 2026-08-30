"""Covers mlb_props_main.run_game_status_check - the --game-status-check
diagnostic that cross-checks MLB's own authoritative real-time game status,
straight from the MLB Stats API, independent of The Odds API's commence_time
(confirmed live 2026-08-29 to be wrong for at least one real game - see this
function's docstring).
"""

from datetime import date
from unittest.mock import patch

from mlb_props_main import run_game_status_check


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.requested = []

    def get(self, url, params=None, timeout=None):
        self.requested.append((url, params))
        return _FakeResponse(self.payload)


def _patch_session(monkeypatch, payload):
    monkeypatch.setattr("mlb_props_main.build_retrying_session", lambda: _FakeSession(payload))


def test_no_games_reports_clearly(monkeypatch):
    _patch_session(monkeypatch, {"dates": []})

    text = run_game_status_check(date(2026, 8, 29))

    assert "No games found" in text


def test_lists_final_in_progress_and_scheduled_games(monkeypatch):
    payload = {
        "dates": [
            {
                "games": [
                    {
                        "teams": {
                            "away": {"team": {"name": "Kansas City Royals"}},
                            "home": {"team": {"name": "Cleveland Guardians"}},
                        },
                        "status": {"detailedState": "In Progress"},
                        "linescore": {"currentInningOrdinal": "7th", "inningState": "Top"},
                    },
                    {
                        "teams": {
                            "away": {"team": {"name": "Boston Red Sox"}},
                            "home": {"team": {"name": "New York Yankees"}},
                        },
                        "status": {"detailedState": "Final"},
                    },
                    {
                        "teams": {
                            "away": {"team": {"name": "Arizona Diamondbacks"}},
                            "home": {"team": {"name": "San Francisco Giants"}},
                        },
                        "status": {"detailedState": "Scheduled"},
                    },
                ]
            }
        ]
    }
    _patch_session(monkeypatch, payload)

    text = run_game_status_check(date(2026, 8, 29))

    assert "Kansas City Royals @ Cleveland Guardians: In Progress (Top 7th)" in text
    assert "Boston Red Sox @ New York Yankees: Final" in text
    assert "Arizona Diamondbacks @ San Francisco Giants: Scheduled" in text


def test_unparsable_game_entry_does_not_crash_the_whole_check(monkeypatch):
    payload = {"dates": [{"games": [{"gamePk": 12345}]}]}  # missing "teams" entirely
    _patch_session(monkeypatch, payload)

    text = run_game_status_check(date(2026, 8, 29))

    assert "unparsable" in text
    assert "12345" in text
