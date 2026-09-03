"""Covers mlb_props_main.run_lineup_diagnostic - the --lineup-diagnostic
that reports each of --date's real games' confirmed-lineup status, since
schedule.py's MlbStatsApiScheduleProvider._confirmed_lineup_batters reads
a real boxscore field (`battingOrder`) never confirmed against a real
response (see that method's docstring).
"""

from datetime import date

from mlb_props.schedule import ProbableMatchup
from mlb_props_main import run_lineup_diagnostic


class _FakeScheduleProvider:
    def __init__(self, matchups):
        self._matchups = matchups

    def get_slate(self, game_date):
        return self._matchups


def _patch_schedule(monkeypatch, matchups):
    monkeypatch.setattr("mlb_props_main.MlbStatsApiScheduleProvider", lambda include_rosters=True: _FakeScheduleProvider(matchups))


def test_reports_no_games_clearly(monkeypatch):
    _patch_schedule(monkeypatch, [])

    text = run_lineup_diagnostic(date(2026, 8, 29))

    assert "0 real game(s)" in text


def test_reports_confirmed_vs_active_roster_counts(monkeypatch):
    matchups = [
        ProbableMatchup(
            away_team="Team A", home_team="Team B", venue="Park One", away_pitcher="P1", home_pitcher="P2",
            game_time_utc="2026-08-29T23:00:00Z", away_batters=["Real Starter"], home_batters=["Real Starter 2"],
            lineup_source="confirmed",
        ),
        ProbableMatchup(
            away_team="Team C", home_team="Team D", venue="Park Two", away_pitcher="P3", home_pitcher="P4",
            game_time_utc="2026-08-30T02:00:00Z", away_batters=["Roster Guy"] * 9, home_batters=["Roster Guy"] * 9,
            lineup_source="active_roster",
        ),
    ]
    _patch_schedule(monkeypatch, matchups)

    text = run_lineup_diagnostic(date(2026, 8, 29))

    assert "2 real game(s)" in text
    assert "1 of 2 game(s) had a real confirmed lineup" in text
    assert "lineup_source=confirmed" in text
    assert "lineup_source=active_roster" in text
    assert "Team A @ Team B" in text
