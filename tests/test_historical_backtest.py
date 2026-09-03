"""Covers mlb_props/historical_backtest.py - the one piece of this
project's model (hot/cold z-score + clearance rate) that can be checked
against real MLB history without lookahead risk. See that module's
docstring for exactly why the rest of the model can't be, yet.
"""

from datetime import date, timedelta

import pandas as pd

from mlb_props.historical_backtest import (
    collect_hot_streak_observations,
    fetch_boxscore_batters,
    summarize_hot_streak_backtest,
)
from mlb_props.schedule import ProbableMatchup


class _FakeBoxscoreResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeBoxscoreSession:
    def __init__(self, payload_by_game_pk):
        self.payload_by_game_pk = payload_by_game_pk
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        for game_pk, payload in self.payload_by_game_pk.items():
            if url.endswith(f"/game/{game_pk}/boxscore"):
                return _FakeBoxscoreResponse(payload)
        raise AssertionError(f"Unexpected URL requested: {url}")


def _boxscore_payload(away_players, home_players):
    def side(players):
        return {"players": {f"ID{i}": p for i, p in enumerate(players)}}

    return {"teams": {"away": side(away_players), "home": side(home_players)}}


def _player_entry(name, plate_appearances):
    return {"person": {"fullName": name}, "stats": {"batting": {"plateAppearances": plate_appearances}}}


def test_fetch_boxscore_batters_returns_real_batters_with_a_plate_appearance():
    payload = _boxscore_payload(
        away_players=[_player_entry("Away Starter", 4), _player_entry("Away Bench", 0)],
        home_players=[_player_entry("Home Starter", 3)],
    )
    session = _FakeBoxscoreSession({12345: payload})

    batters = fetch_boxscore_batters(session, 12345)

    assert set(batters) == {"Away Starter", "Home Starter"}  # the bench player with 0 PA is excluded


def test_fetch_boxscore_batters_skips_unparsable_entries():
    payload = {"teams": {"away": {"players": {"ID1": {"totally": "unrecognized shape"}}}, "home": {"players": {}}}}
    session = _FakeBoxscoreSession({999: payload})

    assert fetch_boxscore_batters(session, 999) == []  # skipped, not a crash


class _FakeSchedule:
    def __init__(self, matchups_by_date):
        self.matchups_by_date = matchups_by_date

    def get_slate(self, game_date):
        return self.matchups_by_date.get(game_date, [])


class _FakePyb:
    """Records every real statcast_batter call so a test can assert a
    player's log is fetched exactly ONCE for the whole backtest window,
    not once per date they appear in it (the real bug this module's
    redesign fixes - see collect_hot_streak_observations' docstring).
    """

    def __init__(self, id_df, logs_by_player_id):
        self._id_df = id_df
        self._logs_by_player_id = logs_by_player_id
        self.statcast_batter_calls = []

    def playerid_lookup(self, last, first):
        return self._id_df

    def statcast_batter(self, start, end, player_id):
        self.statcast_batter_calls.append((start, end, player_id))
        return self._logs_by_player_id.get(player_id, pd.DataFrame({"game_date": [], "events": []}))


def test_collect_hot_streak_observations_uses_the_day_before_as_of_no_lookahead():
    game_date = date(2026, 8, 20)
    matchup = ProbableMatchup(away_team="A", home_team="B", venue="Park", away_pitcher=None, home_pitcher=None, game_pk=555)
    schedule = _FakeSchedule({game_date: [matchup]})
    session = _FakeBoxscoreSession({555: _boxscore_payload([_player_entry("Hot Player", 4)], [])})
    # A real home run ON game_date itself, plus real earlier games. If
    # as_of correctly excludes game_date's own game, only the earlier
    # games count toward the hot-streak signal - not this one.
    log = pd.DataFrame(
        {
            "game_date": ["2026-08-15", "2026-08-20"],
            "events": ["single", "home_run"],
            "woba_value": [0.9, 2.0],
            "woba_denom": [1, 1],
        }
    )
    pyb = _FakePyb(pd.DataFrame({"key_mlbam": [111]}), {111: log})

    observations = collect_hot_streak_observations(schedule, pyb, [game_date], date(2026, 3, 1), session)

    assert len(observations) == 1
    obs = observations[0]
    assert obs.player == "Hot Player"
    assert obs.got_hr is True  # the real outcome for game_date itself, still correctly reported
    assert obs.got_2plus_tb is True
    # the exact real detail that keeps this non-circular: the log fetch
    # covers the whole window (not narrowed per as_of), fetched exactly
    # once - the game-date exclusion happens inside heat_index_from_log,
    # not by asking for a different date range per call.
    assert pyb.statcast_batter_calls == [("2026-03-01", "2026-08-20", 111)]


def test_collect_hot_streak_observations_fetches_each_players_log_only_once():
    # The real bug this module's redesign fixes: a player appearing in
    # multiple real games across the window used to trigger a separate
    # full-season log re-fetch for each one.
    d1, d2 = date(2026, 8, 20), date(2026, 8, 21)
    m1 = ProbableMatchup(away_team="A", home_team="B", venue="Park", away_pitcher=None, home_pitcher=None, game_pk=1)
    m2 = ProbableMatchup(away_team="A", home_team="C", venue="Park2", away_pitcher=None, home_pitcher=None, game_pk=2)
    schedule = _FakeSchedule({d1: [m1], d2: [m2]})
    session = _FakeBoxscoreSession(
        {
            1: _boxscore_payload([_player_entry("Repeat Player", 4)], []),
            2: _boxscore_payload([_player_entry("Repeat Player", 3)], []),
        }
    )
    log = pd.DataFrame(
        {
            "game_date": ["2026-08-20", "2026-08-21"],
            "events": ["strikeout", "single"],
            "woba_value": [0.0, 0.9],
            "woba_denom": [1, 1],
        }
    )
    pyb = _FakePyb(pd.DataFrame({"key_mlbam": [222]}), {222: log})

    observations = collect_hot_streak_observations(schedule, pyb, [d1, d2], date(2026, 3, 1), session)

    assert len(observations) == 2  # both real games still produce a real observation
    assert len(pyb.statcast_batter_calls) == 1  # but the log itself was only fetched once


def test_collect_hot_streak_observations_skips_a_player_whose_id_cant_be_resolved():
    game_date = date(2026, 8, 20)
    matchup = ProbableMatchup(away_team="A", home_team="B", venue="Park", away_pitcher=None, home_pitcher=None, game_pk=555)
    schedule = _FakeSchedule({game_date: [matchup]})
    session = _FakeBoxscoreSession({555: _boxscore_payload([_player_entry("Nobody Real", 4)], [])})
    pyb = _FakePyb(pd.DataFrame({"key_mlbam": []}), {})  # no real mlbam id found

    observations = collect_hot_streak_observations(schedule, pyb, [game_date], date(2026, 3, 1), session)

    assert observations == []


def test_collect_hot_streak_observations_skips_a_player_with_no_real_outcome_that_day():
    # The boxscore says they batted, but the fetched log has no real
    # plate-appearance row for that exact date - "unknown stays unknown,"
    # never a guessed outcome.
    game_date = date(2026, 8, 20)
    matchup = ProbableMatchup(away_team="A", home_team="B", venue="Park", away_pitcher=None, home_pitcher=None, game_pk=555)
    schedule = _FakeSchedule({game_date: [matchup]})
    session = _FakeBoxscoreSession({555: _boxscore_payload([_player_entry("Mystery Player", 4)], [])})
    log = pd.DataFrame({"game_date": ["2026-08-15"], "events": ["single"], "woba_value": [0.9], "woba_denom": [1]})
    pyb = _FakePyb(pd.DataFrame({"key_mlbam": [333]}), {333: log})

    observations = collect_hot_streak_observations(schedule, pyb, [game_date], date(2026, 3, 1), session)

    assert observations == []


def test_collect_hot_streak_observations_skips_a_matchup_with_no_real_game_pk():
    game_date = date(2026, 8, 20)
    matchup = ProbableMatchup(away_team="A", home_team="B", venue="Park", away_pitcher=None, home_pitcher=None, game_pk=None)
    schedule = _FakeSchedule({game_date: [matchup]})
    session = _FakeBoxscoreSession({})
    pyb = _FakePyb(pd.DataFrame({"key_mlbam": []}), {})

    observations = collect_hot_streak_observations(schedule, pyb, [game_date], date(2026, 3, 1), session)

    assert observations == []
    assert session.calls == []  # never even attempted a boxscore fetch


def test_summarize_hot_streak_backtest_is_honest_with_no_observations():
    text = summarize_hot_streak_backtest([])
    assert "No real observations collected" in text


def test_summarize_hot_streak_backtest_buckets_hot_and_cold_separately():
    from mlb_props.historical_backtest import HotStreakObservation

    def obs(player, z, got_hr):
        return HotStreakObservation(
            game_date="2026-08-20", player=player, z_score=z,
            l15_clear_hr_rate=None, l15_clear_tb2_rate=None, l15_clear_hit_rate=None,
            season_clear_hr_rate=None, season_clear_tb2_rate=None, season_clear_hit_rate=None,
            got_hr=got_hr, got_2plus_tb=False, got_hit=got_hr,
        )

    observations = [
        obs("Hot A", 1.5, True), obs("Hot B", 2.0, True),
        obs("Cold A", -1.5, False), obs("Cold B", -2.0, False),
        obs("Neutral A", 0.2, False),
    ]

    text = summarize_hot_streak_backtest(observations)

    assert "5 real (player, game) observation(s)" in text
    assert "Hot (z >= +1.0)" in text
    assert "Cold (z <= -1.0)" in text
    assert "n=2" in text  # both the hot and cold buckets have exactly 2 real observations each
