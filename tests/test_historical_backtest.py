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
from mlb_props.hot_streak import ClearanceWindow, HeatIndex
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


class _FakeHotStreak:
    """Returns a fixed HeatIndex per (player, as_of) pair - records every
    call so a test can assert the real as_of date used (no lookahead).
    """

    def __init__(self, heat_by_player):
        self.heat_by_player = heat_by_player
        self.calls = []

    def get_heat_index(self, player, as_of=None):
        self.calls.append((player, as_of))
        return self.heat_by_player.get(player, _neutral_heat())


def _heat(z_score=0.0, l15_hr_games=3, l15_games=15, season_hr_games=20, season_games=100):
    return HeatIndex(
        player="x", season_woba=0.330, last7_woba=0.330, last15_woba=0.330, last30_woba=0.330,
        last15_pa=40, z_score=z_score,
        clear_l15=ClearanceWindow(games=l15_games, hr_games=l15_hr_games, tb2_games=6, hit_games=9),
        clear_season=ClearanceWindow(games=season_games, hr_games=season_hr_games, tb2_games=40, hit_games=65),
    )


def _neutral_heat():
    return _heat(z_score=0.0)


class _FakePyb:
    def __init__(self, id_df, logs_by_player_id):
        self._id_df = id_df
        self._logs_by_player_id = logs_by_player_id

    def playerid_lookup(self, last, first):
        return self._id_df

    def statcast_batter(self, start, end, player_id):
        return self._logs_by_player_id.get(player_id, pd.DataFrame({"game_date": [], "events": []}))


def test_collect_hot_streak_observations_uses_the_day_before_as_of_no_lookahead():
    game_date = date(2026, 8, 20)
    matchup = ProbableMatchup(away_team="A", home_team="B", venue="Park", away_pitcher=None, home_pitcher=None, game_pk=555)
    schedule = _FakeSchedule({game_date: [matchup]})
    session = _FakeBoxscoreSession({555: _boxscore_payload([_player_entry("Hot Player", 4)], [])})
    hot_streak = _FakeHotStreak({"Hot Player": _heat(z_score=1.8)})
    pyb = _FakePyb(
        pd.DataFrame({"key_mlbam": [111]}),
        {111: pd.DataFrame({"game_date": ["2026-08-20"], "events": ["home_run"]})},
    )

    observations = collect_hot_streak_observations(schedule, hot_streak, pyb, session, [game_date])

    assert len(observations) == 1
    obs = observations[0]
    assert obs.player == "Hot Player"
    assert obs.z_score == 1.8
    assert obs.got_hr is True
    assert obs.got_2plus_tb is True
    # the exact real detail that keeps this non-circular: as_of is the day
    # BEFORE the game itself, never the game's own date.
    assert hot_streak.calls == [("Hot Player", game_date - timedelta(days=1))]


def test_collect_hot_streak_observations_skips_a_player_whose_outcome_cant_be_resolved():
    game_date = date(2026, 8, 20)
    matchup = ProbableMatchup(away_team="A", home_team="B", venue="Park", away_pitcher=None, home_pitcher=None, game_pk=555)
    schedule = _FakeSchedule({game_date: [matchup]})
    session = _FakeBoxscoreSession({555: _boxscore_payload([_player_entry("Nobody Real", 4)], [])})
    hot_streak = _FakeHotStreak({})
    pyb = _FakePyb(pd.DataFrame({"key_mlbam": []}), {})  # no real mlbam id found

    observations = collect_hot_streak_observations(schedule, hot_streak, pyb, session, [game_date])

    assert observations == []


def test_collect_hot_streak_observations_skips_a_matchup_with_no_real_game_pk():
    game_date = date(2026, 8, 20)
    matchup = ProbableMatchup(away_team="A", home_team="B", venue="Park", away_pitcher=None, home_pitcher=None, game_pk=None)
    schedule = _FakeSchedule({game_date: [matchup]})
    session = _FakeBoxscoreSession({})
    hot_streak = _FakeHotStreak({})
    pyb = _FakePyb(pd.DataFrame({"key_mlbam": []}), {})

    observations = collect_hot_streak_observations(schedule, hot_streak, pyb, session, [game_date])

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
