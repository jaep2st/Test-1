"""Covers pipeline._filter_lines_to_confirmed_pregame_games - the
MLB-Stats-API-backed second safety net on top of the odds provider's own
commence_time filter (see theoddsapi.py). Reproduces the exact real
scenarios confirmed live (2026-08-29):
1. Kansas City @ Cleveland: a genuinely Final game whose odds-provider
   event still looked pregame (a wrong/stale commence_time) - a real price
   for it slipped through the first filter entirely.
2. Boston @ NY Yankees / Arizona @ San Francisco: real doubleheaders where
   The Odds API exposed only ONE event per matchup, with no way to tell
   which specific game (one Final/in-progress, one still pregame) a price
   belonged to.
"""

from mlb_props.pipeline import _filter_lines_to_confirmed_pregame_games
from mlb_props.schedule import ProbableMatchup
from odds_monitor.models import PropLine


def _matchup(away, home, status):
    return ProbableMatchup(away_team=away, home_team=home, venue="Park", away_pitcher=None, home_pitcher=None, status=status)


def _line(event, player="Player X"):
    return PropLine(player=player, team=None, league="mlb", market="batter_hits", side="over", line=0.5, odds=-150, sportsbook="draftkings", event=event)


def test_drops_a_line_for_a_confirmed_final_game_even_if_the_odds_provider_thought_it_was_pregame():
    # Reproduces the exact Jac Caglianone case: the odds provider's own
    # commence_time filter let this line through, but MLB's own schedule
    # says the one real game between these teams today is Final.
    slate = [_matchup("Kansas City Royals", "Cleveland Guardians", "Final")]
    lines = [_line("Kansas City Royals @ Cleveland Guardians")]

    kept = _filter_lines_to_confirmed_pregame_games(lines, slate)

    assert kept == []


def test_keeps_a_line_for_a_doubleheaders_still_pregame_second_game():
    # The odds provider only exposed one event for this matchup - the
    # cross-check must still keep the price, since MLB's schedule confirms
    # a real game between these teams is genuinely still upcoming.
    slate = [
        _matchup("Boston Red Sox", "New York Yankees", "Final"),
        _matchup("Boston Red Sox", "New York Yankees", "Pre-Game"),
    ]
    lines = [_line("Boston Red Sox @ New York Yankees")]

    kept = _filter_lines_to_confirmed_pregame_games(lines, slate)

    assert kept == lines


def test_drops_a_line_for_a_doubleheader_where_both_games_are_done():
    slate = [
        _matchup("Boston Red Sox", "New York Yankees", "Final"),
        _matchup("Boston Red Sox", "New York Yankees", "In Progress"),
    ]
    lines = [_line("Boston Red Sox @ New York Yankees")]

    kept = _filter_lines_to_confirmed_pregame_games(lines, slate)

    assert kept == []


def test_unknown_status_is_treated_as_pregame_not_dropped():
    slate = [_matchup("Team A", "Team B", None)]
    lines = [_line("Team A @ Team B")]

    kept = _filter_lines_to_confirmed_pregame_games(lines, slate)

    assert kept == lines


def test_a_matchup_missing_from_the_slate_entirely_is_kept_not_dropped():
    slate = [_matchup("Team A", "Team B", "Final")]
    lines = [_line("Team C @ Team D")]

    kept = _filter_lines_to_confirmed_pregame_games(lines, slate)

    assert kept == lines


def test_scheduled_and_preview_statuses_also_count_as_pregame():
    slate = [
        _matchup("Team A", "Team B", "Scheduled"),
        _matchup("Team C", "Team D", "Preview"),
    ]
    lines = [_line("Team A @ Team B"), _line("Team C @ Team D")]

    kept = _filter_lines_to_confirmed_pregame_games(lines, slate)

    assert kept == lines


def test_normal_pregame_game_is_unaffected():
    slate = [_matchup("Team A", "Team B", "Pre-Game")]
    lines = [_line("Team A @ Team B")]

    kept = _filter_lines_to_confirmed_pregame_games(lines, slate)

    assert kept == lines
