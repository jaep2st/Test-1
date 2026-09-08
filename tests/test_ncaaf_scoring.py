from datetime import date

from ncaaf.context import GameContext
from ncaaf.ratings import TeamRating
from ncaaf.schedule import Matchup
from ncaaf.scoring import compute_game_score


def _matchup(**overrides):
    base = dict(
        season=2026, week=3, season_type="regular", away_team="Away U", home_team="Home U",
        away_conference=None, home_conference=None, venue="Home Stadium", venue_id=1,
        neutral_site=False, conference_game=False, start_date_utc=None, completed=False,
        home_points=None, away_points=None,
    )
    base.update(overrides)
    return Matchup(**base)


def _rating(team, power=0.0, offense=0.0, defense=0.0):
    return TeamRating(
        team=team, season=2026, conference=None, power_rating=power, offense_rating=offense,
        defense_rating=defense, special_teams_rating=0.0, elo=1500.0, sp_overall=power, sos=None, games_played=5,
    )


def _neutral_context(**overrides):
    base = dict(
        venue="Home Stadium", neutral_site=False, is_dome=False, home_field_advantage_pts=0.0,
        altitude_bonus_pts=0.0, home_rest_days=None, away_rest_days=None, rest_edge_pts=0.0,
        travel_miles=None, travel_edge_pts=0.0, weather=None, weather_total_adjustment_pct=0.0,
    )
    base.update(overrides)
    return GameContext(**base)


def test_none_when_a_team_has_no_rating():
    m = _matchup()
    assert compute_game_score(m, None, _rating("Away U"), _neutral_context()) is None
    assert compute_game_score(m, _rating("Home U"), None, _neutral_context()) is None


def test_evenly_matched_neutral_game_is_a_coinflip():
    m = _matchup(neutral_site=True)
    score = compute_game_score(m, _rating("Home U"), _rating("Away U"), _neutral_context())
    assert abs(score.predicted_margin) < 1e-9
    assert abs(score.home_win_prob - 0.5) < 1e-9


def test_stronger_home_team_is_favored():
    m = _matchup()
    score = compute_game_score(m, _rating("Home U", power=14.0), _rating("Away U", power=0.0), _neutral_context())
    assert score.predicted_margin > 0
    assert score.home_win_prob > 0.5


def test_home_field_advantage_shifts_margin_in_favor_of_home():
    m = _matchup()
    no_hfa = compute_game_score(m, _rating("Home U"), _rating("Away U"), _neutral_context(home_field_advantage_pts=0.0))
    with_hfa = compute_game_score(m, _rating("Home U"), _rating("Away U"), _neutral_context(home_field_advantage_pts=2.3))
    assert with_hfa.predicted_margin > no_hfa.predicted_margin
    assert with_hfa.home_win_prob > no_hfa.home_win_prob


def test_cover_prob_favors_home_as_spread_gets_easier():
    m = _matchup()
    score = compute_game_score(m, _rating("Home U", power=7.0), _rating("Away U"), _neutral_context())
    # Home is favored by ~7; covering a small home spread (-1.5) should be
    # much more likely than covering a big one (-13.5).
    assert score.home_cover_prob(-1.5) > score.home_cover_prob(-13.5)
    assert score.away_cover_prob(-1.5) == round(1.0 - score.home_cover_prob(-1.5), 4)


def test_over_prob_increases_with_higher_offense_ratings():
    m = _matchup()
    low_total = compute_game_score(m, _rating("Home U", offense=-5, defense=-5), _rating("Away U", offense=-5, defense=-5), _neutral_context())
    high_total = compute_game_score(m, _rating("Home U", offense=10, defense=-10), _rating("Away U", offense=10, defense=-10), _neutral_context())
    assert high_total.predicted_total > low_total.predicted_total
    assert high_total.over_prob(50.0) > low_total.over_prob(50.0)


def test_weather_adjustment_lowers_predicted_total():
    m = _matchup()
    calm = compute_game_score(m, _rating("Home U"), _rating("Away U"), _neutral_context(weather_total_adjustment_pct=0.0))
    windy = compute_game_score(m, _rating("Home U"), _rating("Away U"), _neutral_context(weather_total_adjustment_pct=-5.0))
    assert windy.predicted_total < calm.predicted_total


def test_probabilities_stay_within_bounds():
    m = _matchup()
    score = compute_game_score(m, _rating("Home U", power=40.0), _rating("Away U", power=-40.0), _neutral_context())
    assert 0.0 <= score.home_win_prob <= 1.0
    assert 0.0 <= score.home_cover_prob(-1.5) <= 1.0
    assert 0.0 <= score.over_prob(50.0) <= 1.0
