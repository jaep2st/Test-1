from ncaaf.context import GameContext
from ncaaf.edges import MIN_BOOKS_FOR_MARKET_AGREE, build_game_edges, effective_tier, rank_candidates
from ncaaf.market import MARKET_H2H, MARKET_SPREADS, GameLine
from ncaaf.ratings import TeamRating
from ncaaf.schedule import Matchup
from ncaaf.scoring import compute_game_score


def _matchup():
    return Matchup(
        season=2026, week=3, season_type="regular", away_team="Away U", home_team="Home U",
        away_conference=None, home_conference=None, venue="Home Stadium", venue_id=1,
        neutral_site=False, conference_game=False, start_date_utc=None, completed=False,
        home_points=None, away_points=None,
    )


def _score(power_diff=10.0):
    m = _matchup()
    home = TeamRating(team="Home U", season=2026, conference=None, power_rating=power_diff, offense_rating=0.0,
                       defense_rating=0.0, special_teams_rating=0.0, elo=1500.0, sp_overall=power_diff, sos=None, games_played=5)
    away = TeamRating(team="Away U", season=2026, conference=None, power_rating=0.0, offense_rating=0.0,
                       defense_rating=0.0, special_teams_rating=0.0, elo=1500.0, sp_overall=0.0, sos=None, games_played=5)
    ctx = GameContext(venue="Home Stadium", neutral_site=False, is_dome=False, home_field_advantage_pts=0.0,
                       altitude_bonus_pts=0.0, home_rest_days=None, away_rest_days=None, rest_edge_pts=0.0,
                       travel_miles=None, travel_edge_pts=0.0, weather=None, weather_total_adjustment_pct=0.0)
    return compute_game_score(m, home, away, ctx)


def _line(**overrides):
    base = dict(event="Away U @ Home U", home_team="Home U", away_team="Away U", commence_time=None,
                market=MARKET_SPREADS, side="home", team="Home U", point=-3.5, odds=-110, sportsbook="draftkings")
    base.update(overrides)
    return GameLine(**base)


def test_agree_tier_needs_two_independent_books():
    score = _score(power_diff=14.0)  # model likes home a lot
    lines = [
        # Home is a big favorite by the model but the market only sees a
        # small spread - real value on home, quoted by two agreeing books.
        _line(sportsbook="draftkings", side="home", point=-1.5, odds=-110),
        _line(sportsbook="draftkings", side="away", point=1.5, odds=-110),
        _line(sportsbook="fanduel", side="home", point=-1.5, odds=-115),
        _line(sportsbook="fanduel", side="away", point=1.5, odds=-105),
    ]
    candidates = build_game_edges({score.event: score}, lines)
    home_spread = next(c for c in candidates if c.market == MARKET_SPREADS and c.side == "home")
    assert home_spread.books_quoting >= MIN_BOOKS_FOR_MARKET_AGREE
    assert home_spread.tier == "agree"


def test_single_book_never_qualifies_as_agree_even_with_a_real_edge():
    score = _score(power_diff=14.0)
    lines = [
        _line(sportsbook="draftkings", side="home", point=-1.5, odds=-110),
        _line(sportsbook="draftkings", side="away", point=1.5, odds=-110),
    ]
    candidates = build_game_edges({score.event: score}, lines)
    home_spread = next(c for c in candidates if c.market == MARKET_SPREADS and c.side == "home")
    assert home_spread.books_quoting == 1
    assert home_spread.tier != "agree"
    assert effective_tier("agree", 1) == "model_only"
    assert effective_tier("agree", 2) == "agree"


def test_single_sided_market_falls_back_to_model_only_ev():
    score = _score(power_diff=14.0)
    lines = [_line(sportsbook="draftkings", side="home", point=-1.5, odds=-110)]  # no "away" side anywhere
    candidates = build_game_edges({score.event: score}, lines)
    home_spread = next(c for c in candidates if c.market == MARKET_SPREADS and c.side == "home")
    assert home_spread.market_fair_prob is None
    assert home_spread.ev_percent_model is not None
    assert home_spread.tier == "model_only_single_sided"


def test_h2h_side_probabilities_come_from_home_win_prob():
    score = _score(power_diff=14.0)
    lines = [
        _line(market=MARKET_H2H, side="home", point=None, odds=-400, sportsbook="draftkings"),
        _line(market=MARKET_H2H, side="away", point=None, odds=+320, sportsbook="draftkings"),
    ]
    candidates = build_game_edges({score.event: score}, lines)
    home_ml = next(c for c in candidates if c.market == MARKET_H2H and c.side == "home")
    away_ml = next(c for c in candidates if c.market == MARKET_H2H and c.side == "away")
    assert home_ml.model_prob == score.home_win_prob
    assert round(away_ml.model_prob + home_ml.model_prob, 4) == 1.0


def test_rank_candidates_never_drops_priced_candidates_at_default_min_ev():
    score = _score(power_diff=-14.0)  # model actually dislikes home this time
    lines = [
        _line(sportsbook="draftkings", side="home", point=-20.5, odds=-110),
        _line(sportsbook="draftkings", side="away", point=20.5, odds=-110),
    ]
    candidates = build_game_edges({score.event: score}, lines)
    ranked = rank_candidates(candidates, min_ev_percent=0.0)
    assert len(ranked) == len(candidates)


def test_rank_candidates_filters_when_min_ev_explicitly_raised():
    score = _score(power_diff=14.0)
    lines = [
        _line(sportsbook="draftkings", side="home", point=-1.5, odds=-110),
        _line(sportsbook="draftkings", side="away", point=1.5, odds=-110),
        _line(market=MARKET_H2H, side="home", point=None, odds=-105, sportsbook="draftkings"),
        _line(market=MARKET_H2H, side="away", point=None, odds=-105, sportsbook="draftkings"),
    ]
    candidates = build_game_edges({score.event: score}, lines)
    ranked_all = rank_candidates(candidates, min_ev_percent=0.0)
    ranked_filtered = rank_candidates(candidates, min_ev_percent=1000.0)  # nothing realistically clears this
    assert len(ranked_filtered) < len(ranked_all)
