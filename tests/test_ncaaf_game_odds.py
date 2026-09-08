from ncaaf.game_odds import find_fair_game_prices
from ncaaf.market import MARKET_H2H, MARKET_SPREADS, MARKET_TOTALS, GameLine


def _line(**overrides):
    base = dict(
        event="Away U @ Home U", home_team="Home U", away_team="Away U", commence_time=None,
        market=MARKET_SPREADS, side="home", team="Home U", point=-3.5, odds=-110, sportsbook="draftkings",
    )
    base.update(overrides)
    return GameLine(**base)


def test_spread_home_away_pair_across_opposite_signed_points():
    lines = [
        _line(sportsbook="draftkings", side="home", team="Home U", point=-6.5, odds=-110),
        _line(sportsbook="draftkings", side="away", team="Away U", point=6.5, odds=-110),
        # A clear outlier on the away side.
        _line(sportsbook="fanduel", side="home", team="Home U", point=-6.5, odds=-130),
        _line(sportsbook="fanduel", side="away", team="Away U", point=6.5, odds=+180),
    ]
    results = find_fair_game_prices(lines)
    away_result = next(r for r in results if r.side == "away")
    assert away_result.best_line.sportsbook == "fanduel"
    assert away_result.best_line.odds == 180
    assert away_result.ev_percent > 0


def test_spread_does_not_pair_different_point_numbers():
    lines = [
        _line(sportsbook="draftkings", side="home", point=-6.5, odds=-110),
        _line(sportsbook="draftkings", side="away", point=6.5, odds=-110),
        _line(sportsbook="fanduel", side="home", point=-3.0, odds=-110),
        _line(sportsbook="fanduel", side="away", point=3.0, odds=-110),
    ]
    results = find_fair_game_prices(lines)
    home_points = sorted({r.point for r in results if r.side == "home"})
    assert home_points == [-6.5, -3.0]


def test_totals_pair_over_under_at_same_point():
    lines = [
        _line(market=MARKET_TOTALS, side="over", team=None, point=54.5, odds=-105, sportsbook="draftkings"),
        _line(market=MARKET_TOTALS, side="under", team=None, point=54.5, odds=-115, sportsbook="draftkings"),
        _line(market=MARKET_TOTALS, side="over", team=None, point=54.5, odds=+120, sportsbook="fanduel"),
        _line(market=MARKET_TOTALS, side="under", team=None, point=54.5, odds=-140, sportsbook="fanduel"),
    ]
    results = find_fair_game_prices(lines)
    over_result = next(r for r in results if r.side == "over")
    assert over_result.best_line.sportsbook == "fanduel"
    assert over_result.best_line.odds == 120


def test_h2h_pairs_home_away_with_no_point():
    lines = [
        _line(market=MARKET_H2H, side="home", team="Home U", point=None, odds=-150, sportsbook="draftkings"),
        _line(market=MARKET_H2H, side="away", team="Away U", point=None, odds=+130, sportsbook="draftkings"),
    ]
    results = find_fair_game_prices(lines)
    assert len(results) == 2
    home_result = next(r for r in results if r.side == "home")
    assert 0.0 < home_result.fair_prob < 1.0


def test_ignores_lines_with_no_odds():
    lines = [_line(odds=None)]
    assert find_fair_game_prices(lines) == []


def test_single_sided_market_produces_no_fair_price():
    lines = [_line(side="home", odds=-110)]  # no "away" side quoted anywhere
    assert find_fair_game_prices(lines) == []
