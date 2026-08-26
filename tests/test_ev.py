from odds_monitor.ev import (
    american_to_decimal,
    american_to_implied_prob,
    decimal_to_american,
    devig_two_way,
    find_fair_prices,
    model_ev_percent,
)
from odds_monitor.models import PropLine


def _line(**overrides):
    base = dict(
        player="Test Player",
        team=None,
        league="mlb",
        market="player_home_runs",
        side="yes",
        line=0.5,
        odds=-120,
        sportsbook="draftkings",
        event="NYY @ BAL",
    )
    base.update(overrides)
    return PropLine(**base)


def test_american_to_decimal_positive_and_negative():
    assert american_to_decimal(150) == 2.5
    assert round(american_to_decimal(-150), 6) == round(1 + 100 / 150, 6)


def test_american_to_implied_prob_matches_known_values():
    assert american_to_implied_prob(100) == 0.5
    assert round(american_to_implied_prob(-110), 4) == round(110 / 210, 4)


def test_decimal_to_american_round_trip_sign():
    assert decimal_to_american(2.5) == 150
    assert decimal_to_american(1.5) == -200


def test_devig_two_way_sums_to_one():
    fair_a, fair_b = devig_two_way(0.55, 0.55)
    assert round(fair_a + fair_b, 9) == 1.0
    assert fair_a == fair_b  # symmetric input -> symmetric fair probabilities


def test_find_fair_prices_flags_best_price_and_computes_ev():
    lines = [
        _line(sportsbook="draftkings", side="yes", odds=-120),
        _line(sportsbook="draftkings", side="no", odds=-110),
        _line(sportsbook="fanduel", side="yes", odds=+150),  # clear outlier: much better price
        _line(sportsbook="fanduel", side="no", odds=-180),
    ]
    results = find_fair_prices(lines)
    yes_result = next(r for r in results if r.side == "yes")
    assert yes_result.best_line.sportsbook == "fanduel"
    assert yes_result.best_line.odds == 150
    assert yes_result.ev_percent > 0  # fanduel's price should look like value vs. the devigged consensus


def test_find_fair_prices_ignores_single_sided_groups():
    lines = [_line(sportsbook="draftkings", side="yes", odds=-120)]  # no "no" side quoted anywhere
    assert find_fair_prices(lines) == []


def test_find_fair_prices_ignores_lines_without_odds():
    lines = [_line(odds=None)]
    assert find_fair_prices(lines) == []


def test_model_ev_percent_positive_when_model_more_confident_than_price_implies():
    # -110 implies ~52.4%; a model that thinks it's 70% likely should show positive EV.
    assert model_ev_percent(0.70, -110) > 0
    # A model that agrees it's a coinflip should show negative EV on a -110 price (the vig).
    assert model_ev_percent(0.50, -110) < 0
