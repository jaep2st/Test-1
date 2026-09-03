"""Covers mlb_props/betting.py: Kelly-fraction math and the recommended-
bets builder that turns real +EV EdgeCandidates into concrete unit-sized
recommendations. See that module's docstring for the conservatism choices
(fractional Kelly, floor/cap, tier-based multiplier) this locks in.
"""

import pytest

from mlb_props.betting import (
    MAX_UNITS,
    MIN_EV_PERCENT_TO_RECOMMEND,
    MIN_UNITS,
    breakeven_price,
    build_recommended_bets,
    kelly_fraction,
    recommend_units,
)
from odds_monitor.ev import american_to_implied_prob
from mlb_props.edges import EdgeCandidate
from mlb_props.pipeline import SlateReport
from odds_monitor.models import PropLine


def test_kelly_fraction_is_positive_for_a_real_edge():
    # +150 (decimal 2.5) with a 50% true win probability is a real edge
    # (implied prob at that price is 40%).
    assert kelly_fraction(0.50, 2.5) > 0


def test_kelly_fraction_is_zero_or_negative_with_no_real_edge():
    # -300 (decimal 1.333) implies ~75% - a 60% true probability is worse
    # than the market's own price, no real edge.
    assert kelly_fraction(0.60, 1.333) <= 0


def test_breakeven_price_is_even_money_at_a_coin_flip():
    assert breakeven_price(0.5) == 100


def test_breakeven_price_round_trips_back_to_the_same_probability():
    # By construction, the implied probability of the breakeven price itself
    # (no vig involved - it's a single number, not a two-sided market) must
    # equal the true probability it was derived from.
    for true_prob in (0.05, 0.2, 0.43, 0.6, 0.91):
        price = breakeven_price(true_prob)
        assert price is not None
        assert american_to_implied_prob(price) == pytest.approx(true_prob, abs=1e-3)


def test_breakeven_price_is_none_outside_a_real_probability_range():
    assert breakeven_price(0.0) is None
    assert breakeven_price(1.0) is None
    assert breakeven_price(-0.1) is None
    assert breakeven_price(1.5) is None


def test_recommend_units_returns_none_when_no_real_edge():
    assert recommend_units(0.10, -300, "agree") is None


def test_recommend_units_never_below_the_floor_once_it_clears_the_bar():
    # A tiny-but-real edge should floor to MIN_UNITS, not round to 0.
    units = recommend_units(0.15, 600, "agree")
    assert units is not None
    assert units >= MIN_UNITS


def test_recommend_units_never_exceeds_the_cap():
    # A huge edge (well-priced favorite vs. a high true probability)
    # should cap at MAX_UNITS regardless of the raw Kelly math.
    units = recommend_units(0.85, -110, "agree")
    assert units == MAX_UNITS


def test_speculative_tier_sized_more_conservatively_than_agree_for_the_same_edge():
    agree_units = recommend_units(0.60, 200, "agree")
    speculative_units = recommend_units(0.60, 200, "model_only")
    assert agree_units is not None and speculative_units is not None
    assert speculative_units <= agree_units


def test_recommend_units_rounds_to_the_nearest_half():
    units = recommend_units(0.30, 250, "agree")
    assert units is not None
    assert (units * 2) == int(units * 2)  # exact half-unit


def _edge(player, market, ev_percent_model, tier_inputs, price=200, book="draftkings", event="Team A @ Team B"):
    return EdgeCandidate(
        player=player,
        market=market,
        event=event,
        model_score=70.0,
        model_prob=tier_inputs.get("model_prob", 0.40),
        market_fair_prob=tier_inputs.get("market_fair_prob"),
        best_line=PropLine(player=player, team=None, league="mlb", market=market, side="yes", line=0.5, odds=price, sportsbook=book, event=event),
        ev_percent_model=ev_percent_model,
        ev_percent_market=tier_inputs.get("ev_percent_market"),
        edge_vs_market=tier_inputs.get("edge_vs_market"),
        price_spread_percent=None,
        books_quoting=2,
        park="Test Park",
        wind_out_mph=0.0,
        temp_f=70.0,
        is_dome=False,
        weather_boost_pct=0.0,
    )


def _agree_edge(player, market, ev_percent_model=10.0, **kwargs):
    return _edge(
        player, market, ev_percent_model,
        {"model_prob": 0.40, "market_fair_prob": 0.30, "ev_percent_market": 5.0, "edge_vs_market": 0.10, **kwargs},
    )


def _model_only_edge(player, market, ev_percent_model=10.0, **kwargs):
    return _edge(player, market, ev_percent_model, {"model_prob": 0.40, "market_fair_prob": None, **kwargs})


def _report(hr_edges=None, tb_edges=None, hits_edges=None):
    from datetime import date

    return SlateReport(
        game_date=date(2026, 8, 31),
        slate=[],
        matchup_environments=[],
        hot_batters=[],
        hr_edges=hr_edges or [],
        tb_edges=tb_edges or [],
        hits_edges=hits_edges or [],
    )


def test_build_recommended_bets_includes_the_matching_breakeven_price():
    edge = _agree_edge("Player Z", "batter_home_runs")
    strong, _ = build_recommended_bets(_report(hr_edges=[edge]))
    assert len(strong) == 1
    assert strong[0].breakeven == breakeven_price(strong[0].model_prob)


def test_build_recommended_bets_splits_by_tier():
    strong_edge = _agree_edge("Player A", "batter_home_runs")
    speculative_edge = _model_only_edge("Player B", "batter_hits")
    strong, speculative = build_recommended_bets(_report(hr_edges=[strong_edge], hits_edges=[speculative_edge]))
    assert [r.player for r in strong] == ["Player A"]
    assert [r.player for r in speculative] == ["Player B"]


def test_build_recommended_bets_excludes_candidates_below_the_ev_threshold():
    below_bar = _agree_edge("Player C", "batter_home_runs", ev_percent_model=MIN_EV_PERCENT_TO_RECOMMEND - 0.1)
    strong, speculative = build_recommended_bets(_report(hr_edges=[below_bar]))
    assert strong == []
    assert speculative == []


def test_build_recommended_bets_excludes_candidates_with_no_market_data():
    no_market = EdgeCandidate(
        player="Player D", market="batter_home_runs", event="Team A @ Team B", model_score=70.0, model_prob=0.15,
        market_fair_prob=None, best_line=None, ev_percent_model=None, ev_percent_market=None, edge_vs_market=None,
        price_spread_percent=None, books_quoting=0, park="Test Park", wind_out_mph=0.0, temp_f=70.0, is_dome=False,
        weather_boost_pct=0.0,
    )
    strong, speculative = build_recommended_bets(_report(hr_edges=[no_market]))
    assert strong == []
    assert speculative == []


def test_build_recommended_bets_sorts_by_ev_percent_descending():
    lower = _agree_edge("Player E", "batter_home_runs", ev_percent_model=5.0)
    higher = _agree_edge("Player F", "batter_total_bases", ev_percent_model=15.0)
    strong, _ = build_recommended_bets(_report(hr_edges=[lower], tb_edges=[higher]))
    assert [r.player for r in strong] == ["Player F", "Player E"]


def test_build_recommended_bets_combines_all_three_markets():
    hr = _agree_edge("Player G", "batter_home_runs")
    tb = _agree_edge("Player H", "batter_total_bases")
    hits = _agree_edge("Player I", "batter_hits")
    strong, _ = build_recommended_bets(_report(hr_edges=[hr], tb_edges=[tb], hits_edges=[hits]))
    assert {r.player for r in strong} == {"Player G", "Player H", "Player I"}


