"""Covers mlb_props/betting.py: Kelly-fraction math and the recommended-
bets builder that turns real +EV EdgeCandidates into concrete unit-sized
recommendations. See that module's docstring for the conservatism choices
(fractional Kelly, floor/cap, tier-based multiplier) this locks in.
"""

from mlb_props.betting import (
    MAX_UNITS,
    MIN_EV_PERCENT_FOR_LIVE,
    MIN_EV_PERCENT_TO_RECOMMEND,
    MIN_UNITS,
    build_live_value_bets,
    build_recommended_bets,
    kelly_fraction,
    recommend_units,
)
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


def _live_line(player, side, price, book, market="batter_home_runs", event="Team X @ Team Y", is_live=True):
    return PropLine(player=player, team=None, league="mlb", market=market, side=side, line=0.5, odds=price, sportsbook=book, event=event, is_live=is_live)


def test_build_live_value_bets_finds_real_cross_book_value():
    lines = [
        _live_line("Player A", "yes", 900, "draftkings"),
        _live_line("Player A", "no", -900, "draftkings"),
        _live_line("Player A", "yes", 650, "betmgm"),
        _live_line("Player A", "no", -1200, "betmgm"),
    ]
    bets = build_live_value_bets(lines)
    assert len(bets) == 1
    assert bets[0].player == "Player A"
    assert bets[0].best_price == 900
    assert bets[0].best_book == "draftkings"
    assert bets[0].ev_percent >= MIN_EV_PERCENT_FOR_LIVE
    assert bets[0].units >= MIN_UNITS


def test_build_live_value_bets_ignores_pregame_lines():
    lines = [
        _live_line("Player A", "yes", 900, "draftkings", is_live=False),
        _live_line("Player A", "no", -900, "draftkings", is_live=False),
    ]
    assert build_live_value_bets(lines) == []


def test_build_live_value_bets_ignores_single_book_lines_with_no_way_to_devig():
    lines = [_live_line("Player A", "yes", 900, "draftkings")]
    assert build_live_value_bets(lines) == []


def test_build_live_value_bets_excludes_edges_below_the_higher_live_bar():
    # A small, real edge that would clear the pregame 3% bar but not the
    # higher 5% live-specific one.
    lines = [
        _live_line("Player A", "yes", 150, "draftkings"),
        _live_line("Player A", "no", -155, "draftkings"),
        _live_line("Player A", "yes", 145, "betmgm"),
        _live_line("Player A", "no", -150, "betmgm"),
    ]
    bets = build_live_value_bets(lines)
    assert all(b.ev_percent >= MIN_EV_PERCENT_FOR_LIVE for b in bets)


def test_build_live_value_bets_ignores_a_longer_shot_point_tier_quoted_by_the_same_side():
    # Confirmed live: betrivers' batter_home_runs market posts "Over" at
    # multiple real point tiers (0.5, 1.5, 2.5) for the same player at
    # once. Only the standard 0.5 ("1+ HR") tier is this project's real
    # market - a longer-shot tier (e.g. 1.5, "2+ HR") must never leak into
    # the results just because find_fair_prices could de-vig it too.
    lines = [
        _live_line("Player A", "yes", 900, "draftkings", market="batter_home_runs"),
        _live_line("Player A", "no", -900, "draftkings", market="batter_home_runs"),
        _live_line("Player A", "yes", 650, "betmgm", market="batter_home_runs"),
        _live_line("Player A", "no", -1200, "betmgm", market="batter_home_runs"),
    ]
    longer_shot_tier = [
        PropLine(player="Player A", team=None, league="mlb", market="batter_home_runs", side="yes", line=1.5, odds=2900, sportsbook="betrivers", event="Team X @ Team Y", is_live=True),
        PropLine(player="Player A", team=None, league="mlb", market="batter_home_runs", side="yes", line=1.5, odds=2500, sportsbook="fanduel", event="Team X @ Team Y", is_live=True),
        PropLine(player="Player A", team=None, league="mlb", market="batter_home_runs", side="no", line=1.5, odds=-3500, sportsbook="betrivers", event="Team X @ Team Y", is_live=True),
        PropLine(player="Player A", team=None, league="mlb", market="batter_home_runs", side="no", line=1.5, odds=-3000, sportsbook="fanduel", event="Team X @ Team Y", is_live=True),
    ]
    bets = build_live_value_bets(lines + longer_shot_tier)
    assert len(bets) == 1
    assert bets[0].best_price == 900  # the standard 0.5 tier's price, never the 1.5 tier's inflated one


def test_build_live_value_bets_finds_nothing_when_only_a_non_standard_tier_is_two_sided():
    # Same real-world shape as above, but with NO standard-tier pricing at
    # all - only a longer-shot tier is de-vig-able. Must return nothing,
    # not silently substitute the wrong tier.
    lines = [
        PropLine(player="Player A", team=None, league="mlb", market="batter_home_runs", side="yes", line=1.5, odds=2900, sportsbook="betrivers", event="Team X @ Team Y", is_live=True),
        PropLine(player="Player A", team=None, league="mlb", market="batter_home_runs", side="yes", line=1.5, odds=2500, sportsbook="fanduel", event="Team X @ Team Y", is_live=True),
        PropLine(player="Player A", team=None, league="mlb", market="batter_home_runs", side="no", line=1.5, odds=-3500, sportsbook="betrivers", event="Team X @ Team Y", is_live=True),
        PropLine(player="Player A", team=None, league="mlb", market="batter_home_runs", side="no", line=1.5, odds=-3000, sportsbook="fanduel", event="Team X @ Team Y", is_live=True),
    ]
    assert build_live_value_bets(lines) == []


def test_build_live_value_bets_sorted_by_ev_descending():
    lines = [
        _live_line("Player A", "yes", 900, "draftkings", event="Game 1"),
        _live_line("Player A", "no", -900, "draftkings", event="Game 1"),
        _live_line("Player A", "yes", 650, "betmgm", event="Game 1"),
        _live_line("Player A", "no", -1200, "betmgm", event="Game 1"),
        _live_line("Player B", "yes", 400, "draftkings", event="Game 2"),
        _live_line("Player B", "no", -500, "draftkings", event="Game 2"),
        _live_line("Player B", "yes", 250, "betmgm", event="Game 2"),
        _live_line("Player B", "no", -700, "betmgm", event="Game 2"),
    ]
    bets = build_live_value_bets(lines)
    assert len(bets) == 2
    assert bets[0].ev_percent >= bets[1].ev_percent
