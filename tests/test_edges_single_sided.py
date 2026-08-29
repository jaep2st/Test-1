"""Covers the single-sided-market fallback in mlb_props/edges.py: confirmed
live (run #12's diagnostics) that a real book quotes MLB's home-run prop as
"Over 0.5" only, with no "Under" leg - find_fair_prices() can't de-vig that
(needs two sides), so without this fallback a genuine real price would never
appear in the report at all.
"""

from odds_monitor.ev import find_fair_prices
from odds_monitor.models import PropLine

from mlb_props.edges import build_hits_edges, build_hr_edges, build_total_bases_edges
from mlb_props.market import MARKET_HITS, MARKET_HOME_RUN, MARKET_TOTAL_BASES
from mlb_props.scoring import HitsScoreResult, HRScoreResult, TotalBasesScoreResult


def _hr_score(player="Aaron Judge", model_prob=0.15):
    return HRScoreResult(
        player=player,
        score=60.0,
        model_prob=model_prob,
        components={},
        park="Yankee Stadium",
        wind_out_mph=5.0,
        temp_f=75.0,
        is_dome=False,
        weather_boost_pct=2.0,
    )


def _tb_score(player="Aaron Judge", model_prob=0.4):
    return TotalBasesScoreResult(
        player=player,
        score=55.0,
        model_prob=model_prob,
        components={},
        park="Yankee Stadium",
        wind_out_mph=5.0,
        temp_f=75.0,
        is_dome=False,
        weather_boost_pct=2.0,
    )


def _single_sided_hr_line(player="Aaron Judge", odds=350, book="betrivers"):
    return PropLine(
        player=player, team=None, league="mlb", market=MARKET_HOME_RUN, side="yes",
        line=0.5, odds=odds, sportsbook=book, event="Houston Astros @ New York Yankees",
    )


def test_single_sided_hr_market_still_produces_a_priced_edge():
    scores = [_hr_score()]
    lines = [_single_sided_hr_line()]
    fair_prices = find_fair_prices(lines)  # no "no" side quoted -> empty
    assert fair_prices == []

    edges = build_hr_edges(scores, fair_prices, lines, event_lookup={})

    assert len(edges) == 1
    edge = edges[0]
    assert edge.has_market_data
    assert edge.best_line.odds == 350
    assert edge.market_fair_prob is None
    assert edge.ev_percent_market is None
    assert edge.edge_vs_market is None
    assert edge.ev_percent_model is not None
    assert edge.books_quoting == 1


def test_single_sided_fallback_ignores_a_longer_shot_multi_HR_line():
    # Confirmed live (run #13): a real book posts multiple point values
    # under the exact same market/side ("Over 0.5", "Over 1.5" HRs, both
    # outcome name "Over") - picking "best price" across all of them used
    # to silently swap in the far-less-likely 2+ HR line's payout (huge
    # odds) instead of the standard 1+ HR line actually being scored.
    scores = [_hr_score()]
    lines = [
        _single_sided_hr_line(odds=350, book="betrivers"),  # standard 1+ HR (line=0.5)
        PropLine(
            player="Aaron Judge", team=None, league="mlb", market=MARKET_HOME_RUN, side="yes",
            line=1.5, odds=19900, sportsbook="betrivers", event="Houston Astros @ New York Yankees",
        ),  # a longer-shot 2+ HR line - must NOT be picked
    ]
    edges = build_hr_edges(scores, find_fair_prices(lines), lines, event_lookup={})

    assert edges[0].best_line.odds == 350
    assert edges[0].best_line.line == 0.5


def test_single_sided_fallback_picks_the_best_priced_book():
    scores = [_hr_score()]
    lines = [
        _single_sided_hr_line(odds=250, book="draftkings"),
        _single_sided_hr_line(odds=400, book="betrivers"),  # better payout
    ]
    edges = build_hr_edges(scores, find_fair_prices(lines), lines, event_lookup={})

    assert edges[0].best_line.sportsbook == "betrivers"
    assert edges[0].best_line.odds == 400
    assert edges[0].books_quoting == 2


def test_two_sided_market_still_prefers_the_devigged_fair_price():
    # Both sides quoted by one book -> find_fair_prices should produce a
    # real FairPrice, and that path (not the single-sided fallback) wins.
    lines = [
        _single_sided_hr_line(odds=350, book="fanduel"),
        PropLine(
            player="Aaron Judge", team=None, league="mlb", market=MARKET_HOME_RUN, side="no",
            line=0.5, odds=-450, sportsbook="fanduel", event="Houston Astros @ New York Yankees",
        ),
    ]
    edges = build_hr_edges([_hr_score()], find_fair_prices(lines), lines, event_lookup={})

    assert edges[0].market_fair_prob is not None
    assert edges[0].ev_percent_market is not None


def test_no_price_at_all_still_falls_back_to_model_only():
    edges = build_hr_edges([_hr_score()], [], [], event_lookup={})
    assert not edges[0].has_market_data
    assert edges[0].best_line is None


def test_total_bases_single_sided_fallback_too():
    lines = [
        PropLine(
            player="Aaron Judge", team=None, league="mlb", market=MARKET_TOTAL_BASES, side="over",
            line=1.5, odds=120, sportsbook="betrivers", event="Houston Astros @ New York Yankees",
        )
    ]
    edges = build_total_bases_edges([_tb_score()], find_fair_prices(lines), lines, event_lookup={})
    assert edges[0].has_market_data
    assert edges[0].market_fair_prob is None
    assert edges[0].ev_percent_model is not None


def _hits_score(player="Aaron Judge", model_prob=0.65):
    return HitsScoreResult(
        player=player,
        score=55.0,
        model_prob=model_prob,
        components={},
        park="Yankee Stadium",
        wind_out_mph=5.0,
        temp_f=75.0,
        is_dome=False,
        weather_boost_pct=2.0,
    )


def test_hits_two_sided_market_produces_a_devigged_fair_price():
    lines = [
        PropLine(
            player="Aaron Judge", team=None, league="mlb", market=MARKET_HITS, side="over",
            line=0.5, odds=-140, sportsbook="draftkings", event="Houston Astros @ New York Yankees",
        ),
        PropLine(
            player="Aaron Judge", team=None, league="mlb", market=MARKET_HITS, side="under",
            line=0.5, odds=110, sportsbook="draftkings", event="Houston Astros @ New York Yankees",
        ),
    ]
    edges = build_hits_edges([_hits_score()], find_fair_prices(lines), lines, event_lookup={})

    assert edges[0].has_market_data
    assert edges[0].market_fair_prob is not None
    assert edges[0].ev_percent_market is not None


def test_hits_single_sided_fallback():
    lines = [
        PropLine(
            player="Aaron Judge", team=None, league="mlb", market=MARKET_HITS, side="over",
            line=0.5, odds=-135, sportsbook="betrivers", event="Houston Astros @ New York Yankees",
        )
    ]
    edges = build_hits_edges([_hits_score()], find_fair_prices(lines), lines, event_lookup={})
    assert edges[0].has_market_data
    assert edges[0].market_fair_prob is None
    assert edges[0].ev_percent_model is not None


def test_hits_no_price_falls_back_to_model_only():
    edges = build_hits_edges([_hits_score()], [], [], event_lookup={})
    assert not edges[0].has_market_data
    assert edges[0].best_line is None


def test_hits_edge_uses_the_standard_line_not_a_longer_shot_tier():
    # Confirmed live (2026-08-29): a real book posts genuine two-sided
    # pricing at multiple point tiers (1+ hits AND 2+ hits) - the standard
    # "1+ hits" (0.5) fair price must win here, not the much-longer-shot
    # "2+ hits" (1.5) tier's fair price, even though find_fair_prices()
    # now correctly returns a FairPrice for each tier (see test_ev.py).
    lines = [
        PropLine(player="Aaron Judge", team=None, league="mlb", market=MARKET_HITS, side="over",
                  line=0.5, odds=-165, sportsbook="draftkings", event="e"),
        PropLine(player="Aaron Judge", team=None, league="mlb", market=MARKET_HITS, side="under",
                  line=0.5, odds=140, sportsbook="draftkings", event="e"),
        PropLine(player="Aaron Judge", team=None, league="mlb", market=MARKET_HITS, side="over",
                  line=1.5, odds=225, sportsbook="draftkings", event="e"),
        PropLine(player="Aaron Judge", team=None, league="mlb", market=MARKET_HITS, side="under",
                  line=1.5, odds=-290, sportsbook="draftkings", event="e"),
    ]
    edges = build_hits_edges([_hits_score()], find_fair_prices(lines), lines, event_lookup={})

    assert edges[0].best_line.line == 0.5
    assert edges[0].best_line.odds == -165
    assert edges[0].market_fair_prob > 0.55
