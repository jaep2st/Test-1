"""Covers the single-sided-market fallback in mlb_props/edges.py: confirmed
live (run #12's diagnostics) that a real book quotes MLB's home-run prop as
"Over 0.5" only, with no "Under" leg - find_fair_prices() can't de-vig that
(needs two sides), so without this fallback a genuine real price would never
appear in the report at all.
"""

from odds_monitor.ev import find_fair_prices
from odds_monitor.models import PropLine

from mlb_props.edges import build_hits_edges, build_hr_edges, build_total_bases_edges, rank_candidates
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


def test_rank_candidates_never_drops_a_priced_candidate_at_the_default_min_ev():
    # Confirmed live (2026-08-29): a real HR price existed for players
    # whose model probability was too low to justify the long-shot payout
    # (negative EV(mdl)) - at the documented default (min_ev_percent=0.0,
    # "show all"), those real prices vanished from the report entirely
    # instead of just being ranked lower. A real market price is exactly
    # the information this report exists to surface; the default must
    # never delete it.
    scores = [_hr_score(player="Bryce Eldridge", model_prob=0.08)]  # low model prob
    lines = [_single_sided_hr_line(player="Bryce Eldridge", odds=440, book="betrivers")]  # long-shot real price
    edges = build_hr_edges(scores, find_fair_prices(lines), lines, event_lookup={})
    assert edges[0].ev_percent_model < 0.0  # confirms this candidate is the negative-EV case being tested

    ranked = rank_candidates(edges, min_ev_percent=0.0)

    assert len(ranked) == 1
    assert ranked[0].has_market_data
    assert ranked[0].player == "Bryce Eldridge"


def test_rank_candidates_still_filters_when_a_positive_threshold_is_explicit():
    scores = [_hr_score(player="Bryce Eldridge", model_prob=0.08)]
    lines = [_single_sided_hr_line(player="Bryce Eldridge", odds=440, book="betrivers")]
    edges = build_hr_edges(scores, find_fair_prices(lines), lines, event_lookup={})

    ranked = rank_candidates(edges, min_ev_percent=5.0)

    assert ranked == []


def test_a_single_book_two_sided_market_never_earns_the_agree_tier():
    # Real user report (2026-09-03): a report scored Angel Genao's 1+ HR
    # prop as tier == "agree" ("STRONG BET") off ESPN BET's price alone
    # (books_quoting=1 - the only book that had posted the market yet).
    # find_fair_prices() can produce a real FairPrice from just one book's
    # two-sided quote, but one book's own number is not a "the market
    # agrees" signal - Fanatics posted +1500 for the exact same bet not
    # long after, more than 60% better than the +900 this project called
    # "strong". A real edge against a thin, one-book price should still
    # surface (it's real, live market data), just not with false
    # cross-book confidence.
    lines = [
        _single_sided_hr_line(odds=650, book="draftkings"),  # yes
        PropLine(
            player="Aaron Judge", team=None, league="mlb", market=MARKET_HOME_RUN, side="no",
            line=0.5, odds=-900, sportsbook="draftkings", event="Houston Astros @ New York Yankees",
        ),
    ]
    edges = build_hr_edges([_hr_score(model_prob=0.20)], find_fair_prices(lines), lines, event_lookup={})

    edge = edges[0]
    assert edge.books_quoting == 1
    assert edge.market_fair_prob is not None  # a real FairPrice did get computed
    assert edge.ev_percent_model is not None and edge.ev_percent_model > 0
    assert edge.edge_vs_market is not None and edge.edge_vs_market > 0
    assert edge.tier == "model_only"  # not "agree" - one book isn't a consensus


def test_a_second_independent_book_does_earn_the_agree_tier():
    # Same real edge as above, but now a second book (Fanatics) also
    # quotes both sides - a genuine second opinion, so this really is
    # "the market agrees" now.
    lines = [
        _single_sided_hr_line(odds=650, book="draftkings"),
        PropLine(
            player="Aaron Judge", team=None, league="mlb", market=MARKET_HOME_RUN, side="no",
            line=0.5, odds=-900, sportsbook="draftkings", event="Houston Astros @ New York Yankees",
        ),
        _single_sided_hr_line(odds=600, book="fanatics"),
        PropLine(
            player="Aaron Judge", team=None, league="mlb", market=MARKET_HOME_RUN, side="no",
            line=0.5, odds=-800, sportsbook="fanatics", event="Houston Astros @ New York Yankees",
        ),
    ]
    edges = build_hr_edges([_hr_score(model_prob=0.20)], find_fair_prices(lines), lines, event_lookup={})

    edge = edges[0]
    assert edge.books_quoting == 2
    assert edge.tier == "agree"


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
