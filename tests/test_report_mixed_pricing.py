"""Confirmed live (run #14): real market coverage is genuinely partial - a
book quotes some candidates and not others, never the whole scored field.
mlb_props/report.py's plain-text table used to be all-or-nothing (any
priced row present meant every unpriced candidate vanished from the report
entirely, even though they're still real, ranked candidates) - this covers
the fix: priced rows first, then backfill remaining slots with model-only
rows so the full picture stays visible.
"""

from mlb_props.edges import EdgeCandidate
from mlb_props.report import render_hr_props
from odds_monitor.models import PropLine


def _priced(player="Mike Trout"):
    line = PropLine(
        player=player, team=None, league="mlb", market="batter_home_runs", side="yes",
        line=0.5, odds=350, sportsbook="betrivers", event="e",
    )
    return EdgeCandidate(
        player=player, market="batter_home_runs", event="e", model_score=60.0, model_prob=0.15,
        market_fair_prob=None, best_line=line, ev_percent_model=12.5, ev_percent_market=None,
        edge_vs_market=None, price_spread_percent=None, books_quoting=1,
        park="x", wind_out_mph=0.0, temp_f=70.0, is_dome=False, weather_boost_pct=0.0,
    )


def _unpriced(player):
    return EdgeCandidate(
        player=player, market="batter_home_runs", event="e", model_score=55.0, model_prob=0.12,
        market_fair_prob=None, best_line=None, ev_percent_model=None, ev_percent_market=None,
        edge_vs_market=None, price_spread_percent=None, books_quoting=0,
        park="x", wind_out_mph=0.0, temp_f=70.0, is_dome=False, weather_boost_pct=0.0,
    )


def test_unpriced_candidates_still_show_when_some_are_priced():
    edges = [_priced("Mike Trout")] + [_unpriced(f"Player {i}") for i in range(5)]
    text = render_hr_props(edges, top=15)

    assert "Mike Trout" in text
    for i in range(5):
        assert f"Player {i}" in text
    assert "model-only" in text


def test_backfill_respects_top_limit():
    edges = [_priced("Mike Trout")] + [_unpriced(f"Player {i}") for i in range(10)]
    text = render_hr_props(edges, top=3)

    assert "Mike Trout" in text
    shown = sum(f"Player {i}" in text for i in range(10))
    assert shown == 2  # top=3 minus the 1 priced row


def test_all_unpriced_still_uses_the_original_model_only_format():
    edges = [_unpriced(f"Player {i}") for i in range(3)]
    text = render_hr_props(edges, top=15)
    assert "no market prices matched" in text
    for i in range(3):
        assert f"Player {i}" in text
