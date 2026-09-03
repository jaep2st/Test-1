"""Covers lineup_source threading end-to-end: schedule.py's ProbableMatchup
-> pipeline.py's _resolve_batters/run_pipeline -> edges.py's EdgeCandidate ->
results.py's PickRecord. See ProbableMatchup.lineup_source's docstring for
why this exists - a real, honest per-game signal for whether a pick was
scored against MLB's actual confirmed starting lineup or the active-roster
proxy, surfaced so a reader can judge information-freshness for themselves.
"""

from datetime import date, datetime, timezone

from odds_monitor.ev import find_fair_prices
from odds_monitor.models import PropLine

from mlb_props.context import MockParkWeatherProvider
from mlb_props.edges import build_hr_edges
from mlb_props.hot_streak import MockHotStreakProvider
from mlb_props.market import MARKET_HOME_RUN
from mlb_props.matchup import MockMatchupProvider
from mlb_props.pipeline import _resolve_batters, run_pipeline
from mlb_props.results import record_picks
from mlb_props.schedule import ProbableMatchup, ScheduleProvider
from mlb_props.scoring import HRScoreResult
from mlb_props.statcast import MockStatcastProvider


def _hr_score(player="Aaron Judge", model_prob=0.15):
    return HRScoreResult(
        player=player, score=60.0, model_prob=model_prob, components={},
        park="Yankee Stadium", wind_out_mph=5.0, temp_f=75.0, is_dome=False, weather_boost_pct=2.0,
    )


def test_resolve_batters_carries_the_real_game_lineup_source():
    confirmed_game = ProbableMatchup(
        away_team="A", home_team="B", venue="Park", away_pitcher="P1", home_pitcher="P2",
        away_batters=["Confirmed Batter"], home_batters=["Confirmed Batter 2"], lineup_source="confirmed",
    )
    active_roster_game = ProbableMatchup(
        away_team="C", home_team="D", venue="Park 2", away_pitcher="P3", home_pitcher="P4",
        away_batters=["Roster Batter"], home_batters=["Roster Batter 2"], lineup_source="active_roster",
    )

    ctx = _resolve_batters([confirmed_game, active_roster_game], extra_batters=None)

    assert ctx["Confirmed Batter"]["lineup_source"] == "confirmed"
    assert ctx["Confirmed Batter 2"]["lineup_source"] == "confirmed"
    assert ctx["Roster Batter"]["lineup_source"] == "active_roster"


def test_resolve_batters_marks_extra_batters_as_active_roster():
    game = ProbableMatchup(away_team="A", home_team="B", venue="Park", away_pitcher="P1", home_pitcher="P2")
    ctx = _resolve_batters([game], extra_batters=["Extra Batter"])
    assert ctx["Extra Batter"]["lineup_source"] == "active_roster"


def test_build_hr_edges_uses_the_real_lineup_source_lookup():
    score = _hr_score()
    lines = [
        PropLine(player="Aaron Judge", team=None, league="mlb", market=MARKET_HOME_RUN, side="yes", line=0.5, odds=200, sportsbook="dk", event="Away @ Home"),
        PropLine(player="Aaron Judge", team=None, league="mlb", market=MARKET_HOME_RUN, side="no", line=0.5, odds=-250, sportsbook="dk", event="Away @ Home"),
    ]
    fair_prices = find_fair_prices(lines)
    edges = build_hr_edges([score], fair_prices, lines, {"Aaron Judge": "Away @ Home"}, {"Aaron Judge": "confirmed"})
    assert edges[0].lineup_source == "confirmed"


def test_build_hr_edges_defaults_to_active_roster_without_a_lookup():
    score = _hr_score()
    edges = build_hr_edges([score], [], [], {"Aaron Judge": "Away @ Home"})
    assert edges[0].lineup_source == "active_roster"


def _no_market_candidate(lineup_source="active_roster"):
    from mlb_props.edges import EdgeCandidate

    return EdgeCandidate(
        player="Player A", market="batter_home_runs", event="Team A @ Team B", model_score=50.0, model_prob=0.12,
        market_fair_prob=None, best_line=None, ev_percent_model=None, ev_percent_market=None, edge_vs_market=None,
        price_spread_percent=None, books_quoting=0, park="Test Park", wind_out_mph=0.0, temp_f=70.0, is_dome=False,
        weather_boost_pct=0.0, lineup_source=lineup_source,
    )


def test_prop_row_shows_the_confirmed_lineup_badge():
    from mlb_props.html_report import _prop_row

    row_html = _prop_row(_no_market_candidate("confirmed"), None, "hr")
    assert "Lineup confirmed" in row_html
    assert "lineup-confirmed" in row_html


def test_prop_row_shows_the_active_roster_badge_by_default():
    from mlb_props.html_report import _prop_row

    row_html = _prop_row(_no_market_candidate("active_roster"), None, "hr")
    assert "lineup not posted yet" in row_html
    assert "lineup-projected" in row_html


def test_reco_row_shows_the_real_lineup_source_badge():
    from mlb_props.betting import RecommendedBet
    from mlb_props.html_report import _reco_row

    r = RecommendedBet(
        player="Player A", market="batter_home_runs", market_label="1+ HR", event="Team A @ Team B",
        tier="agree", model_prob=0.20, market_fair_prob=0.15, edge_vs_market=0.05, ev_percent_model=10.0,
        best_price=200, best_book="draftkings", books_quoting=2, units=1.0, full_kelly_percent=4.0, breakeven=400,
        lineup_source="confirmed",
    )
    row_html = _reco_row(r, "2026-08-20")
    assert "Lineup confirmed" in row_html


def test_record_picks_carries_lineup_source_through_to_the_jsonl(tmp_path):
    schedule_stub = _confirmed_schedule_stub()
    report = run_pipeline(
        game_date=date(2026, 8, 26),
        schedule=schedule_stub,
        statcast=MockStatcastProvider(seed=1),
        matchup_provider=MockMatchupProvider(seed=1),
        hot_streak=MockHotStreakProvider(seed=1),
        park_weather=MockParkWeatherProvider(seed=1),
        odds=_NoOddsStub(),
    )
    assert report.hr_edges, "expected at least one real scored HR candidate"
    assert all(e.lineup_source == "confirmed" for e in report.hr_edges)

    out_path = str(tmp_path / "picks.jsonl")
    n = record_picks(report, out_path, recorded_at=datetime(2026, 8, 26, 18, tzinfo=timezone.utc))
    assert n > 0
    from mlb_props.results import load_picks

    picks = load_picks(out_path)
    assert any(p.lineup_source == "confirmed" for p in picks)


class _NoOddsStub:
    def fetch_player_props(self, league):
        return []


def _confirmed_schedule_stub():
    class _Stub(ScheduleProvider):
        def get_slate(self, game_date):
            return [
                ProbableMatchup(
                    away_team="New York Yankees",
                    home_team="Baltimore Orioles",
                    venue="Camden Yards",
                    away_pitcher="Kevin Gausman",
                    home_pitcher="Grayson Rodriguez",
                    away_batters=["Aaron Judge", "Juan Soto"],
                    home_batters=["Gunnar Henderson"],
                    lineup_source="confirmed",
                )
            ]

    return _Stub()
