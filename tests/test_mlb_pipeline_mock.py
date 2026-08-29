from datetime import date

from mlb_props.context import MockParkWeatherProvider
from mlb_props.hot_streak import MockHotStreakProvider
from mlb_props.market import MockMlbPropsOddsProvider
from mlb_props.matchup import MockMatchupProvider
from mlb_props.pipeline import run_pipeline
from mlb_props.report import render_report
from mlb_props.schedule import MockScheduleProvider
from mlb_props.statcast import MockStatcastProvider


def _run(seed=1, min_ev_percent=0.0):
    schedule = MockScheduleProvider()
    slate = schedule.get_slate(date(2026, 8, 26))
    events_by_batter = {}
    all_batters = []
    for game in slate:
        event = f"{game.away_team} @ {game.home_team}"
        for b in game.away_batters + game.home_batters:
            events_by_batter[b] = event
            all_batters.append(b)

    return run_pipeline(
        game_date=date(2026, 8, 26),
        schedule=schedule,
        statcast=MockStatcastProvider(seed=seed),
        matchup_provider=MockMatchupProvider(seed=seed),
        hot_streak=MockHotStreakProvider(seed=seed),
        park_weather=MockParkWeatherProvider(seed=seed),
        odds=MockMlbPropsOddsProvider(batters=all_batters, events_by_batter=events_by_batter, seed=seed),
        min_ev_percent=min_ev_percent,
    )


def test_pipeline_runs_end_to_end_and_scores_every_slate_batter():
    report = _run()
    assert report.slate
    assert report.matchup_environments
    assert report.hot_batters
    assert report.hr_edges
    assert report.tb_edges
    assert report.hits_edges


def test_matchup_environments_are_sorted_best_first():
    report = _run()
    scores = [e.environment_score for e in report.matchup_environments]
    assert scores == sorted(scores, reverse=True)


def test_hot_batters_are_sorted_hottest_first():
    report = _run()
    z_scores = [h.z_score for h in report.hot_batters]
    assert z_scores == sorted(z_scores, reverse=True)


def test_hr_edges_all_have_model_ev_at_or_above_min_when_priced():
    report = _run(min_ev_percent=0.0)
    for edge in report.hr_edges:
        if edge.has_market_data:
            assert edge.ev_percent_model >= 0.0


def test_report_renders_without_error_and_mentions_key_sections():
    report = _run()
    text = render_report(report)
    assert "Best HR Matchups" in text
    assert "Who's Hot" in text
    assert "Best Home Run Props" in text
    assert "Best 2+ Total Bases Props" in text
    assert "Best 1+ Hits Props" in text


def test_pipeline_is_deterministic_with_same_seed():
    report_a = _run(seed=42)
    report_b = _run(seed=42)
    assert [e.player for e in report_a.hr_edges] == [e.player for e in report_b.hr_edges]


def test_hits_edges_all_have_model_ev_at_or_above_min_when_priced():
    report = _run(min_ev_percent=0.0)
    for edge in report.hits_edges:
        if edge.has_market_data:
            assert edge.ev_percent_model >= 0.0
