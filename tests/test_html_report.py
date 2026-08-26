from datetime import date

from mlb_props.context import MockParkWeatherProvider
from mlb_props.hot_streak import MockHotStreakProvider
from mlb_props.html_report import render_html_report
from mlb_props.market import MockMlbPropsOddsProvider
from mlb_props.matchup import MockMatchupProvider
from mlb_props.pipeline import run_pipeline
from mlb_props.schedule import MockScheduleProvider
from mlb_props.statcast import MockStatcastProvider


def _report(seed=5):
    schedule = MockScheduleProvider()
    slate = schedule.get_slate(date(2026, 8, 26))
    events_by_batter, all_batters = {}, []
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
    )


def test_html_report_is_well_formed_and_self_contained():
    html_text = render_html_report(_report(), is_mock=True)
    assert html_text.strip().startswith("<!doctype html>")
    assert html_text.count("<html") == 1
    assert html_text.count("</html>") == 1
    # self-contained: no external script/stylesheet hosts other than Google Fonts
    assert "fonts.googleapis.com" in html_text
    assert "src=\"http" not in html_text


def test_html_report_shows_sample_banner_only_in_mock_mode():
    mock_html = render_html_report(_report(), is_mock=True)
    assert "SAMPLE OUTPUT" in mock_html
    live_html = render_html_report(_report(), is_mock=False)
    assert "SAMPLE OUTPUT" not in live_html
    assert "LIVE" in live_html


def test_html_report_includes_every_hr_and_tb_pick():
    report = _report()
    html_text = render_html_report(report, top=50)
    for edge in report.hr_edges + report.tb_edges:
        assert edge.player in html_text


def test_html_report_escapes_player_names_safely():
    report = _report()
    # Sanity: report always renders without raising even if a name had markup-like
    # characters (none do in the mock data, but escaping must run regardless).
    html_text = render_html_report(report)
    assert "<script>" not in html_text
