from ncaaf.context import MockVenueProvider, MockWeatherProvider
from ncaaf.html_report import render_html_report
from ncaaf.market import MockGameOddsProvider
from ncaaf.performance_report import render_performance_report
from ncaaf.pipeline import run_pipeline
from ncaaf.ratings import MockRatingsProvider
from ncaaf.schedule import MockScheduleProvider


def _report(seed=1):
    schedule = MockScheduleProvider()
    slate = schedule.get_week_slate(2026, 3)
    return run_pipeline(
        2026, 3, schedule, MockRatingsProvider(), MockVenueProvider(), MockWeatherProvider(seed=seed),
        MockGameOddsProvider(slate, seed=seed),
    )


def test_html_report_renders_without_error_and_includes_key_sections():
    html_text = render_html_report(_report(), top=10, is_mock=True)
    assert "<!doctype html>" in html_text.lower()
    assert "Recommended Bets" in html_text
    assert "Predicted Scores" in html_text
    assert "Ranked +EV Picks" in html_text
    assert "SAMPLE DATA" in html_text


def test_html_report_live_status_bug_differs_from_mock():
    mock_html = render_html_report(_report(), is_mock=True)
    live_html = render_html_report(_report(), is_mock=False)
    assert "SAMPLE DATA" in mock_html
    assert "LIVE DATA" in live_html


def test_performance_report_renders_with_no_history():
    html_text = render_performance_report("/tmp/definitely-does-not-exist-ncaaf-data")
    assert "<!doctype html>" in html_text.lower()
    assert "0 picks recorded" in html_text
