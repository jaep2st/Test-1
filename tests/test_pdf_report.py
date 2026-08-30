from datetime import date

import pytest

from mlb_props.context import MockParkWeatherProvider
from mlb_props.hot_streak import MockHotStreakProvider
from mlb_props.market import MockMlbPropsOddsProvider
from mlb_props.matchup import MockMatchupProvider
from mlb_props.pipeline import run_pipeline
from mlb_props.schedule import MockScheduleProvider
from mlb_props.statcast import MockStatcastProvider

reportlab = pytest.importorskip("reportlab", reason="reportlab not installed - --pdf-out is an optional feature")
from mlb_props.pdf_report import render_pdf_report  # noqa: E402


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


def test_render_pdf_report_produces_a_real_pdf_file(tmp_path):
    out = tmp_path / "report.pdf"
    render_pdf_report(_report(), str(out), is_mock=True)
    assert out.exists()
    data = out.read_bytes()
    assert data.startswith(b"%PDF-")
    assert len(data) > 1000  # a real multi-table document, not an empty shell


def test_render_pdf_report_handles_a_report_with_no_priced_edges(tmp_path):
    # No odds provider configured (NoOddsProvider-equivalent: an odds
    # provider that returns nothing) - every candidate should fall back to
    # the model-only table without crashing.
    from odds_monitor.providers.base import OddsProvider

    class _EmptyOdds(OddsProvider):
        def fetch_player_props(self, league):
            return []

    schedule = MockScheduleProvider()
    report = run_pipeline(
        game_date=date(2026, 8, 26),
        schedule=schedule,
        statcast=MockStatcastProvider(seed=5),
        matchup_provider=MockMatchupProvider(seed=5),
        hot_streak=MockHotStreakProvider(seed=5),
        park_weather=MockParkWeatherProvider(seed=5),
        odds=_EmptyOdds(),
    )
    out = tmp_path / "report_no_odds.pdf"
    render_pdf_report(report, str(out), is_mock=True)
    assert out.read_bytes().startswith(b"%PDF-")
