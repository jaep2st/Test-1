"""Covers mlb_props_main.py's new --record-picks/--performance-out/
--resolve-results/--record-clv flags at the CLI dispatch level (main()) -
the underlying logic itself (record_picks, resolve_results_for_date,
record_closing_odds, render_performance_report) is covered directly in
test_results.py/test_backtest.py/test_performance_report.py.
"""

import json
import os

import mlb_props_main
from mlb_props_main import main


def test_record_picks_and_performance_out_work_together_in_mock_mode(tmp_path, capsys):
    data_dir = tmp_path / "data"
    perf_out = tmp_path / "performance.html"
    rc = main(
        [
            "--mock",
            "--mock-seed",
            "3",
            "--date",
            "2026-08-20",
            "--record-picks",
            "--data-dir",
            str(data_dir),
            "--performance-out",
            str(perf_out),
        ]
    )
    assert rc == 0
    picks_path = data_dir / "picks" / "2026-08-20.jsonl"
    assert picks_path.exists()
    lines = picks_path.read_text().strip().splitlines()
    assert len(lines) > 0
    record = json.loads(lines[0])
    assert record["game_date"] == "2026-08-20"
    assert perf_out.exists()
    assert perf_out.read_text().strip().startswith("<!doctype html>")


def test_record_picks_is_additive_and_never_overwrites_prior_snapshots(tmp_path):
    data_dir = tmp_path / "data"
    for _ in range(2):
        rc = main(["--mock", "--mock-seed", "3", "--date", "2026-08-20", "--record-picks", "--data-dir", str(data_dir)])
        assert rc == 0
    picks_path = data_dir / "picks" / "2026-08-20.jsonl"
    lines = picks_path.read_text().strip().splitlines()
    # Two full snapshots of the same slate, not deduped/overwritten - see
    # PickRecord.recorded_at's docstring in results.py.
    assert len(lines) % 2 == 0
    assert len(lines) > 0


def test_record_clv_works_in_mock_mode_against_previously_recorded_picks(tmp_path):
    data_dir = tmp_path / "data"
    rc = main(["--mock", "--mock-seed", "3", "--date", "2026-08-20", "--record-picks", "--data-dir", str(data_dir)])
    assert rc == 0
    rc = main(["--mock", "--mock-seed", "3", "--date", "2026-08-20", "--record-clv", "--data-dir", str(data_dir)])
    assert rc == 0
    clv_path = data_dir / "clv" / "2026-08-20.jsonl"
    assert clv_path.exists()


def test_resolve_results_reports_a_clean_configuration_error_without_pybaseball(tmp_path, capsys):
    # This sandbox doesn't have pybaseball installed - the CLI should fail
    # with a clear message, not a raw traceback (same posture as every
    # other pybaseball-dependent real-data flag in this project).
    rc = main(["--resolve-results", "--date", "2026-08-20", "--data-dir", str(tmp_path / "data")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Configuration error" in err
    assert "pybaseball" in err


def test_record_clv_without_any_odds_key_still_runs_gracefully(tmp_path):
    # No --mock, no odds key configured: build_providers degrades to
    # NoOddsProvider (see mlb_props_main.build_providers) rather than
    # erroring, so record_clv should complete with zero records, not crash.
    data_dir = tmp_path / "data"
    os.makedirs(data_dir / "picks")
    (data_dir / "picks" / "2026-08-20.jsonl").write_text("")
    rc = main(["--record-clv", "--date", "2026-08-20", "--data-dir", str(data_dir)])
    assert rc == 0
    assert (data_dir / "clv" / "2026-08-20.jsonl").exists() is False  # zero records -> nothing written


def test_mock_mode_never_attempts_a_live_bets_fetch(tmp_path):
    # --mock has no real odds key at all - the live cross-book value scan
    # must be skipped entirely (honest empty state on the page), not
    # attempted against a provider that can't serve it.
    html_out = tmp_path / "index.html"
    rc = main(["--mock", "--mock-seed", "3", "--date", "2026-08-20", "--html-out", str(html_out)])
    assert rc == 0
    assert "No real live cross-book value" in html_out.read_text()


def test_run_live_value_scan_fetches_only_live_and_returns_built_bets(monkeypatch):
    # Unit-tests the extracted helper directly (mlb_props_main.
    # run_live_value_scan) rather than through main()'s full pipeline -
    # main() without --mock builds real (network-calling) schedule/
    # statcast/matchup/weather providers that have no place in a unit
    # test; this function is exactly the seam that keeps the live-bets
    # wiring testable without any of that.
    from mlb_props.betting import LiveValueBet

    class _FakeLiveProvider:
        def __init__(self, api_key, books):
            self.api_key = api_key
            self.books = books

        def fetch_player_props(self, league, only_live=False):
            assert only_live is True
            return ["sentinel-line"]

    fake_bet = LiveValueBet(
        player="Test Player", market="batter_home_runs", market_label="1+ HR", event="Team A @ Team B",
        fair_prob=0.15, best_price=650, best_book="draftkings", books_used=2, ev_percent=9.5, units=0.5, breakeven=567,
    )
    monkeypatch.setattr(mlb_props_main, "TheOddsApiProvider", _FakeLiveProvider)
    monkeypatch.setattr(mlb_props_main, "build_live_value_bets", lambda lines: [fake_bet] if lines == ["sentinel-line"] else [])

    bets = mlb_props_main.run_live_value_scan("test-key", None)

    assert bets == [fake_bet]


def test_html_report_renders_live_bets_passed_to_it():
    from mlb_props.betting import LiveValueBet
    from mlb_props.html_report import render_html_report
    from mlb_props.pipeline import SlateReport
    from datetime import date as _date

    live_bet = LiveValueBet(
        player="Test Player", market="batter_home_runs", market_label="1+ HR", event="Team A @ Team B",
        fair_prob=0.15, best_price=650, best_book="draftkings", books_used=2, ev_percent=9.5, units=0.5, breakeven=567,
    )
    empty_report = SlateReport(
        game_date=_date(2026, 8, 20), slate=[], matchup_environments=[], hot_batters=[],
        hr_edges=[], tb_edges=[], hits_edges=[],
    )
    html_text = render_html_report(empty_report, live_bets=[live_bet])
    assert "Test Player" in html_text
    assert "+650" in html_text
    assert "draftkings" in html_text
    assert "beat +567" in html_text
