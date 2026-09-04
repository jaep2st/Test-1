import runpy
import sys

from mlb_props_main import main, parse_args


def test_mlb_props_cli_mock_runs_without_error(capsys):
    exit_code = main(["--mock", "--mock-seed", "1", "--log-level", "WARNING"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Best Home Run Props" in out


def test_mlb_props_cli_missing_api_key_degrades_gracefully(monkeypatch, caplog):
    # Without any odds API key and without --mock, real providers are built
    # (they need real network access to do anything, which isn't available
    # in a test), so this exercises that construction doesn't raise and that
    # the missing-key path is a warning, not a hard failure.
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("BETSTAMP_API_KEY", raising=False)
    monkeypatch.delenv("BALLPARKPAL_API_KEY", raising=False)
    from mlb_props_main import build_providers, parse_args

    args = parse_args(["--date", "2026-08-26"])
    schedule, statcast, matchup, hot_streak, park_weather, odds, ballparkpal = build_providers(args)
    assert odds.fetch_player_props("mlb") == []
    assert ballparkpal.get_hitter_park_factor("Aaron Judge", args.game_date) is None


def test_parse_args_defaults():
    args = parse_args(["--mock"])
    assert args.min_ev == 0.0
    assert args.top == 15
    assert args.mock is True


def test_run_historical_hot_streak_backtest_persists_the_real_run(tmp_path, monkeypatch):
    # pybaseball isn't installed in this dev environment (see requirements.txt
    # and mlb_props_main.py's own ImportError guard) - stub it out via
    # sys.modules so `import pybaseball as pyb` inside the function under
    # test succeeds without needing the real package.
    import sys
    import types
    from datetime import date

    monkeypatch.setitem(sys.modules, "pybaseball", types.ModuleType("pybaseball"))

    import mlb_props.historical_backtest as hb_module
    from mlb_props.historical_backtest import HotStreakObservation, load_historical_backtest_runs

    def fake_collect(schedule, pyb, game_dates, season_start, session, timeout=10.0):
        return [
            HotStreakObservation(
                game_date="2026-08-20", player="Hot A", z_score=1.5,
                l15_clear_hr_rate=None, l15_clear_tb2_rate=None, l15_clear_hit_rate=None,
                season_clear_hr_rate=None, season_clear_tb2_rate=None, season_clear_hit_rate=None,
                got_hr=True, got_2plus_tb=False, got_hit=True,
            )
        ]

    monkeypatch.setattr(hb_module, "collect_hot_streak_observations", fake_collect)

    from mlb_props_main import run_historical_hot_streak_backtest

    text = run_historical_hot_streak_backtest(date(2026, 8, 15), date(2026, 8, 20), data_dir=str(tmp_path))

    assert "1 real (player, game) observation(s)" in text
    assert f"Persisted this run's real summary to {tmp_path}/historical_backtest/runs.jsonl" in text
    runs = load_historical_backtest_runs(str(tmp_path / "historical_backtest" / "runs.jsonl"))
    assert len(runs) == 1
    assert runs[0].n_observations == 1
    assert runs[0].start_date == "2026-08-15"


def test_run_historical_hot_streak_backtest_skips_persistence_without_a_data_dir(tmp_path, monkeypatch):
    import sys
    import types
    from datetime import date

    monkeypatch.setitem(sys.modules, "pybaseball", types.ModuleType("pybaseball"))

    import mlb_props.historical_backtest as hb_module

    monkeypatch.setattr(hb_module, "collect_hot_streak_observations", lambda *a, **k: [])

    from mlb_props_main import run_historical_hot_streak_backtest

    text = run_historical_hot_streak_backtest(date(2026, 8, 15), date(2026, 8, 20), data_dir=None)

    assert "Persisted" not in text
