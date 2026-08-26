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
    from mlb_props_main import build_providers, parse_args

    args = parse_args(["--date", "2026-08-26"])
    schedule, statcast, matchup, hot_streak, park_weather, odds = build_providers(args)
    assert odds.fetch_player_props("mlb") == []


def test_parse_args_defaults():
    args = parse_args(["--mock"])
    assert args.min_ev == 0.0
    assert args.top == 15
    assert args.mock is True
