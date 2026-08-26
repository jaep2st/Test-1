import runpy
import sys

from mlb_props_main import main, parse_args


def test_mlb_props_cli_mock_runs_without_error(capsys):
    exit_code = main(["--mock", "--mock-seed", "1", "--log-level", "WARNING"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Best Home Run Props" in out


def test_mlb_props_cli_missing_api_key_is_a_config_error(monkeypatch):
    monkeypatch.delenv("BETSTAMP_API_KEY", raising=False)
    exit_code = main(["--date", "2026-08-26", "--log-level", "WARNING"])
    assert exit_code == 2


def test_parse_args_defaults():
    args = parse_args(["--mock"])
    assert args.min_ev == 0.0
    assert args.top == 15
    assert args.mock is True
