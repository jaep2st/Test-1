from odds_monitor.cli import main, parse_args


def test_cli_mock_once_runs_without_error():
    exit_code = main(["--mock", "--mock-seed", "1", "--once", "--log-level", "WARNING"])
    assert exit_code == 0


def test_cli_missing_api_key_is_a_config_error(monkeypatch):
    monkeypatch.delenv("BETSTAMP_API_KEY", raising=False)
    exit_code = main(["--once", "--log-level", "WARNING"])
    assert exit_code == 2


def test_parse_args_defaults():
    args = parse_args([])
    assert args.leagues is None
    assert args.min_spread == 2.0
    assert args.min_prob_spread == 8.0
    assert args.interval == 300.0
    assert args.notify == []


def test_cli_mock_once_covers_mlb_home_run_props():
    exit_code = main(["--mock", "--mock-seed", "11", "--league", "mlb", "--once", "--log-level", "WARNING"])
    assert exit_code == 0
