import os

import pytest

from ncaaf_main import build_providers, main, parse_args


def test_ncaaf_cli_mock_runs_without_error(capsys):
    exit_code = main(["--mock", "--season", "2026", "--week", "3", "--mock-seed", "1", "--log-level", "WARNING"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "PREDICTED SCORES" in out


def test_ncaaf_cli_mock_writes_html_and_records_picks(tmp_path):
    html_out = tmp_path / "board.html"
    data_dir = tmp_path / "data"
    exit_code = main(
        ["--mock", "--season", "2026", "--week", "3", "--mock-seed", "2", "--html-out", str(html_out),
         "--record-picks", "--data-dir", str(data_dir), "--log-level", "WARNING"]
    )
    assert exit_code == 0
    assert html_out.exists()
    assert "NCAAF" in html_out.read_text()
    picks_file = data_dir / "picks" / "2026-wk03.jsonl"
    assert picks_file.exists()
    assert picks_file.read_text().strip()


def test_ncaaf_cli_real_mode_without_cfbd_key_fails_gracefully(monkeypatch, capsys):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    exit_code = main(["--season", "2026", "--week", "3"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "CFBD_API_KEY" in err or "College Football Data" in err


def test_build_providers_raises_without_cfbd_key(monkeypatch):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    args = parse_args(["--season", "2026", "--week", "3"])
    with pytest.raises(ValueError, match="College Football Data"):
        build_providers(args)


def test_parse_args_defaults():
    args = parse_args(["--mock"])
    assert args.min_ev == 0.0
    assert args.top == 40
    assert args.mock is True
    assert args.data_dir == "data/ncaaf"


def test_ncaaf_cli_performance_only_requires_output_path():
    exit_code = main(["--performance-only"])
    assert exit_code == 2


def test_ncaaf_cli_performance_only_renders_from_empty_data_dir(tmp_path):
    perf_out = tmp_path / "performance.html"
    exit_code = main(["--performance-only", "--performance-out", str(perf_out), "--data-dir", str(tmp_path / "nope")])
    assert exit_code == 0
    assert perf_out.exists()
    assert "Performance" in perf_out.read_text()
