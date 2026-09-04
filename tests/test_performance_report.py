"""Covers mlb_props/performance_report.py - renders without error on both
an empty data directory (day one, before any results exist) and a
populated one, and shows the real numbers it computed rather than hiding
them.
"""

import os
from datetime import datetime, timezone

from mlb_props.performance_report import REFIT_READY_DAYS, render_performance_report
from mlb_props.results import ClvRecord, GameOutcome, PickRecord, _append_jsonl


def _pick(player, game_date="2026-08-20", **overrides):
    defaults = dict(
        game_date=game_date,
        recorded_at="2026-08-20T18:00:00+00:00",
        player=player,
        market="batter_home_runs",
        event="Team A @ Team B",
        tier="agree",
        model_score=70.0,
        model_prob=0.15,
        bp_model_prob=None,
        market_fair_prob=0.10,
        best_price=650,
        best_book="draftkings",
        ev_percent_model=25.0,
        ev_percent_market=15.0,
        edge_vs_market=0.05,
        books_quoting=4,
    )
    defaults.update(overrides)
    return PickRecord(**defaults)


def test_renders_without_error_on_a_completely_empty_data_dir(tmp_path):
    html = render_performance_report(str(tmp_path))
    assert html.strip().startswith("<!doctype html>")
    assert "No resolved picks yet" in html
    assert f"{REFIT_READY_DAYS} days" in html  # the not-ready-yet framing
    assert "Weight refit check" in html
    assert "No resolved pick yet carries real component features" in html
    assert "Market blend check" in html
    assert "No resolved pick yet carries a real market_fair_prob" in html
    assert "Real hit rate by lineup source" in html
    assert "Historical hot-streak signal backtest" in html
    assert 'No historical backtest run yet' in html
    assert "Methodology" in html


def test_shows_the_real_historical_backtest_run_when_one_was_recorded(tmp_path):
    from datetime import date

    from mlb_props.historical_backtest import build_historical_backtest_run, record_historical_backtest_run

    def obs(player, z, got_hr):
        from mlb_props.historical_backtest import HotStreakObservation

        return HotStreakObservation(
            game_date="2026-08-20", player=player, z_score=z,
            l15_clear_hr_rate=None, l15_clear_tb2_rate=None, l15_clear_hit_rate=None,
            season_clear_hr_rate=None, season_clear_tb2_rate=None, season_clear_hit_rate=None,
            got_hr=got_hr, got_2plus_tb=False, got_hit=got_hr,
        )

    observations = [obs("Hot A", 1.5, True), obs("Cold A", -1.5, False)]
    run = build_historical_backtest_run(observations, date(2026, 8, 15), date(2026, 9, 3))
    record_historical_backtest_run(run, str(tmp_path / "historical_backtest" / "runs.jsonl"))

    html = render_performance_report(str(tmp_path))

    assert "2026-08-15" in html
    assert "2026-09-03" in html
    assert "n=2 real (player, game) observations" in html
    assert "No historical backtest run yet" not in html


def test_clv_tiles_lead_the_at_a_glance_tiles(tmp_path):
    # CLV, not win rate, is the standard proxy for real betting skill (see
    # the Methodology section) - the tiles should say so first.
    html = render_performance_report(str(tmp_path))
    clv_pos = html.index("Mean CLV")
    hit_rate_pos = html.index("Real hit rate")
    assert clv_pos < hit_rate_pos


def test_renders_real_numbers_from_populated_data(tmp_path):
    os.makedirs(tmp_path / "picks")
    os.makedirs(tmp_path / "results")
    os.makedirs(tmp_path / "clv")
    _append_jsonl([_pick("Player A")], str(tmp_path / "picks" / "2026-08-20.jsonl"))
    _append_jsonl(
        [GameOutcome(game_date="2026-08-20", player="Player A", got_hr=True, got_2plus_tb=True, got_hit=True)],
        str(tmp_path / "results" / "2026-08-20.jsonl"),
    )
    _append_jsonl(
        [
            ClvRecord(
                game_date="2026-08-20", recorded_at="x", player="Player A", market="batter_home_runs",
                event="Team A @ Team B", pick_price=650, pick_book="draftkings", closing_price=550,
                closing_book="draftkings", clv_percent=15.4,
            )
        ],
        str(tmp_path / "clv" / "2026-08-20.jsonl"),
    )
    html = render_performance_report(str(tmp_path), generated_at=datetime(2026, 8, 21, tzinfo=timezone.utc))
    assert "Player A" in html
    assert "100.0%" in html  # real hit rate tile
    assert "+15.4%" in html  # mean CLV tile
    assert "1 real day" in html
    # 2026-08-20T18:00:00+00:00 (the _pick fixture's default recorded_at)
    # is 14:00 ET (EDT, UTC-4) - both the raw pick-log cell and the real
    # hit-rate-by-hour breakdown should show that real converted time.
    assert "2026-08-20 14:00 ET" in html
    assert "Real hit rate by hour recorded (ET)" in html
    assert "14:00 ET" in html


def test_pick_log_shows_the_real_caesars_brand_not_the_legacy_api_key(tmp_path):
    # Confirmed live (2026-09-04): The Odds API's real key for Caesars
    # Sportsbook is `williamhill_us` - see market.book_display_name. The
    # pick log only shows resolved picks (joined against a real outcome).
    os.makedirs(tmp_path / "picks")
    os.makedirs(tmp_path / "results")
    _append_jsonl([_pick("Player A", best_book="williamhill_us")], str(tmp_path / "picks" / "2026-08-20.jsonl"))
    _append_jsonl(
        [GameOutcome(game_date="2026-08-20", player="Player A", got_hr=True, got_2plus_tb=True, got_hit=True)],
        str(tmp_path / "results" / "2026-08-20.jsonl"),
    )
    html = render_performance_report(str(tmp_path))
    assert '<td class="book" data-k="book">Caesars</td>' in html


def test_renders_the_weight_refit_comparison_when_components_exist():
    import tempfile

    from mlb_props.refit import MIN_PICKS_TO_FIT

    tmp_path = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp_path, "picks"))
    os.makedirs(os.path.join(tmp_path, "results"))

    picks, outcomes = [], []
    n = MIN_PICKS_TO_FIT + 20
    for i in range(n):
        won = i % 2 == 0
        picks.append(
            _pick(
                f"Player {i}",
                components={
                    "barrel_pct": 90.0 if won else 10.0,
                    "hard_hit_pct": 50.0,
                    "avg_exit_velo": 50.0,
                    "hr_fb_pct": 50.0,
                    "pull_air_pct": 50.0,
                    "platoon_edge": 50.0,
                    "pitcher_allowed": 50.0,
                    "pitch_mix_edge": 50.0,
                    "park_factor": 50.0,
                    "weather_boost": 50.0,
                    "hot_streak": 50.0,
                },
            )
        )
        outcomes.append(GameOutcome(game_date="2026-08-20", player=f"Player {i}", got_hr=won, got_2plus_tb=False, got_hit=False))
    _append_jsonl(picks, os.path.join(tmp_path, "picks", "2026-08-20.jsonl"))
    _append_jsonl(outcomes, os.path.join(tmp_path, "results", "2026-08-20.jsonl"))

    html = render_performance_report(tmp_path)
    assert "Weight refit check" in html
    assert "1+ HR" in html
    assert "Barrel Pct" in html  # a real component name in the comparison table
    # A real, informative component (barrel_pct) vs. an uninformative flat
    # model_prob should measurably beat the current model here.
    assert "Fitted weights measurably beat the current hand-set ones" in html


def test_renders_the_market_blend_comparison_when_market_data_exists():
    import tempfile

    from mlb_props.refit import MIN_PICKS_TO_FIT

    tmp_path = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp_path, "picks"))
    os.makedirs(os.path.join(tmp_path, "results"))

    picks, outcomes = [], []
    n = MIN_PICKS_TO_FIT + 20
    for i in range(n):
        won = i % 2 == 0
        # model_prob is a constant, uninformative 0.5 for every pick;
        # market_fair_prob perfectly separates won/lost - the blend fit
        # should weight almost entirely toward the market and measurably
        # beat pure model_prob on real held-out data.
        picks.append(_pick(f"Player {i}", model_prob=0.5, market_fair_prob=0.95 if won else 0.05))
        outcomes.append(GameOutcome(game_date="2026-08-20", player=f"Player {i}", got_hr=won, got_2plus_tb=False, got_hit=False))
    _append_jsonl(picks, os.path.join(tmp_path, "picks", "2026-08-20.jsonl"))
    _append_jsonl(outcomes, os.path.join(tmp_path, "results", "2026-08-20.jsonl"))

    html = render_performance_report(tmp_path)
    assert "Market blend check" in html
    assert "1+ HR" in html
    assert "measurably beats pure model_prob" in html


def test_escapes_malicious_player_name_in_pick_log(tmp_path):
    os.makedirs(tmp_path / "picks")
    os.makedirs(tmp_path / "results")
    malicious = "<script>alert(1)</script>"
    _append_jsonl([_pick(malicious)], str(tmp_path / "picks" / "2026-08-20.jsonl"))
    _append_jsonl(
        [GameOutcome(game_date="2026-08-20", player=malicious, got_hr=True, got_2plus_tb=True, got_hit=True)],
        str(tmp_path / "results" / "2026-08-20.jsonl"),
    )
    html = render_performance_report(str(tmp_path))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
