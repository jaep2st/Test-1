"""Covers mlb_props/backtest.py: joining recorded picks against real
resolved outcomes, calibration bucketing, CLV summary, and hit-rate
breakdowns. All synthetic fixtures built directly (no filesystem/network) -
the JSONL round-trip itself is covered in test_results.py.
"""

from mlb_props.backtest import (
    calibration_buckets,
    clv_summary,
    hit_rate_by_market,
    hit_rate_by_run_hour,
    hit_rate_by_tier,
    latest_results_by_key,
    recorded_at_et,
    resolve_picks,
)
from mlb_props.results import ClvRecord, GameOutcome, PickRecord


def _pick(player, market="batter_home_runs", model_prob=0.15, tier="agree", recorded_at="2026-08-20T18:00:00+00:00", game_date="2026-08-20"):
    return PickRecord(
        game_date=game_date,
        recorded_at=recorded_at,
        player=player,
        market=market,
        event="Team A @ Team B",
        tier=tier,
        model_score=70.0,
        model_prob=model_prob,
        bp_model_prob=None,
        market_fair_prob=0.10,
        best_price=650,
        best_book="draftkings",
        ev_percent_model=25.0,
        ev_percent_market=15.0,
        edge_vs_market=0.05,
        books_quoting=4,
    )


def _outcome(player, got_hr=False, got_2plus_tb=False, got_hit=False, game_date="2026-08-20"):
    return GameOutcome(game_date=game_date, player=player, got_hr=got_hr, got_2plus_tb=got_2plus_tb, got_hit=got_hit)


def test_resolve_picks_joins_pick_to_its_real_outcome():
    picks = [_pick("Player A", model_prob=0.20)]
    results = [_outcome("Player A", got_hr=True)]
    resolved = resolve_picks(picks, results)
    assert len(resolved) == 1
    assert resolved[0].won is True
    assert resolved[0].pick.player == "Player A"


def test_resolve_picks_excludes_picks_with_no_resolved_outcome_yet():
    picks = [_pick("Player A")]
    resolved = resolve_picks(picks, [])
    assert resolved == []


def test_resolve_picks_keeps_only_the_latest_same_day_pick_snapshot():
    picks = [
        _pick("Player A", model_prob=0.10, recorded_at="2026-08-20T11:00:00+00:00"),
        _pick("Player A", model_prob=0.30, recorded_at="2026-08-20T18:00:00+00:00"),
    ]
    results = [_outcome("Player A", got_hr=True)]
    resolved = resolve_picks(picks, results)
    assert len(resolved) == 1
    assert resolved[0].pick.model_prob == 0.30


def test_latest_results_by_key_takes_the_last_appended_resolution():
    results = [_outcome("Player A", got_hr=False), _outcome("Player A", got_hr=True)]
    by_key = latest_results_by_key(results)
    assert by_key[("player a", "2026-08-20")].got_hr is True


def test_calibration_buckets_reports_predicted_vs_actual_per_decile():
    picks = [_pick("Player A", model_prob=0.15), _pick("Player B", model_prob=0.18)]
    results = [_outcome("Player A", got_hr=True), _outcome("Player B", got_hr=False)]
    resolved = resolve_picks(picks, results)
    buckets = calibration_buckets(resolved, n_buckets=10)
    assert len(buckets) == 10
    decile_1 = next(b for b in buckets if b.lo == 0.1)
    assert decile_1.n == 2
    assert decile_1.actual_rate == 0.5


def test_calibration_buckets_marks_empty_ranges_with_zero_n_not_dropped():
    resolved = resolve_picks([_pick("Player A", model_prob=0.95)], [_outcome("Player A", got_hr=True)])
    buckets = calibration_buckets(resolved, n_buckets=10)
    empty_bucket = next(b for b in buckets if b.lo == 0.1)
    assert empty_bucket.n == 0
    assert empty_bucket.actual_rate is None


def test_clv_summary_on_no_records_returns_none_not_zero():
    summary = clv_summary([])
    assert summary.n == 0
    assert summary.mean_clv_percent is None
    assert summary.beat_close_percent is None


def test_clv_summary_computes_mean_and_beat_close_share():
    records = [
        ClvRecord(game_date="2026-08-20", recorded_at="x", player="A", market="batter_home_runs", event="e", pick_price=650, pick_book="dk", closing_price=550, closing_book="dk", clv_percent=10.0),
        ClvRecord(game_date="2026-08-20", recorded_at="x", player="B", market="batter_home_runs", event="e", pick_price=-140, pick_book="fd", closing_price=-120, closing_book="fd", clv_percent=-5.0),
    ]
    summary = clv_summary(records)
    assert summary.n == 2
    assert summary.mean_clv_percent == 2.5
    assert summary.beat_close_percent == 50.0


def test_hit_rate_by_market_groups_correctly():
    picks = [_pick("A", market="batter_home_runs"), _pick("B", market="batter_hits")]
    results = [_outcome("A", got_hr=True), _outcome("B", got_hit=False)]
    resolved = resolve_picks(picks, results)
    groups = {g.key: g for g in hit_rate_by_market(resolved)}
    assert groups["batter_home_runs"].hit_rate == 1.0
    assert groups["batter_hits"].hit_rate == 0.0


def test_hit_rate_by_tier_groups_correctly():
    picks = [_pick("A", tier="agree"), _pick("B", tier="model_only")]
    results = [_outcome("A", got_hr=True), _outcome("B", got_hr=True)]
    resolved = resolve_picks(picks, results)
    groups = {g.key: g for g in hit_rate_by_tier(resolved)}
    assert groups["agree"].n == 1
    assert groups["model_only"].n == 1


def test_recorded_at_et_converts_real_utc_to_us_eastern():
    # 22:32 UTC on 2026-08-20 is EDT (UTC-4) - 18:32 ET.
    pick = _pick("A", recorded_at="2026-08-20T22:32:00+00:00")
    et = recorded_at_et(pick)
    assert et.hour == 18
    assert et.minute == 32
    assert et.tzinfo is not None


def test_hit_rate_by_run_hour_groups_by_the_real_recorded_hour():
    picks = [
        _pick("A", recorded_at="2026-08-20T15:00:00+00:00"),  # 11:00 ET
        _pick("B", recorded_at="2026-08-20T22:30:00+00:00"),  # 18:00 ET
        _pick("C", recorded_at="2026-08-20T15:10:00+00:00"),  # 11:00 ET - same bucket as A despite a different minute
    ]
    results = [_outcome("A", got_hr=True), _outcome("B", got_hr=False), _outcome("C", got_hr=False)]
    resolved = resolve_picks(picks, results)
    groups = {g.key: g for g in hit_rate_by_run_hour(resolved)}
    assert groups["11:00 ET"].n == 2
    assert groups["11:00 ET"].hit_rate == 0.5
    assert groups["18:00 ET"].n == 1
    assert groups["18:00 ET"].hit_rate == 0.0


def test_hit_rate_by_run_hour_counts_a_manual_off_schedule_run_honestly():
    # A pick recorded at an hour outside this project's four scheduled
    # crons (11am/12pm/6:30pm/10:30pm ET) - e.g. a manual workflow_dispatch
    # at 3pm ET - must still get its own real bucket, never be dropped or
    # folded into the nearest scheduled hour.
    picks = [_pick("A", recorded_at="2026-08-20T19:00:00+00:00")]  # 15:00 ET
    results = [_outcome("A", got_hr=True)]
    resolved = resolve_picks(picks, results)
    groups = {g.key: g for g in hit_rate_by_run_hour(resolved)}
    assert "15:00 ET" in groups
    assert groups["15:00 ET"].n == 1
