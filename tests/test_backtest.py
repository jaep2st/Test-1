"""Covers mlb_props/backtest.py: joining recorded picks against real
resolved outcomes, calibration bucketing, CLV summary, and hit-rate
breakdowns. All synthetic fixtures built directly (no filesystem/network) -
the JSONL round-trip itself is covered in test_results.py.
"""

from mlb_props.backtest import (
    calibration_buckets,
    clv_summary,
    hit_rate_by_lineup_source,
    hit_rate_by_market,
    hit_rate_by_run_hour,
    hit_rate_by_tier,
    latest_results_by_key,
    recorded_at_et,
    resolve_picks,
    units_by_date,
    units_ledger,
    units_summary,
)
from mlb_props.results import ClvRecord, GameOutcome, PickRecord


def _pick(
    player,
    market="batter_home_runs",
    model_prob=0.15,
    tier="agree",
    recorded_at="2026-08-20T18:00:00+00:00",
    game_date="2026-08-20",
    lineup_source="active_roster",
    best_price=650,
    ev_percent_model=25.0,
    books_quoting=4,
):
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
        best_price=best_price,
        best_book="draftkings",
        ev_percent_model=ev_percent_model,
        ev_percent_market=15.0,
        edge_vs_market=0.05,
        books_quoting=books_quoting,
        lineup_source=lineup_source,
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


def test_resolve_picks_keeps_each_days_pick_in_a_real_multi_day_series():
    # Same two teams (event string never carries a date - see PickRecord.key)
    # on three consecutive days, same player/market each day - a completely
    # normal MLB series. Confirmed live (2026-09-07) that deduping picks
    # across every recorded day with PickRecord.key alone (player, market,
    # event - no game_date) collapsed a real series down to only its last
    # day, discarding 89% of all recorded pick snapshots from every
    # Performance-page stat. Each day's own pick and own real outcome must
    # both survive independently.
    picks = [
        _pick("Player A", model_prob=0.10, recorded_at="2026-08-20T18:00:00+00:00", game_date="2026-08-20"),
        _pick("Player A", model_prob=0.20, recorded_at="2026-08-21T18:00:00+00:00", game_date="2026-08-21"),
        _pick("Player A", model_prob=0.30, recorded_at="2026-08-22T18:00:00+00:00", game_date="2026-08-22"),
    ]
    results = [
        _outcome("Player A", got_hr=True, game_date="2026-08-20"),
        _outcome("Player A", got_hr=False, game_date="2026-08-21"),
        _outcome("Player A", got_hr=True, game_date="2026-08-22"),
    ]
    resolved = resolve_picks(picks, results)
    assert len(resolved) == 3
    by_date = {r.pick.game_date: r for r in resolved}
    assert by_date["2026-08-20"].won is True
    assert by_date["2026-08-20"].pick.model_prob == 0.10
    assert by_date["2026-08-21"].won is False
    assert by_date["2026-08-22"].won is True


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


def test_hit_rate_by_lineup_source_groups_correctly():
    picks = [_pick("A", lineup_source="confirmed"), _pick("B", lineup_source="active_roster")]
    results = [_outcome("A", got_hr=True), _outcome("B", got_hr=False)]
    resolved = resolve_picks(picks, results)
    groups = {g.key: g for g in hit_rate_by_lineup_source(resolved)}
    assert groups["confirmed"].hit_rate == 1.0
    assert groups["active_roster"].hit_rate == 0.0


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


def test_units_ledger_only_counts_picks_that_clear_the_real_bet_bar():
    # Pick A: real edge, "agree" tier, books_quoting=4 -> a genuine
    # recommended bet, quarter-Kelly sized. Pick B: below
    # MIN_EV_PERCENT_TO_RECOMMEND (3.0) -> never a real bet, must be
    # excluded even though it's a resolved, won pick.
    picks = [
        _pick("Player A", model_prob=0.40, best_price=200, tier="agree", books_quoting=4, ev_percent_model=25.0),
        _pick("Player B", model_prob=0.40, best_price=200, tier="agree", books_quoting=4, ev_percent_model=1.0),
    ]
    results = [_outcome("Player A", got_hr=True), _outcome("Player B", got_hr=True)]
    resolved = resolve_picks(picks, results)
    ledger = units_ledger(resolved)

    assert len(ledger) == 1
    assert ledger[0].player == "Player A"
    assert ledger[0].units == 2.5
    assert ledger[0].won is True
    assert ledger[0].net_units == 5.0


def test_units_ledger_records_a_real_loss_as_negative_net_units():
    picks = [_pick("Player A", model_prob=0.40, best_price=200, tier="agree", books_quoting=4)]
    results = [_outcome("Player A", got_hr=False)]
    resolved = resolve_picks(picks, results)
    ledger = units_ledger(resolved)

    assert len(ledger) == 1
    assert ledger[0].won is False
    assert ledger[0].units == 2.5
    assert ledger[0].net_units == -2.5


def test_units_ledger_reads_tier_through_effective_tier_not_the_raw_field():
    # A stale "agree" (books_quoting=1, below MIN_BOOKS_FOR_MARKET_AGREE)
    # must be sized at the speculative fraction, not quarter-Kelly - see
    # edges.effective_tier's docstring.
    picks = [_pick("Player A", model_prob=0.40, best_price=200, tier="agree", books_quoting=1)]
    results = [_outcome("Player A", got_hr=True)]
    resolved = resolve_picks(picks, results)
    ledger = units_ledger(resolved)

    assert len(ledger) == 1
    assert ledger[0].tier == "model_only"  # corrected, not the raw "agree"
    assert ledger[0].units == 1.5  # speculative (1/8-Kelly) sizing, not 2.5 (quarter-Kelly)


def test_units_ledger_excludes_a_pick_with_no_real_price():
    picks = [_pick("Player A", model_prob=0.40, best_price=None, tier="no_market")]
    results = [_outcome("Player A", got_hr=True)]
    resolved = resolve_picks(picks, results)
    assert units_ledger(resolved) == []


def test_units_summary_splits_strong_vs_speculative_and_computes_roi():
    picks = [
        _pick("Player A", model_prob=0.40, best_price=200, tier="agree", books_quoting=4, game_date="2026-08-20"),
        _pick("Player B", model_prob=0.25, best_price=400, tier="model_only", books_quoting=1, game_date="2026-08-20"),
    ]
    results = [_outcome("Player A", got_hr=True, game_date="2026-08-20"), _outcome("Player B", got_hr=False, game_date="2026-08-20")]
    resolved = resolve_picks(picks, results)
    summary = units_summary(units_ledger(resolved))

    assert summary.n_bets == 2
    assert summary.strong_n_bets == 1
    assert summary.strong_net_units == 5.0
    assert summary.speculative_n_bets == 1
    assert summary.speculative_net_units == -1.0
    assert summary.net_units == 4.0
    assert summary.total_units_staked == 3.5  # 2.5 + 1.0
    assert summary.roi_percent == round(4.0 / 3.5 * 100.0, 1)


def test_units_summary_on_no_real_bets_returns_zeros_not_none_crash():
    summary = units_summary([])
    assert summary.n_bets == 0
    assert summary.net_units == 0.0
    assert summary.roi_percent is None


def test_units_by_date_tracks_a_real_running_total_across_days():
    picks = [
        _pick("Player A", model_prob=0.40, best_price=200, tier="agree", books_quoting=4,
              recorded_at="2026-08-20T18:00:00+00:00", game_date="2026-08-20"),
        _pick("Player A", model_prob=0.40, best_price=200, tier="agree", books_quoting=4,
              recorded_at="2026-08-21T18:00:00+00:00", game_date="2026-08-21"),
    ]
    results = [
        _outcome("Player A", got_hr=True, game_date="2026-08-20"),
        _outcome("Player A", got_hr=False, game_date="2026-08-21"),
    ]
    resolved = resolve_picks(picks, results)
    daily = units_by_date(units_ledger(resolved))

    assert len(daily) == 2
    assert daily[0].game_date == "2026-08-20"
    assert daily[0].net_units == 5.0
    assert daily[0].cumulative_units == 5.0
    assert daily[1].game_date == "2026-08-21"
    assert daily[1].net_units == -2.5
    assert daily[1].cumulative_units == 2.5
