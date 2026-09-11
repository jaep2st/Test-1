"""Covers mlb_props/backtest.py: joining recorded picks against real
resolved outcomes, calibration bucketing, CLV summary, and hit-rate
breakdowns. All synthetic fixtures built directly (no filesystem/network) -
the JSONL round-trip itself is covered in test_results.py.
"""

from mlb_props.backtest import (
    calibration_buckets,
    clv_summary,
    edge_confidence,
    hit_rate_by_lineup_source,
    hit_rate_by_market,
    hit_rate_by_run_hour,
    hit_rate_by_tier,
    last_priced_pick_by_key,
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


def test_resolve_picks_prefers_the_latest_priced_snapshot_over_a_later_unpriced_one():
    # Confirmed live (2026-09-08, Coby Mayo/Guardians@Orioles): a real
    # confirmed agree-tier pick at 4pm ET had its market pulled entirely by
    # 6:36pm ET once the game reached "Warmup" (best_price=None,
    # tier="no_market") - a completely normal, expected market-closing
    # pattern near first pitch. The later, unpriced snapshot must not erase
    # the earlier real, bettable price from the historical record - "latest
    # wins" would otherwise silently drop a real recommended bet from every
    # stat on the Performance page.
    picks = [
        _pick("Player A", best_price=650, tier="agree", recorded_at="2026-08-20T16:00:00+00:00"),
        _pick("Player A", best_price=None, tier="no_market", recorded_at="2026-08-20T22:36:00+00:00"),
    ]
    results = [_outcome("Player A", got_hr=True)]
    resolved = resolve_picks(picks, results)
    assert len(resolved) == 1
    assert resolved[0].pick.best_price == 650
    assert resolved[0].pick.tier == "agree"


def test_resolve_picks_keeps_the_latest_when_neither_snapshot_was_ever_priced():
    picks = [
        _pick("Player A", best_price=None, tier="no_market", recorded_at="2026-08-20T11:00:00+00:00"),
        _pick("Player A", best_price=None, tier="no_market", recorded_at="2026-08-20T18:00:00+00:00"),
    ]
    results = [_outcome("Player A", got_hr=True)]
    resolved = resolve_picks(picks, results)
    assert len(resolved) == 1
    assert resolved[0].pick.recorded_at == "2026-08-20T18:00:00+00:00"


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
    assert ledger[0].units == 0.5  # speculative (1/16-Kelly) sizing, not 2.5 (quarter-Kelly)


def test_units_ledger_excludes_a_pick_with_no_real_price():
    picks = [_pick("Player A", model_prob=0.40, best_price=None, tier="no_market")]
    results = [_outcome("Player A", got_hr=True)]
    resolved = resolve_picks(picks, results)
    assert units_ledger(resolved) == []


def test_units_ledger_dedupes_correlated_legs_on_the_same_player_same_day():
    # Same real player, same real game date, two different markets both
    # clearing the bet bar - the exact 2026-09-10 correlated-double-stake
    # pattern (see _dedupe_best_leg_per_player_per_day's docstring). Only
    # the higher-EV% leg should end up staked, not both.
    picks = [
        _pick("Player A", market="batter_hits", tier="agree", books_quoting=4, ev_percent_model=10.0, game_date="2026-08-20"),
        _pick("Player A", market="batter_total_bases", tier="agree", books_quoting=4, ev_percent_model=25.0, game_date="2026-08-20"),
    ]
    results = [_outcome("Player A", got_hr=True, game_date="2026-08-20")]
    resolved = resolve_picks(picks, results)
    ledger = units_ledger(resolved)

    assert len(ledger) == 1
    assert ledger[0].market == "batter_total_bases"
    assert ledger[0].ev_percent_model == 25.0


def test_units_ledger_prefers_agree_leg_over_higher_ev_model_only_leg():
    picks = [
        _pick("Player A", market="batter_hits", tier="agree", books_quoting=4, ev_percent_model=10.0, game_date="2026-08-20"),
        _pick("Player A", market="batter_total_bases", tier="model_only", books_quoting=1, ev_percent_model=40.0, game_date="2026-08-20"),
    ]
    results = [_outcome("Player A", got_hr=True, game_date="2026-08-20")]
    resolved = resolve_picks(picks, results)
    ledger = units_ledger(resolved)

    assert len(ledger) == 1
    assert ledger[0].market == "batter_hits"
    assert ledger[0].tier == "agree"


def test_units_ledger_does_not_dedupe_the_same_player_across_different_days():
    picks = [
        _pick("Player A", tier="agree", books_quoting=4, game_date="2026-08-20"),
        _pick("Player A", tier="agree", books_quoting=4, game_date="2026-08-21"),
    ]
    results = [
        _outcome("Player A", got_hr=True, game_date="2026-08-20"),
        _outcome("Player A", got_hr=True, game_date="2026-08-21"),
    ]
    resolved = resolve_picks(picks, results)
    assert len(units_ledger(resolved)) == 2


def test_units_ledger_marks_best_bets_capped_and_ev_floored_per_day():
    # 6 real "agree" candidates clearing BEST_BETS_MIN_EV_PERCENT (8.0) on
    # one day, plus one speculative pick that must never qualify - only the
    # top BEST_BETS_MAX_COUNT (5) by EV% get is_best_bet=True.
    picks = [
        _pick(f"Player {i}", market="batter_hits", tier="agree", books_quoting=4, ev_percent_model=ev, game_date="2026-08-20")
        for i, ev in enumerate([30.0, 25.0, 20.0, 15.0, 10.0, 9.0], start=1)
    ]
    picks.append(
        _pick("Player Spec", market="batter_hits", tier="model_only", books_quoting=1, ev_percent_model=50.0, game_date="2026-08-20")
    )
    results = [_outcome(p.player, got_hr=True, game_date="2026-08-20") for p in picks]
    resolved = resolve_picks(picks, results)
    ledger = units_ledger(resolved)

    best = {r.player for r in ledger if r.is_best_bet}
    assert best == {"Player 1", "Player 2", "Player 3", "Player 4", "Player 5"}
    assert "Player 6" not in best  # 6th-highest EV%, gets capped out at 5
    assert "Player Spec" not in best  # speculative tier never qualifies


def test_units_ledger_best_bets_excludes_agree_picks_below_the_real_ev_floor():
    picks = [
        _pick("Player Strong", market="batter_hits", tier="agree", books_quoting=4, ev_percent_model=10.0, game_date="2026-08-20"),
        _pick("Player Weak", market="batter_hits", tier="agree", books_quoting=4, ev_percent_model=5.0, game_date="2026-08-20"),
    ]
    results = [_outcome(p.player, got_hr=True, game_date="2026-08-20") for p in picks]
    resolved = resolve_picks(picks, results)
    ledger = units_ledger(resolved)

    best = {r.player for r in ledger if r.is_best_bet}
    assert best == {"Player Strong"}  # Player Weak clears MIN_EV_PERCENT_TO_RECOMMEND but not BEST_BETS_MIN_EV_PERCENT


def test_units_summary_reports_best_bets_as_a_subset_of_strong():
    picks = [
        _pick("Player A", tier="agree", books_quoting=4, ev_percent_model=25.0, best_price=200, model_prob=0.40, game_date="2026-08-20"),
        _pick("Player B", tier="agree", books_quoting=4, ev_percent_model=5.0, best_price=200, model_prob=0.40, game_date="2026-08-20"),
    ]
    results = [
        _outcome("Player A", got_hr=True, game_date="2026-08-20"),
        _outcome("Player B", got_hr=False, game_date="2026-08-20"),
    ]
    resolved = resolve_picks(picks, results)
    summary = units_summary(units_ledger(resolved))

    assert summary.strong_n_bets == 2
    assert summary.best_bets_n_bets == 1  # only Player A clears the 8.0 EV% Best Bets floor
    assert summary.best_bets_net_units == 5.0


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
    assert summary.speculative_net_units == -0.5
    assert summary.net_units == 4.5
    assert summary.total_units_staked == 3.0  # 2.5 + 0.5
    assert summary.roi_percent == round(4.5 / 3.0 * 100.0, 1)


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


def test_last_priced_pick_by_key_keeps_the_priced_snapshot_over_a_later_unpriced_one():
    # html_report.py's stale-pregame-price fallback (a started game whose
    # roster panel falls back to the last real price this project recorded
    # for it today) is built from this function - it must never surface
    # the priceless "market pulled near first pitch" snapshot instead of
    # the real, once-bettable price. Same real scenario as
    # test_resolve_picks_prefers_the_latest_priced_snapshot_over_a_later_unpriced_one.
    picks = [
        _pick("Player A", best_price=650, tier="agree", recorded_at="2026-08-20T16:00:00+00:00"),
        _pick("Player A", best_price=None, tier="no_market", recorded_at="2026-08-20T22:36:00+00:00"),
    ]
    latest = last_priced_pick_by_key(picks)
    assert len(latest) == 1
    pick = latest[picks[0].key]
    assert pick.best_price == 650
    assert pick.tier == "agree"


def test_last_priced_pick_by_key_keys_by_player_market_event_not_game_date():
    picks = [_pick("Player A", best_price=650, market="batter_home_runs")]
    latest = last_priced_pick_by_key(picks)
    assert set(latest.keys()) == {("player a", "batter_home_runs", "team a @ team b")}


# --- edge_confidence: is a tier's real per-bet edge statistically
# distinguishable from zero yet, or still indistinguishable from noise. ---


def _strong_ledger(n, won):
    # model_prob=0.40 @ +200 "agree" always sizes to 2.5u (quarter-Kelly)
    # and pays 5.0u on a win / -2.5u on a loss - identical every time, so
    # a run of all-wins or all-losses gives a perfectly tight (zero-
    # variance) confidence interval, the clearest possible test case.
    picks = [
        _pick(f"Player {i}", model_prob=0.40, best_price=200, tier="agree", books_quoting=4, game_date="2026-08-20")
        for i in range(n)
    ]
    results = [_outcome(f"Player {i}", got_hr=won, game_date="2026-08-20") for i in range(n)]
    return units_ledger(resolve_picks(picks, results))


def test_edge_confidence_reports_not_enough_data_below_two_bets():
    strong, speculative = edge_confidence(_strong_ledger(1, True))
    assert strong.n_bets == 1
    assert strong.significant is None
    assert speculative.n_bets == 0
    assert speculative.significant is None


def test_edge_confidence_flags_a_real_positive_edge_when_the_ci_excludes_zero():
    ledger = _strong_ledger(20, True)  # 20 identical wins - zero variance, CI = [5.0, 5.0]
    strong, speculative = edge_confidence(ledger)
    assert strong.tier_label == "Strong (agree)"
    assert strong.n_bets == 20
    assert strong.significant is True
    assert strong.ci_lo_95 > 0


def test_edge_confidence_flags_a_real_negative_edge_when_the_ci_excludes_zero():
    ledger = _strong_ledger(20, False)  # 20 identical losses - zero variance, CI = [-2.5, -2.5]
    strong, speculative = edge_confidence(ledger)
    assert strong.significant is False
    assert strong.ci_hi_95 < 0


def test_edge_confidence_says_not_proven_yet_when_the_ci_still_contains_zero():
    # A real mix of wins and losses at a small sample - real variance,
    # nowhere near enough data to say anything with 95% confidence.
    picks = [
        _pick(f"Player {i}", model_prob=0.40, best_price=200, tier="agree", books_quoting=4, game_date="2026-08-20")
        for i in range(4)
    ]
    results = [
        _outcome("Player 0", got_hr=True, game_date="2026-08-20"),
        _outcome("Player 1", got_hr=False, game_date="2026-08-20"),
        _outcome("Player 2", got_hr=True, game_date="2026-08-20"),
        _outcome("Player 3", got_hr=False, game_date="2026-08-20"),
    ]
    ledger = units_ledger(resolve_picks(picks, results))
    strong, speculative = edge_confidence(ledger)
    assert strong.n_bets == 4
    assert strong.significant is None
    assert strong.ci_lo_95 < 0 < strong.ci_hi_95


def test_edge_confidence_splits_strong_and_speculative_independently():
    strong_picks = [
        _pick("Player S1", model_prob=0.40, best_price=200, tier="agree", books_quoting=4, game_date="2026-08-20"),
    ]
    speculative_picks = [
        _pick("Player M1", model_prob=0.40, best_price=200, tier="model_only", books_quoting=4, game_date="2026-08-20"),
    ]
    results = [
        _outcome("Player S1", got_hr=True, game_date="2026-08-20"),
        _outcome("Player M1", got_hr=False, game_date="2026-08-20"),
    ]
    ledger = units_ledger(resolve_picks(strong_picks + speculative_picks, results))
    strong, speculative = edge_confidence(ledger)
    assert strong.n_bets == 1
    assert speculative.n_bets == 1
    assert strong.mean_net_units_per_bet > 0
    assert speculative.mean_net_units_per_bet < 0
