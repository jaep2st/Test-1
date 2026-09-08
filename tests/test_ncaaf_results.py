from ncaaf.backtest import latest_results_by_key, resolve_picks
from ncaaf.results import GameOutcome, PickRecord, hit_for, latest_pick_per_key


def _pick(**overrides):
    base = dict(
        season=2026, week=3, recorded_at="2026-09-15T12:00:00+00:00", event="Away U @ Home U",
        market="spreads", side="home", team="Home U", point=-6.5, tier="agree", model_prob=0.65,
        market_fair_prob=0.55, best_price=-110, best_book="draftkings", ev_percent_model=8.0,
        ev_percent_market=5.0, edge_vs_market=0.10, books_quoting=2, predicted_margin=10.0, predicted_total=52.0,
    )
    base.update(overrides)
    return PickRecord(**base)


def _outcome(home_points, away_points, **overrides):
    base = dict(season=2026, week=3, event="Away U @ Home U", home_points=home_points, away_points=away_points)
    base.update(overrides)
    return GameOutcome(**base)


def test_spread_home_favorite_covers():
    pick = _pick(market="spreads", side="home", point=-6.5)
    assert hit_for(pick, _outcome(home_points=30, away_points=20)) is True  # won by 10, covers -6.5
    assert hit_for(pick, _outcome(home_points=27, away_points=24)) is False  # won by 3, doesn't cover -6.5


def test_spread_away_side_is_the_inverse():
    pick = _pick(market="spreads", side="away", point=6.5)
    assert hit_for(pick, _outcome(home_points=30, away_points=20)) is False
    assert hit_for(pick, _outcome(home_points=27, away_points=24)) is True


def test_spread_exact_push_is_none():
    pick = _pick(market="spreads", side="home", point=-3.0)
    assert hit_for(pick, _outcome(home_points=23, away_points=20)) is None


def test_h2h_favorite_and_underdog():
    home_pick = _pick(market="h2h", side="home", point=None)
    away_pick = _pick(market="h2h", side="away", point=None)
    outcome = _outcome(home_points=24, away_points=21)
    assert hit_for(home_pick, outcome) is True
    assert hit_for(away_pick, outcome) is False


def test_totals_over_under_and_push():
    over_pick = _pick(market="totals", side="over", point=45.5)
    under_pick = _pick(market="totals", side="under", point=45.5)
    assert hit_for(over_pick, _outcome(home_points=24, away_points=24)) is True  # total 48 > 45.5
    assert hit_for(under_pick, _outcome(home_points=24, away_points=24)) is False
    push_pick = _pick(market="totals", side="over", point=48.0)
    assert hit_for(push_pick, _outcome(home_points=24, away_points=24)) is None


def test_latest_pick_per_key_keeps_the_most_recent_snapshot():
    early = _pick(recorded_at="2026-09-14T12:00:00+00:00", best_price=-110)
    late = _pick(recorded_at="2026-09-15T18:00:00+00:00", best_price=-120)
    latest = latest_pick_per_key([early, late])
    assert latest[early.key].best_price == -120


def test_latest_results_by_key_keeps_events_from_different_weeks_separate():
    # The same two teams meeting again in a later week/season must not
    # silently overwrite an earlier real result for the same event string.
    week3 = _outcome(28, 14, week=3)
    week9 = _outcome(10, 31, week=9, event="Away U @ Home U")
    by_key = latest_results_by_key([week3, week9])
    assert by_key[("away u @ home u", 2026, 3)].home_points == 28
    assert by_key[("away u @ home u", 2026, 9)].home_points == 10


def test_resolve_picks_matches_pick_and_outcome_by_season_and_week():
    pick_w3 = _pick(week=3, market="h2h", side="home", point=None)
    pick_w9 = _pick(week=9, market="h2h", side="home", point=None, recorded_at="2026-11-01T12:00:00+00:00")
    outcome_w3 = _outcome(30, 10, week=3)  # home wins
    outcome_w9 = _outcome(10, 30, week=9)  # home loses

    resolved = resolve_picks([pick_w3, pick_w9], [outcome_w3, outcome_w9])
    by_week = {r.pick.week: r.won for r in resolved}
    assert by_week[3] is True
    assert by_week[9] is False
