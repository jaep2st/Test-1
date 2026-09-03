"""Covers mlb_props/refit.py: the real logistic-regression weight refit
against resolved picks' recorded components. All synthetic fixtures built
directly (no filesystem/network) - reuses test_backtest.py's ResolvedPick/
PickRecord/GameOutcome joining, already covered there.
"""

import itertools
from dataclasses import replace
from datetime import date, timedelta

from mlb_props.backtest import ResolvedPick
from mlb_props.refit import MIN_PICKS_TO_FIT, fit_all_market_blends, fit_market_blend, refit_all_markets, refit_market
from mlb_props.results import PickRecord


def _pick(player, market="batter_home_runs", model_prob=0.15, components=None, game_date="2026-08-20", market_fair_prob=0.10):
    return PickRecord(
        game_date=game_date,
        recorded_at=f"{game_date}T18:00:00+00:00",
        player=player,
        market=market,
        event="Team A @ Team B",
        tier="agree",
        model_score=70.0,
        model_prob=model_prob,
        bp_model_prob=None,
        market_fair_prob=market_fair_prob,
        best_price=650,
        best_book="draftkings",
        ev_percent_model=25.0,
        ev_percent_market=15.0,
        edge_vs_market=0.05,
        books_quoting=4,
        components=components or {},
    )


def test_refit_market_returns_none_for_an_unrecognized_market():
    resolved = [ResolvedPick(pick=_pick("Player A", market="not_a_real_market", components={"x": 50.0}), won=True)]
    assert refit_market("not_a_real_market", resolved) is None


def test_refit_market_returns_none_when_no_resolved_pick_has_real_components():
    # Every pick recorded before the components field existed - {} is the
    # honest default, not a fittable feature set.
    resolved = [ResolvedPick(pick=_pick(f"Player {i}"), won=(i % 2 == 0)) for i in range(50)]
    assert refit_market("batter_home_runs", resolved) is None


def test_refit_market_flags_a_small_sample_as_unreliable_but_still_returns_a_result():
    resolved = [
        ResolvedPick(pick=_pick(f"Player {i}", components={"barrel_pct": 50.0}), won=(i % 2 == 0)) for i in range(10)
    ]
    result = refit_market("batter_home_runs", resolved)
    assert result is not None
    assert result.reliable is False
    assert result.n_train + result.n_test == 10


def _perfect_predictor_rows(n_won, n_lost):
    """Real (in the sense of a real logistic-regression exercise, not
    fabricated market data) synthetic rows where barrel_pct alone
    perfectly separates won/lost, every other HR component held at a
    constant, uninformative 50.0 - proves the fit actually learns real
    signal from real features, not just noise.

    Interleaved across distinct, increasing real `game_date`s (rather
    than all winners then all losers on one date) so refit_market's
    walk-forward split - see refit.py's `_time_ordered_split` - produces
    a held-out tail that's actually representative of both classes, the
    same way a real multi-week sample would be. Clustering every winner
    before every loser on a single date would make the held-out slice
    artificially one-sided, which isn't a real backtesting scenario.
    """
    winners = [
        ResolvedPick(pick=_pick(f"Winner {i}", components={"barrel_pct": 95.0, "hard_hit_pct": 50.0}), won=True)
        for i in range(n_won)
    ]
    losers = [
        ResolvedPick(pick=_pick(f"Loser {i}", components={"barrel_pct": 5.0, "hard_hit_pct": 50.0}), won=False)
        for i in range(n_lost)
    ]
    interleaved = [r for pair in itertools.zip_longest(winners, losers) for r in pair if r is not None]
    start = date(2026, 6, 1)
    resolved = []
    for i, r in enumerate(interleaved):
        game_date = (start + timedelta(days=i)).isoformat()
        resolved.append(ResolvedPick(pick=replace(r.pick, game_date=game_date, recorded_at=f"{game_date}T18:00:00+00:00"), won=r.won))
    return resolved


def test_refit_learns_a_real_perfect_predictor_component():
    resolved = _perfect_predictor_rows(40, 40)
    result = refit_market("batter_home_runs", resolved)
    assert result is not None
    assert result.reliable is True
    # barrel_pct should dominate every other component's fitted importance -
    # it's the only one that actually explains the real outcome.
    assert result.fitted_importance["barrel_pct"] == max(result.fitted_importance.values())
    assert result.fitted_importance["barrel_pct"] > 0.5
    # A near-perfect real separator should achieve genuinely low held-out
    # log-loss, not just a numerically-lower-than-chance one.
    assert result.fitted_test_log_loss is not None
    assert result.fitted_test_log_loss < 0.4


def test_refit_compares_honestly_against_the_current_recorded_model_prob():
    # The CURRENT model's own recorded model_prob is a near-perfect
    # predictor here; the real components carry no signal at all (held
    # constant across every row) - a fit trained only on noise should NOT
    # beat that, and improves_on_current must say so honestly.
    resolved = []
    for i in range(40):
        resolved.append(
            ResolvedPick(pick=_pick(f"Winner {i}", model_prob=0.98, components={"barrel_pct": 50.0}), won=True)
        )
    for i in range(40):
        resolved.append(
            ResolvedPick(pick=_pick(f"Loser {i}", model_prob=0.02, components={"barrel_pct": 50.0}), won=False)
        )
    result = refit_market("batter_home_runs", resolved)
    assert result is not None
    assert result.current_test_log_loss is not None
    assert result.current_test_log_loss < 0.3  # the current model is genuinely near-perfect here
    assert result.improves_on_current is False


def test_refit_all_markets_only_includes_markets_with_real_components():
    resolved = _perfect_predictor_rows(30, 30)  # batter_home_runs only
    resolved.append(ResolvedPick(pick=_pick("No Features", market="batter_total_bases"), won=True))
    results = refit_all_markets(resolved)
    assert [r.market for r in results] == ["batter_home_runs"]


def test_min_picks_to_fit_is_a_real_positive_floor():
    # Guards against an accidental 0/negative constant silently disabling
    # the reliable/unreliable distinction entirely.
    assert MIN_PICKS_TO_FIT > 0


def test_refit_market_splits_chronologically_not_randomly():
    # Real walk-forward guarantee: every training row's game_date must be
    # no later than every test row's game_date, even when the input list
    # itself isn't sorted (a real data/picks/*.jsonl load order isn't
    # guaranteed to be date-sorted either). A random-shuffle split (the
    # old behavior) would routinely violate this.
    from mlb_props.refit import TEST_FRACTION, _time_ordered_split

    dates = [f"2026-0{6 if i < 60 else 7}-{(i % 28) + 1:02d}" for i in range(80)]
    resolved = [
        ResolvedPick(pick=_pick(f"Player {i}", components={"barrel_pct": 50.0}, game_date=d), won=(i % 3 == 0))
        for i, d in enumerate(dates)
    ]
    import random

    shuffled = resolved[:]
    random.Random(1).shuffle(shuffled)

    train, test = _time_ordered_split(shuffled, TEST_FRACTION)
    assert train and test
    latest_train_date = max(r.pick.game_date for r in train)
    earliest_test_date = min(r.pick.game_date for r in test)
    assert latest_train_date <= earliest_test_date


# ---------------------------------------------------------------------------
# fit_market_blend / fit_all_market_blends
# ---------------------------------------------------------------------------


def _blend_rows(n_won, n_lost, winner_model_prob, loser_model_prob, winner_market_prob, loser_market_prob, market="batter_home_runs"):
    """Same interleave-across-real-dates shape as _perfect_predictor_rows
    above (see that helper's docstring for why), but varying model_prob/
    market_fair_prob directly instead of scoring components - what
    fit_market_blend actually fits against.
    """
    winners = [
        ResolvedPick(
            pick=_pick(f"Winner {i}", market=market, model_prob=winner_model_prob, market_fair_prob=winner_market_prob),
            won=True,
        )
        for i in range(n_won)
    ]
    losers = [
        ResolvedPick(
            pick=_pick(f"Loser {i}", market=market, model_prob=loser_model_prob, market_fair_prob=loser_market_prob),
            won=False,
        )
        for i in range(n_lost)
    ]
    interleaved = [r for pair in itertools.zip_longest(winners, losers) for r in pair if r is not None]
    start = date(2026, 6, 1)
    resolved = []
    for i, r in enumerate(interleaved):
        game_date = (start + timedelta(days=i)).isoformat()
        resolved.append(ResolvedPick(pick=replace(r.pick, game_date=game_date, recorded_at=f"{game_date}T18:00:00+00:00"), won=r.won))
    return resolved


def test_fit_market_blend_returns_none_for_an_unrecognized_market():
    resolved = [ResolvedPick(pick=_pick("Player A", market="not_a_real_market"), won=True)]
    assert fit_market_blend("not_a_real_market", resolved) is None


def test_fit_market_blend_returns_none_when_no_resolved_pick_has_a_real_market_fair_prob():
    # Single-sided markets (see EdgeCandidate's module docstring) have no
    # market-side probability to blend with at all.
    resolved = [
        ResolvedPick(pick=_pick(f"Player {i}", market_fair_prob=None), won=(i % 2 == 0)) for i in range(50)
    ]
    assert fit_market_blend("batter_home_runs", resolved) is None


def test_fit_market_blend_flags_a_small_sample_as_unreliable_but_still_returns_a_result():
    resolved = _blend_rows(3, 3, 0.5, 0.5, 0.6, 0.4)
    result = fit_market_blend("batter_home_runs", resolved)
    assert result is not None
    assert result.reliable is False
    assert result.n_train + result.n_test == 6


def test_fit_market_blend_prefers_the_market_when_market_is_the_real_predictor():
    # market_fair_prob perfectly separates won/lost; model_prob is a
    # constant, uninformative 0.5 for every row - a real fit should weight
    # almost entirely toward the market side (best_alpha near 0) and
    # genuinely beat pure model_prob's held-out log-loss.
    resolved = _blend_rows(
        n_won=40, n_lost=40, winner_model_prob=0.5, loser_model_prob=0.5, winner_market_prob=0.95, loser_market_prob=0.05
    )
    result = fit_market_blend("batter_home_runs", resolved)
    assert result is not None
    assert result.reliable is True
    assert result.best_alpha <= 0.2
    assert result.improves_on_model_only is True


def test_fit_market_blend_prefers_the_model_when_model_is_the_real_predictor():
    # The mirror image: model_prob separates the outcome, market_fair_prob
    # is constant/uninformative - best_alpha should land near 1 (pure
    # model), and there's no real reason to expect it to beat model_prob
    # alone since model_prob already *is* the real signal here.
    resolved = _blend_rows(
        n_won=40, n_lost=40, winner_model_prob=0.95, loser_model_prob=0.05, winner_market_prob=0.5, loser_market_prob=0.5
    )
    result = fit_market_blend("batter_home_runs", resolved)
    assert result is not None
    assert result.reliable is True
    assert result.best_alpha >= 0.8


def test_fit_all_market_blends_only_includes_markets_with_real_market_data():
    resolved = _blend_rows(30, 30, 0.5, 0.5, 0.95, 0.05)  # batter_home_runs only
    resolved.append(ResolvedPick(pick=_pick("No Market Data", market="batter_total_bases", market_fair_prob=None), won=True))
    results = fit_all_market_blends(resolved)
    assert [r.market for r in results] == ["batter_home_runs"]
