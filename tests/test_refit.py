"""Covers mlb_props/refit.py: the real logistic-regression weight refit
against resolved picks' recorded components. All synthetic fixtures built
directly (no filesystem/network) - reuses test_backtest.py's ResolvedPick/
PickRecord/GameOutcome joining, already covered there.
"""

from mlb_props.backtest import ResolvedPick
from mlb_props.refit import MIN_PICKS_TO_FIT, refit_all_markets, refit_market
from mlb_props.results import PickRecord


def _pick(player, market="batter_home_runs", model_prob=0.15, components=None):
    return PickRecord(
        game_date="2026-08-20",
        recorded_at="2026-08-20T18:00:00+00:00",
        player=player,
        market=market,
        event="Team A @ Team B",
        tier="agree",
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
    """
    resolved = []
    for i in range(n_won):
        resolved.append(
            ResolvedPick(pick=_pick(f"Winner {i}", components={"barrel_pct": 95.0, "hard_hit_pct": 50.0}), won=True)
        )
    for i in range(n_lost):
        resolved.append(
            ResolvedPick(pick=_pick(f"Loser {i}", components={"barrel_pct": 5.0, "hard_hit_pct": 50.0}), won=False)
        )
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
