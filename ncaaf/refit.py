"""Real, data-driven weight refitting against actual resolved outcomes -
the `ncaaf` counterpart to `mlb_props/refit.py`. Fits a logistic regression
per market (spreads/h2h/totals) on `scoring.py`'s own real components
(`PickRecord.components`, sourced from `EdgeCandidate.components`), and a
real log-odds market-blend weight (Bill Benter's documented technique
combining a fundamentals model with the market's own price) - see
`mlb_props/refit.py`'s module docstring for the full rationale, which
applies here unchanged: walk-forward (chronological by `(season, week)`,
never a random shuffle) train/test split, a real held-out log-loss
comparison against the CURRENT live model, and a **proposal, never a live
behavior change** - nothing here touches `scoring.py`/`context.py`'s real
constants automatically.

Unlike MLB's three markets (each with its own distinct component set),
every ncaaf market shares the exact same `components` shape (see
`scoring.compute_game_score`) - it's a whole-game model, not a market-
specific score - so `MARKET_COMPONENT_KEYS` below is the same tuple for
all three, fit independently per market anyway since each market's real
outcome (a cover, a moneyline win, an over/under) can genuinely weight the
same components differently in practice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .backtest import ResolvedPick

_COMPONENT_KEYS: Tuple[str, ...] = (
    "power_diff",
    "home_field_advantage_pts",
    "altitude_bonus_pts",
    "rest_edge_pts",
    "travel_edge_pts",
    "weather_total_adjustment_pct",
    "home_offense_vs_away_defense",
    "away_offense_vs_home_defense",
)
MARKET_COMPONENT_KEYS: Dict[str, Tuple[str, ...]] = {m: _COMPONENT_KEYS for m in ("spreads", "h2h", "totals")}

L2_PENALTY = 0.05
LEARNING_RATE = 0.3
MAX_ITERATIONS = 2000
CONVERGENCE_TOL = 1e-7
MIN_PICKS_TO_FIT = 40
TEST_FRACTION = 0.25


def _sigmoid(z: float) -> float:
    if z < -700:
        return 0.0
    if z > 700:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p: float) -> float:
    eps = 1e-9
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


def _time_ordered_split(resolved: Sequence[ResolvedPick], test_fraction: float) -> Tuple[List[ResolvedPick], List[ResolvedPick]]:
    ordered = sorted(resolved, key=lambda r: (r.pick.season, r.pick.week))
    if len(ordered) < 4:
        return ordered, []
    n_test = round(len(ordered) * test_fraction)
    if n_test == 0:
        return ordered, []
    split_idx = len(ordered) - n_test
    return ordered[:split_idx], ordered[split_idx:]


def _predict(weights: Dict[str, float], bias: float, components: Dict[str, float], keys: Sequence[str]) -> float:
    # Components here are already real point-scale values (not 0-100
    # normalized like mlb_props' scoring.py) - dividing by 10 keeps
    # gradients on a comparably stable numeric scale without needing to
    # know each component's exact native range up front.
    z = bias + sum(weights[k] * (components.get(k, 0.0) / 10.0) for k in keys)
    return _sigmoid(z)


def _log_loss(predictions: Sequence[Tuple[float, bool]]) -> Optional[float]:
    if not predictions:
        return None
    eps = 1e-9
    total = 0.0
    for p, won in predictions:
        p = min(1.0 - eps, max(eps, p))
        total += -(math.log(p) if won else math.log(1.0 - p))
    return round(total / len(predictions), 4)


def _fit_logistic(train_rows: Sequence[Tuple[Dict[str, float], bool]], keys: Sequence[str]) -> Tuple[Dict[str, float], float]:
    weights = {k: 0.0 for k in keys}
    bias = 0.0
    n = len(train_rows)
    if n == 0:
        return weights, bias
    prev_loss: Optional[float] = None
    for _ in range(MAX_ITERATIONS):
        grad_w = {k: 0.0 for k in keys}
        grad_b = 0.0
        for components, won in train_rows:
            p = _predict(weights, bias, components, keys)
            err = p - (1.0 if won else 0.0)
            for k in keys:
                grad_w[k] += err * (components.get(k, 0.0) / 10.0)
            grad_b += err
        for k in keys:
            grad_w[k] = grad_w[k] / n + L2_PENALTY * weights[k]
            weights[k] -= LEARNING_RATE * grad_w[k]
        bias -= LEARNING_RATE * (grad_b / n)
        loss = _log_loss([(_predict(weights, bias, c, keys), won) for c, won in train_rows])
        if prev_loss is not None and loss is not None and abs(prev_loss - loss) < CONVERGENCE_TOL:
            break
        prev_loss = loss
    return weights, bias


def _normalize_importance(weights: Dict[str, float]) -> Dict[str, float]:
    abs_weights = {k: abs(v) for k, v in weights.items()}
    total = sum(abs_weights.values())
    if total <= 0:
        return {k: round(1.0 / len(weights), 4) for k in weights}
    return {k: round(v / total, 4) for k, v in abs_weights.items()}


@dataclass(frozen=True)
class RefitResult:
    market: str
    n_train: int
    n_test: int
    reliable: bool
    fitted_importance: Dict[str, float]
    fitted_test_log_loss: Optional[float]
    current_test_log_loss: Optional[float]
    improves_on_current: Optional[bool]


_IMPROVEMENT_MARGIN = 0.02


def refit_market(market: str, resolved: List[ResolvedPick]) -> Optional[RefitResult]:
    keys = MARKET_COMPONENT_KEYS.get(market)
    if keys is None:
        return None
    eligible = [r for r in resolved if r.pick.market == market and r.pick.components]
    if not eligible:
        return None

    train, test = _time_ordered_split(eligible, TEST_FRACTION)
    train_rows = [(r.pick.components, r.won) for r in train]
    test_rows = [(r.pick.components, r.won, r.pick.model_prob) for r in test]

    weights, bias = _fit_logistic(train_rows, keys)

    fitted_test_loss = _log_loss([(_predict(weights, bias, c, keys), won) for c, won, _mp in test_rows])
    current_test_loss = _log_loss([(mp, won) for _c, won, mp in test_rows])

    improves_on_current: Optional[bool] = None
    if fitted_test_loss is not None and current_test_loss is not None:
        improves_on_current = (current_test_loss - fitted_test_loss) >= _IMPROVEMENT_MARGIN

    return RefitResult(
        market=market, n_train=len(train_rows), n_test=len(test_rows), reliable=len(train_rows) >= MIN_PICKS_TO_FIT,
        fitted_importance=_normalize_importance(weights), fitted_test_log_loss=fitted_test_loss,
        current_test_log_loss=current_test_loss, improves_on_current=improves_on_current,
    )


def refit_all_markets(resolved: List[ResolvedPick]) -> List[RefitResult]:
    results = []
    for market in ("spreads", "h2h", "totals"):
        result = refit_market(market, resolved)
        if result is not None:
            results.append(result)
    return results


_BLEND_ALPHA_GRID: Tuple[float, ...] = tuple(round(i / 10, 1) for i in range(11))


def _blended_prob(alpha: float, model_prob: float, market_prob: float) -> float:
    return _sigmoid(alpha * _logit(model_prob) + (1.0 - alpha) * _logit(market_prob))


@dataclass(frozen=True)
class BlendResult:
    market: str
    n_train: int
    n_test: int
    reliable: bool
    best_alpha: float
    blended_test_log_loss: Optional[float]
    model_only_test_log_loss: Optional[float]
    market_only_test_log_loss: Optional[float]
    improves_on_model_only: Optional[bool]


def fit_market_blend(market: str, resolved: List[ResolvedPick]) -> Optional[BlendResult]:
    if market not in MARKET_COMPONENT_KEYS:
        return None
    eligible = [r for r in resolved if r.pick.market == market and r.pick.market_fair_prob is not None]
    if not eligible:
        return None

    train, test = _time_ordered_split(eligible, TEST_FRACTION)
    train_rows = [(r.pick.model_prob, r.pick.market_fair_prob, r.won) for r in train]
    test_rows = [(r.pick.model_prob, r.pick.market_fair_prob, r.won) for r in test]

    best_alpha = 1.0
    best_train_loss: Optional[float] = None
    for alpha in _BLEND_ALPHA_GRID:
        loss = _log_loss([(_blended_prob(alpha, mp, mkp), won) for mp, mkp, won in train_rows])
        if loss is not None and (best_train_loss is None or loss < best_train_loss):
            best_train_loss = loss
            best_alpha = alpha

    blended_loss = _log_loss([(_blended_prob(best_alpha, mp, mkp), won) for mp, mkp, won in test_rows])
    model_only_loss = _log_loss([(mp, won) for mp, _mkp, won in test_rows])
    market_only_loss = _log_loss([(mkp, won) for _mp, mkp, won in test_rows])

    improves_on_model_only: Optional[bool] = None
    if blended_loss is not None and model_only_loss is not None:
        improves_on_model_only = (model_only_loss - blended_loss) >= _IMPROVEMENT_MARGIN

    return BlendResult(
        market=market, n_train=len(train_rows), n_test=len(test_rows), reliable=len(train_rows) >= MIN_PICKS_TO_FIT,
        best_alpha=best_alpha, blended_test_log_loss=blended_loss, model_only_test_log_loss=model_only_loss,
        market_only_test_log_loss=market_only_loss, improves_on_model_only=improves_on_model_only,
    )


def fit_all_market_blends(resolved: List[ResolvedPick]) -> List[BlendResult]:
    results = []
    for market in ("spreads", "h2h", "totals"):
        result = fit_market_blend(market, resolved)
        if result is not None:
            results.append(result)
    return results
