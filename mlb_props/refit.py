"""Real, data-driven weight refitting - the actual mechanism behind the
Performance page's "N real days of resolved picks" note (see
performance_report.REFIT_READY_DAYS). Fits each market's component
weights via L2-regularized logistic regression against real resolved
outcomes, using the exact same 0-100-normalized component values
scoring.py already computes for every candidate (see
`EdgeCandidate.components`/`PickRecord.components`) - never synthetic or
assumed data. Also fits a real market-blended probability (see
`fit_market_blend` below) - the other real, validated proposal this
module produces.

Before this module existed, "21 real days" could never actually trigger
anything: every scored candidate's raw per-component feature values were
computed every run (scoring.py's *ScoreResult.components) and then
thrown away - dumped into an unstructured "CANDIDATE_DETAIL" log line
(pipeline.py) and never persisted anywhere a real refit could read them
back from. Only the final blended model_score/model_prob ever made it
into a PickRecord. No amount of accumulated days could have refit
anything without the raw features to regress against - see edges.py's
and results.py's `components` fields, added alongside this module for
exactly that reason.

Deliberately a PROPOSAL, not a live behavior change: neither
`refit_market` nor `fit_market_blend` ever touches scoring.py's
HR_WEIGHTS/TB_WEIGHTS/HITS_WEIGHTS or betting.py's live sizing - each
returns a side-by-side comparison (the fit's real held-out log-loss vs.
the CURRENT hand-set model's real held-out log-loss, using each pick's
already-recorded model_prob for the latter - the exact number a real
reader actually saw, not a recomputation) for a human to review before
deciding whether to adopt it. Automatically swapping live scoring off an
unreviewed fit - however large the sample - is exactly the kind of
overconfident move this project has consistently avoided elsewhere
(fractional Kelly sizing, "unknown stays unknown", the Live bets
execution-risk disclosure, etc.).

Every held-out split in this module is walk-forward (chronological by
`pick.game_date`), never a random shuffle: a random split on time-series
picks lets a fit "see" outcomes from dates after the ones it's tested
against, which is exactly the kind of leakage that makes a backtest look
better than real deployment ever would - the same real-world failure
mode documented sports-betting/quant-trading validation practice calls
out (train strictly on the past, test strictly on what came after). See
`_time_ordered_split` below.

No numpy/scikit-learn: this project stays zero-dependency by convention
(see html_report.py's inline-SVG calibration chart, vanilla-JS sort/
filter/search instead of a framework) - plain-Python batch gradient
descent (and, for the single-parameter blend fit, a plain grid search)
is entirely adequate at the data volumes this ever runs on (hundreds to
low thousands of resolved picks, not millions), and keeps the workflow's
install step exactly as it is today.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .backtest import ResolvedPick
from .market import MARKET_HITS, MARKET_HOME_RUN, MARKET_TOTAL_BASES
from .scoring import HITS_WEIGHTS, HR_WEIGHTS, TB_WEIGHTS

# Each real market's component keys, in scoring.py's own canonical order -
# imported directly from HR_WEIGHTS/TB_WEIGHTS/HITS_WEIGHTS rather than
# duplicated here, so this module can never silently drift out of sync
# with a real change to scoring.py's own component set.
MARKET_COMPONENT_KEYS: Dict[str, Tuple[str, ...]] = {
    MARKET_HOME_RUN: tuple(HR_WEIGHTS.keys()),
    MARKET_TOTAL_BASES: tuple(TB_WEIGHTS.keys()),
    MARKET_HITS: tuple(HITS_WEIGHTS.keys()),
}
_CURRENT_WEIGHTS_BY_MARKET: Dict[str, Dict[str, float]] = {
    MARKET_HOME_RUN: HR_WEIGHTS,
    MARKET_TOTAL_BASES: TB_WEIGHTS,
    MARKET_HITS: HITS_WEIGHTS,
}

L2_PENALTY = 0.05  # shrinks fitted weights toward 0 (and, after renormalizing for display, toward uniform) - a real guard against overfitting a small real sample, not a knob tuned against this project's own data
LEARNING_RATE = 0.3
MAX_ITERATIONS = 2000
CONVERGENCE_TOL = 1e-7
MIN_PICKS_TO_FIT = 40  # a logistic fit over up to 11 real features (or, for fit_market_blend, the single real alpha parameter) needs a real floor - below this the fit is still computed and returned, but flagged unreliable, never presented as if trustworthy
TEST_FRACTION = 0.25  # a real held-out check, never just training-set fit quality


def _sigmoid(z: float) -> float:
    if z < -700:  # avoid a float overflow in math.exp on an extreme logit
        return 0.0
    if z > 700:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p: float) -> float:
    """Inverse of `_sigmoid` - log-odds of a real probability, clamped
    away from the exact 0/1 boundary (an honest real probability is never
    exactly certain) so the result always stays finite."""
    eps = 1e-9
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


def _time_ordered_split(resolved: Sequence[ResolvedPick], test_fraction: float) -> Tuple[List[ResolvedPick], List[ResolvedPick]]:
    """Walk-forward split: train on the earlier `1 - test_fraction` share
    of `resolved` by real `pick.game_date`, test on the later share -
    never a random shuffle. See this module's docstring for why a random
    split is the wrong tool for time-series picks like these (it lets a
    fit "see" outcomes from dates after the ones it's evaluated against).
    A stable sort keeps same-day picks in their original recorded order
    as a tiebreaker, so this is fully deterministic - no seed needed.
    Below 4 real rows there's no meaningful held-out slice; everything
    goes to train and the test half is empty, same posture the old
    random-split code used for a tiny sample.
    """
    ordered = sorted(resolved, key=lambda r: r.pick.game_date)
    if len(ordered) < 4:
        return ordered, []
    n_test = round(len(ordered) * test_fraction)
    if n_test == 0:
        return ordered, []
    split_idx = len(ordered) - n_test
    return ordered[:split_idx], ordered[split_idx:]


def _predict(weights: Dict[str, float], bias: float, components: Dict[str, float], keys: Sequence[str]) -> float:
    z = bias + sum(weights[k] * (components.get(k, 0.0) / 100.0) for k in keys)
    return _sigmoid(z)


def _log_loss(predictions: Sequence[Tuple[float, bool]]) -> Optional[float]:
    """Mean binary cross-entropy of real (predicted_prob, won) pairs -
    lower is better; 0 is a perfect (and, on any real sample, suspicious)
    fit. `None` for an empty sequence, never a fabricated 0.0.
    """
    if not predictions:
        return None
    eps = 1e-9
    total = 0.0
    for p, won in predictions:
        p = min(1.0 - eps, max(eps, p))
        total += -(math.log(p) if won else math.log(1.0 - p))
    return round(total / len(predictions), 4)


def _fit_logistic(train_rows: Sequence[Tuple[Dict[str, float], bool]], keys: Sequence[str]) -> Tuple[Dict[str, float], float]:
    """Batch gradient descent, L2-regularized, on real (components, won)
    rows. Components are already 0-100-normalized (scoring.py's
    `_normalize`), rescaled to 0-1 here purely for stable gradients -
    doesn't change what's being fit, just its numeric scale.
    """
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
                grad_w[k] += err * (components.get(k, 0.0) / 100.0)
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
    """Rescales fitted logistic coefficients into the same non-negative,
    sum-to-1 "importance" convention scoring.py's hand-set *_WEIGHTS use,
    purely for a like-for-like side-by-side display - NOT a claim that
    these numbers can be pasted directly into scoring.py as a drop-in
    replacement: a logistic model's sigmoid(w.x + b) is a different
    function from scoring.py's normalize-then-piecewise-linear-calibrate
    pipeline, even when both assign similar relative importance to the
    same features.
    """
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
    reliable: bool  # False below MIN_PICKS_TO_FIT real training rows - shown, never hidden
    fitted_importance: Dict[str, float]  # normalized |logistic weight| per component, for comparison only - see _normalize_importance's docstring
    current_weights: Dict[str, float]  # scoring.py's live hand-set weights for the same market, for direct comparison
    fitted_test_log_loss: Optional[float]  # the fitted model's real held-out log-loss; None if n_test == 0
    current_test_log_loss: Optional[float]  # the CURRENT hand-set model's real held-out log-loss, from each pick's already-recorded model_prob - the honest, apples-to-apples baseline
    improves_on_current: Optional[bool]  # True only if the fit's held-out log-loss beats the current model's by a real (not noise-level) margin; None if either side has no held-out result to compare


_IMPROVEMENT_MARGIN = 0.02  # the fit's log-loss must be at least this much lower (not just numerically lower) to call it a real improvement, not sampling noise on a small held-out set


def refit_market(market: str, resolved: List[ResolvedPick]) -> Optional[RefitResult]:
    """Fits `market`'s component weights via logistic regression against
    real resolved outcomes for that market (walk-forward split - see
    `_time_ordered_split`), and compares its real held-out log-loss to
    the CURRENT hand-set model's real held-out log-loss.

    Returns `None` if `market` isn't one of this project's three real
    markets, or if no resolved pick for it carries any real `components`
    (every pick recorded before that field existed - see
    PickRecord.components's docstring - silently has none to fit from).
    A market with real components but fewer than MIN_PICKS_TO_FIT still
    returns a `RefitResult`, just with `reliable=False` - callers should
    show it, clearly labeled as not yet actionable, never suppress it.
    """
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
        market=market,
        n_train=len(train_rows),
        n_test=len(test_rows),
        reliable=len(train_rows) >= MIN_PICKS_TO_FIT,
        fitted_importance=_normalize_importance(weights),
        current_weights=dict(_CURRENT_WEIGHTS_BY_MARKET[market]),
        fitted_test_log_loss=fitted_test_loss,
        current_test_log_loss=current_test_loss,
        improves_on_current=improves_on_current,
    )


def refit_all_markets(resolved: List[ResolvedPick]) -> List[RefitResult]:
    """`refit_market` for each of this project's three real markets, in
    scoring.py's own market order - skips a market with no resolved picks
    carrying real components at all (see `refit_market`), never a
    fabricated empty result for it.
    """
    results = []
    for market in (MARKET_HOME_RUN, MARKET_TOTAL_BASES, MARKET_HITS):
        result = refit_market(market, resolved)
        if result is not None:
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# Market-blended probability - Bill Benter's real, documented technique:
# combine a fundamentals-based model with the market's own odds (which
# already encode real public information) rather than treating either
# alone as ground truth. `model_prob` and `market_fair_prob` already sit
# side by side on every EdgeCandidate/PickRecord - this fits the one real
# parameter (how much weight each side should get) against actual
# resolved outcomes instead of guessing it, using the exact same
# walk-forward discipline and "proposal, not live" posture as the weight
# refit above.
# ---------------------------------------------------------------------------

# Log-odds pooling weights to search - a single real scalar, so a plain
# grid is simpler and more transparent than gradient descent here; each
# step is a genuinely different blend, not false precision.
_BLEND_ALPHA_GRID: Tuple[float, ...] = tuple(round(i / 10, 1) for i in range(11))  # 0.0, 0.1, ..., 1.0


def _blended_prob(alpha: float, model_prob: float, market_prob: float) -> float:
    """Standard log-odds pooling: alpha=1.0 is pure model_prob, alpha=0.0
    is pure market_prob, and everything between is a real weighted
    combination of the two - never a naive linear average of
    probabilities, which distorts badly near 0/1."""
    return _sigmoid(alpha * _logit(model_prob) + (1.0 - alpha) * _logit(market_prob))


@dataclass(frozen=True)
class BlendResult:
    market: str
    n_train: int
    n_test: int
    reliable: bool  # False below MIN_PICKS_TO_FIT real training rows - shown, never hidden
    best_alpha: float  # the real, walk-forward-validated weight on model_prob (1.0 - best_alpha on market_fair_prob) that minimized held-out log-loss on the training split
    blended_test_log_loss: Optional[float]  # best_alpha's blend, real held-out log-loss
    model_only_test_log_loss: Optional[float]  # pure model_prob (alpha=1.0), the current live behavior, on the same held-out rows - the honest baseline
    market_only_test_log_loss: Optional[float]  # pure market_fair_prob (alpha=0.0) on the same held-out rows, for context
    improves_on_model_only: Optional[bool]  # True only if the blend's held-out log-loss beats pure model_prob's by a real (not noise-level) margin; None if either side has no held-out result to compare


def fit_market_blend(market: str, resolved: List[ResolvedPick]) -> Optional[BlendResult]:
    """Fits `market`'s real log-odds blend weight (see `_blended_prob`)
    against real resolved outcomes, and compares its real held-out
    log-loss to pure `model_prob` (today's live behavior) and pure
    `market_fair_prob`, on the exact same walk-forward-held-out rows.

    Returns `None` if `market` isn't one of this project's three real
    markets, or if no resolved pick for it carries a real
    `market_fair_prob` (single-sided markets - see EdgeCandidate's
    module docstring - have no market-side probability to blend with at
    all). A market with real data but fewer than MIN_PICKS_TO_FIT still
    returns a `BlendResult`, just with `reliable=False` - callers should
    show it, clearly labeled as not yet actionable, never suppress it.
    """
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
        market=market,
        n_train=len(train_rows),
        n_test=len(test_rows),
        reliable=len(train_rows) >= MIN_PICKS_TO_FIT,
        best_alpha=best_alpha,
        blended_test_log_loss=blended_loss,
        model_only_test_log_loss=model_only_loss,
        market_only_test_log_loss=market_only_loss,
        improves_on_model_only=improves_on_model_only,
    )


def fit_all_market_blends(resolved: List[ResolvedPick]) -> List[BlendResult]:
    """`fit_market_blend` for each of this project's three real markets,
    in scoring.py's own market order - skips a market with no resolved
    picks carrying a real `market_fair_prob` at all (see
    `fit_market_blend`), never a fabricated empty result for it.
    """
    results = []
    for market in (MARKET_HOME_RUN, MARKET_TOTAL_BASES, MARKET_HITS):
        result = fit_market_blend(market, resolved)
        if result is not None:
            results.append(result)
    return results
