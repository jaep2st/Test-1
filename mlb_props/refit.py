"""Real, data-driven weight refitting - the actual mechanism behind the
Performance page's "N real days of resolved picks" note (see
performance_report.REFIT_READY_DAYS). Fits each market's component
weights via L2-regularized logistic regression against real resolved
outcomes, using the exact same 0-100-normalized component values
scoring.py already computes for every candidate (see
`EdgeCandidate.components`/`PickRecord.components`) - never synthetic or
assumed data.

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

Deliberately a PROPOSAL, not a live behavior change: `refit_market`
never touches scoring.py's HR_WEIGHTS/TB_WEIGHTS/HITS_WEIGHTS - it
returns a side-by-side comparison (the fit's real held-out log-loss vs.
the CURRENT hand-set model's real held-out log-loss, using each pick's
already-recorded model_prob for the latter - the exact number a real
reader actually saw, not a recomputation) for a human to review before
deciding whether to adopt it. Automatically swapping live scoring off an
unreviewed fit - however large the sample - is exactly the kind of
overconfident move this project has consistently avoided elsewhere
(fractional Kelly sizing, "unknown stays unknown", the Live bets
execution-risk disclosure, etc.).

No numpy/scikit-learn: this project stays zero-dependency by convention
(see html_report.py's inline-SVG calibration chart, vanilla-JS sort/
filter/search instead of a framework) - plain-Python batch gradient
descent is entirely adequate at the data volumes this ever runs on
(hundreds to low thousands of resolved picks, not millions), and keeps
the workflow's install step exactly as it is today.
"""

from __future__ import annotations

import math
import random
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
MIN_PICKS_TO_FIT = 40  # a logistic fit over up to 11 real features needs a real floor - below this the fit is still computed and returned, but flagged unreliable, never presented as if trustworthy
TEST_FRACTION = 0.25  # a real held-out check, never just training-set fit quality
RANDOM_SEED = 20260101  # fixed, not wall-clock - refitting twice on the same recorded data must give the same split and the same result, so a run is reproducible from the data alone


def _sigmoid(z: float) -> float:
    if z < -700:  # avoid a float overflow in math.exp on an extreme logit
        return 0.0
    if z > 700:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


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
    real resolved outcomes for that market, and compares its real
    held-out log-loss to the CURRENT hand-set model's real held-out
    log-loss.

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
    rows = [(r.pick.components, r.won, r.pick.model_prob) for r in resolved if r.pick.market == market and r.pick.components]
    if not rows:
        return None

    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n_test = round(len(indices) * TEST_FRACTION) if len(indices) >= 4 else 0
    test_idx = set(indices[:n_test])
    train_rows = [(rows[i][0], rows[i][1]) for i in indices if i not in test_idx]
    test_rows = [rows[i] for i in indices if i in test_idx]

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
