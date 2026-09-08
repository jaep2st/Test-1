"""Turns this project's own +EV game-line candidates into a concrete "what
to bet, how much" recommendation - the `mlb_props/betting.py` counterpart
for whole-game bets. Same fractional-Kelly sizing, same conservative
posture, for the same reason: `scoring.py`'s model_prob is a transparent,
hand-built statistical estimate, not a trained/calibrated one, and full
Kelly off an overconfident probability can recommend a dangerously large
position if the model's real edge is smaller than it looks.

1 unit = 1% of bankroll - see `mlb_props/betting.py`'s module docstring for
the same convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from odds_monitor.ev import american_to_decimal, decimal_to_american

from .edges import EdgeCandidate

STRONG_KELLY_MULTIPLIER = 0.25  # quarter-Kelly for tier == "agree"
SPECULATIVE_KELLY_MULTIPLIER = 0.125  # 1/8-Kelly for model-only tiers
MIN_EV_PERCENT_TO_RECOMMEND = 3.0
MIN_UNITS = 0.5
MAX_UNITS = 3.0
UNIT_ROUNDING = 0.5

_MARKET_LABELS = {"spreads": "Against the Spread", "h2h": "Moneyline", "totals": "Total"}


def breakeven_price(true_prob: float) -> Optional[int]:
    if true_prob <= 0.0 or true_prob >= 1.0:
        return None
    return decimal_to_american(1.0 / true_prob)


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    p = model_prob
    q = 1.0 - p
    return (b * p - q) / b


_KELLY_MULTIPLIER_BY_TIER = {"agree": STRONG_KELLY_MULTIPLIER}


def recommend_units(model_prob: float, odds: int, tier: str) -> Optional[float]:
    decimal_odds = american_to_decimal(odds)
    full_kelly = kelly_fraction(model_prob, decimal_odds)
    if full_kelly <= 0:
        return None
    multiplier = _KELLY_MULTIPLIER_BY_TIER.get(tier, SPECULATIVE_KELLY_MULTIPLIER)
    units = full_kelly * multiplier * 100.0
    units = max(MIN_UNITS, min(MAX_UNITS, units))
    return round(units / UNIT_ROUNDING) * UNIT_ROUNDING


@dataclass(frozen=True)
class RecommendedBet:
    event: str
    market: str
    market_label: str
    selection: str
    tier: str
    model_prob: float
    market_fair_prob: Optional[float]
    edge_vs_market: Optional[float]
    ev_percent_model: float
    best_price: int
    best_book: str
    books_quoting: int
    units: float
    full_kelly_percent: float
    breakeven: Optional[int]
    predicted_margin: float
    predicted_total: float
    components: Dict[str, float] = field(default_factory=dict)


def _to_recommendation(e: EdgeCandidate) -> Optional[RecommendedBet]:
    if not e.has_market_data or e.ev_percent_model is None or e.ev_percent_model < MIN_EV_PERCENT_TO_RECOMMEND:
        return None
    units = recommend_units(e.model_prob, e.best_line.odds, e.tier)
    if units is None:
        return None
    decimal_odds = american_to_decimal(e.best_line.odds)
    return RecommendedBet(
        event=e.event,
        market=e.market,
        market_label=_MARKET_LABELS.get(e.market, e.market),
        selection=e.selection_label,
        tier=e.tier,
        model_prob=e.model_prob,
        market_fair_prob=e.market_fair_prob,
        edge_vs_market=e.edge_vs_market,
        ev_percent_model=e.ev_percent_model,
        best_price=e.best_line.odds,
        best_book=e.best_line.sportsbook,
        books_quoting=e.books_quoting,
        units=units,
        full_kelly_percent=round(kelly_fraction(e.model_prob, decimal_odds) * 100.0, 2),
        breakeven=breakeven_price(e.model_prob),
        predicted_margin=e.predicted_margin,
        predicted_total=e.predicted_total,
        components=e.components,
    )


def build_recommended_bets(candidates: List[EdgeCandidate]) -> Tuple[List[RecommendedBet], List[RecommendedBet]]:
    """Returns (strong, speculative), each sorted by EV%% descending - see
    `mlb_props/betting.py`'s identical function docstring for what each
    tier means.
    """
    recs = [r for r in (_to_recommendation(c) for c in candidates) if r is not None]
    strong = sorted((r for r in recs if r.tier == "agree"), key=lambda r: r.ev_percent_model, reverse=True)
    speculative = sorted((r for r in recs if r.tier != "agree"), key=lambda r: r.ev_percent_model, reverse=True)
    return strong, speculative
