"""Turns this project's own +EV props into a concrete "what to bet, how
much" recommendation - the direct answer to "what should I actually bet
tonight" that the ranked prop tables alone don't give (they show every
candidate's EV%, but never say which ones clear a real bar or how big a
position to take).

Unit sizing uses fractional Kelly, deliberately conservative on purpose:
this project's model_prob is a hand-weighted heuristic (see scoring.py),
not a calibrated prediction. Full Kelly off an overconfident probability
estimate can recommend a bet size that's dangerously large if the model's
real edge is smaller than it looks - fractional Kelly is the standard way
sharp bettors size into genuine uncertainty about their own edge, not
just variance in outcomes. Two further, more conservative-than-typical
choices on top of that:

1. A "strong" pick (tier == "agree" - both this project's own model AND
   a real cross-book no-vig consensus, from at least
   edges.MIN_BOOKS_FOR_MARKET_AGREE independent books, see value) is sized
   at quarter-Kelly (0.25x) - a single book's price doesn't qualify as
   "the market agrees" no matter how good the math looks against it; see
   that constant's docstring for a real case where trusting one book's
   early number this way missed a much better price a second book posted
   shortly after. A "speculative" pick (model_only/
   model_only_single_sided - only this project's own heuristic sees it,
   with no market corroboration) is sized at an extra half of that
   (0.125x, effectively 1/8-Kelly) - Kelly math assumes model_prob IS the
   true probability, and there's real, disclosed reason to trust that
   assumption less when nothing else confirms it.
2. A hard floor and cap (MIN_UNITS/MAX_UNITS below) regardless of what
   the raw Kelly math says, so one overconfident model_prob can't
   recommend an outsized position, and a marginal-but-real edge doesn't
   round down to a meaningless size.

1 unit = 1% of bankroll, the standard convention in sports-betting
write-ups - this project has no way to know your actual bankroll, only
relative sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from odds_monitor.ev import american_to_decimal, decimal_to_american

from .edges import EdgeCandidate
from .market import MARKET_HITS, MARKET_HOME_RUN, MARKET_TOTAL_BASES
from .pipeline import SlateReport

# Real bet-sizing constants, every one deliberately conservative - see
# module docstring for why each exists.
STRONG_KELLY_MULTIPLIER = 0.25  # quarter-Kelly for tier == "agree"
SPECULATIVE_KELLY_MULTIPLIER = 0.125  # 1/8-Kelly for model-only tiers - see docstring point 1
MIN_EV_PERCENT_TO_RECOMMEND = 3.0  # below this, "edge" is noise-level against a hand-tuned heuristic, not a real recommendation
MIN_UNITS = 0.5  # smallest recommended size once a pick clears the bar - anything smaller isn't worth a distinct position
MAX_UNITS = 3.0  # hard cap regardless of what Kelly says - protects against a single overconfident model_prob
UNIT_ROUNDING = 0.5  # rounded to the nearest half-unit for legibility

_MARKET_LABELS = {
    MARKET_HOME_RUN: "1+ HR",
    MARKET_TOTAL_BASES: "2+ Total Bases",
    MARKET_HITS: "1+ Hits",
}


def breakeven_price(true_prob: float) -> Optional[int]:
    """The exact American price at which a bet on `true_prob` has EV% == 0 -
    the real "number to beat," useful precisely because every price on this
    page is a snapshot: by the time you check your actual sportsbook, the
    real price may have moved. You don't need this project to re-fetch
    anything to answer "is it still a good bet" - just compare your book's
    current price to this one. A price at least as good (a bigger plus
    number, or a less-negative minus number) is still +EV against this
    project's own probability estimate; anything worse than this number no
    longer is, even if it once was when this page was generated.

    None when `true_prob` is at or outside (0, 1) - no finite fair price
    exists there (shouldn't happen for a real probability estimate, but
    "unknown stays unknown" rather than a nonsense number).
    """
    if true_prob <= 0.0 or true_prob >= 1.0:
        return None
    return decimal_to_american(1.0 / true_prob)


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Full Kelly fraction of bankroll for a bet at `decimal_odds` if
    `model_prob` is the true win probability: f* = (bp - q) / b, where b
    is the net odds (decimal_odds - 1), p = model_prob, q = 1 - p.
    Zero or negative when there's no real edge (or odds <= 1.0, i.e. no
    real payout) - callers should treat that as "don't bet," never size a
    negative position.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    p = model_prob
    q = 1.0 - p
    return (b * p - q) / b


_KELLY_MULTIPLIER_BY_TIER = {
    "agree": STRONG_KELLY_MULTIPLIER,
}


def recommend_units(model_prob: float, odds: int, tier: str) -> Optional[float]:
    """Fractional-Kelly bet size, in units (1 unit = 1% of bankroll - see
    module docstring). `tier` selects how conservative the fraction is:
    quarter-Kelly for "agree", 1/8-Kelly for anything else (model_only/
    model_only_single_sided) - see `_KELLY_MULTIPLIER_BY_TIER`
    and the module/section docstrings for why each tier gets the fraction
    it does. Returns None when there's no real edge to size (full Kelly
    <= 0) - never a negative or zero unit count.

    Floored at MIN_UNITS and capped at MAX_UNITS regardless of the raw
    Kelly math, then rounded to the nearest UNIT_ROUNDING for legibility.
    The real, unrounded full-Kelly percentage is always available via
    `kelly_fraction()` directly for anyone who wants to size it
    themselves without these guardrails.
    """
    decimal_odds = american_to_decimal(odds)
    full_kelly = kelly_fraction(model_prob, decimal_odds)
    if full_kelly <= 0:
        return None
    multiplier = _KELLY_MULTIPLIER_BY_TIER.get(tier, SPECULATIVE_KELLY_MULTIPLIER)
    units = full_kelly * multiplier * 100.0  # 1 unit = 1% of bankroll
    units = max(MIN_UNITS, min(MAX_UNITS, units))
    return round(units / UNIT_ROUNDING) * UNIT_ROUNDING


@dataclass(frozen=True)
class RecommendedBet:
    player: str
    market: str
    market_label: str
    event: str
    tier: str
    model_prob: float
    market_fair_prob: Optional[float]
    edge_vs_market: Optional[float]
    ev_percent_model: float
    best_price: int
    best_book: str
    books_quoting: int
    units: float
    # The real, unrounded full-Kelly bankroll percentage this recommendation
    # was derived from (before the tier multiplier/floor/cap/rounding above) -
    # shown alongside `units` so the math behind the recommendation is never
    # hidden, only made more conservative.
    full_kelly_percent: float
    # The exact price at which this bet stops being +EV against model_prob -
    # see breakeven_price()'s docstring. None only if model_prob is ever
    # somehow outside (0, 1), which shouldn't happen in practice.
    breakeven: Optional[int]
    # The real per-component scoring breakdown behind model_score/model_prob
    # (see EdgeCandidate.components' docstring) - carried through so the
    # page can show *why* this specific bet is recommended, not just that
    # it is. {} for a candidate that predates this field.
    components: Dict[str, float] = field(default_factory=dict)
    # "confirmed" if this recommendation was scored against MLB's real,
    # posted starting lineup; "active_roster" (the honest default) if it
    # was scored against the active-roster proxy instead - see
    # edges.py's EdgeCandidate.lineup_source.
    lineup_source: str = "active_roster"


def _to_recommendation(e: EdgeCandidate) -> Optional[RecommendedBet]:
    if not e.has_market_data or e.ev_percent_model is None or e.ev_percent_model < MIN_EV_PERCENT_TO_RECOMMEND:
        return None
    units = recommend_units(e.model_prob, e.best_line.odds, e.tier)
    if units is None:
        return None
    decimal_odds = american_to_decimal(e.best_line.odds)
    return RecommendedBet(
        player=e.player,
        market=e.market,
        market_label=_MARKET_LABELS.get(e.market, e.market),
        event=e.event,
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
        components=e.components,
        lineup_source=e.lineup_source,
    )


def build_recommended_bets(report: SlateReport) -> Tuple[List[RecommendedBet], List[RecommendedBet]]:
    """Every real, positive-EV candidate across all three markets that
    clears MIN_EV_PERCENT_TO_RECOMMEND, split into two confidence tiers,
    each sorted by EV% descending:

    - strong: tier == "agree" - both this project's own model and the
      market's own cross-book no-vig consensus see value. The higher-
      confidence half of this list, sized more aggressively (still only
      quarter-Kelly - see module docstring).
    - speculative: everything else that still clears the bar
      (model_only/model_only_single_sided) - only this project's own
      heuristic sees it, with no market corroboration. Real, but sized
      more conservatively and should carry more scrutiny before betting.

    Returns (strong, speculative). A candidate the pregame/EV filters
    already excluded (no market data, negative EV, below the real-edge
    bar) never appears in either list - this is a strict subset of what
    the ranked prop tables already show, not a new source of picks.
    """
    candidates = [c for edges in (report.hr_edges, report.tb_edges, report.hits_edges) for c in edges]
    recs = [r for r in (_to_recommendation(c) for c in candidates) if r is not None]
    strong = sorted((r for r in recs if r.tier == "agree"), key=lambda r: r.ev_percent_model, reverse=True)
    speculative = sorted((r for r in recs if r.tier != "agree"), key=lambda r: r.ev_percent_model, reverse=True)
    return strong, speculative
