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
   with no market corroboration) is sized at 1/16-Kelly (0.0625x) - Kelly
   math assumes model_prob IS the true probability, and there's real,
   disclosed reason to trust that assumption less when nothing else
   confirms it. Confirmed against this project's own real recorded
   history (2026-09-09, 112 real resolved Speculative bets): realized win
   rate (30.4%) came in BELOW what the market's own price required just
   to break even (35.9%), net -11.4u - not "less edge than Strong," zero
   validated edge so far. This tier was sized at 1/8-Kelly until that
   real evidence came in; halved again in response (see
   backtest.hit_rate_by_tier / units_summary for the numbers this is
   based on, and performance_report.py's speculative_warning for where a
   live version of this same check is surfaced).
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

from .edges import EdgeCandidate, effective_tier
from .market import MARKET_HITS, MARKET_HOME_RUN, MARKET_TOTAL_BASES
from .pipeline import SlateReport
from .results import PickRecord

# Real bet-sizing constants, every one deliberately conservative - see
# module docstring for why each exists.
STRONG_KELLY_MULTIPLIER = 0.25  # quarter-Kelly for tier == "agree"
SPECULATIVE_KELLY_MULTIPLIER = 0.0625  # 1/16-Kelly for model-only tiers - see docstring point 1 (real data, 2026-09-09: this tier realized below the market's own breakeven)
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

    Deduped to one leg per player across the whole board first: the same
    player recommended in more than one market the same day (1+ Hits AND
    2+ Total Bases, say) isn't two independent bets - both ride on the
    same at-bats, so staking both compounds one correlated outcome into
    what looks like real diversification. Confirmed live (2026-09-10): 21
    of 42 real Strong-tier players that day had 2+ correlated legs
    recommended at once; deduping to one leg per player would have
    roughly halved that day's real tracked loss on the exact same picks
    (see backtest.py's matching dedup for the units-tracking side of
    this fix). Keeps the single highest-priority leg per player: a real
    cross-book "agree" leg beats a model-only one on the same player even
    at lower EV% (the confirmation matters more than the number here),
    then the higher EV% within the same tier.
    """
    candidates = [c for edges in (report.hr_edges, report.tb_edges, report.hits_edges) for c in edges]
    recs = [r for r in (_to_recommendation(c) for c in candidates) if r is not None]

    best_by_player: Dict[str, RecommendedBet] = {}
    for r in recs:
        key = r.player.strip().lower()
        current = best_by_player.get(key)
        if current is None:
            best_by_player[key] = r
            continue
        is_agree = r.tier == "agree"
        cur_is_agree = current.tier == "agree"
        if is_agree and not cur_is_agree:
            best_by_player[key] = r
        elif is_agree == cur_is_agree and r.ev_percent_model > current.ev_percent_model:
            best_by_player[key] = r
    deduped = list(best_by_player.values())

    strong = sorted((r for r in deduped if r.tier == "agree"), key=lambda r: r.ev_percent_model, reverse=True)
    speculative = sorted((r for r in deduped if r.tier != "agree"), key=lambda r: r.ev_percent_model, reverse=True)
    return strong, speculative


# A real ask: too many recommended picks makes the day-to-day units swing
# hard to grind through even when the underlying picks are individually
# fine (see units_ledger's docstring on correlated same-player legs for
# half of that swing - this is the other half: even after dedup, a full
# Strong list on a busy slate can be 40+ real bets, a lot of simultaneous
# variance). Best bets is a further, tighter slice of an already-deduped
# Strong list for someone who wants fewer, higher-conviction plays to
# track a steadier daily/monthly/yearly number - not a claim these
# specific bets are more likely to win (see backtest.edge_confidence:
# neither tier has a statistically proven edge yet at this project's
# current sample size), only that they're this run's most convicted
# subset of an already-real bar.
BEST_BETS_MAX_COUNT = 5
BEST_BETS_MIN_EV_PERCENT = 8.0


def best_bets(strong: List[RecommendedBet]) -> List[RecommendedBet]:
    """The tightest real subset of `strong` (already deduped to one leg
    per player, already sorted by EV% descending - see
    `build_recommended_bets`): at most BEST_BETS_MAX_COUNT picks, and
    only ones clearing BEST_BETS_MIN_EV_PERCENT - a real bar well above
    the noise-level MIN_EV_PERCENT_TO_RECOMMEND every Strong pick already
    clears. A quiet day with nothing that convicted honestly returns
    fewer than the cap, never padded out with a weaker pick to hit a
    round number.
    """
    return [r for r in strong if r.ev_percent_model >= BEST_BETS_MIN_EV_PERCENT][:BEST_BETS_MAX_COUNT]


@dataclass(frozen=True)
class WithdrawnRecommendation:
    """A pick an earlier run *today* actually recommended (cleared the
    same real bar as `build_recommended_bets`) that this run no longer
    does. Real user report: a recommendation that just silently vanishes
    between runs is indistinguishable from "did I imagine that" - this
    keeps it visible, with the real reason it dropped, rather than
    deleting it the moment it stops qualifying. Never a new source of
    picks to bet - a note not to bet this one if you saw it earlier."""

    player: str
    market: str
    market_label: str
    event: str
    prior_tier: str
    prior_price: int
    prior_book: str
    prior_ev_percent: float
    prior_units: float
    # ISO 8601 UTC - PickRecord.recorded_at of the last run that actually
    # recommended this pick, so "earlier today" has a real timestamp.
    recorded_at: str
    reason: str


def _pick_was_recommended(p: PickRecord) -> bool:
    """Same real bar `_to_recommendation` checks on a live `EdgeCandidate`,
    applied to an already-recorded `PickRecord` snapshot - reads tier
    through `effective_tier` for the same reason `units_ledger` does (see
    that function's docstring): a stale "agree" recorded before
    MIN_BOOKS_FOR_MARKET_AGREE existed/changed shouldn't count as a real
    strong recommendation today.
    """
    if p.best_price is None or p.best_book is None:
        return False
    if p.ev_percent_model is None or p.ev_percent_model < MIN_EV_PERCENT_TO_RECOMMEND:
        return False
    tier = effective_tier(p.tier, p.books_quoting)
    return recommend_units(p.model_prob, p.best_price, tier) is not None


def _reason_for_withdrawal(prior: PickRecord, current: Optional[EdgeCandidate]) -> str:
    """Plain-language, honest explanation of why a once-recommended pick
    no longer is - "whatever that reason is," so it's never just a bare
    "no longer recommended" with nothing to check it against."""
    if current is None:
        return "not scored this run (game/market may be off today's slate)"
    if not current.has_market_data:
        return "market closed - no price available now (game may have started)"
    clauses = []
    if current.best_line and current.best_line.odds != prior.best_price:
        clauses.append(f"price moved {prior.best_price:+d} → {current.best_line.odds:+d}")
    cur_tier = effective_tier(current.tier, current.books_quoting)
    if effective_tier(prior.tier, prior.books_quoting) == "agree" and cur_tier != "agree":
        clauses.append(f"lost book consensus - tier dropped to {cur_tier.replace('_', ' ')}")
    if current.ev_percent_model is None:
        clauses.append("no longer shows a positive edge")
    elif current.ev_percent_model < MIN_EV_PERCENT_TO_RECOMMEND:
        clauses.append(f"edge dropped to {current.ev_percent_model:.1f}% (below the {MIN_EV_PERCENT_TO_RECOMMEND:.0f}% bar)")
    if not clauses:
        clauses.append("no longer sizes to a real recommended position")
    return "; ".join(clauses)


def withdrawn_recommendations(report: SlateReport, prior_picks: List[PickRecord]) -> List[WithdrawnRecommendation]:
    """Every pick a prior run recorded *today* that actually cleared the
    real recommend bar at the time (see `_pick_was_recommended`) but this
    run's own candidates no longer do - the line moved, the tier lost its
    cross-book consensus, the market closed once the game started, or the
    edge just evaporated. Kept visible (not silently dropped) with the
    real reason, so seeing an old recommendation later today never means
    accidentally betting something that stopped being good hours ago.

    `prior_picks` is every pick recorded by earlier runs today, loaded
    before this run's own candidates were recorded - same data,
    same timing, as html_report.py's stale-price fallback (see
    mlb_props_main.py's call site).
    """
    current_by_key: Dict[Tuple[str, str, str], EdgeCandidate] = {}
    for c in report.hr_edges + report.tb_edges + report.hits_edges:
        current_by_key[(c.player.strip().lower(), c.market.lower(), c.event.lower())] = c

    strong, speculative = build_recommended_bets(report)
    currently_recommended = {(r.player.strip().lower(), r.market.lower(), r.event.lower()) for r in strong + speculative}

    latest_prior: Dict[Tuple[str, str, str], PickRecord] = {}
    for p in prior_picks:
        prev = latest_prior.get(p.key)
        if prev is None or p.recorded_at > prev.recorded_at:
            latest_prior[p.key] = p

    withdrawn = []
    for key, prior in latest_prior.items():
        if key in currently_recommended or not _pick_was_recommended(prior):
            continue
        tier = effective_tier(prior.tier, prior.books_quoting)
        units = recommend_units(prior.model_prob, prior.best_price, tier) or 0.0
        withdrawn.append(
            WithdrawnRecommendation(
                player=prior.player,
                market=prior.market,
                market_label=_MARKET_LABELS.get(prior.market, prior.market),
                event=prior.event,
                prior_tier=tier,
                prior_price=prior.best_price,
                prior_book=prior.best_book,
                prior_ev_percent=prior.ev_percent_model,
                prior_units=units,
                recorded_at=prior.recorded_at,
                reason=_reason_for_withdrawal(prior, current_by_key.get(key)),
            )
        )
    return sorted(withdrawn, key=lambda w: w.recorded_at, reverse=True)
