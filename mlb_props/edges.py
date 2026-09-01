"""Combine our composite model scores with the market's own no-vig
consensus (from `odds_monitor.ev`) into ranked +EV candidates.

Two independent signals feed every candidate, when the market allows it:
1. **Model edge** - our HR/total-bases score says this player's probability
   is higher than what the best available price pays out for (EV computed
   against our own `model_prob`). This only needs one real price, so it
   works even for a single-sided market (see below).
2. **Market edge** - regardless of our model, one book's price is
   meaningfully better than the cross-book no-vig consensus (classic line
   shopping / +EV, computed by `odds_monitor.ev.find_fair_prices`). This
   needs a genuinely two-sided market (both "yes"/"no" or "over"/"under"
   quoted by at least one book) to de-vig - see that function's docstring.

A candidate flagged by both is the strongest kind of spot: our fundamentals
say the market's consensus is underpricing it, *and* there's a specific book
offering a price better than that consensus.

**Single-sided markets:** confirmed live (see `odds_monitor/providers/
theoddsapi.py`'s home-run-market diagnostics) that at least one real book
quotes MLB's home-run prop as "Over 0.5" only, with no "Under" leg at all -
a common shape for "anytime" props. `find_fair_prices` can't de-vig that (it
needs two sides), so those real prices would otherwise never appear despite
being genuine, live market data. `_build_edges` falls back to the best
available single-sided price when no de-vigged `FairPrice` match exists,
computing model-vs-price EV only (market-edge fields stay `None` - there's
no no-vig consensus to compare against without a second side).

That fallback is restricted to the standard "1+ HR"/"2+ total bases" line
(`HOME_RUN_LINE_FOR_1PLUS`/`TOTAL_BASES_LINE_FOR_2PLUS` in `market.py`) on
purpose: confirmed live that a single real book can post several point
values under the exact same market/side (e.g. "Over 0.5", "Over 1.5", "Over
2.5" HRs, all outcome name "Over") - without this filter, picking "best
price" across every line collapses onto whichever is the longest shot (2+
or 3+ HRs), producing wildly inflated EV% on a real number for the wrong
bet entirely. Only lines matching the standard line are considered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from odds_monitor.ev import FairPrice, american_to_decimal, model_ev_percent
from odds_monitor.models import PropLine

from .market import (
    HITS_LINE_FOR_1PLUS,
    HOME_RUN_LINE_FOR_1PLUS,
    MARKET_HITS,
    MARKET_HOME_RUN,
    MARKET_TOTAL_BASES,
    TOTAL_BASES_LINE_FOR_2PLUS,
)
from .scoring import HitsScoreResult, HRScoreResult, TotalBasesScoreResult


@dataclass(frozen=True)
class EdgeCandidate:
    player: str
    market: str
    event: str
    model_score: float
    model_prob: float
    market_fair_prob: Optional[float]
    best_line: Optional[PropLine]
    ev_percent_model: Optional[float]  # EV% of the best price if our model_prob is right
    ev_percent_market: Optional[float]  # EV% of the best price vs. the market's own no-vig consensus
    edge_vs_market: Optional[float]  # model_prob - market_fair_prob; positive = we like it more than the market does
    price_spread_percent: Optional[float]  # best vs. worst book gap, for line-shopping context
    books_quoting: int
    park: str
    wind_out_mph: float
    temp_f: Optional[float]
    is_dome: bool
    weather_boost_pct: float
    # Ballpark Pal's own independent per-game model probability, when
    # configured (see mlb_props/ballparkpal.py) - None for TB (no honest
    # analog exists for that market) and for HR/Hits when not configured
    # or Ballpark Pal has no data for this matchup. A genuine second
    # opinion, never blended into model_prob/ev_percent_model themselves.
    bp_model_prob: Optional[float] = None
    # Each scoring component's raw 0-100-normalized contribution (see
    # scoring.py's HR_WEIGHTS/TB_WEIGHTS/HITS_WEIGHTS and the matching
    # *ScoreResult.components) - previously computed every run and only
    # ever dumped into an unstructured log line, never actually captured
    # anywhere a future weight refit could use. Threaded through here so
    # results.py can persist it on PickRecord - see mlb_props/refit.py,
    # which is the reason this field exists at all. Defaults to {} so
    # older code/tests constructing an EdgeCandidate without it still work.
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def has_market_data(self) -> bool:
        return self.best_line is not None

    @property
    def tier(self) -> str:
        """Which of the report's four labeled buckets this candidate falls
        into - shared by html_report.py's row styling and results.py's pick
        recording (see that module), so a pick's recorded tier always
        matches what the published report actually showed for it that run.
        "agree" = model_prob beats the best price AND that price beats the
        market's own no-vig consensus - the strongest kind of spot. See
        this module's docstring for the two independent signals.
        """
        if not self.has_market_data:
            return "no_market"
        both_agree = (
            self.ev_percent_model is not None
            and self.ev_percent_model > 0
            and self.edge_vs_market is not None
            and self.edge_vs_market > 0
        )
        if both_agree:
            return "agree"
        if self.market_fair_prob is None:
            return "model_only_single_sided"
        return "model_only"

    @property
    def weather_note(self) -> str:
        if self.is_dome:
            return f"{self.park}: roof closed, no wind effect"
        wind_dir = "out" if self.wind_out_mph > 0 else ("in" if self.wind_out_mph < 0 else "calm")
        temp = f"{self.temp_f:.0f}F" if self.temp_f is not None else "n/a"
        return f"{self.park}: wind {abs(self.wind_out_mph):.0f}mph {wind_dir}, {temp} ({self.weather_boost_pct:+.1f}% HR odds)"

    def describe(self) -> str:
        if not self.has_market_data:
            return (
                f"{self.player} ({self.event}) - {self.market}: model {self.model_score:.0f}/100 "
                f"({self.model_prob:.1%}) - no market price found | {self.weather_note}"
            )
        return (
            f"{self.player} ({self.event}) - {self.market}: model {self.model_score:.0f}/100 "
            f"({self.model_prob:.1%}) vs market fair {self.market_fair_prob:.1%} "
            f"[edge {self.edge_vs_market:+.1%}] | best price {self.best_line.sportsbook} "
            f"{self.best_line.odds:+d} (EV {self.ev_percent_model:+.1f}% by our model, "
            f"{self.ev_percent_market:+.1f}% vs. market consensus) | "
            f"{self.books_quoting} books, {self.price_spread_percent:.1f}pt price spread | {self.weather_note}"
        )


def _fair_price_lookup(fair_prices: List[FairPrice], market: str, side: str, expected_line: float) -> Dict[str, FairPrice]:
    """Restricted to `expected_line` for the same reason `_single_sided_lookup`
    below is: `find_fair_prices` now returns one `FairPrice` per real point
    tier it finds two-sided pricing for (see `odds_monitor.ev._pair_key`'s
    docstring) - a player can have a real, correctly-devigged fair price for
    both "1+ hits" (0.5) and "2+ hits" (1.5) at once, and only the standard
    line this pipeline actually scores should ever surface here.
    """
    return {
        fp.player.strip().lower(): fp
        for fp in fair_prices
        if fp.market.lower() == market.lower() and fp.side.lower() == side.lower() and abs(fp.line - expected_line) < 1e-6
    }


def _single_sided_lookup(
    lines: List[PropLine], market: str, side: str, expected_line: float
) -> Dict[str, "tuple[PropLine, int]"]:
    """Best real price per player for a market/side that `find_fair_prices`
    couldn't de-vig (no second side quoted anywhere) - see this module's
    docstring. Maps lowercased player name -> (best line, distinct books
    quoting that side). `None`-odds lines are ignored, same as
    `find_fair_prices`.

    Restricted to `expected_line` (the standard "1+ HR"/"2+ total bases"
    point value) - see this module's docstring for why: a real book can
    post several point values under the same market/side, and picking
    "best price" across all of them silently swaps in a much longer-shot
    bet than the one actually being scored/reported.
    """
    by_player: Dict[str, List[PropLine]] = {}
    for line in lines:
        if (
            line.odds is not None
            and line.market.lower() == market.lower()
            and line.side.lower() == side.lower()
            and abs(line.line - expected_line) < 1e-6
        ):
            by_player.setdefault(line.player.strip().lower(), []).append(line)
    return {
        player: (max(group, key=lambda l: american_to_decimal(l.odds)), len({l.sportsbook for l in group}))
        for player, group in by_player.items()
    }


def _build_edges(
    scores: List,
    market: str,
    side: str,
    expected_line: float,
    fair_prices: List[FairPrice],
    lines: List[PropLine],
    event_lookup: Dict[str, str],
) -> List[EdgeCandidate]:
    lookup = _fair_price_lookup(fair_prices, market, side, expected_line)
    single_sided = _single_sided_lookup(lines, market, side, expected_line)
    candidates: List[EdgeCandidate] = []
    for result in scores:
        key = result.player.strip().lower()
        fp = lookup.get(key)
        if fp is None:
            single = single_sided.get(key)
            best_line, books = single if single else (None, 0)
            event = event_lookup.get(result.player, best_line.event if best_line else "")
            candidates.append(
                EdgeCandidate(
                    player=result.player,
                    market=market,
                    event=event,
                    model_score=result.score,
                    model_prob=result.model_prob,
                    market_fair_prob=None,
                    best_line=best_line,
                    ev_percent_model=round(model_ev_percent(result.model_prob, best_line.odds), 1) if best_line else None,
                    ev_percent_market=None,
                    edge_vs_market=None,
                    price_spread_percent=None,
                    books_quoting=books,
                    park=result.park,
                    wind_out_mph=result.wind_out_mph,
                    temp_f=result.temp_f,
                    is_dome=result.is_dome,
                    weather_boost_pct=result.weather_boost_pct,
                    bp_model_prob=result.bp_model_prob,
                    components=result.components,
                )
            )
            continue
        event = event_lookup.get(result.player, fp.event)
        candidates.append(
            EdgeCandidate(
                player=result.player,
                market=market,
                event=event,
                model_score=result.score,
                model_prob=result.model_prob,
                market_fair_prob=fp.fair_prob,
                best_line=fp.best_line,
                ev_percent_model=round(model_ev_percent(result.model_prob, fp.best_line.odds), 1),
                ev_percent_market=round(fp.ev_percent, 1),
                edge_vs_market=round(result.model_prob - fp.fair_prob, 4),
                price_spread_percent=round(fp.price_spread_percent, 1),
                books_quoting=fp.books_used,
                park=result.park,
                wind_out_mph=result.wind_out_mph,
                temp_f=result.temp_f,
                is_dome=result.is_dome,
                weather_boost_pct=result.weather_boost_pct,
                bp_model_prob=result.bp_model_prob,
                components=result.components,
            )
        )
    return candidates


def build_hr_edges(
    scores: List[HRScoreResult], fair_prices: List[FairPrice], lines: List[PropLine], event_lookup: Dict[str, str]
) -> List[EdgeCandidate]:
    return _build_edges(scores, MARKET_HOME_RUN, "yes", HOME_RUN_LINE_FOR_1PLUS, fair_prices, lines, event_lookup)


def build_total_bases_edges(
    scores: List[TotalBasesScoreResult], fair_prices: List[FairPrice], lines: List[PropLine], event_lookup: Dict[str, str]
) -> List[EdgeCandidate]:
    return _build_edges(scores, MARKET_TOTAL_BASES, "over", TOTAL_BASES_LINE_FOR_2PLUS, fair_prices, lines, event_lookup)


def build_hits_edges(
    scores: List[HitsScoreResult], fair_prices: List[FairPrice], lines: List[PropLine], event_lookup: Dict[str, str]
) -> List[EdgeCandidate]:
    return _build_edges(scores, MARKET_HITS, "over", HITS_LINE_FOR_1PLUS, fair_prices, lines, event_lookup)


def rank_candidates(candidates: List[EdgeCandidate], min_ev_percent: float = 0.0) -> List[EdgeCandidate]:
    """Best spots first: prioritize candidates where both our model *and*
    the market's own cross-book consensus agree there's value, then fall
    back to model-only or market-only signal.

    Confirmed live (2026-08-29): at the documented default (`min_ev_percent
    =0.0`, "show all" per the CLI's --min-ev help text), this used to
    silently DROP any priced candidate whose model-implied EV was negative
    - not demote it, remove it from the returned list entirely, so it
    showed up neither in the priced table nor the model-only fallback
    (which only sees candidates still in this list). Real market data
    (a genuine BetRivers home-run price) vanished without a trace for
    several players this way, on every run, because every real workflow
    dispatch passes --min-ev 0. A real price - even one our model doesn't
    like - is exactly the information this report exists to surface, so
    the default must never delete it. `min_ev_percent` now only filters
    when a caller explicitly raises it above 0 to declutter a big table;
    at 0 (or below), nothing with real market data is ever dropped.
    """

    def sort_key(c: EdgeCandidate):
        if not c.has_market_data:
            return (-1, c.model_score)
        both_agree = c.ev_percent_model is not None and c.ev_percent_model > 0 and c.edge_vs_market is not None and c.edge_vs_market > 0
        return (2 if both_agree else (1 if c.ev_percent_model and c.ev_percent_model > 0 else 0), c.ev_percent_model or 0)

    filtered = [
        c for c in candidates
        if not c.has_market_data or min_ev_percent <= 0.0 or (c.ev_percent_model or -999) >= min_ev_percent
    ]
    return sorted(filtered, key=sort_key, reverse=True)
