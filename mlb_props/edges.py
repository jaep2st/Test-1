"""Combine our composite model scores with the market's own no-vig
consensus (from `odds_monitor.ev`) into ranked +EV candidates.

Two independent signals feed every candidate:
1. **Model edge** - our HR/total-bases score says this player's probability
   is higher than what the best available price pays out for (EV computed
   against our own `model_prob`).
2. **Market edge** - regardless of our model, one book's price is
   meaningfully better than the cross-book no-vig consensus (classic line
   shopping / +EV, computed by `odds_monitor.ev.find_fair_prices`).

A candidate flagged by both is the strongest kind of spot: our fundamentals
say the market's consensus is underpricing it, *and* there's a specific book
offering a price better than that consensus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from odds_monitor.ev import FairPrice, model_ev_percent
from odds_monitor.models import PropLine

from .market import MARKET_HOME_RUN, MARKET_TOTAL_BASES
from .scoring import HRScoreResult, TotalBasesScoreResult


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

    @property
    def has_market_data(self) -> bool:
        return self.best_line is not None

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


def _fair_price_lookup(fair_prices: List[FairPrice], market: str, side: str) -> Dict[str, FairPrice]:
    return {
        fp.player.strip().lower(): fp
        for fp in fair_prices
        if fp.market.lower() == market.lower() and fp.side.lower() == side.lower()
    }


def _build_edges(
    scores: List, market: str, side: str, fair_prices: List[FairPrice], event_lookup: Dict[str, str]
) -> List[EdgeCandidate]:
    lookup = _fair_price_lookup(fair_prices, market, side)
    candidates: List[EdgeCandidate] = []
    for result in scores:
        key = result.player.strip().lower()
        fp = lookup.get(key)
        event = event_lookup.get(result.player, fp.event if fp else "")
        if fp is None:
            candidates.append(
                EdgeCandidate(
                    player=result.player,
                    market=market,
                    event=event,
                    model_score=result.score,
                    model_prob=result.model_prob,
                    market_fair_prob=None,
                    best_line=None,
                    ev_percent_model=None,
                    ev_percent_market=None,
                    edge_vs_market=None,
                    price_spread_percent=None,
                    books_quoting=0,
                    park=result.park,
                    wind_out_mph=result.wind_out_mph,
                    temp_f=result.temp_f,
                    is_dome=result.is_dome,
                    weather_boost_pct=result.weather_boost_pct,
                )
            )
            continue
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
            )
        )
    return candidates


def build_hr_edges(
    scores: List[HRScoreResult], fair_prices: List[FairPrice], event_lookup: Dict[str, str]
) -> List[EdgeCandidate]:
    return _build_edges(scores, MARKET_HOME_RUN, "yes", fair_prices, event_lookup)


def build_total_bases_edges(
    scores: List[TotalBasesScoreResult], fair_prices: List[FairPrice], event_lookup: Dict[str, str]
) -> List[EdgeCandidate]:
    return _build_edges(scores, MARKET_TOTAL_BASES, "over", fair_prices, event_lookup)


def rank_candidates(candidates: List[EdgeCandidate], min_ev_percent: float = 0.0) -> List[EdgeCandidate]:
    """Best spots first: prioritize candidates where both our model *and*
    the market's own cross-book consensus agree there's value, then fall
    back to model-only or market-only signal.
    """

    def sort_key(c: EdgeCandidate):
        if not c.has_market_data:
            return (-1, c.model_score)
        both_agree = c.ev_percent_model is not None and c.ev_percent_model > 0 and c.edge_vs_market is not None and c.edge_vs_market > 0
        return (2 if both_agree else (1 if c.ev_percent_model and c.ev_percent_model > 0 else 0), c.ev_percent_model or 0)

    filtered = [c for c in candidates if not c.has_market_data or (c.ev_percent_model or -999) >= min_ev_percent]
    return sorted(filtered, key=sort_key, reverse=True)
