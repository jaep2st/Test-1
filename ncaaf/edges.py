"""Combines this project's own predicted probabilities (`scoring.py`) with
the market's own no-vig consensus (`game_odds.py`) into ranked +EV game-
line candidates - the `mlb_props/edges.py` counterpart for whole-game bets
instead of player props.

The one real structural difference from player props: a prop has one fixed
"recommended side" (e.g. always the "over"/"yes" leg), but a game line has
no such fixed side - either team can be the model's real pick against the
spread, on the moneyline, or for the total. Every side of every market this
project has a real prediction and a real market price for becomes its own
candidate here; `rank_candidates` sorts the good ones to the top rather
than this module pre-selecting one side per market.

Same two independent signals as `mlb_props/edges.py`:
1. **Model edge** - our predicted probability says a price pays out more
   than it should.
2. **Market edge** - regardless of our model, one book's price beats the
   real cross-book no-vig consensus (classic line shopping).
A candidate flagged by both (`tier == "agree"`, with at least
`MIN_BOOKS_FOR_MARKET_AGREE` independent books behind the consensus - see
that constant's docstring in `mlb_props/edges.py` for the real incident
that established this bar) is the strongest kind of spot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .game_odds import GameFairPrice, american_to_decimal, find_fair_game_prices, model_ev_percent
from .market import MARKET_H2H, MARKET_SPREADS, MARKET_TOTALS, GameLine
from .scoring import GameScore

# Same real-incident-driven bar as mlb_props/edges.py's identical constant:
# a single book's price is that book's own opening number, not a real
# cross-book consensus.
MIN_BOOKS_FOR_MARKET_AGREE = 2


def effective_tier(tier: str, books_quoting: int) -> str:
    if tier == "agree" and books_quoting < MIN_BOOKS_FOR_MARKET_AGREE:
        return "model_only"
    return tier


@dataclass(frozen=True)
class EdgeCandidate:
    event: str
    market: str  # "spreads" / "h2h" / "totals"
    side: str  # "home"/"away" or "over"/"under"
    team: Optional[str]  # actual team name for spreads/h2h; None for totals
    point: Optional[float]
    model_prob: float
    market_fair_prob: Optional[float]
    best_line: Optional[GameLine]
    ev_percent_model: Optional[float]
    ev_percent_market: Optional[float]
    edge_vs_market: Optional[float]
    price_spread_percent: Optional[float]
    books_quoting: int
    predicted_margin: float
    predicted_total: float
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def has_market_data(self) -> bool:
        return self.best_line is not None

    @property
    def selection_label(self) -> str:
        if self.market == MARKET_TOTALS:
            point = f"{self.point:g}" if self.point is not None else "?"
            return f"{self.side.title()} {point}"
        point = f" {self.point:+g}" if self.market == MARKET_SPREADS and self.point is not None else ""
        return f"{self.team or self.side.title()}{point}"

    @property
    def tier(self) -> str:
        if not self.has_market_data:
            return "no_market"
        both_agree = (
            self.ev_percent_model is not None
            and self.ev_percent_model > 0
            and self.edge_vs_market is not None
            and self.edge_vs_market > 0
            and self.books_quoting >= MIN_BOOKS_FOR_MARKET_AGREE
        )
        if both_agree:
            return "agree"
        if self.market_fair_prob is None:
            return "model_only_single_sided"
        return "model_only"

    def describe(self) -> str:
        base = f"{self.event} - {self.market} {self.selection_label}: model {self.model_prob:.1%}"
        if not self.has_market_data:
            return base + " - no market price found"
        return (
            base + f" vs market fair {self.market_fair_prob:.1%} [edge {self.edge_vs_market:+.1%}] | "
            f"best price {self.best_line.sportsbook} {self.best_line.odds:+d} "
            f"(EV {self.ev_percent_model:+.1f}% by our model, {self.ev_percent_market:+.1f}% vs. market consensus) | "
            f"{self.books_quoting} books"
        )


def _model_prob_for(score: GameScore, market: str, side: str, point: Optional[float]) -> Optional[float]:
    if market == MARKET_H2H:
        return score.home_win_prob if side == "home" else round(1.0 - score.home_win_prob, 4)
    if market == MARKET_SPREADS:
        if point is None:
            return None
        home_spread = point if side == "home" else -point
        return score.home_cover_prob(home_spread) if side == "home" else score.away_cover_prob(home_spread)
    if market == MARKET_TOTALS:
        if point is None:
            return None
        return score.over_prob(point) if side == "over" else score.under_prob(point)
    return None


def _single_sided_lookup(lines: List[GameLine], fair_price_keys: set) -> Dict[tuple, tuple]:
    """Best real price per (event, market, side, point) that
    `find_fair_game_prices` couldn't de-vig (no opposite side quoted by any
    common book) - the game-line analog of `mlb_props/edges.py`'s
    `_single_sided_lookup`. Maps key -> (best_line, distinct books).
    """
    by_key: Dict[tuple, List[GameLine]] = {}
    for line in lines:
        if line.odds is None:
            continue
        key = (line.event.strip().lower(), line.market, line.side, round(line.point, 2) if line.point is not None else None)
        if key in fair_price_keys:
            continue
        by_key.setdefault(key, []).append(line)
    return {
        key: (max(group, key=lambda l: american_to_decimal(l.odds)), len({l.sportsbook for l in group}))
        for key, group in by_key.items()
    }


def build_game_edges(scores: Dict[str, GameScore], lines: List[GameLine]) -> List[EdgeCandidate]:
    """`scores` keyed by `GameScore.event`. Builds one `EdgeCandidate` per
    real, priced (event, market, side, point) this project has both a
    prediction and a market price for.
    """
    fair_prices = find_fair_game_prices(lines)
    fair_by_key: Dict[tuple, GameFairPrice] = {
        (fp.event.strip().lower(), fp.market, fp.side, round(fp.point, 2) if fp.point is not None else None): fp
        for fp in fair_prices
    }
    single_sided = _single_sided_lookup(lines, set(fair_by_key.keys()))

    candidates: List[EdgeCandidate] = []
    seen_keys = set(fair_by_key.keys()) | set(single_sided.keys())
    for key in seen_keys:
        event_lower, market, side, point = key
        score = None
        for ev, s in scores.items():
            if ev.strip().lower() == event_lower:
                score = s
                break
        if score is None:
            continue
        model_prob = _model_prob_for(score, market, side, point)
        if model_prob is None:
            continue

        fp = fair_by_key.get(key)
        if fp is not None:
            team = fp.team
            best_line = fp.best_line
            market_fair_prob = fp.fair_prob
            ev_model = round(model_ev_percent(model_prob, best_line.odds), 1)
            ev_market = round(fp.ev_percent, 1)
            edge_vs_market = round(model_prob - fp.fair_prob, 4)
            price_spread_percent = round(fp.price_spread_percent, 1)
            books = fp.books_used
        else:
            single = single_sided.get(key)
            if single is None:
                continue
            best_line, books = single
            team = best_line.team
            market_fair_prob = None
            ev_model = round(model_ev_percent(model_prob, best_line.odds), 1)
            ev_market = None
            edge_vs_market = None
            price_spread_percent = None

        candidates.append(
            EdgeCandidate(
                event=score.event,
                market=market,
                side=side,
                team=team,
                point=point,
                model_prob=model_prob,
                market_fair_prob=market_fair_prob,
                best_line=best_line,
                ev_percent_model=ev_model,
                ev_percent_market=ev_market,
                edge_vs_market=edge_vs_market,
                price_spread_percent=price_spread_percent,
                books_quoting=books,
                predicted_margin=score.predicted_margin,
                predicted_total=score.predicted_total,
                components=score.components,
            )
        )
    return candidates


def rank_candidates(candidates: List[EdgeCandidate], min_ev_percent: float = 0.0) -> List[EdgeCandidate]:
    def sort_key(c: EdgeCandidate):
        if not c.has_market_data:
            return (-1, c.model_prob)
        both_agree = c.ev_percent_model is not None and c.ev_percent_model > 0 and c.edge_vs_market is not None and c.edge_vs_market > 0
        return (2 if both_agree else (1 if c.ev_percent_model and c.ev_percent_model > 0 else 0), c.ev_percent_model or 0)

    filtered = [
        c for c in candidates
        if not c.has_market_data or min_ev_percent <= 0.0 or (c.ev_percent_model or -999) >= min_ev_percent
    ]
    return sorted(filtered, key=sort_key, reverse=True)
