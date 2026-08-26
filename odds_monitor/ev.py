"""Odds math: American <-> probability/decimal conversions, no-vig
("de-vigged") fair-probability estimation from a two-way market, and
expected-value calculations used to flag +EV prices.

This module is sport-agnostic - it works on any two-way market (over/under,
yes/no, home run/no home run) expressed as `PropLine`s that share a `key`.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple

from .models import PropLine


def american_to_decimal(odds: int) -> float:
    """American odds -> decimal (payout multiple, including the stake)."""
    if odds > 0:
        return 1.0 + odds / 100.0
    if odds < 0:
        return 1.0 + 100.0 / (-odds)
    raise ValueError("American odds cannot be 0")


def american_to_implied_prob(odds: int) -> float:
    """American odds -> the *vig-included* implied probability of that side."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return (-odds) / (-odds + 100.0)
    raise ValueError("American odds cannot be 0")


def decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be > 1.0")
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def devig_two_way(prob_a: float, prob_b: float) -> Tuple[float, float]:
    """Multiplicative (proportional) de-vig: scale two raw implied
    probabilities that sum to > 1.0 (because of the book's vig) down so they
    sum to exactly 1.0, preserving their ratio. This is the simplest and most
    common no-vig method; it slightly under/over-states fair probability for
    heavily skewed lines vs. more sophisticated methods (e.g. Shin), but is a
    solid default for player props.
    """
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError("Probabilities must be positive")
    return prob_a / total, prob_b / total


@dataclass(frozen=True)
class FairPrice:
    """A player-prop side's consensus no-vig fair probability, plus the best
    price a bettor could actually get for it right now.
    """

    player: str
    market: str
    side: str
    event: str
    fair_prob: float  # consensus (median across books) no-vig probability
    books_used: int  # how many books had both sides quoted, for the devig
    best_line: PropLine  # the single best price available for this side
    best_decimal: float
    ev_percent: float  # (fair_prob * best_decimal - 1) * 100
    worst_line: Optional[PropLine]  # worst price offered, for line-shopping context
    price_spread_percent: float  # gap between best and worst book's implied prob


def _pair_key(line: PropLine) -> Tuple[str, str, str, str]:
    """Same as PropLine.key but without the side, so we can pair over/under
    (or yes/no) quotes from the same book together for a per-book devig.
    """
    return (line.player.strip().lower(), line.league.lower(), line.market.lower(), line.event.lower())


def find_fair_prices(lines: Iterable[PropLine]) -> List[FairPrice]:
    """For every (player, market, event) with two-sided odds quoted by at
    least one book, compute a consensus no-vig fair probability per side and
    the best/worst price a bettor can get for it, sorted by EV% descending.

    Lines missing `odds` are ignored (this needs prices, not just point
    values).
    """
    by_pair: Dict[Tuple[str, str, str, str], Dict[str, List[PropLine]]] = {}
    for line in lines:
        if line.odds is None:
            continue
        pair = by_pair.setdefault(_pair_key(line), {})
        pair.setdefault(line.side.lower(), []).append(line)

    results: List[FairPrice] = []
    for pair_lines in by_pair.values():
        sides = list(pair_lines.keys())
        if len(sides) != 2:
            continue  # need exactly two sides (over/under or yes/no) to devig
        side_a, side_b = sides
        by_book_a = {l.sportsbook: l for l in pair_lines[side_a]}
        by_book_b = {l.sportsbook: l for l in pair_lines[side_b]}
        common_books = set(by_book_a) & set(by_book_b)
        if not common_books:
            continue

        fair_a_samples: List[float] = []
        fair_b_samples: List[float] = []
        for book in common_books:
            raw_a = american_to_implied_prob(by_book_a[book].odds)
            raw_b = american_to_implied_prob(by_book_b[book].odds)
            fair_a, fair_b = devig_two_way(raw_a, raw_b)
            fair_a_samples.append(fair_a)
            fair_b_samples.append(fair_b)

        consensus = {side_a: median(fair_a_samples), side_b: median(fair_b_samples)}

        for side, side_lines in pair_lines.items():
            fair_prob = consensus[side]
            priced = sorted(side_lines, key=lambda l: american_to_decimal(l.odds), reverse=True)
            best = priced[0]
            worst = priced[-1] if len(priced) > 1 else None
            best_decimal = american_to_decimal(best.odds)
            ev_percent = (fair_prob * best_decimal - 1.0) * 100.0
            price_spread_percent = 0.0
            if worst is not None:
                price_spread_percent = (
                    american_to_implied_prob(best.odds) - american_to_implied_prob(worst.odds)
                ) * -100.0
                # best price = lowest implied prob for the bettor; report the
                # gap as a positive number of percentage points.
                price_spread_percent = abs(price_spread_percent)
            results.append(
                FairPrice(
                    player=best.player,
                    market=best.market,
                    side=side,
                    event=best.event,
                    fair_prob=fair_prob,
                    books_used=len(common_books),
                    best_line=best,
                    best_decimal=best_decimal,
                    ev_percent=ev_percent,
                    worst_line=worst,
                    price_spread_percent=price_spread_percent,
                )
            )

    results.sort(key=lambda f: f.ev_percent, reverse=True)
    return results


def model_ev_percent(model_prob: float, odds: int) -> float:
    """EV% of a price if `model_prob` (our own probability estimate, e.g.
    from `mlb_props.scoring`) is correct, regardless of what the market's
    no-vig price implies.
    """
    return (model_prob * american_to_decimal(odds) - 1.0) * 100.0
