"""No-vig fair-price math for game-line markets (spreads, moneylines,
totals), reusing the same American<->probability primitives
`odds_monitor.ev` already provides (and `mlb_props` already relies on) -
this module is the game-line-shaped counterpart to that module's
`find_fair_prices`, needed because a game line's two sides aren't
symmetric the way a player prop's over/under is: a spread's home/away legs
carry opposite-signed point values (home -6.5 vs away +6.5), so pairing
them for a per-book devig needs a point-normalization step
`odds_monitor.ev._pair_key` doesn't have any use for.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple

from odds_monitor.ev import american_to_decimal, american_to_implied_prob, devig_two_way, model_ev_percent  # noqa: F401 - model_ev_percent re-exported for callers

from .market import GameLine, MARKET_SPREADS


@dataclass(frozen=True)
class GameFairPrice:
    """One side's consensus no-vig fair probability for a specific game/
    market/point, plus the best real price available for it right now -
    the game-line analog of `odds_monitor.ev.FairPrice`.
    """

    event: str
    market: str
    side: str  # "home"/"away" or "over"/"under"
    team: Optional[str]
    point: Optional[float]  # this side's own real point value, as actually offered (unlike the internal pairing key, never sign-normalized)
    fair_prob: float
    books_used: int
    best_line: GameLine
    best_decimal: float
    ev_percent: float
    worst_line: Optional[GameLine]
    price_spread_percent: float


def _normalized_point(line: GameLine) -> float:
    """The pairing key's point value: for `spreads`, home/away legs are
    sign-mirrors of the same real number (home -6.5 / away +6.5), so this
    normalizes both onto the home-team's sign so they land in the same
    pairing group. `totals` legs already share one real point value
    (over/under 54.5), so no normalization is needed there. `h2h` has no
    real point at all - a constant placeholder groups both sides together.
    """
    if line.point is None:
        return 0.0
    if line.market == MARKET_SPREADS:
        return round(line.point if line.side == "home" else -line.point, 2)
    return round(line.point, 2)


def _pair_key(line: GameLine) -> Tuple[str, str, float]:
    return (line.event.strip().lower(), line.market, _normalized_point(line))


def find_fair_game_prices(lines: Iterable[GameLine]) -> List[GameFairPrice]:
    """For every (event, market, point) with two-sided odds quoted by at
    least one book, computes a consensus no-vig fair probability per side
    (per-book devig, median across books - see `odds_monitor.ev.
    find_fair_prices`'s identical method for the reasoning) and the best/
    worst real price available for it. Lines with `odds is None` are
    ignored.
    """
    by_pair: Dict[Tuple[str, str, float], Dict[str, List[GameLine]]] = {}
    for line in lines:
        if line.odds is None:
            continue
        pair = by_pair.setdefault(_pair_key(line), {})
        pair.setdefault(line.side.lower(), []).append(line)

    results: List[GameFairPrice] = []
    for pair_lines in by_pair.values():
        sides = list(pair_lines.keys())
        if len(sides) != 2:
            continue
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
                price_spread_percent = abs(
                    (american_to_implied_prob(best.odds) - american_to_implied_prob(worst.odds)) * -100.0
                )
            results.append(
                GameFairPrice(
                    event=best.event,
                    market=best.market,
                    side=side,
                    team=best.team,
                    point=best.point,
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
