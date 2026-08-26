"""Discrepancy detection: group lines by (player, league, market, side, event)
and flag groups whose spread meets a threshold.

Two kinds of props need two kinds of comparison:
  - Line props (over/under a point value, e.g. player_points) - compare the
    point line itself. See find_discrepancies.
  - Binary props (Yes/No, e.g. player_home_runs "to hit a home run") - there
    is no line to compare, only a price, so compare implied win probability
    instead. See find_odds_discrepancies.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .models import BINARY_SIDES, LINE_SIDES, Discrepancy, OddsDiscrepancy, PropLine
from .odds_math import implied_probability_pct


def _group_by_key(lines: Iterable[PropLine]) -> Dict[Tuple[str, str, str, str, str], List[PropLine]]:
    groups: Dict[Tuple[str, str, str, str, str], List[PropLine]] = defaultdict(list)
    for line in lines:
        groups[line.key].append(line)
    return groups


def find_discrepancies(lines: Iterable[PropLine], min_spread: float = 2.0) -> List[Discrepancy]:
    """Return one Discrepancy per line-based prop (over/under a point value)
    where the widest cross-book point gap is >= min_spread, sorted from
    largest spread to smallest. Binary (Yes/No) props are ignored - they
    carry no comparable line - see find_odds_discrepancies for those.
    """
    groups = _group_by_key(line for line in lines if line.side.lower() in LINE_SIDES)

    discrepancies: List[Discrepancy] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        low = min(group, key=lambda l: l.line)
        high = max(group, key=lambda l: l.line)
        spread = high.line - low.line
        if spread >= min_spread:
            discrepancies.append(
                Discrepancy(
                    player=low.player,
                    league=low.league,
                    market=low.market,
                    side=low.side,
                    event=low.event,
                    spread=spread,
                    low=low,
                    high=high,
                    all_lines=tuple(group),
                )
            )

    discrepancies.sort(key=lambda d: d.spread, reverse=True)
    return discrepancies


def find_odds_discrepancies(lines: Iterable[PropLine], min_prob_spread: float = 8.0) -> List[OddsDiscrepancy]:
    """Return one OddsDiscrepancy per binary (Yes/No) prop where the widest
    cross-book gap in implied win probability is >= min_prob_spread
    percentage points, sorted from largest gap to smallest. Line-based props
    are ignored - see find_discrepancies for those.
    """
    groups = _group_by_key(
        line for line in lines if line.side.lower() in BINARY_SIDES and line.odds is not None
    )

    discrepancies: List[OddsDiscrepancy] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        priced = [(line, implied_probability_pct(line.odds)) for line in group]
        best_line, best_prob = min(priced, key=lambda pair: pair[1])
        worst_line, worst_prob = max(priced, key=lambda pair: pair[1])
        spread = worst_prob - best_prob
        if spread >= min_prob_spread:
            discrepancies.append(
                OddsDiscrepancy(
                    player=best_line.player,
                    league=best_line.league,
                    market=best_line.market,
                    side=best_line.side,
                    event=best_line.event,
                    prob_spread=spread,
                    best=best_line,
                    worst=worst_line,
                    best_prob_pct=best_prob,
                    worst_prob_pct=worst_prob,
                    all_lines=tuple(group),
                )
            )

    discrepancies.sort(key=lambda d: d.prob_spread, reverse=True)
    return discrepancies
