"""Discrepancy detection: group lines by (player, league, market, side, event)
and flag groups whose max/min point spread meets a threshold.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .models import Discrepancy, PropLine


def find_discrepancies(lines: Iterable[PropLine], min_spread: float = 2.0) -> List[Discrepancy]:
    """Return one Discrepancy per prop where the widest cross-book point gap
    is >= min_spread, sorted from largest spread to smallest.
    """
    groups: Dict[Tuple[str, str, str, str, str], List[PropLine]] = defaultdict(list)
    for line in lines:
        groups[line.key].append(line)

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
