"""Shared MLBAM player-id lookup - used by any real (`pybaseball`-backed)
provider that needs to call Baseball Savant with a numeric player id
instead of a name (matchup/platoon splits, pitch-mix, hot-streak windows).

Split out so the lookup logic and its cache-per-name behavior live in one
place instead of being copy-pasted across `matchup.py`, `statcast.py`, and
`hot_streak.py`.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def lookup_mlbam_id(pyb, full_name: str, cache: Dict[str, Optional[int]]) -> Optional[int]:
    """Resolve a "First Last" name to Baseball Savant's numeric player id
    via `pybaseball.playerid_lookup`. Caches both hits and misses in the
    caller-supplied `cache` dict (keyed by the exact name passed in) so a
    name that repeatedly fails to resolve doesn't retry the lookup every
    time - a real, if imperfect, tradeoff given how many times a single
    pitcher's name gets looked up across a slate.
    """
    if full_name in cache:
        return cache[full_name]

    parts = full_name.strip().split(" ", 1)
    if len(parts) != 2:
        logger.warning("Can't split %r into first/last name for playerid_lookup", full_name)
        cache[full_name] = None
        return None
    first, last = parts

    try:
        result = pyb.playerid_lookup(last, first)
    except Exception:
        logger.exception("playerid_lookup failed for %r", full_name)
        cache[full_name] = None
        return None

    if result.empty:
        logger.warning("playerid_lookup found no MLBAM id for %r", full_name)
        cache[full_name] = None
        return None

    player_id = int(result.iloc[0]["key_mlbam"])
    cache[full_name] = player_id
    return player_id
