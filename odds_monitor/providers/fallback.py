"""Falls back to a second real odds provider when the primary one fails
systemically, instead of quietly degrading straight to no market data.

Confirmed live (2026-08-29): a run against The Odds API returned 401
Unauthorized on all 17 per-event odds requests in a row - almost certainly
a free-tier credit quota exhausted by repeated same-day runs - while the
rest of the pipeline (Statcast, matchup, weather) worked fine. Before this,
that failure meant every prop that run showed model-only, with no market
price or EV% at all, even though this project already ships a second real
provider (`BetstampProvider`) that was simply never tried because
`mlb_props_main.build_providers()` always preferred The Odds API whenever
its key was configured, with no fallback path.
"""

from __future__ import annotations

import logging
from typing import List

from ..models import PropLine
from .base import OddsProvider
from .theoddsapi import OddsFetchFailed

logger = logging.getLogger(__name__)


class FallbackOddsProvider(OddsProvider):
    """Tries `primary` first; on `OddsFetchFailed` (a systemic failure -
    see that exception's docstring), falls back to `secondary` and returns
    its result instead. A legitimately empty result from `primary` (e.g. no
    props posted yet for today's slate) is NOT a failure and is returned
    as-is, without touching `secondary` - this only kicks in when the
    primary provider could not fetch at all, not when it fetched
    successfully and found nothing.

    Any other exception from `primary`, or any exception from `secondary`
    itself, propagates to the caller unchanged - this class only adds one
    extra attempt, not a new error-handling policy. `mlb_props.pipeline`
    and `odds_monitor.scheduler` both already catch broadly around
    `fetch_player_props`, so a `secondary` failure still degrades to
    model-only rankings exactly as a lone provider's failure always has.
    """

    def __init__(self, primary: OddsProvider, secondary: OddsProvider):
        self.primary = primary
        self.secondary = secondary

    def fetch_player_props(self, league: str) -> List[PropLine]:
        try:
            return self.primary.fetch_player_props(league)
        except OddsFetchFailed:
            logger.warning(
                "%s failed systemically for league=%s - falling back to %s",
                type(self.primary).__name__,
                league,
                type(self.secondary).__name__,
                exc_info=True,
            )
            return self.secondary.fetch_player_props(league)
