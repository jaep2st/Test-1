"""'Who's hot': rolling-window recent form vs. season baseline, expressed as
a z-score so a hot streak in wOBA is comparable to one in HR rate.

Real data: `pybaseball.batting_stats_range(start, end)` (FanGraphs custom
date-range splits) gives wOBA/ISO/HR/PA directly for any window, which is
much simpler than re-deriving it from pitch-level Statcast logs.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Rough league-average wOBA and its typical game-to-game standard deviation
# over a 15-day window for an everyday player - used to turn a raw wOBA
# surge into a z-score. These are reasonable approximations, not derived
# from a live pull.
LEAGUE_AVG_WOBA = 0.315
WOBA_15D_STDEV = 0.045


@dataclass(frozen=True)
class HeatIndex:
    player: str
    season_woba: float
    last7_woba: float
    last15_woba: float
    last30_woba: float
    last15_pa: int
    z_score: float  # how far last15_woba sits above season_woba, in stdev units

    @property
    def label(self) -> str:
        if self.z_score >= 1.5:
            return "scorching"
        if self.z_score >= 0.75:
            return "hot"
        if self.z_score <= -1.5:
            return "ice cold"
        if self.z_score <= -0.75:
            return "cold"
        return "steady"


class HotStreakProvider(ABC):
    @abstractmethod
    def get_heat_index(self, player: str, as_of: Optional[date] = None) -> HeatIndex:
        raise NotImplementedError


class PybaseballHotStreakProvider(HotStreakProvider):
    """Requires `pip install pybaseball pandas` and network access to
    FanGraphs (via pybaseball's `batting_stats_range`). Not exercised live
    in this build environment - verify column names with `--log-level
    DEBUG` before relying on it.
    """

    def __init__(self, season_start: date):
        self.season_start = season_start

    def _pyb(self):
        try:
            import pybaseball  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("pybaseball is required. Install with `pip install pybaseball pandas`.") from exc
        return pybaseball

    def _range_woba(self, pyb, player: str, start: date, end: date) -> "tuple[float, int]":
        try:
            df = pyb.batting_stats_range(start.isoformat(), end.isoformat())
        except Exception:
            logger.exception("batting_stats_range(%s, %s) failed", start, end)
            return LEAGUE_AVG_WOBA, 0
        match = df[df["Name"].str.lower() == player.strip().lower()]
        if match.empty:
            return LEAGUE_AVG_WOBA, 0
        row = match.iloc[0]
        woba = float(row.get("wOBA", LEAGUE_AVG_WOBA) or LEAGUE_AVG_WOBA)
        pa = int(row.get("PA", 0) or 0)
        return woba, pa

    def get_heat_index(self, player: str, as_of: Optional[date] = None) -> HeatIndex:
        pyb = self._pyb()
        today = as_of or date.today()
        season_woba, _ = self._range_woba(pyb, player, self.season_start, today)
        last7_woba, _ = self._range_woba(pyb, player, today - timedelta(days=7), today)
        last15_woba, last15_pa = self._range_woba(pyb, player, today - timedelta(days=15), today)
        last30_woba, _ = self._range_woba(pyb, player, today - timedelta(days=30), today)
        z = (last15_woba - season_woba) / WOBA_15D_STDEV if season_woba else 0.0
        return HeatIndex(
            player=player,
            season_woba=season_woba,
            last7_woba=last7_woba,
            last15_woba=last15_woba,
            last30_woba=last30_woba,
            last15_pa=last15_pa,
            z_score=round(z, 2),
        )


class MockHotStreakProvider(HotStreakProvider):
    """Synthetic recent-form data - no network calls."""

    def __init__(self, seed=None):
        import random

        self._rng = random.Random(seed)

    def get_heat_index(self, player: str, as_of: Optional[date] = None) -> HeatIndex:
        season_woba = round(self._rng.uniform(0.290, 0.400), 3)
        drift = self._rng.gauss(0, WOBA_15D_STDEV * 1.3)
        last15_woba = round(max(0.150, season_woba + drift), 3)
        last7_woba = round(max(0.150, last15_woba + self._rng.gauss(0, 0.03)), 3)
        last30_woba = round(max(0.150, (season_woba + last15_woba) / 2 + self._rng.gauss(0, 0.02)), 3)
        z = round((last15_woba - season_woba) / WOBA_15D_STDEV, 2)
        return HeatIndex(
            player=player,
            season_woba=season_woba,
            last7_woba=last7_woba,
            last15_woba=last15_woba,
            last30_woba=last30_woba,
            last15_pa=self._rng.randint(40, 65),
            z_score=z,
        )
