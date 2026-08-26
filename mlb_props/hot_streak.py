"""'Who's hot': rolling-window recent form vs. season baseline, expressed as
a z-score so a hot streak in wOBA is comparable to one in HR rate.

Two real implementations:
- `StatcastHotStreakProvider` (the default) derives rolling wOBA windows
  directly from Baseball Savant pitch-level data - the same
  `woba_value`/`woba_denom` columns `mlb_props.matchup` already uses. This
  is the one actually wired up by `mlb_props_main.py`.
- `PybaseballHotStreakProvider` uses `pybaseball.batting_stats_range`
  (FanGraphs custom date-range splits) instead, which is simpler but
  proved unusable in practice: a live run found FanGraphs returning 403 to
  every request from a GitHub Actions runner (see git history / the
  mlb-props-report workflow's run log), so every player came back with the
  same neutral placeholder instead of real data. Kept for reference/local
  use where FanGraphs isn't blocked, but Baseball Savant is the more
  reliable source for this project's actual use case.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Optional

from ._ids import lookup_mlbam_id

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


class StatcastHotStreakProvider(HotStreakProvider):
    """Rolling wOBA windows computed from a player's own season of
    Baseball Savant pitch-level data (`pybaseball.statcast_batter`), the
    same source `mlb_props.matchup.PybaseballMatchupProvider` pulls for
    platoon/BvP - not FanGraphs. Requires `pip install pybaseball pandas`
    and network access to Baseball Savant.
    """

    def __init__(self, season_start: date):
        self.season_start = season_start
        self._id_cache: Dict[str, Optional[int]] = {}

    def _pyb(self):
        try:
            import pybaseball  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("pybaseball is required. Install with `pip install pybaseball pandas`.") from exc
        return pybaseball

    @staticmethod
    def _woba(rows) -> Optional[float]:
        if rows is None or len(rows) == 0 or "woba_value" not in rows.columns or "woba_denom" not in rows.columns:
            return None
        denom = rows["woba_denom"].sum()
        if not denom:
            return None
        return round(float(rows["woba_value"].sum() / denom), 3)

    def get_heat_index(self, player: str, as_of: Optional[date] = None) -> HeatIndex:
        pyb = self._pyb()
        today = as_of or date.today()
        neutral = HeatIndex(player, LEAGUE_AVG_WOBA, LEAGUE_AVG_WOBA, LEAGUE_AVG_WOBA, LEAGUE_AVG_WOBA, 0, 0.0)

        player_id = lookup_mlbam_id(pyb, player, self._id_cache)
        if player_id is None:
            return neutral

        try:
            log = pyb.statcast_batter(self.season_start.isoformat(), min(today, date(today.year, 11, 30)).isoformat(), player_id)
        except Exception:
            logger.exception("statcast_batter fetch failed for %r", player)
            return neutral
        if log is None or log.empty or "events" not in log.columns or "game_date" not in log.columns:
            return neutral

        pa_rows = log[log["events"].notna()]
        if pa_rows.empty:
            return neutral

        import pandas as pd  # local import: only needed for this real (non-mock) provider

        game_dates = pd.to_datetime(pa_rows["game_date"])

        def woba_and_pa_since(days: Optional[int]) -> "tuple[Optional[float], int]":
            if days is None:
                subset = pa_rows
            else:
                cutoff = pd.Timestamp(today - timedelta(days=days))
                subset = pa_rows[game_dates >= cutoff]
            return self._woba(subset), len(subset)

        season_woba, _ = woba_and_pa_since(None)
        last30_woba, _ = woba_and_pa_since(30)
        last15_woba, last15_pa = woba_and_pa_since(15)
        last7_woba, _ = woba_and_pa_since(7)

        season_woba = season_woba if season_woba is not None else LEAGUE_AVG_WOBA
        z = (last15_woba - season_woba) / WOBA_15D_STDEV if last15_woba is not None else 0.0

        return HeatIndex(
            player=player,
            season_woba=season_woba,
            last7_woba=last7_woba if last7_woba is not None else season_woba,
            last15_woba=last15_woba if last15_woba is not None else season_woba,
            last30_woba=last30_woba if last30_woba is not None else season_woba,
            last15_pa=last15_pa,
            z_score=round(z, 2),
        )


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
