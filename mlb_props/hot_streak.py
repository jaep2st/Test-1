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

# WOBA_15D_STDEV above is calibrated for an everyday player's typical real
# plate-appearance count over a 15-day/15-game window - not a fixed
# per-player constant. A genuinely smaller sample (a part-time player, a
# recent callup, someone a few games off the IL) has a noisier last15_woba
# than that calibration assumes, so a hot streak built on a handful of real
# PA used to swing z_score exactly as hard as one built on a full sample -
# a real bug (3-for-5 read as "scorching", same as a real 20-for-55).
# Standard error scales with 1/sqrt(PA), so `_shrunk_woba_stdev` below
# scales WOBA_15D_STDEV up the same way whenever last15_pa falls short of
# this reference, pulling z_score toward 0 in proportion to how little real
# data backs it - the smaller the sample, the less it can move the score.
# ~4.3 PA/game (a real, typical MLB rate) * 15 team-games.
REFERENCE_PA_FOR_15D_STDEV = 58


def _shrunk_woba_stdev(last15_pa: int) -> float:
    """The effective stdev to divide a last15-vs-season wOBA gap by, scaled
    up (never down - a bigger-than-reference sample doesn't make the base
    calibration itself more precise) for a below-reference real PA count.
    See REFERENCE_PA_FOR_15D_STDEV's docstring.
    """
    if last15_pa <= 0:
        return float("inf")
    reference_pa = min(last15_pa, REFERENCE_PA_FOR_15D_STDEV)
    return WOBA_15D_STDEV * (REFERENCE_PA_FOR_15D_STDEV / reference_pa) ** 0.5


# Real total-base value per Statcast `events` outcome. Only PA-ending
# events appear in the per-PA log this is computed from (strikeout, walk,
# HBP, sac fly, field_out, double_play, ...); anything not a key here
# contributes 0 bases and doesn't count as a hit for that plate appearance.
_HIT_BASES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}


@dataclass(frozen=True)
class ClearanceWindow:
    """Real per-game outcome counts over a window of actual games played -
    literal 'did this player clear the line in this specific game', not a
    rolling average or a market-implied estimate. `games` is real games
    played in the window (can be fewer than the window's nominal size early
    in a season or right off an IL stint) - always carried alongside the
    counts so a 2-for-3 sample is never confused with an 8-for-15 one.
    """

    games: int
    hr_games: int
    tb2_games: int
    hit_games: int

    @property
    def hr_rate(self) -> Optional[float]:
        return round(self.hr_games / self.games, 3) if self.games else None

    @property
    def tb2_rate(self) -> Optional[float]:
        return round(self.tb2_games / self.games, 3) if self.games else None

    @property
    def hit_rate(self) -> Optional[float]:
        return round(self.hit_games / self.games, 3) if self.games else None


def game_outcomes_from_events(games_events: "list[list[str]]") -> "list[tuple[bool, bool, bool]]":
    """`games_events`: one list of real Statcast `events` values per game
    actually played, oldest first (one list entry per plate appearance in
    that game). Returns one (got_hr, got_2plus_tb, got_hit) tuple per game,
    same order - the literal outcome real bettors would have cashed or
    missed that day.
    """
    outcomes = []
    for events in games_events:
        bases = sum(_HIT_BASES.get(e, 0) for e in events)
        got_hit = any(e in _HIT_BASES for e in events)
        got_hr = any(e == "home_run" for e in events)
        outcomes.append((got_hr, bases >= 2, got_hit))
    return outcomes


def clearance_windows_from_outcomes(
    outcomes: "list[tuple[bool, bool, bool]]",
) -> "tuple[Optional[ClearanceWindow], Optional[ClearanceWindow], Optional[ClearanceWindow], Optional[ClearanceWindow]]":
    """`outcomes`: real per-game (got_hr, got_2plus_tb, got_hit) tuples,
    oldest game first (see `game_outcomes_from_events`). Returns
    (last5, last10, last15, season) windows - "last N" means the N most
    recently *played* real games, not N calendar days, so a scheduled
    off-day never dilutes the window the way a day-based cutoff would.
    `None` for a window with zero games available (e.g. a rookie's first
    week), matching this project's "unknown stays None, never a fake zero"
    convention.
    """

    def window(n: Optional[int]) -> Optional[ClearanceWindow]:
        subset = outcomes if n is None else outcomes[-n:]
        if not subset:
            return None
        return ClearanceWindow(
            games=len(subset),
            hr_games=sum(1 for hr, _, _ in subset if hr),
            tb2_games=sum(1 for _, tb2, _ in subset if tb2),
            hit_games=sum(1 for _, _, h in subset if h),
        )

    return window(5), window(10), window(15), window(None)


@dataclass(frozen=True)
class HeatIndex:
    player: str
    season_woba: float
    last7_woba: float
    last15_woba: float
    last30_woba: float
    last15_pa: int
    z_score: float  # how far last15_woba sits above season_woba, in stdev units
    # Real per-game clearance counts (see ClearanceWindow) - None when
    # unavailable (no per-game log for this provider/player), never a
    # guessed zero. Populated only by StatcastHotStreakProvider, which has
    # the real per-PA-with-game-date log this needs; computed from the same
    # fetch already made for the wOBA fields above, at no extra network cost.
    clear_l5: Optional["ClearanceWindow"] = None
    clear_l10: Optional["ClearanceWindow"] = None
    clear_l15: Optional["ClearanceWindow"] = None
    clear_season: Optional["ClearanceWindow"] = None

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
        # Shrunk toward 0 for a below-reference last15_pa - see
        # _shrunk_woba_stdev's docstring. A player with zero real
        # plate appearances in the window (last15_woba is None) already
        # falls through to the honest neutral z=0.0, not a guessed streak.
        z = (last15_woba - season_woba) / _shrunk_woba_stdev(last15_pa) if last15_woba is not None else 0.0

        # Real per-game clearance counts, grouped from the same per-PA log
        # already fetched above - no extra network cost. String `game_date`
        # sorts chronologically same as a parsed date would (ISO format),
        # so grouping directly on it (rather than the parsed `game_dates`
        # Series) keeps games in real chronological order without a second
        # column merge.
        games_events = [list(grp["events"]) for _, grp in pa_rows.groupby("game_date", sort=True)]
        outcomes = game_outcomes_from_events(games_events)
        clear_l5, clear_l10, clear_l15, clear_season = clearance_windows_from_outcomes(outcomes)

        return HeatIndex(
            player=player,
            season_woba=season_woba,
            last7_woba=last7_woba if last7_woba is not None else season_woba,
            last15_woba=last15_woba if last15_woba is not None else season_woba,
            last30_woba=last30_woba if last30_woba is not None else season_woba,
            last15_pa=last15_pa,
            z_score=round(z, 2),
            clear_l5=clear_l5,
            clear_l10=clear_l10,
            clear_l15=clear_l15,
            clear_season=clear_season,
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
        # See StatcastHotStreakProvider's use of _shrunk_woba_stdev above -
        # same shrinkage, same reasoning, kept consistent between the two
        # real providers even though this one isn't the one actually wired
        # up (see class docstring).
        z = (last15_woba - season_woba) / _shrunk_woba_stdev(last15_pa) if season_woba else 0.0
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

    def _synth_window(self, games: int, hr_pct: float) -> ClearanceWindow:
        """A HR implies 2+ TB implies a hit, so sample the counts nested in
        that order rather than independently - keeps synthetic windows
        internally consistent the way a real box-score-derived one always is.
        """
        hr_games = sum(1 for _ in range(games) if self._rng.random() < hr_pct)
        tb2_games = hr_games + sum(1 for _ in range(games - hr_games) if self._rng.random() < 0.18)
        hit_games = tb2_games + sum(1 for _ in range(games - tb2_games) if self._rng.random() < 0.45)
        return ClearanceWindow(games=games, hr_games=hr_games, tb2_games=tb2_games, hit_games=hit_games)

    def get_heat_index(self, player: str, as_of: Optional[date] = None) -> HeatIndex:
        season_woba = round(self._rng.uniform(0.290, 0.400), 3)
        drift = self._rng.gauss(0, WOBA_15D_STDEV * 1.3)
        last15_woba = round(max(0.150, season_woba + drift), 3)
        last7_woba = round(max(0.150, last15_woba + self._rng.gauss(0, 0.03)), 3)
        last30_woba = round(max(0.150, (season_woba + last15_woba) / 2 + self._rng.gauss(0, 0.02)), 3)
        z = round((last15_woba - season_woba) / WOBA_15D_STDEV, 2)
        hr_pct = max(0.03, min(0.30, 0.10 + z * 0.04))
        return HeatIndex(
            player=player,
            season_woba=season_woba,
            last7_woba=last7_woba,
            last15_woba=last15_woba,
            last30_woba=last30_woba,
            last15_pa=self._rng.randint(40, 65),
            z_score=z,
            clear_l5=self._synth_window(5, hr_pct),
            clear_l10=self._synth_window(10, hr_pct),
            clear_l15=self._synth_window(15, hr_pct),
            clear_season=self._synth_window(self._rng.randint(80, 140), 0.10),
        )
