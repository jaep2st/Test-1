"""Matchup-specific edges: platoon splits, batter-vs-pitcher history, and
pitch-mix fit (how well a batter's pitch-type performance lines up with
what the opposing pitcher actually throws).

Real data comes from Statcast pitch-level data via `pybaseball.statcast_batter`
(every pitch a batter has seen this season, with `p_throws`, `pitcher`,
`pitch_type`, and the `woba_value`/`woba_denom` needed to compute wOBA
splits) and `pybaseball.statcast_pitcher` (a pitcher's pitch-type usage).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

LEAGUE_AVG_WOBA = 0.315  # rough modern-era MLB league-average wOBA; used as a neutral baseline


@dataclass(frozen=True)
class MatchupProfile:
    """Everything about how this specific batter has historically handled
    this specific pitcher, this pitcher's throwing hand, and this pitcher's
    pitch mix.
    """

    batter: str
    pitcher: str

    # Platoon: batter's wOBA vs the pitcher's throwing hand this season.
    platoon_woba: float
    platoon_pa: int

    # Batter-vs-pitcher history (often small-sample; weight lightly).
    bvp_pa: int
    bvp_hr: int
    bvp_avg: float
    bvp_slg: float

    # How much better/worse the batter's wOBA is against the specific pitch
    # types this pitcher actually throws (usage-weighted), relative to the
    # batter's own season-average wOBA. Positive = favorable pitch mix.
    pitch_mix_edge: float


class MatchupProvider(ABC):
    @abstractmethod
    def get_matchup(
        self, batter: str, bats: str, pitcher: str, pitcher_throws: str, pitcher_pitch_mix: Dict[str, float]
    ) -> MatchupProfile:
        raise NotImplementedError


class PybaseballMatchupProvider(MatchupProvider):
    """Computes real platoon/BvP/pitch-mix edges from season Statcast
    pitch-level logs. Requires `pip install pybaseball pandas` and network
    access to Baseball Savant. Not exercised live in this build environment
    (see `mlb_props/statcast.py` docstring) - verify column names with
    `--log-level DEBUG` before relying on it.
    """

    def __init__(self, year: int):
        self.year = year
        self._id_cache: Dict[str, int] = {}

    def _pyb(self):
        try:
            import pybaseball  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("pybaseball is required. Install with `pip install pybaseball pandas`.") from exc
        return pybaseball

    def _lookup_id(self, full_name: str) -> Optional[int]:
        if full_name in self._id_cache:
            return self._id_cache[full_name]
        pyb = self._pyb()
        parts = full_name.strip().split(" ", 1)
        if len(parts) != 2:
            return None
        first, last = parts
        try:
            result = pyb.playerid_lookup(last, first)
        except Exception:
            logger.exception("playerid_lookup failed for %r", full_name)
            return None
        if result.empty:
            return None
        pid = int(result.iloc[0]["key_mlbam"])
        self._id_cache[full_name] = pid
        return pid

    def get_matchup(
        self, batter: str, bats: str, pitcher: str, pitcher_throws: str, pitcher_pitch_mix: Dict[str, float]
    ) -> MatchupProfile:
        pyb = self._pyb()
        batter_id = self._lookup_id(batter)
        pitcher_id = self._lookup_id(pitcher)
        if batter_id is None:
            logger.warning("Could not resolve MLBAM id for batter %r", batter)
            return MatchupProfile(batter, pitcher, LEAGUE_AVG_WOBA, 0, 0, 0, 0.0, 0.0, 0.0)

        start = f"{self.year}-03-01"
        end = f"{self.year}-11-30"
        try:
            log = pyb.statcast_batter(start, end, batter_id)
        except Exception:
            logger.exception("statcast_batter fetch failed for %r", batter)
            return MatchupProfile(batter, pitcher, LEAGUE_AVG_WOBA, 0, 0, 0, 0.0, 0.0, 0.0)

        pa_rows = log[log["events"].notna()] if "events" in log.columns else log.iloc[0:0]

        # Platoon split: wOBA vs the pitcher's throwing hand.
        opp_hand = "L" if pitcher_throws == "L" else "R"
        platoon_rows = pa_rows[pa_rows.get("p_throws", opp_hand) == opp_hand] if "p_throws" in pa_rows.columns else pa_rows
        platoon_woba = self._woba(platoon_rows)
        platoon_pa = len(platoon_rows)

        # Batter vs. this specific pitcher.
        bvp_rows = pa_rows[pa_rows.get("pitcher") == pitcher_id] if pitcher_id and "pitcher" in pa_rows.columns else pa_rows.iloc[0:0]
        bvp_pa = len(bvp_rows)
        bvp_hr = int((bvp_rows.get("events") == "home_run").sum()) if bvp_pa else 0
        bvp_hits = int(bvp_rows.get("events").isin(["single", "double", "triple", "home_run"]).sum()) if bvp_pa else 0
        bvp_avg = round(bvp_hits / bvp_pa, 3) if bvp_pa else 0.0
        bvp_slg = round(self._total_bases(bvp_rows) / bvp_pa, 3) if bvp_pa else 0.0

        # Pitch-mix edge: batter's wOBA against the pitch types this pitcher
        # actually throws (usage-weighted), relative to the batter's own
        # season wOBA - i.e. is this batter better *than his own average*
        # against the specific mix he's about to face.
        pitch_mix_edge = self._pitch_mix_edge(pa_rows, pitcher_pitch_mix)

        return MatchupProfile(
            batter=batter,
            pitcher=pitcher,
            platoon_woba=platoon_woba,
            platoon_pa=platoon_pa,
            bvp_pa=bvp_pa,
            bvp_hr=bvp_hr,
            bvp_avg=bvp_avg,
            bvp_slg=bvp_slg,
            pitch_mix_edge=pitch_mix_edge,
        )

    @staticmethod
    def _woba(rows) -> float:
        if rows is None or len(rows) == 0 or "woba_value" not in rows.columns or "woba_denom" not in rows.columns:
            return LEAGUE_AVG_WOBA
        denom = rows["woba_denom"].sum()
        if not denom:
            return LEAGUE_AVG_WOBA
        return round(rows["woba_value"].sum() / denom, 3)

    @staticmethod
    def _total_bases(rows) -> int:
        if rows is None or len(rows) == 0 or "events" not in rows.columns:
            return 0
        weights = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
        return int(sum(weights.get(e, 0) for e in rows["events"]))

    def _pitch_mix_edge(self, pa_rows, pitcher_pitch_mix: Dict[str, float]) -> float:
        usable_mix = dict(pitcher_pitch_mix)
        if not usable_mix or pa_rows is None or len(pa_rows) == 0 or "pitch_type" not in pa_rows.columns:
            return 0.0
        overall_woba = self._woba(pa_rows)
        weighted = 0.0
        weight_sum = 0.0
        for pitch_type, usage in usable_mix.items():
            subset = pa_rows[pa_rows["pitch_type"] == pitch_type]
            if len(subset) < 5:  # too little sample against this pitch type to trust
                continue
            weighted += self._woba(subset) * usage
            weight_sum += usage
        if weight_sum == 0:
            return 0.0
        return round((weighted / weight_sum) - overall_woba, 4)


class MockMatchupProvider(MatchupProvider):
    """Synthetic matchup edges - no network calls. Numbers are randomized
    (seeded) within realistic ranges; not real scouting data.
    """

    def __init__(self, seed=None):
        import random

        self._rng = random.Random(seed)

    def get_matchup(
        self, batter: str, bats: str, pitcher: str, pitcher_throws: str, pitcher_pitch_mix: Dict[str, float]
    ) -> MatchupProfile:
        platoon_woba = round(self._rng.uniform(0.260, 0.400), 3)
        bvp_pa = self._rng.choice([0, 0, 3, 5, 8, 12, 18])
        bvp_hr = self._rng.choices([0, 1, 2], weights=[0.7, 0.25, 0.05])[0] if bvp_pa else 0
        bvp_avg = round(self._rng.uniform(0.150, 0.400), 3) if bvp_pa else 0.0
        bvp_slg = round(bvp_avg + self._rng.uniform(0.0, 0.350), 3) if bvp_pa else 0.0
        pitch_mix_edge = round(self._rng.uniform(-0.045, 0.045), 4)
        return MatchupProfile(
            batter=batter,
            pitcher=pitcher,
            platoon_woba=platoon_woba,
            platoon_pa=self._rng.randint(80, 300),
            bvp_pa=bvp_pa,
            bvp_hr=bvp_hr,
            bvp_avg=bvp_avg,
            bvp_slg=bvp_slg,
            pitch_mix_edge=pitch_mix_edge,
        )
