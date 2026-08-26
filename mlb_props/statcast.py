"""Batted-ball quality profiles: barrel rate, hard-hit rate, exit velocity,
launch angle, and the expected-stat/quality-of-contact metrics that matter
most for spotting home run and extra-base-hit spots.

Real data comes from Baseball Savant via the `pybaseball` package:
- `statcast_batter_exitvelo_barrels(year)` / `statcast_pitcher_exitvelo_barrels(year)`
  for barrel%, hard-hit%, avg exit velo, avg launch angle, sweet-spot%.
- `statcast_batter_expected_stats(year)` / `statcast_pitcher_expected_stats(year)`
  for xwOBA, xSLG, xBA (contact-quality-adjusted outcomes, less luck-driven
  than actual results).
- `pitching_stats(year)` (FanGraphs, also bundled with pybaseball) for HR/9,
  FB%, and HR/FB% allowed.

`pybaseball` and `pandas` are optional dependencies - only required for
`PybaseballStatcastProvider`. `MockStatcastProvider` needs neither and is
what `--mock` mode uses.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatterProfile:
    """One hitter's season-to-date batted-ball quality."""

    player: str
    team: str
    bats: str  # "L", "R", or "S" (switch)
    pa: int
    ab: int
    hr: int
    barrel_pct: float  # % of batted balls hit 98+ mph at the optimal launch angle window
    hard_hit_pct: float  # % of batted balls hit 95+ mph
    avg_exit_velo: float  # mph
    avg_launch_angle: float  # degrees
    sweet_spot_pct: float  # % of batted balls with launch angle 8-32 degrees
    pull_air_pct: float  # % of fly balls/line drives pulled - strongly correlated with HR power
    hr_fb_pct: float  # HR per fly ball, %
    iso: float  # isolated power = SLG - AVG
    xwoba: float
    xslg: float


@dataclass(frozen=True)
class PitcherProfile:
    """One pitcher's season-to-date quality of contact allowed, plus arsenal."""

    player: str
    team: str
    throws: str  # "L" or "R"
    ip: float
    barrel_pct_allowed: float
    hard_hit_pct_allowed: float
    avg_exit_velo_allowed: float
    hr_per_9: float
    hr_fb_pct_allowed: float
    xwoba_allowed: float
    xslg_allowed: float
    pitch_mix: Dict[str, float]  # e.g. {"FF": 0.42, "SL": 0.24, "CH": 0.14, ...} usage shares, sum ~1.0


class StatcastProvider(ABC):
    """Source of batter/pitcher batted-ball-quality profiles."""

    @abstractmethod
    def batter_profile(self, player: str) -> Optional[BatterProfile]:
        raise NotImplementedError

    @abstractmethod
    def pitcher_profile(self, player: str) -> Optional[PitcherProfile]:
        raise NotImplementedError


class PybaseballStatcastProvider(StatcastProvider):
    """Pulls real season Statcast leaderboards from Baseball Savant via
    `pybaseball`. Requires `pip install pybaseball pandas` and outbound
    network access to `baseballsavant.mlb.com`.

    NOTE: this provider was written against pybaseball's documented public
    functions (`statcast_batter_exitvelo_barrels`,
    `statcast_pitcher_exitvelo_barrels`, `statcast_batter_expected_stats`,
    `statcast_pitcher_expected_stats`, `pitching_stats`,
    `playerid_lookup`), but a live pull against Baseball Savant was not
    possible from the environment this was built in (outbound access to
    `baseballsavant.mlb.com` was blocked there). Before relying on it,
    run once with `--log-level DEBUG`, inspect the DataFrame column names
    for your installed pybaseball version, and adjust `_COLUMN_ALIASES`
    below if they've changed.
    """

    _COLUMN_ALIASES: Dict[str, tuple] = {
        "barrel_pct": ("brl_percent", "barrel_batted_rate"),
        "hard_hit_pct": ("hard_hit_percent",),
        "avg_exit_velo": ("avg_hit_speed", "exit_velocity_avg"),
        "avg_launch_angle": ("avg_hit_angle", "launch_angle_avg"),
        "sweet_spot_pct": ("sweet_spot_percent",),
        "xwoba": ("xwoba",),
        "xslg": ("xslg",),
        "pitch_mix": ("pitch_type", "pitch_usage"),
    }

    def __init__(self, year: int, min_bbe: int = 25, min_ip: float = 10.0):
        self.year = year
        self.min_bbe = min_bbe
        self.min_ip = min_ip
        self._batter_cache: Optional["object"] = None
        self._pitcher_cache: Optional["object"] = None

    def _pyb(self):
        try:
            import pybaseball  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "pybaseball is required for PybaseballStatcastProvider. "
                "Install it with `pip install pybaseball pandas`."
            ) from exc
        return pybaseball

    def _pick(self, row, field: str):
        for col in self._COLUMN_ALIASES.get(field, ()):
            if col in row and row[col] is not None:
                return row[col]
        return None

    def batter_profile(self, player: str) -> Optional[BatterProfile]:
        pyb = self._pyb()
        try:
            if self._batter_cache is None:
                self._batter_cache = pyb.statcast_batter_exitvelo_barrels(self.year, minBBE=self.min_bbe)
            expected = pyb.statcast_batter_expected_stats(self.year, minPA=self.min_bbe)
        except Exception:
            logger.exception("Failed to fetch Statcast batter leaderboard for %s", player)
            return None

        df = self._batter_cache
        match = df[df["player_name"].str.lower() == player.strip().lower()]
        if match.empty:
            logger.warning("No Statcast batter row found for %r", player)
            return None
        row = match.iloc[0].to_dict()

        exp_row: Dict = {}
        try:
            exp_match = expected[expected["player_name"].str.lower() == player.strip().lower()]
            if not exp_match.empty:
                exp_row = exp_match.iloc[0].to_dict()
        except Exception:
            logger.debug("No expected-stats row for %r", player)

        try:
            return BatterProfile(
                player=player,
                team=str(row.get("team_name", row.get("team", ""))),
                bats=str(row.get("bat_side", row.get("stand", "R")))[:1] or "R",
                pa=int(row.get("pa", exp_row.get("pa", 0)) or 0),
                ab=int(row.get("ab", 0) or 0),
                hr=int(row.get("home_run", row.get("hr", 0)) or 0),
                barrel_pct=float(self._pick(row, "barrel_pct") or 0.0),
                hard_hit_pct=float(self._pick(row, "hard_hit_pct") or 0.0),
                avg_exit_velo=float(self._pick(row, "avg_exit_velo") or 0.0),
                avg_launch_angle=float(self._pick(row, "avg_launch_angle") or 0.0),
                sweet_spot_pct=float(self._pick(row, "sweet_spot_pct") or 0.0),
                pull_air_pct=float(row.get("pull_percent", 0.0) or 0.0),
                hr_fb_pct=float(row.get("hr_fb_ratio", row.get("hr_fb", 0.0)) or 0.0),
                iso=float(exp_row.get("iso", 0.0) or 0.0),
                xwoba=float(self._pick(exp_row, "xwoba") or 0.0),
                xslg=float(self._pick(exp_row, "xslg") or 0.0),
            )
        except (KeyError, TypeError, ValueError):
            logger.exception("Could not parse Statcast row for %r - check _COLUMN_ALIASES", player)
            return None

    def pitcher_profile(self, player: str) -> Optional[PitcherProfile]:
        pyb = self._pyb()
        try:
            if self._pitcher_cache is None:
                self._pitcher_cache = pyb.statcast_pitcher_exitvelo_barrels(self.year, minBBE=self.min_bbe)
            expected = pyb.statcast_pitcher_expected_stats(self.year, minPA=self.min_bbe)
            fg = pyb.pitching_stats(self.year, qual=self.min_ip)
        except Exception:
            logger.exception("Failed to fetch Statcast pitcher leaderboard for %s", player)
            return None

        df = self._pitcher_cache
        match = df[df["player_name"].str.lower() == player.strip().lower()]
        if match.empty:
            logger.warning("No Statcast pitcher row found for %r", player)
            return None
        row = match.iloc[0].to_dict()

        exp_row: Dict = {}
        try:
            exp_match = expected[expected["player_name"].str.lower() == player.strip().lower()]
            if not exp_match.empty:
                exp_row = exp_match.iloc[0].to_dict()
        except Exception:
            logger.debug("No expected-stats row for %r", player)

        hr9 = 0.0
        try:
            fg_match = fg[fg["Name"].str.lower() == player.strip().lower()]
            if not fg_match.empty:
                fg_row = fg_match.iloc[0].to_dict()
                hr9 = float(fg_row.get("HR/9", 0.0) or 0.0)
        except Exception:
            logger.debug("No FanGraphs pitching row for %r", player)

        try:
            return PitcherProfile(
                player=player,
                team=str(row.get("team_name", row.get("team", ""))),
                throws=str(row.get("pitch_hand", row.get("p_throws", "R")))[:1] or "R",
                ip=float(row.get("ip", 0.0) or 0.0),
                barrel_pct_allowed=float(self._pick(row, "barrel_pct") or 0.0),
                hard_hit_pct_allowed=float(self._pick(row, "hard_hit_pct") or 0.0),
                avg_exit_velo_allowed=float(self._pick(row, "avg_exit_velo") or 0.0),
                hr_per_9=hr9,
                hr_fb_pct_allowed=float(row.get("hr_fb_ratio", row.get("hr_fb", 0.0)) or 0.0),
                xwoba_allowed=float(self._pick(exp_row, "xwoba") or 0.0),
                xslg_allowed=float(self._pick(exp_row, "xslg") or 0.0),
                pitch_mix={},  # populate via mlb_props.matchup's pitch-mix lookup
            )
        except (KeyError, TypeError, ValueError):
            logger.exception("Could not parse Statcast pitcher row for %r - check _COLUMN_ALIASES", player)
            return None


class MockStatcastProvider(StatcastProvider):
    """Synthetic batted-ball profiles for a handful of illustrative names.
    No network calls, no dependencies. Numbers are randomized (seeded) within
    realistic MLB full-season ranges - they are NOT real stats and should
    never be presented as such. Use this only to exercise the pipeline.
    """

    _REALISTIC_BATTER_RANGES = dict(
        barrel_pct=(3.0, 22.0),
        hard_hit_pct=(28.0, 58.0),
        avg_exit_velo=(85.0, 95.5),
        avg_launch_angle=(6.0, 22.0),
        sweet_spot_pct=(22.0, 45.0),
        pull_air_pct=(12.0, 48.0),
        hr_fb_pct=(6.0, 34.0),
        iso=(0.100, 0.320),
        xwoba=(0.290, 0.430),
        xslg=(0.360, 0.650),
    )
    _REALISTIC_PITCHER_RANGES = dict(
        barrel_pct_allowed=(3.0, 12.0),
        hard_hit_pct_allowed=(28.0, 46.0),
        avg_exit_velo_allowed=(86.0, 91.5),
        hr_per_9=(0.6, 2.1),
        hr_fb_pct_allowed=(6.0, 18.0),
        xwoba_allowed=(0.280, 0.360),
        xslg_allowed=(0.360, 0.470),
    )

    def __init__(self, seed: Optional[int] = None):
        import random

        self._rng = random.Random(seed)

    def _rand_range(self, lo: float, hi: float) -> float:
        return round(self._rng.uniform(lo, hi), 3)

    def batter_profile(self, player: str) -> Optional[BatterProfile]:
        r = self._REALISTIC_BATTER_RANGES
        pa = self._rng.randint(300, 600)
        ab = int(pa * 0.88)
        return BatterProfile(
            player=player,
            team=self._rng.choice(["NYY", "LAD", "ATL", "HOU", "PHI", "BAL", "SEA", "TEX"]),
            bats=self._rng.choice(["L", "R", "R", "S"]),
            pa=pa,
            ab=ab,
            hr=int(ab * self._rand_range(*r["hr_fb_pct"]) / 100 * 0.35),
            barrel_pct=self._rand_range(*r["barrel_pct"]),
            hard_hit_pct=self._rand_range(*r["hard_hit_pct"]),
            avg_exit_velo=self._rand_range(*r["avg_exit_velo"]),
            avg_launch_angle=self._rand_range(*r["avg_launch_angle"]),
            sweet_spot_pct=self._rand_range(*r["sweet_spot_pct"]),
            pull_air_pct=self._rand_range(*r["pull_air_pct"]),
            hr_fb_pct=self._rand_range(*r["hr_fb_pct"]),
            iso=self._rand_range(*r["iso"]),
            xwoba=self._rand_range(*r["xwoba"]),
            xslg=self._rand_range(*r["xslg"]),
        )

    def pitcher_profile(self, player: str) -> Optional[PitcherProfile]:
        r = self._REALISTIC_PITCHER_RANGES
        pitch_types = ["FF", "SI", "SL", "CH", "CU", "FC", "ST"]
        usages = [self._rng.random() for _ in range(self._rng.randint(3, 5))]
        total = sum(usages)
        mix = {pt: round(u / total, 3) for pt, u in zip(self._rng.sample(pitch_types, len(usages)), usages)}
        return PitcherProfile(
            player=player,
            team=self._rng.choice(["NYM", "SD", "MIN", "CLE", "MIL", "ARI", "TB", "DET"]),
            throws=self._rng.choice(["L", "R", "R", "R"]),
            ip=round(self._rng.uniform(40, 160), 1),
            barrel_pct_allowed=self._rand_range(*r["barrel_pct_allowed"]),
            hard_hit_pct_allowed=self._rand_range(*r["hard_hit_pct_allowed"]),
            avg_exit_velo_allowed=self._rand_range(*r["avg_exit_velo_allowed"]),
            hr_per_9=self._rand_range(*r["hr_per_9"]),
            hr_fb_pct_allowed=self._rand_range(*r["hr_fb_pct_allowed"]),
            xwoba_allowed=self._rand_range(*r["xwoba_allowed"]),
            xslg_allowed=self._rand_range(*r["xslg_allowed"]),
            pitch_mix=mix,
        )
