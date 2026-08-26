"""Batted-ball quality profiles: barrel rate, hard-hit rate, exit velocity,
launch angle, and the expected-stat/quality-of-contact metrics that matter
most for spotting home run and extra-base-hit spots.

Real data comes from Baseball Savant via the `pybaseball` package:
- `statcast_batter_exitvelo_barrels(year)` / `statcast_pitcher_exitvelo_barrels(year)`
  for barrel%, hard-hit%, avg exit velo, avg launch angle, sweet-spot%.
- `statcast_batter_expected_stats(year)` / `statcast_pitcher_expected_stats(year)`
  for xwOBA, xSLG, xBA (contact-quality-adjusted outcomes, less luck-driven
  than actual results).
- `statcast_pitcher(start, end, player_id)` (pitch-level Baseball Savant log)
  for pitch-mix, throwing hand, and HR/9 + HR/FB% allowed - all derived
  directly from real batted-ball events, not FanGraphs (whose `pitching_stats`
  leaderboard is blocked outright from some hosting providers, e.g. GitHub
  Actions runners hit 403s scraping it).
- `statcast_batter(start, end, player_id)` (pitch-level Baseball Savant log,
  batter side) for a batter's own real HR/FB% - see `enrich_batted_ball()`,
  called only for the phase-2 prefiltered candidates in pipeline.py, not
  every roster batter.

`pybaseball` and `pandas` are optional dependencies - only required for
`PybaseballStatcastProvider`. `MockStatcastProvider` needs neither and is
what `--mock` mode uses.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import date
from typing import Dict, List, Optional

from ._ids import lookup_mlbam_id

logger = logging.getLogger(__name__)

# Column that identifies a player varies by which Baseball Savant CSV
# endpoint pybaseball is wrapping: some leaderboards use a plain
# "player_name" column, others (confirmed via a live run against the
# `/leaderboard/statcast` endpoint behind statcast_*_exitvelo_barrels) use
# "last_name, first_name" - literally "Last, First" as one column, comma
# included. Tried in order; add more if a future pybaseball/Savant version
# uses something else (check `--log-level DEBUG` output, which logs the
# real column names on every lookup).
_NAME_COLUMN_CANDIDATES = ("player_name", "last_name, first_name", "Name", "name")


def _find_player_row(df, player: str):
    """Best-effort, never-raises row lookup by full name across whichever
    of `_NAME_COLUMN_CANDIDATES` the given DataFrame actually has. Returns
    an empty slice (falsy via `.empty`) if nothing matches or the frame's
    schema doesn't include any recognized name column.
    """
    try:
        logger.debug("Matching %r against columns: %s", player, list(df.columns))
        target = player.strip().lower()
        last_first = None
        parts = target.rsplit(" ", 1)
        if len(parts) == 2:
            last_first = f"{parts[1]}, {parts[0]}"  # "Last, First" ordering, lowercased

        for col in _NAME_COLUMN_CANDIDATES:
            if col not in df.columns:
                continue
            values = df[col].astype(str).str.lower()
            match = df[values == target]
            if match.empty and last_first:
                match = df[values == last_first]
            if not match.empty:
                return match
        logger.warning(
            "No recognized player-name column for %r among %s - skipping this row", player, list(df.columns)
        )
    except Exception:
        logger.exception("Player lookup failed unexpectedly for %r", player)
    return df.iloc[0:0]


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

    def enrich_batted_ball(self, batter: BatterProfile) -> BatterProfile:
        """Optionally fill in additional per-batter batted-ball fields that
        are too expensive to compute for every roster batter up front (see
        `pipeline.py`'s two-phase scoring) - call this only for the
        prefiltered top candidates in phase 2, not the full-roster phase-1
        prefilter pass. Default: no-op (return the profile unchanged);
        override where a real per-player fetch is worth the extra cost.
        """
        return batter


class PybaseballStatcastProvider(StatcastProvider):
    """Pulls real season Statcast leaderboards from Baseball Savant via
    `pybaseball`. Requires `pip install pybaseball pandas` and outbound
    network access to `baseballsavant.mlb.com`.

    Column names below were confirmed against a live Baseball Savant pull
    (via `--log-level DEBUG`'s "Matching ... against columns" lines from a
    real run): `statcast_*_exitvelo_barrels(year)` returns
    `['last_name, first_name', 'player_id', 'attempts', 'avg_hit_angle',
    'anglesweetspotpercent', 'max_hit_speed', 'avg_hit_speed', 'ev50',
    'fbld', 'gb', 'max_distance', 'avg_distance', 'avg_hr_distance',
    'ev95plus', 'ev95percent', 'barrels', 'brl_percent', 'brl_pa']` - no
    team, bats/throws, PA/AB/HR, pull%, or HR/FB% columns at all, so those
    fields fall back to their defaults regardless of aliasing.
    `statcast_*_expected_stats(year)` returns `['last_name, first_name',
    'player_id', 'year', 'pa', 'bip', 'ba', 'est_ba',
    'est_ba_minus_ba_diff', 'slg', 'est_slg', 'est_slg_minus_slg_diff',
    'woba', 'est_woba', 'est_woba_minus_woba_diff', 'era', 'xera',
    'era_minus_xera_diff']` - no `iso` column; computed from `slg`/`ba`
    below instead. If a future pybaseball/Savant version changes these,
    the same debug log line will show the new columns.
    """

    _COLUMN_ALIASES: Dict[str, tuple] = {
        "barrel_pct": ("brl_percent", "barrel_batted_rate"),
        "hard_hit_pct": ("ev95percent", "hard_hit_percent"),
        "avg_exit_velo": ("avg_hit_speed", "exit_velocity_avg"),
        "avg_launch_angle": ("avg_hit_angle", "launch_angle_avg"),
        "sweet_spot_pct": ("anglesweetspotpercent", "sweet_spot_percent"),
        "xwoba": ("est_woba", "xwoba"),
        "xslg": ("est_slg", "xslg"),
        "actual_slg": ("slg",),
        "actual_ba": ("ba",),
        "pitch_mix": ("pitch_type", "pitch_usage"),
    }

    def __init__(self, year: int, min_bbe: int = 25, min_ip: float = 10.0):
        self.year = year
        self.min_bbe = min_bbe
        self.min_ip = min_ip
        self._batter_cache: Optional["object"] = None
        self._batter_expected_cache: Optional["object"] = None
        self._pitcher_cache: Optional["object"] = None
        self._pitcher_expected_cache: Optional["object"] = None
        self._id_cache: Dict[str, Optional[int]] = {}
        self._pitch_mix_cache: Dict[str, Dict[str, float]] = {}
        self._throws_cache: Dict[str, Optional[str]] = {}
        self._hr9_cache: Dict[str, float] = {}
        self._hr_fb_allowed_cache: Dict[str, float] = {}
        # A team's ~9 roster batters all share the same 1-2 probable
        # pitchers, so without memoizing by name, pitcher_profile() (now a
        # non-trivial fetch, with the pitch-mix lookup below) would redo
        # the same work ~9x per game.
        self._pitcher_profile_cache: Dict[str, Optional["PitcherProfile"]] = {}

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

    def _iso(self, row: Dict) -> float:
        """ISO = SLG - AVG. Not a column Baseball Savant's expected-stats
        leaderboard provides directly (confirmed live - see
        _COLUMN_ALIASES' docstring note) - computed from actual SLG/BA
        instead, both of which are.
        """
        slg = self._pick(row, "actual_slg")
        ba = self._pick(row, "actual_ba")
        if slg is None or ba is None:
            return 0.0
        return float(slg) - float(ba)

    def batter_profile(self, player: str) -> Optional[BatterProfile]:
        pyb = self._pyb()
        try:
            if self._batter_cache is None:
                self._batter_cache = pyb.statcast_batter_exitvelo_barrels(self.year, minBBE=self.min_bbe)
            if self._batter_expected_cache is None:
                self._batter_expected_cache = pyb.statcast_batter_expected_stats(self.year, minPA=self.min_bbe)
        except Exception:
            logger.exception("Failed to fetch Statcast batter leaderboard for %s", player)
            return None

        expected = self._batter_expected_cache
        df = self._batter_cache
        match = _find_player_row(df, player)
        if match.empty:
            logger.warning("No Statcast batter row found for %r", player)
            return None
        row = match.iloc[0].to_dict()

        exp_row: Dict = {}
        exp_match = _find_player_row(expected, player)
        if not exp_match.empty:
            exp_row = exp_match.iloc[0].to_dict()

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
                # pull_air_pct and hr_fb_pct default to 0.0 here (confirmed
                # live): neither Baseball Savant leaderboard used in this
                # method carries a pull% or HR/FB% column - those are
                # normally FanGraphs fields (Pull%, HR/FB), and FanGraphs is
                # blocked from GitHub Actions (see hot_streak.py's
                # StatcastHotStreakProvider docstring). hr_fb_pct gets a real
                # value later, but only for phase-2 candidates - see
                # `enrich_batted_ball()`. pull_air_pct has no such fix (see
                # that method's docstring for why) and stays a known gap.
                pull_air_pct=float(row.get("pull_percent", 0.0) or 0.0),
                hr_fb_pct=float(row.get("hr_fb_ratio", row.get("hr_fb", 0.0)) or 0.0),
                iso=self._iso(exp_row),
                xwoba=float(self._pick(exp_row, "xwoba") or 0.0),
                xslg=float(self._pick(exp_row, "xslg") or 0.0),
            )
        except (KeyError, TypeError, ValueError):
            logger.exception("Could not parse Statcast row for %r - check _COLUMN_ALIASES", player)
            return None

    def pitcher_profile(self, player: str) -> Optional[PitcherProfile]:
        if player in self._pitcher_profile_cache:
            return self._pitcher_profile_cache[player]

        result = self._pitcher_profile_uncached(player)
        self._pitcher_profile_cache[player] = result
        return result

    def enrich_batted_ball(self, batter: BatterProfile) -> BatterProfile:
        """Fills in `hr_fb_pct` (real HR-per-fly-ball rate) from a per-batter
        Statcast pitch-level log - the same real HR/fly-ball computation
        already proven for pitchers in `_pitcher_arsenal`, applied here to
        the batter's own batted balls instead of what they allowed.

        Deliberately NOT called from `batter_profile()`: that method runs
        for every roster batter in pipeline.py's cheap phase-1 prefilter
        (up to ~150-250+ players on a full slate), and a full-season
        pitch-level fetch per batter at that scale would reintroduce the
        exact per-player network cost the two-phase architecture exists to
        avoid. Call this only on the phase-2 prefiltered candidates.

        `pull_air_pct` stays at its default here on purpose: Statcast pitch
        logs carry raw hit-location coordinates (`hc_x`/`hc_y`), not a
        ready-made "pulled" flag, and deriving pull side from those requires
        a spray-angle formula this class has no way to verify against real
        data (this dev environment has no network access - see the module
        docstring). Shipping a guessed formula as a real number would be
        worse than the honest, documented 0.0 default it replaces.
        """
        try:
            pyb = self._pyb()
            player_id = lookup_mlbam_id(pyb, batter.player, self._id_cache)
            if player_id is None:
                return batter
            start = f"{self.year}-03-01"
            end = min(date.today(), date(self.year, 11, 30)).isoformat()
            pitches = pyb.statcast_batter(start, end, player_id)
            if pitches is None or pitches.empty or "events" not in pitches.columns:
                return batter
            pa_rows = pitches[pitches["events"].notna()]
            if pa_rows.empty or "bb_type" not in pa_rows.columns:
                return batter
            fb_count = int((pa_rows["bb_type"] == "fly_ball").sum())
            if not fb_count:
                return batter
            hr_count = int((pa_rows["events"] == "home_run").sum())
            hr_fb = round(hr_count / fb_count * 100, 1)
            return replace(batter, hr_fb_pct=hr_fb)
        except Exception:
            logger.warning(
                "Batted-ball enrichment failed for %r - hr_fb_pct stays at its default", batter.player, exc_info=True
            )
            return batter

    # Roughly the modern-MLB average number of plate appearances a pitcher
    # sees per 9 innings - used to convert a real HR-per-PA rate (which we
    # can compute directly from pitch-level data) into a HR/9 estimate,
    # since the exitvelo/expected-stats leaderboards used elsewhere in this
    # class don't carry innings-pitched at all (confirmed live).
    _PA_PER_9_INNINGS = 38.3

    def _pitcher_arsenal(self, pyb, player: str) -> "tuple[Dict[str, float], Optional[str], float, float]":
        """Everything about a pitcher that isn't on the exitvelo/expected-
        stats leaderboards, all pulled from one fetch of their own season
        of Statcast pitch-level data and cached together:
        - pitch-type usage%, e.g. {"FF": 0.42, "SL": 0.24, ...} - feeds
          `mlb_props.matchup`'s pitch-mix-edge component.
        - real throwing hand - `row.get("pitch_hand", ...)` elsewhere in
          this class silently defaults to "R" for every pitcher, including
          real lefties, which corrupts the platoon-edge component
          downstream for every batter facing one.
        - HR/9 (estimated as HR-per-PA-faced * 38.3) - this used to come
          from FanGraphs' pitching_stats(), which returns 403 to every
          request from a GitHub Actions runner (confirmed live), so it
          silently defaulted to 0.0 for every pitcher, every run. This
          estimate is real, if approximate (PA/9 varies somewhat by role
          and league).
        - HR/FB% allowed, computed the same way from batted-ball type.
        Best-effort throughout: any piece that can't be computed just
        means that component defaults to neutral, same as any other
        missing signal.
        """
        if player in self._pitch_mix_cache:
            return (
                self._pitch_mix_cache[player],
                self._throws_cache.get(player),
                self._hr9_cache.get(player, 0.0),
                self._hr_fb_allowed_cache.get(player, 0.0),
            )
        mix: Dict[str, float] = {}
        throws: Optional[str] = None
        hr9 = 0.0
        hr_fb_allowed = 0.0
        try:
            player_id = lookup_mlbam_id(pyb, player, self._id_cache)
            if player_id is not None:
                start = f"{self.year}-03-01"
                end = min(date.today(), date(self.year, 11, 30)).isoformat()
                pitches = pyb.statcast_pitcher(start, end, player_id)
                if pitches is not None and not pitches.empty:
                    if "pitch_type" in pitches.columns:
                        counts = pitches["pitch_type"].dropna().value_counts(normalize=True)
                        mix = {str(k): round(float(v), 4) for k, v in counts.items()}
                    if "p_throws" in pitches.columns:
                        hand_values = pitches["p_throws"].dropna()
                        if not hand_values.empty:
                            throws = str(hand_values.iloc[0])[:1] or None
                    if "events" in pitches.columns:
                        pa_rows = pitches[pitches["events"].notna()]
                        pa_count = len(pa_rows)
                        hr_count = int((pa_rows["events"] == "home_run").sum())
                        if pa_count:
                            hr9 = round((hr_count / pa_count) * self._PA_PER_9_INNINGS, 3)
                        if "bb_type" in pa_rows.columns:
                            fb_count = int((pa_rows["bb_type"] == "fly_ball").sum())
                            if fb_count:
                                hr_fb_allowed = round(hr_count / fb_count * 100, 1)
        except Exception:
            logger.warning("Pitch-arsenal fetch failed for %r - pitch_mix/throws/HR9 will default to neutral", player, exc_info=True)
        self._pitch_mix_cache[player] = mix
        self._throws_cache[player] = throws
        self._hr9_cache[player] = hr9
        self._hr_fb_allowed_cache[player] = hr_fb_allowed
        return mix, throws, hr9, hr_fb_allowed

    def _pitcher_profile_uncached(self, player: str) -> Optional[PitcherProfile]:
        pyb = self._pyb()
        # These two Baseball Savant leaderboards are the essential data for
        # this profile - if either fails, there's nothing useful to return.
        try:
            if self._pitcher_cache is None:
                self._pitcher_cache = pyb.statcast_pitcher_exitvelo_barrels(self.year, minBBE=self.min_bbe)
            if self._pitcher_expected_cache is None:
                self._pitcher_expected_cache = pyb.statcast_pitcher_expected_stats(self.year, minPA=self.min_bbe)
        except Exception:
            logger.exception("Failed to fetch Statcast pitcher leaderboard for %s", player)
            return None

        expected = self._pitcher_expected_cache
        df = self._pitcher_cache
        match = _find_player_row(df, player)
        if match.empty:
            logger.warning("No Statcast pitcher row found for %r", player)
            return None
        row = match.iloc[0].to_dict()

        exp_row: Dict = {}
        exp_match = _find_player_row(expected, player)
        if not exp_match.empty:
            exp_row = exp_match.iloc[0].to_dict()

        # HR/9 and HR/FB%-allowed used to come from FanGraphs' pitching_stats()
        # leaderboard - a separate, less reliable source than Baseball Savant
        # that's blocked outright from some hosting providers (GitHub Actions
        # runners hit 403s scraping it), which silently left every pitcher's
        # HR/9 at 0.0. Both are now derived from the same statcast_pitcher()
        # pitch-level log already fetched for pitch-mix/throws below, at no
        # extra network cost and with no FanGraphs dependency.
        pitch_mix, real_throws, hr9, hr_fb_allowed = self._pitcher_arsenal(pyb, player)
        fallback_throws = str(row.get("pitch_hand", row.get("p_throws", "R")))[:1] or "R"

        try:
            return PitcherProfile(
                player=player,
                team=str(row.get("team_name", row.get("team", ""))),
                throws=real_throws or fallback_throws,
                ip=float(row.get("ip", 0.0) or 0.0),
                barrel_pct_allowed=float(self._pick(row, "barrel_pct") or 0.0),
                hard_hit_pct_allowed=float(self._pick(row, "hard_hit_pct") or 0.0),
                avg_exit_velo_allowed=float(self._pick(row, "avg_exit_velo") or 0.0),
                hr_per_9=hr9,
                hr_fb_pct_allowed=hr_fb_allowed,
                xwoba_allowed=float(self._pick(exp_row, "xwoba") or 0.0),
                xslg_allowed=float(self._pick(exp_row, "xslg") or 0.0),
                pitch_mix=pitch_mix,
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
