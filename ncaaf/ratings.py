"""Each FBS team's real power rating for a season: a blend of two
independent signals -

1. **SP+** (`GET /ratings/sp` via `cfbd.py`) - Bill Connelly's public,
   opponent-adjusted efficiency rating, split into overall/offense/defense/
   special-teams. Season-level, not point-in-time (see the lookahead-bias
   caveat in `historical_backtest.py`'s module docstring).
2. **Elo** (`elo.py`) - this project's own rating, computed purely from
   final scores of games already played, genuinely point-in-time by
   construction.

A third signal, **PPA** (predicted points added, `GET /ppa/teams`), blends
into the offense/defense split specifically - a different methodology
(play-by-play efficiency) than SP+'s, so agreement between the two is a
real, if imperfect, cross-check.

Blending method: each raw signal is z-scored across that season's real team
pool (mean 0, std 1) before blending, sidestepping the need to know each
signal's exact native units - the blended z-score is then rescaled to real
point-units using a realistic std-dev of FBS team quality
(`POWER_RATING_POINTS_STD`/`OFFENSE_POINTS_STD`/`DEFENSE_POINTS_STD`, chosen
from realistic real SP+ score spreads, not fit against this project's own
results yet). This is a transparent, hand-weighted ensemble - not a trained/
calibrated model - see `scoring.py`'s module docstring for the same caveat
applied to the final margin/total prediction built on top of it.

**Sign-convention disclosure:** CFBD's real, live JSON shape for
`defense.rating`/`ppa` defense fields was not confirmed against a live
response while building this (no outbound network access in this dev
environment - see the rest of this project's real-provider modules for the
same disclosed situation, e.g. `mlb_props/statcast.py`). The constants below
assume Bill Connelly's publicly documented convention (a lower raw
`defense.rating` = fewer points allowed than average = a better defense) and
PPA's own convention (a more negative raw defense PPA = a better defense,
since PPA is computed from the offense's perspective). If a real run's
`--log-level DEBUG` output shows a defense you know is elite (e.g. a top-5
scoring defense) landing with a below-average normalized `defense_rating`,
flip `INVERT_SP_DEFENSE`/`INVERT_PPA_DEFENSE` below.
"""

from __future__ import annotations

import logging
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from .elo import BASE_RATING, GameResult, compute_elo_ratings

logger = logging.getLogger(__name__)

INVERT_SP_DEFENSE = True  # see module docstring - flip if live data shows this is backwards
INVERT_PPA_DEFENSE = True

W_SP_OVERALL, W_ELO = 0.6, 0.4
W_SP_OFFENSE, W_PPA_OFFENSE = 0.6, 0.4
W_SP_DEFENSE, W_PPA_DEFENSE = 0.6, 0.4

# Realistic real-world std-dev of FBS team quality, in points-per-game above/
# below average - chosen from publicly known SP+ score spreads (top teams
# ~+25 to +30, bottom teams ~-25 to -30, most of the ~130-team FBS field
# within +/-15), not fit against this project's own real results yet (see
# `refit.py` for how that fit would eventually happen once enough real
# resolved picks exist).
POWER_RATING_POINTS_STD = 13.5
OFFENSE_POINTS_STD = 9.5
DEFENSE_POINTS_STD = 9.5

# Elo's own implied "points above average", using elo.py's public 25-Elo-
# points-per-expected-point conversion, anchored at elo.py's BASE_RATING as
# the league-average team.
_ELO_POINTS_DIVISOR = 25.0


@dataclass(frozen=True)
class TeamRating:
    team: str
    season: int
    conference: Optional[str]
    power_rating: float  # blended overall, points above a league-average FBS team
    offense_rating: float  # blended offense, points above average (higher = better)
    defense_rating: float  # blended defense, points fewer than average allowed (higher = better)
    special_teams_rating: float  # SP+'s own special-teams rating, points above average; 0.0 if unavailable
    elo: float  # this project's own point-in-time Elo (see elo.py)
    sp_overall: Optional[float]  # raw CFBD SP+ overall rating, for transparency/debugging
    sos: Optional[float]  # SP+'s own strength-of-schedule figure, if provided
    games_played: int  # real completed games this team has played this season, as of this rating


class RatingsProvider(ABC):
    @abstractmethod
    def get_ratings(self, season: int) -> Dict[str, TeamRating]:
        raise NotImplementedError


def _zscore(values: Dict[str, float]) -> Dict[str, float]:
    present = {k: v for k, v in values.items() if v is not None}
    if len(present) < 2:
        return {k: 0.0 for k in values}
    mean = statistics.mean(present.values())
    stdev = statistics.pstdev(present.values()) or 1.0
    return {k: ((v - mean) / stdev if v is not None else 0.0) for k, v in values.items()}


class CfbdRatingsProvider(RatingsProvider):
    def __init__(self, client=None, elo_seasons_back: int = 1):
        from .cfbd import CfbdClient

        self.client = client or CfbdClient()
        # How many prior real seasons of games to include when computing
        # Elo - carries real signal forward through the season-transition
        # regression in elo.py, rather than starting every team from a
        # blank BASE_RATING slate on day one of the current season.
        # Defaults to 1 (not more): each extra season costs 2 more real
        # `client.games()` calls per `get_ratings()` run (regular +
        # postseason), against CFBD's free-tier cap of 1,000 calls/month
        # (see collegefootballdata.com/key) - raise it if you're on a paid
        # Patreon tier and want a better-warmed-up rating.
        self.elo_seasons_back = elo_seasons_back

    def _elo_ratings(self, season: int) -> "tuple[Dict[str, float], Dict[str, int]]":
        """Returns `(elo_ratings, games_played)` - `games_played` is a real
        side-effect tally of the CURRENT season's completed regular-season
        games, reusing the exact same `client.games(season, "regular")`
        fetch this method already needs for Elo instead of `get_ratings`
        re-fetching it separately. CFBD's free tier caps real usage at
        1,000 calls/month (see collegefootballdata.com/key) - a weekly-
        cadence sport has no real need to re-fetch the same season's games
        twice in one run just to avoid threading one extra return value.
        """
        games: List[GameResult] = []
        games_played: Dict[str, int] = {}
        for year in range(season - self.elo_seasons_back, season + 1):
            for raw in self.client.games(year, season_type="regular"):
                if not raw.get("completed"):
                    continue
                if year == season:
                    for side in ("homeTeam", "awayTeam"):
                        team = raw.get(side)
                        if team:
                            games_played[team] = games_played.get(team, 0) + 1
                try:
                    games.append(
                        GameResult(
                            season=year,
                            week=raw["week"],
                            home_team=raw["homeTeam"],
                            away_team=raw["awayTeam"],
                            home_points=int(raw["homePoints"]),
                            away_points=int(raw["awayPoints"]),
                            neutral_site=bool(raw.get("neutralSite", False)),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    logger.warning("Skipping unparsable completed game for Elo: %r", raw)
            # Postseason games carry real signal too (a bowl/CFP result is a
            # real game), included the same way.
            for raw in self.client.games(year, season_type="postseason"):
                if not raw.get("completed"):
                    continue
                try:
                    games.append(
                        GameResult(
                            season=year,
                            week=100 + raw.get("week", 0),  # sorts after every real-season week
                            home_team=raw["homeTeam"],
                            away_team=raw["awayTeam"],
                            home_points=int(raw["homePoints"]),
                            away_points=int(raw["awayPoints"]),
                            neutral_site=bool(raw.get("neutralSite", False)),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    logger.warning("Skipping unparsable completed postseason game for Elo: %r", raw)
        return compute_elo_ratings(games), games_played

    def get_ratings(self, season: int) -> Dict[str, TeamRating]:
        sp_rows = self.client.ratings_sp(season)
        ppa_rows = self.client.ppa_teams(season)
        elo, games_played = self._elo_ratings(season)

        sp_overall_raw: Dict[str, float] = {}
        sp_offense_raw: Dict[str, float] = {}
        sp_defense_raw: Dict[str, float] = {}
        sp_special_teams: Dict[str, float] = {}
        sos: Dict[str, float] = {}
        conference: Dict[str, Optional[str]] = {}
        for row in sp_rows:
            try:
                team = row["team"]
            except (KeyError, TypeError):
                logger.warning("Skipping unparsable SP+ row: %r", row)
                continue
            conference[team] = row.get("conference")
            if row.get("rating") is not None:
                sp_overall_raw[team] = float(row["rating"])
            offense = row.get("offense") or {}
            defense = row.get("defense") or {}
            special_teams = row.get("specialTeams") or {}
            if offense.get("rating") is not None:
                sp_offense_raw[team] = float(offense["rating"])
            if defense.get("rating") is not None:
                raw_def = float(defense["rating"])
                sp_defense_raw[team] = -raw_def if INVERT_SP_DEFENSE else raw_def
            if special_teams.get("rating") is not None:
                sp_special_teams[team] = float(special_teams["rating"])
            if row.get("sos") is not None:
                sos[team] = float(row["sos"])

        ppa_offense_raw: Dict[str, float] = {}
        ppa_defense_raw: Dict[str, float] = {}
        for row in ppa_rows:
            try:
                team = row["team"]
            except (KeyError, TypeError):
                logger.warning("Skipping unparsable PPA row: %r", row)
                continue
            offense = row.get("offense") or {}
            defense = row.get("defense") or {}
            if offense.get("overall") is not None:
                ppa_offense_raw[team] = float(offense["overall"])
            if defense.get("overall") is not None:
                raw_def = float(defense["overall"])
                ppa_defense_raw[team] = -raw_def if INVERT_PPA_DEFENSE else raw_def

        all_teams = set(sp_overall_raw) | set(elo) | set(ppa_offense_raw) | set(ppa_defense_raw)
        if not all_teams:
            logger.warning("No SP+/PPA/Elo data resolved for season %s - returning no ratings", season)
            return {}

        # Elo's own "points above average" figure, on the same real anchor
        # (BASE_RATING == the league-average team) every team's Elo is
        # computed relative to.
        elo_points = {t: (elo.get(t, BASE_RATING) - BASE_RATING) / _ELO_POINTS_DIVISOR for t in all_teams}

        sp_overall_z = _zscore({t: sp_overall_raw.get(t) for t in all_teams})
        elo_z = _zscore(elo_points)
        sp_offense_z = _zscore({t: sp_offense_raw.get(t) for t in all_teams})
        ppa_offense_z = _zscore({t: ppa_offense_raw.get(t) for t in all_teams})
        sp_defense_z = _zscore({t: sp_defense_raw.get(t) for t in all_teams})
        ppa_defense_z = _zscore({t: ppa_defense_raw.get(t) for t in all_teams})

        ratings: Dict[str, TeamRating] = {}
        for team in all_teams:
            power_z = W_SP_OVERALL * sp_overall_z[team] + W_ELO * elo_z[team]
            offense_z = W_SP_OFFENSE * sp_offense_z[team] + W_PPA_OFFENSE * ppa_offense_z[team]
            defense_z = W_SP_DEFENSE * sp_defense_z[team] + W_PPA_DEFENSE * ppa_defense_z[team]
            ratings[team] = TeamRating(
                team=team,
                season=season,
                conference=conference.get(team),
                power_rating=round(power_z * POWER_RATING_POINTS_STD, 2),
                offense_rating=round(offense_z * OFFENSE_POINTS_STD, 2),
                defense_rating=round(defense_z * DEFENSE_POINTS_STD, 2),
                special_teams_rating=round(sp_special_teams.get(team, 0.0), 2),
                elo=round(elo.get(team, BASE_RATING), 1),
                sp_overall=sp_overall_raw.get(team),
                sos=sos.get(team),
                games_played=games_played.get(team, 0),
            )
        return ratings


class MockRatingsProvider(RatingsProvider):
    """Synthetic ratings for a fixed set of real program names (see
    `schedule.MockScheduleProvider`) - no network calls, no API key.
    Roughly realistic relative strength so `--mock` output looks sane, not
    a claim about any real team's actual current rating.
    """

    _RATINGS = {
        "Ohio State": (24.0, 14.0, 12.0, 1850.0),
        "Michigan": (14.0, 6.0, 10.0, 1720.0),
        "Georgia": (22.0, 13.0, 11.0, 1830.0),
        "Alabama": (18.0, 12.0, 8.0, 1780.0),
        "Texas": (20.0, 13.0, 9.0, 1800.0),
        "Oklahoma": (9.0, 4.0, 6.0, 1620.0),
        "Oregon": (19.0, 14.0, 7.0, 1790.0),
        "Washington": (8.0, 7.0, 3.0, 1600.0),
        "Boise State": (10.0, 8.0, 4.0, 1630.0),
        "Colorado State": (-2.0, 1.0, -1.0, 1500.0),
        "Air Force": (2.0, -1.0, 5.0, 1540.0),
        "Wyoming": (-3.0, -2.0, 0.0, 1490.0),
    }
    _DEFAULT = (0.0, 0.0, 0.0, 1500.0)

    def get_ratings(self, season: int) -> Dict[str, TeamRating]:
        out = {}
        for team, (power, offense, defense, elo) in self._RATINGS.items():
            out[team] = TeamRating(
                team=team,
                season=season,
                conference=None,
                power_rating=power,
                offense_rating=offense,
                defense_rating=defense,
                special_teams_rating=0.0,
                elo=elo,
                sp_overall=power,
                sos=None,
                games_played=6,
            )
        return out

    def get_team(self, team: str, season: int) -> TeamRating:
        power, offense, defense, elo = self._RATINGS.get(team, self._DEFAULT)
        return TeamRating(
            team=team, season=season, conference=None, power_rating=power, offense_rating=offense,
            defense_rating=defense, special_teams_rating=0.0, elo=elo, sp_overall=power, sos=None, games_played=6,
        )
