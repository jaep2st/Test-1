"""Today's MLB slate: games, probable starters, and ballparks.

Real data: the MLB Stats API (`statsapi.mlb.com`), a free, public, no-key
JSON API. `GET /api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=probablePitcher,team,venue`
returns every game for a date with probable pitchers and venue names.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

logger = logging.getLogger(__name__)

MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"


@dataclass(frozen=True)
class ProbableMatchup:
    """One game's context: teams, park, and today's probable starters."""

    away_team: str
    home_team: str
    venue: str
    away_pitcher: Optional[str]
    home_pitcher: Optional[str]
    game_time_utc: Optional[str] = None
    # MLB Stats API only posts the actual starting lineup shortly before
    # first pitch, so well ahead of game time these are populated from each
    # team's active roster instead (all non-pitchers currently on the
    # 26-man active roster) - a reasonable stand-in for "who might play"
    # that's available any time of day, at the cost of including bench
    # players alongside the starters. Callers can still override with their
    # own list via cli --batters.
    away_batters: List[str] = field(default_factory=list)
    home_batters: List[str] = field(default_factory=list)


class ScheduleProvider(ABC):
    @abstractmethod
    def get_slate(self, game_date: date) -> List[ProbableMatchup]:
        raise NotImplementedError


class MlbStatsApiScheduleProvider(ScheduleProvider):
    """Requires network access to `statsapi.mlb.com` (free, no API key).
    Not exercised live in this build environment - verify the response
    shape with `--log-level DEBUG` before relying on it.
    """

    def __init__(self, session=None, timeout: float = 10.0, include_rosters: bool = True, max_batters_per_team: int = 9):
        import requests

        self.session = session or requests.Session()
        self.timeout = timeout
        self.include_rosters = include_rosters
        # The roster endpoint doesn't indicate batting order or who's
        # actually starting today, so this is a size cap for speed (each
        # batter costs several downstream API calls across the pipeline),
        # not a "starters" filter - 9 approximates a lineup's worth without
        # scoring an entire ~13-man position-player bench every run.
        self.max_batters_per_team = max_batters_per_team
        self._roster_cache: dict = {}

    def get_slate(self, game_date: date) -> List[ProbableMatchup]:
        try:
            resp = self.session.get(
                f"{MLB_STATS_API_BASE}/schedule",
                params={
                    "sportId": 1,
                    "date": game_date.isoformat(),
                    "hydrate": "probablePitcher,team,venue",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            logger.exception("MLB Stats API schedule fetch failed for %s", game_date)
            return []

        matchups: List[ProbableMatchup] = []
        for date_block in payload.get("dates", []):
            for game in date_block.get("games", []):
                try:
                    teams = game["teams"]
                    away_team_id = teams["away"]["team"]["id"]
                    home_team_id = teams["home"]["team"]["id"]
                    away = teams["away"]["team"]["name"]
                    home = teams["home"]["team"]["name"]
                    venue = game.get("venue", {}).get("name", "")
                    away_pitcher = teams["away"].get("probablePitcher", {}).get("fullName")
                    home_pitcher = teams["home"].get("probablePitcher", {}).get("fullName")
                    matchups.append(
                        ProbableMatchup(
                            away_team=away,
                            home_team=home,
                            venue=venue,
                            away_pitcher=away_pitcher,
                            home_pitcher=home_pitcher,
                            game_time_utc=game.get("gameDate"),
                            away_batters=self._active_position_players(away_team_id)[: self.max_batters_per_team]
                            if self.include_rosters
                            else [],
                            home_batters=self._active_position_players(home_team_id)[: self.max_batters_per_team]
                            if self.include_rosters
                            else [],
                        )
                    )
                except (KeyError, TypeError):
                    logger.warning("Skipping unparsable schedule entry: %r", game)
        return matchups

    def _active_position_players(self, team_id: int) -> List[str]:
        """All non-pitchers on a team's active (26-man) roster right now."""
        if team_id in self._roster_cache:
            return self._roster_cache[team_id]
        try:
            resp = self.session.get(
                f"{MLB_STATS_API_BASE}/teams/{team_id}/roster",
                params={"rosterType": "active"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            entries = resp.json().get("roster", [])
        except Exception:
            logger.exception("MLB Stats API roster fetch failed for team_id=%s", team_id)
            self._roster_cache[team_id] = []
            return []

        players = []
        for entry in entries:
            try:
                position_type = entry.get("position", {}).get("type", "")
                if position_type == "Pitcher":
                    continue
                players.append(entry["person"]["fullName"])
            except (KeyError, TypeError):
                logger.warning("Skipping unparsable roster entry: %r", entry)
        self._roster_cache[team_id] = players
        return players


class MockScheduleProvider(ScheduleProvider):
    """A small synthetic slate - no network calls. Uses real team/park
    names for realism but is not today's actual schedule.
    """

    _SAMPLE_SLATE = [
        ProbableMatchup(
            away_team="New York Yankees",
            home_team="Baltimore Orioles",
            venue="Camden Yards",
            away_pitcher="Kevin Gausman",
            home_pitcher="Grayson Rodriguez",
            away_batters=["Aaron Judge", "Juan Soto", "Giancarlo Stanton"],
            home_batters=["Gunnar Henderson", "Adley Rutschman", "Anthony Santander"],
        ),
        ProbableMatchup(
            away_team="Cincinnati Reds",
            home_team="Colorado Rockies",
            venue="Coors Field",
            away_pitcher="Hunter Greene",
            home_pitcher="Kyle Freeland",
            away_batters=["Elly De La Cruz", "Spencer Steer"],
            home_batters=["Ryan McMahon", "Ezequiel Tovar"],
        ),
        ProbableMatchup(
            away_team="Los Angeles Dodgers",
            home_team="San Francisco Giants",
            venue="Oracle Park",
            away_pitcher="Yoshinobu Yamamoto",
            home_pitcher="Logan Webb",
            away_batters=["Shohei Ohtani", "Mookie Betts", "Freddie Freeman"],
            home_batters=["Matt Chapman", "Jung Hoo Lee"],
        ),
        ProbableMatchup(
            away_team="Atlanta Braves",
            home_team="Philadelphia Phillies",
            venue="Citizens Bank Park",
            away_pitcher="Spencer Strider",
            home_pitcher="Zack Wheeler",
            away_batters=["Ronald Acuna Jr.", "Matt Olson", "Austin Riley"],
            home_batters=["Kyle Schwarber", "Bryce Harper", "Trea Turner"],
        ),
    ]

    def get_slate(self, game_date: date) -> List[ProbableMatchup]:
        return list(self._SAMPLE_SLATE)
