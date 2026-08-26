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
    # Best-effort expected lineup regulars; MLB Stats API only posts actual
    # lineups shortly before first pitch, so this is often empty well ahead
    # of game time - callers should let the user supply their own roster in
    # that case (see cli --batters).
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

    def __init__(self, session=None, timeout: float = 10.0):
        import requests

        self.session = session or requests.Session()
        self.timeout = timeout

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
                        )
                    )
                except (KeyError, TypeError):
                    logger.warning("Skipping unparsable schedule entry: %r", game)
        return matchups


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
