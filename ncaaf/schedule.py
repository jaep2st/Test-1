"""This week's (or any real week's) NCAA FBS schedule: matchups, venues,
neutral-site flags, and - once final - real final scores.

Real data: the College Football Data API (see `cfbd.py`), specifically
`GET /games`. `current_season_week` uses CFBD's own `/calendar` endpoint to
answer "what real week is it right now" instead of a hardcoded guess about
when the season starts - the season's real start date (and postseason
length) shifts a little every year.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Matchup:
    """One real (or, for `MockScheduleProvider`, synthetic) FBS game."""

    season: int
    week: int
    season_type: str  # "regular" or "postseason"
    away_team: str
    home_team: str
    away_conference: Optional[str]
    home_conference: Optional[str]
    venue: str
    venue_id: Optional[int]
    neutral_site: bool
    conference_game: bool
    start_date_utc: Optional[str]  # ISO 8601, CFBD's own `startDate`
    completed: bool
    home_points: Optional[int]
    away_points: Optional[int]
    cfbd_game_id: Optional[int] = None

    @property
    def event(self) -> str:
        return f"{self.away_team} @ {self.home_team}"


class ScheduleProvider(ABC):
    @abstractmethod
    def get_week_slate(self, season: int, week: int, season_type: str = "regular") -> List[Matchup]:
        raise NotImplementedError


def _parse_game(raw: dict, season: int, week: int, season_type: str) -> Optional[Matchup]:
    try:
        away = raw.get("awayTeam") or raw.get("away_team")
        home = raw.get("homeTeam") or raw.get("home_team")
        if not away or not home:
            return None
        return Matchup(
            season=season,
            week=raw.get("week", week),
            season_type=raw.get("seasonType", season_type),
            away_team=away,
            home_team=home,
            away_conference=raw.get("awayConference"),
            home_conference=raw.get("homeConference"),
            venue=raw.get("venue") or "",
            venue_id=raw.get("venueId"),
            neutral_site=bool(raw.get("neutralSite", False)),
            conference_game=bool(raw.get("conferenceGame", False)),
            start_date_utc=raw.get("startDate"),
            completed=bool(raw.get("completed", False)),
            home_points=raw.get("homePoints"),
            away_points=raw.get("awayPoints"),
            cfbd_game_id=raw.get("id"),
        )
    except (KeyError, TypeError, AttributeError):
        logger.warning("Skipping unparsable CFBD game entry: %r", raw)
        return None


class CfbdScheduleProvider(ScheduleProvider):
    def __init__(self, client=None):
        from .cfbd import CfbdClient

        self.client = client or CfbdClient()

    def get_week_slate(self, season: int, week: int, season_type: str = "regular") -> List[Matchup]:
        raw_games = self.client.games(season, week=week, season_type=season_type)
        matchups = [m for m in (_parse_game(g, season, week, season_type) for g in raw_games) if m is not None]
        # FCS/non-FBS opponents (common in early-season "buy games") have no
        # real power rating to score against - keep them in the slate (a
        # real game a reader might look this report up for) but callers
        # scoring matchups should expect a missing rating for that side and
        # degrade to "no pick" rather than a guessed number, same "unknown
        # stays unknown" posture as the rest of this project.
        return matchups


class MockScheduleProvider(ScheduleProvider):
    """A small synthetic slate - no network calls, no API key. Uses real
    program names for realism but is not any real week's actual schedule.
    """

    _SAMPLE_SLATE = [
        dict(away_team="Michigan", home_team="Ohio State", away_conference="Big Ten", home_conference="Big Ten",
             venue="Ohio Stadium", venue_id=3891, neutral_site=False, conference_game=True),
        dict(away_team="Alabama", home_team="Georgia", away_conference="SEC", home_conference="SEC",
             venue="Sanford Stadium", venue_id=3729, neutral_site=False, conference_game=True),
        dict(away_team="Texas", home_team="Oklahoma", away_conference="SEC", home_conference="SEC",
             venue="Cotton Bowl", venue_id=3745, neutral_site=True, conference_game=True),
        dict(away_team="Oregon", home_team="Washington", away_conference="Big Ten", home_conference="Big Ten",
             venue="Husky Stadium", venue_id=3901, neutral_site=False, conference_game=True),
        dict(away_team="Boise State", home_team="Colorado State", away_conference="Mountain West",
             home_conference="Mountain West", venue="Canvas Stadium", venue_id=4213, neutral_site=False,
             conference_game=True),
        dict(away_team="Air Force", home_team="Wyoming", away_conference="Mountain West",
             home_conference="Mountain West", venue="War Memorial Stadium", venue_id=3801, neutral_site=False,
             conference_game=True),
    ]

    def get_week_slate(self, season: int, week: int, season_type: str = "regular") -> List[Matchup]:
        return [
            Matchup(
                season=season,
                week=week,
                season_type=season_type,
                away_team=g["away_team"],
                home_team=g["home_team"],
                away_conference=g["away_conference"],
                home_conference=g["home_conference"],
                venue=g["venue"],
                venue_id=g["venue_id"],
                neutral_site=g["neutral_site"],
                conference_game=g["conference_game"],
                start_date_utc=None,
                completed=False,
                home_points=None,
                away_points=None,
            )
            for g in self._SAMPLE_SLATE
        ]


def current_season_week(client, today: Optional[date] = None) -> Tuple[int, int, str]:
    """Real `(season, week, season_type)` for `today` (default: today, UTC),
    resolved from CFBD's own `/calendar` endpoint rather than a hardcoded
    guess about the season's start date. Falls back to a rough date-based
    heuristic (regular season roughly runs late August - early December,
    postseason through mid-January) only if the calendar fetch itself fails,
    clearly logged as a fallback rather than silently passed off as real
    calendar data.
    """
    today = today or datetime.now(timezone.utc).date()
    try:
        weeks = client.calendar(today.year if today.month >= 6 else today.year - 1)
        for entry in weeks:
            try:
                start = datetime.fromisoformat(entry["firstGameStart"].replace("Z", "+00:00")).date()
                end = datetime.fromisoformat(entry["lastGameStart"].replace("Z", "+00:00")).date()
            except (KeyError, ValueError, TypeError):
                continue
            if start <= today <= end:
                return int(entry.get("season", today.year)), int(entry["week"]), entry.get("seasonType", "regular")
        # No exact week match (e.g. mid-week between calendar entries, or a
        # true off-season date) - pick the nearest upcoming week if any, so
        # "run this a few days before kickoff" still resolves sensibly.
        upcoming = sorted(
            (e for e in weeks if e.get("firstGameStart")),
            key=lambda e: e["firstGameStart"],
        )
        for entry in upcoming:
            start = datetime.fromisoformat(entry["firstGameStart"].replace("Z", "+00:00")).date()
            if start >= today:
                return int(entry.get("season", today.year)), int(entry["week"]), entry.get("seasonType", "regular")
    except Exception:
        logger.exception("CFBD calendar fetch failed - falling back to a date-based week estimate")

    # Rough fallback only: real season structure shifts year to year, so
    # this is deliberately approximate and only used if the real calendar
    # is unreachable.
    year = today.year
    if today.month < 6:
        year -= 1
    if today.month in (8, 9, 10, 11, 12) or (today.month == 1 and today.day <= 20):
        approx_start = date(year, 8, 24)
        week = max(1, min(15, (today - approx_start).days // 7 + 1)) if today >= approx_start else 1
        return year, week, "regular"
    return year, 1, "regular"
