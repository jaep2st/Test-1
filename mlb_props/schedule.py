"""Today's MLB slate: games, probable starters, and ballparks.

Real data: the MLB Stats API (`statsapi.mlb.com`), a free, public, no-key
JSON API. `GET /api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=probablePitcher,team,venue`
returns every game for a date with probable pitchers and venue names.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"

# How close to a real game's first pitch it's worth spending an extra real
# network call trying for its confirmed starting lineup - MLB typically
# posts lineups 1-4 hours before first pitch (see schedule.py's/README's
# real research on this), so a run many hours out would just waste a call
# on a game that hasn't posted one yet. Wide enough to comfortably cover
# that real window without narrowing it so tight a run right at the edge
# misses a lineup that happened to post a little early.
LINEUP_FETCH_WINDOW_HOURS = 5.0


def _worth_trying_confirmed_lineup(game_time_utc: Optional[str]) -> bool:
    """True when it's worth spending an extra real network call trying
    for this game's confirmed starting lineup - within
    `LINEUP_FETCH_WINDOW_HOURS` of its real first pitch. Fails open
    (`True`) when `game_time_utc` is missing/unparsable - "try anyway" is
    the safer default here, not silently never trying for a game whose
    real start time this project couldn't read.
    """
    if not game_time_utc:
        return True
    try:
        start = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
    except ValueError:
        return True
    hours_until = (start - datetime.now(timezone.utc)).total_seconds() / 3600.0
    return hours_until <= LINEUP_FETCH_WINDOW_HOURS


@dataclass(frozen=True)
class ProbableMatchup:
    """One game's context: teams, park, and today's probable starters."""

    away_team: str
    home_team: str
    venue: str
    away_pitcher: Optional[str]
    home_pitcher: Optional[str]
    game_time_utc: Optional[str] = None
    # MLB Stats API's own real-time status for this specific game
    # ("Pre-Game", "In Progress", "Final", etc. - the same field
    # mlb_props_main.py's --game-status-check surfaces). A doubleheader
    # produces two separate `games` entries for the same two teams here,
    # each with its own status - unlike a third-party odds provider, which
    # confirmed live (2026-08-29) can expose only one event per matchup
    # with no way to tell which specific game a price is for. See
    # pipeline.py's cross-check against this field for why it matters.
    status: Optional[str] = None
    # MLB Stats API only posts the actual starting lineup shortly before
    # first pitch, so well ahead of game time these are populated from each
    # team's active roster instead (all non-pitchers currently on the
    # 26-man active roster) - a reasonable stand-in for "who might play"
    # that's available any time of day, at the cost of including bench
    # players alongside the starters. Callers can still override with their
    # own list via cli --batters.
    away_batters: List[str] = field(default_factory=list)
    home_batters: List[str] = field(default_factory=list)
    # "confirmed" once away_batters/home_batters above are MLB's real,
    # posted starting lineup for this game (see
    # MlbStatsApiScheduleProvider._confirmed_lineup_batters) - "active_roster"
    # (the honest default/fallback) whenever they're still the active-roster
    # proxy above, because the real lineup hasn't posted yet or the fetch
    # failed. Threaded through scoring so a real reader can see, per pick,
    # whether it's scored against MLB's actual starters or a same-day guess -
    # see mlb_props/edges.py's EdgeCandidate.lineup_source.
    lineup_source: str = "active_roster"
    # MLB Stats API's own real per-game ID - unused by the live pipeline
    # (which never needs to look a specific game back up), but the one
    # stable key into other real per-game MLB Stats API endpoints (e.g.
    # the boxscore endpoint mlb_props/historical_backtest.py uses to find
    # who actually batted in a specific past game - a schedule/roster
    # lookup alone can't answer that for a date that isn't today, since
    # away_batters/home_batters above are today's active roster, not a
    # historical one). `None` for MockScheduleProvider's synthetic slate.
    game_pk: Optional[int] = None


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
        from odds_monitor.http_utils import build_retrying_session

        # Retries transient connection failures (see that module's
        # docstring - confirmed live against The Odds API) instead of
        # dropping the whole slate on one dropped connection. Only applied
        # when no session is injected, so tests supplying a fake session
        # are unaffected.
        self.session = session or build_retrying_session()
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
                    game_time_utc = game.get("gameDate")
                    game_pk = game.get("gamePk")

                    away_batters = (
                        self._active_position_players(away_team_id)[: self.max_batters_per_team]
                        if self.include_rosters
                        else []
                    )
                    home_batters = (
                        self._active_position_players(home_team_id)[: self.max_batters_per_team]
                        if self.include_rosters
                        else []
                    )
                    lineup_source = "active_roster"
                    # Real confirmed lineup, when MLB has posted one and
                    # it's close enough to first pitch to be worth trying
                    # (see LINEUP_FETCH_WINDOW_HOURS) - replaces the
                    # active-roster proxy above for this one game only,
                    # never silently swapped in when unavailable.
                    if self.include_rosters and game_pk and _worth_trying_confirmed_lineup(game_time_utc):
                        confirmed = self._confirmed_lineup_batters(game_pk)
                        if confirmed is not None:
                            away_batters, home_batters = confirmed
                            lineup_source = "confirmed"

                    matchups.append(
                        ProbableMatchup(
                            away_team=away,
                            home_team=home,
                            venue=venue,
                            away_pitcher=away_pitcher,
                            home_pitcher=home_pitcher,
                            game_time_utc=game_time_utc,
                            status=game.get("status", {}).get("detailedState"),
                            away_batters=away_batters,
                            home_batters=home_batters,
                            lineup_source=lineup_source,
                            game_pk=game_pk,
                        )
                    )
                except (KeyError, TypeError):
                    logger.warning("Skipping unparsable schedule entry: %r", game)
        return matchups

    def _confirmed_lineup_batters(self, game_pk: int) -> Optional[Tuple[List[str], List[str]]]:
        """Real starting-lineup batters for one specific game, split
        (away, home), once MLB has posted them. Reuses the same real
        `/game/{game_pk}/boxscore` endpoint
        `historical_backtest.fetch_boxscore_batters` already fetches
        successfully post-game (same `teams.{side}.players` traversal),
        but reads a different real field: `battingOrder` - present on a
        player's boxscore entry once that player is in today's actual
        starting lineup, typically 1-4 real hours before first pitch,
        well before any `plateAppearances` exist (which is what that
        other function reads instead, for a completed game). NOT YET
        CONFIRMED against a real response from this sandbox (no network
        access here) - same disclosed-guess posture as every other new
        field this project adds; verify with `--lineup-diagnostic` before
        trusting it fully. Parsing is defensive throughout - an
        unparsable entry is skipped with a logged warning, never crashes
        the whole fetch.

        Returns `None` (not `([], [])`) whenever a real confirmed lineup
        can't be determined - the fetch failed, or MLB hasn't posted a
        real lineup for this game yet (every player entry lacks a real
        `battingOrder`). Callers should fall back to the active-roster
        proxy in that case, never treat `None` as "empty lineup."
        """
        try:
            resp = self.session.get(f"{MLB_STATS_API_BASE}/game/{game_pk}/boxscore", timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            logger.info("No confirmed lineup available yet for game_pk=%s (boxscore fetch failed)", game_pk)
            return None

        # Unconditional real-shape logging (INFO), same convention as
        # BetstampProvider's own payload logging - what --lineup-diagnostic
        # actually reads to confirm/correct the `battingOrder` guess above,
        # not this function's return value.
        away_players = payload.get("teams", {}).get("away", {}).get("players", {})
        sample_id, sample_entry = next(iter(away_players.items()), (None, None)) if isinstance(away_players, dict) else (None, None)
        logger.info(
            "LINEUP_DIAGNOSTIC game_pk=%s boxscore top-level keys=%s away.players sample (id=%s)=%r",
            game_pk,
            sorted(payload.keys()) if isinstance(payload, dict) else type(payload),
            sample_id,
            sample_entry,
        )

        by_side: Dict[str, List[str]] = {"away": [], "home": []}
        for side in ("away", "home"):
            players = payload.get("teams", {}).get(side, {}).get("players", {})
            if not isinstance(players, dict):
                continue
            for _player_id, entry in players.items():
                try:
                    if not entry.get("battingOrder"):
                        continue
                    name = entry.get("person", {}).get("fullName")
                    if name:
                        by_side[side].append(name)
                except (KeyError, TypeError, AttributeError):
                    logger.warning("Skipping unparsable boxscore player entry for game_pk=%s: %r", game_pk, entry)

        if not by_side["away"] and not by_side["home"]:
            return None
        return by_side["away"], by_side["home"]

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
