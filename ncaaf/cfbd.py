"""Thin client for the College Football Data API (collegefootballdata.com) -
this project's primary real data source for schedules, opponent-adjusted
team ratings (SP+), advanced efficiency stats (PPA), venues, and - notably -
real historical betting lines going back many seasons, which is what makes a
genuine backtest against real closing numbers possible (see
`ncaaf/historical_backtest.py`).

Free tier: a personal API key from https://collegefootballdata.com/key
(email signup, key emailed instantly, no card). Authenticated with a Bearer
token on every request - this project never scrapes the site itself.

This is deliberately a thin, generic wrapper (one `_get` plus one method per
real endpoint actually used) rather than a full API client library - every
method returns the raw parsed JSON list CFBD documents for that endpoint;
callers (`schedule.py`, `ratings.py`, `market.py`/`historical_backtest.py`)
own their own defensive parsing of the specific fields they need, same
"skip and log, never crash the whole fetch" convention as every other real
provider in this project (see `odds_monitor/providers/theoddsapi.py`).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from odds_monitor.http_utils import build_retrying_session

logger = logging.getLogger(__name__)

CFBD_API_BASE = "https://api.collegefootballdata.com"


class CfbdRequestFailed(Exception):
    """Raised when a real CFBD request fails outright (network error, bad
    key, rate limit) - as opposed to a request that succeeds but returns an
    empty list (a bye week, an off-season date range), which is not an
    error and is returned as `[]` like any other empty real result.
    """


class CfbdClient:
    """Requires an API key: pass `api_key=` or set `CFBD_API_KEY`. Get one
    free at https://collegefootballdata.com/key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = CFBD_API_BASE,
        session: Optional[requests.Session] = None,
        timeout: float = 20.0,
    ):
        # .strip() guards the same real footgun documented on
        # TheOddsApiProvider/BetstampProvider: a secret pasted with stray
        # surrounding whitespace silently corrupts the Authorization header.
        raw_key = api_key or os.environ.get("CFBD_API_KEY")
        self.api_key = raw_key.strip() if raw_key else raw_key
        if not self.api_key:
            raise ValueError(
                "A College Football Data API key is required. Pass api_key=... or set the "
                "CFBD_API_KEY environment variable. Get one free at https://collegefootballdata.com/key"
            )
        self.base_url = base_url.rstrip("/")
        self.session = session or build_retrying_session()
        self.timeout = timeout

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        try:
            response = self.session.get(f"{self.base_url}{path}", params=params or {}, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.exception("CFBD request failed: %s %r", path, params)
            raise CfbdRequestFailed(f"CFBD request failed: {path} {params}: {exc}") from exc

    def games(
        self, year: int, week: Optional[int] = None, season_type: str = "regular", team: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """GET /games - one entry per game, with scores once final
        (`completed`, `homePoints`, `awayPoints`) and venue/neutral-site
        info. `week=None` returns the whole season.
        """
        params: Dict[str, Any] = {"year": year, "seasonType": season_type}
        if week is not None:
            params["week"] = week
        if team:
            params["team"] = team
        return self._get("/games", params) or []

    def calendar(self, year: int) -> List[Dict[str, Any]]:
        """GET /calendar - each real week's date range for `year`, the
        authoritative source for "what week is it right now" (see
        `schedule.current_season_week`) instead of a hardcoded guess about
        when the season starts.
        """
        return self._get("/calendar", {"year": year}) or []

    def ratings_sp(self, year: int) -> List[Dict[str, Any]]:
        """GET /ratings/sp - Bill Connelly's SP+ opponent-adjusted rating
        per team for `year`: overall + offense/defense/special-teams
        sub-ratings, each expressed in points relative to a national
        average. Season-level (not per-week) - see
        `ratings.py`/`historical_backtest.py` docstrings for the lookahead
        caveat this implies for in-season/backtested use.
        """
        return self._get("/ratings/sp", {"year": year}) or []

    def ppa_teams(self, year: int, season_type: str = "regular") -> List[Dict[str, Any]]:
        """GET /ppa/teams - season-cumulative predicted-points-added (PPA)
        per team, offense and defense, overall and by down/pass-rush split.
        A second, independent efficiency signal alongside SP+ (different
        methodology, same underlying play-by-play data).
        """
        return self._get("/ppa/teams", {"year": year, "seasonType": season_type}) or []

    def lines(
        self, year: int, week: Optional[int] = None, season_type: str = "regular", team: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """GET /lines - real historical betting lines (spread, over/under,
        moneylines) aggregated per game from several real sportsbooks,
        going back many seasons. This is what makes a genuine backtest
        against real closing numbers possible - see
        `ncaaf/historical_backtest.py`. Live/current-week lines should
        still come from The Odds API (`ncaaf/market.py`) for real-time
        cross-book prices; this endpoint is for history.
        """
        params: Dict[str, Any] = {"year": year, "seasonType": season_type}
        if week is not None:
            params["week"] = week
        if team:
            params["team"] = team
        return self._get("/lines", params) or []

    def venues(self) -> List[Dict[str, Any]]:
        """GET /venues - every real venue CFBD knows: name, city/state,
        lat/lon, elevation (feet), and whether it's a dome. Used for home-
        field altitude adjustments, travel-distance estimates, and to skip
        weather effects at domes - see `context.py`.
        """
        return self._get("/venues") or []

    def teams(self, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """GET /teams/fbs - every real FBS team for `year` (defaults to the
        most recent season CFBD has), with conference affiliation.
        """
        params: Dict[str, Any] = {}
        if year:
            params["year"] = year
        return self._get("/teams/fbs", params) or []
