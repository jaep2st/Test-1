"""Real cross-book NCAA football game-line odds (spreads, moneylines,
totals) from The Odds API (https://the-odds-api.com), plus a mock provider
so the whole pipeline is demoable with `--mock` and no API key.

Unlike `mlb_props`' player-prop odds (one `events` list call + one
per-event `odds` call, since props require the per-event odds endpoint),
game-line markets are available from The Odds API's bulk
`GET /sports/{sport}/odds` endpoint in a SINGLE real request for the whole
slate - `regions x markets` credits total, not per game. See
https://the-odds-api.com/sports-odds-data/betting-markets.html for the
real, documented market keys used below.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from odds_monitor.http_utils import build_retrying_session

# Re-exported here so callers only need to import from ncaaf.market for
# sportsbook display names - the underlying table is real, sport-agnostic
# book-key data (see mlb_props/market.py's BOOK_DISPLAY_NAMES docstring for
# how it was built/confirmed).
from mlb_props.market import BOOK_DISPLAY_NAMES, book_display_name  # noqa: F401

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_ncaaf"

MARKET_H2H = "h2h"
MARKET_SPREADS = "spreads"
MARKET_TOTALS = "totals"
ALL_MARKETS = (MARKET_H2H, MARKET_SPREADS, MARKET_TOTALS)


@dataclass(frozen=True)
class GameLine:
    """One sportsbook's price for one side of one game-line market.

    `side` is "home"/"away" for `h2h`/`spreads`, or "over"/"under" for
    `totals`. `point` is the spread number (signed from `side`'s own
    perspective, e.g. a home favorite's spread line is negative) or the
    total number; always `None` for `h2h` (a moneyline has no line).
    """

    event: str  # "Away @ Home"
    home_team: str
    away_team: str
    commence_time: Optional[str]
    market: str
    side: str
    team: Optional[str]  # the actual team name this side refers to (h2h/spreads only)
    point: Optional[float]
    odds: int
    sportsbook: str
    is_live: bool = False
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def key(self) -> tuple:
        return (self.event.strip().lower(), self.market, self.side, round(self.point, 2) if self.point is not None else None)


class GameOddsProvider:
    def fetch_game_odds(self, include_live: bool = False) -> List[GameLine]:
        raise NotImplementedError


class NoGameOddsProvider(GameOddsProvider):
    """No odds API key configured - the rest of the pipeline (ratings,
    context, scoring) still runs and produces model-only predictions
    instead of hard-failing.
    """

    def fetch_game_odds(self, include_live: bool = False) -> List[GameLine]:
        return []


class OddsFetchFailed(Exception):
    """A systemic failure (auth/quota, an outage), not a legitimately
    empty slate (a bye week) - see theoddsapi.py's identically-named
    exception for the same distinction on the player-props side.
    """


class TheOddsApiGameOddsProvider(GameOddsProvider):
    """Requires an API key: pass `api_key=` or set `ODDS_API_KEY`. Get one
    free at https://the-odds-api.com.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        regions: str = "us,us2",
        books: Optional[List[str]] = None,
        timeout: float = 15.0,
        session: Optional[requests.Session] = None,
    ):
        raw_key = api_key or os.environ.get("ODDS_API_KEY")
        self.api_key = raw_key.strip() if raw_key else raw_key
        if not self.api_key:
            raise ValueError(
                "An Odds API key is required. Pass api_key=... or set the ODDS_API_KEY "
                "environment variable. Get one free at https://the-odds-api.com"
            )
        self.base_url = base_url.rstrip("/")
        self.regions = regions
        self.books = set(b.lower() for b in books) if books else None
        self.timeout = timeout
        self.session = session or build_retrying_session()

    def _get(self, path: str, params: Dict[str, Any]) -> Any:
        full_params = dict(params)
        full_params["apiKey"] = self.api_key
        response = self.session.get(f"{self.base_url}{path}", params=full_params, timeout=self.timeout)
        remaining = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")
        if remaining is not None or used is not None:
            logger.info("The Odds API quota after %s: used=%s remaining=%s", path, used, remaining)
        response.raise_for_status()
        return response.json()

    def fetch_game_odds(self, include_live: bool = False) -> List[GameLine]:
        try:
            events = self._get(
                f"/sports/{SPORT_KEY}/odds",
                {"regions": self.regions, "markets": ",".join(ALL_MARKETS), "oddsFormat": "american"},
            )
        except Exception as exc:
            logger.exception("Failed to fetch NCAAF game odds from The Odds API")
            raise OddsFetchFailed(f"Failed to fetch NCAAF game odds: {exc}") from exc

        lines: List[GameLine] = []
        now = datetime.now(timezone.utc)
        skipped_live = 0
        for event in events or []:
            home_team = event.get("home_team")
            away_team = event.get("away_team")
            commence_time = event.get("commence_time")
            if not home_team or not away_team:
                continue
            event_label = f"{away_team} @ {home_team}"
            is_live = False
            if commence_time:
                try:
                    starts_at = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                    is_live = starts_at <= now
                except ValueError:
                    logger.warning("Could not parse commence_time %r for %s", commence_time, event_label)
            if is_live and not include_live:
                skipped_live += 1
                continue
            for bookmaker in event.get("bookmakers", []) or []:
                book_key = str(bookmaker.get("key", "")).lower()
                if self.books and book_key not in self.books:
                    continue
                for market in bookmaker.get("markets", []) or []:
                    market_key = market.get("key")
                    if market_key not in ALL_MARKETS:
                        continue
                    for outcome in market.get("outcomes", []) or []:
                        try:
                            lines.append(
                                self._parse_outcome(outcome, market_key, home_team, away_team, event_label, commence_time, book_key, is_live)
                            )
                        except (KeyError, TypeError, ValueError) as exc:
                            logger.warning("Skipping unparsable outcome on %s/%s/%s: %s (%r)", event_label, book_key, market_key, exc, outcome)
        if skipped_live:
            logger.info("Skipped odds for %d already-started NCAAF event(s) this run", skipped_live)
        return lines

    @staticmethod
    def _parse_outcome(
        outcome: Dict[str, Any],
        market_key: str,
        home_team: str,
        away_team: str,
        event_label: str,
        commence_time: Optional[str],
        book_key: str,
        is_live: bool,
    ) -> GameLine:
        name = str(outcome.get("name", "")).strip()
        price = outcome.get("price")
        if price is None:
            raise KeyError("outcome has no 'price'")
        if market_key == MARKET_TOTALS:
            side = name.lower()
            if side not in ("over", "under"):
                raise ValueError(f"unrecognized totals outcome name {name!r}")
            team = None
            point = outcome.get("point")
            if point is None:
                raise KeyError("totals outcome has no 'point'")
        else:
            if name == home_team:
                side = "home"
            elif name == away_team:
                side = "away"
            else:
                raise ValueError(f"outcome name {name!r} matches neither home ({home_team!r}) nor away ({away_team!r})")
            team = name
            point = outcome.get("point") if market_key == MARKET_SPREADS else None
            if market_key == MARKET_SPREADS and point is None:
                raise KeyError("spreads outcome has no 'point'")
        return GameLine(
            event=event_label,
            home_team=home_team,
            away_team=away_team,
            commence_time=commence_time,
            market=market_key,
            side=side,
            team=team,
            point=float(point) if point is not None else None,
            odds=int(price),
            sportsbook=book_key,
            is_live=is_live,
        )


_MOCK_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "espnbet", "fanatics"]


class MockGameOddsProvider(GameOddsProvider):
    """Synthetic spread/moneyline/total odds for a given slate - no network
    calls, no API key. Each game gets a randomized "true" spread/total, then
    a realistic vig-included American-odds pair per book, with an
    occasional deliberate outlier book so the line-shopping/+EV detector
    has something real to find.
    """

    def __init__(self, matchups, seed: Optional[int] = None, outlier_chance: float = 0.3):
        import random

        self._rng = random.Random(seed)
        self.matchups = list(matchups)
        self.outlier_chance = outlier_chance

    @staticmethod
    def _prob_to_american(prob: float) -> int:
        prob = min(0.95, max(0.05, prob))
        if prob >= 0.5:
            return -round(prob / (1 - prob) * 100)
        return round((1 - prob) / prob * 100)

    def _price_pair(self, true_prob: float) -> "tuple[int, int]":
        vig = self._rng.uniform(0.03, 0.06)
        a = min(0.95, max(0.05, true_prob + vig / 2))
        b = min(0.95, max(0.05, (1 - true_prob) + vig / 2))
        return self._prob_to_american(a), self._prob_to_american(b)

    def fetch_game_odds(self, include_live: bool = False) -> List[GameLine]:
        lines: List[GameLine] = []
        for m in self.matchups:
            event = m.event
            true_spread = round(self._rng.uniform(-21, 21) * 2) / 2.0  # home perspective, half-point increments
            true_total = round(self._rng.uniform(42, 66) * 2) / 2.0
            home_win_prob = min(0.95, max(0.05, 0.5 + true_spread / 40.0))

            books = self._rng.sample(_MOCK_BOOKS, k=self._rng.randint(4, len(_MOCK_BOOKS)))
            outlier_book = self._rng.choice(books) if self._rng.random() < self.outlier_chance else None

            for book in books:
                spread_shift = self._rng.uniform(-0.05, 0.05) if book != outlier_book else self._rng.uniform(0.05, 0.10)
                home_odds, away_odds = self._price_pair(min(0.95, max(0.05, home_win_prob + spread_shift)))
                lines.append(GameLine(event, m.home_team, m.away_team, None, MARKET_SPREADS, "home", m.home_team, -true_spread, home_odds, book))
                lines.append(GameLine(event, m.home_team, m.away_team, None, MARKET_SPREADS, "away", m.away_team, true_spread, away_odds, book))

                ml_shift = self._rng.uniform(-0.03, 0.03) if book != outlier_book else self._rng.uniform(0.04, 0.08)
                home_ml, away_ml = self._price_pair(min(0.95, max(0.05, home_win_prob + ml_shift)))
                lines.append(GameLine(event, m.home_team, m.away_team, None, MARKET_H2H, "home", m.home_team, None, home_ml, book))
                lines.append(GameLine(event, m.home_team, m.away_team, None, MARKET_H2H, "away", m.away_team, None, away_ml, book))

                over_shift = self._rng.uniform(-0.04, 0.04) if book != outlier_book else self._rng.uniform(0.05, 0.09)
                over_odds, under_odds = self._price_pair(min(0.95, max(0.05, 0.5 + over_shift)))
                lines.append(GameLine(event, m.home_team, m.away_team, None, MARKET_TOTALS, "over", None, true_total, over_odds, book))
                lines.append(GameLine(event, m.home_team, m.away_team, None, MARKET_TOTALS, "under", None, true_total, under_odds, book))
        return lines
