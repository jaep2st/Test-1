"""Ballpark Pal API integration: real, per-hitter, per-game park + weather
factors from a domain-specific baseball model - a direct upgrade over this
project's own `context.py`, which uses a static 3-year-rolling park factor
table plus a rough wind/temp heuristic, both applied at the *game* level
(shared across every batter in that game) rather than per-hitter.

API docs were read manually via screenshots of the account holder's own
https://www.ballparkpal.com/api page (2026-08-30) - this host is blocked
from this dev/agent sandbox, same as espn.com/statsapi.mlb.com/the-odds-
api.com/etc. (confirmed live), so nothing here was fetched programmatically
during development. Verify field names with `--log-level DEBUG` on the
first real run, same posture as every other real provider in this project.

Base URL: https://www.ballparkpal.com/api/v1
Auth: `X-API-Key` header (`BALLPARKPAL_API_KEY` env var / --ballparkpal-api-key)
Free tier (per Ballpark Pal's own API Access page, confirmed live
2026-08-30): 15,000 requests/month, 60 requests/minute. This integration
costs exactly one request per report run (one bulk `date=` fetch, cached),
regardless of slate size.

Scope of this integration, deliberately narrow: only `/api/v1/parkfactors/
hitters` is wired into scoring right now. Ballpark Pal's own HR/XBH matchup
probabilities (`/api/v1/matchups`) are NOT used here yet - their exact
probability basis (per-plate-appearance vs. per-game) isn't stated anywhere
in the docs seen so far, and shipping a comparison to this project's own
per-game `model_prob` on a guess risks a silent, misleading unit mismatch
(a real elite hitter's real per-PA HR rate, plausibly ~2-4%, would look like
it's "wildly disagreeing" with our own ~10-23% per-game figure even if both
models actually agree). Revisit once that's confirmed against a real
response.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

logger = logging.getLogger(__name__)

BALLPARKPAL_API_BASE = "https://www.ballparkpal.com/api/v1"


@dataclass(frozen=True)
class HitterParkFactor:
    """One batter's real, modeled park+weather multiplier for one game,
    from Ballpark Pal. `home_runs` is the combined (stadium + weather)
    multiplier (e.g. 1.08 = +8% HR odds vs. neutral) - directly comparable
    to this project's own `park_hr_factor` (100 = neutral) once scaled by
    100 (1.08 -> 108). `home_runs_stadium` isolates the park-only
    component (no weather applied); `home_runs_weather` is weather's own
    contribution, expressed as a deviation (combined minus stadium-only,
    e.g. 0.02 = weather added ~2% on top of the stadium factor) - NOT
    itself a multiplier, per Ballpark Pal's docs. Any may be `None` if
    Ballpark Pal hasn't generated that projection for this player/game yet.
    """

    player_name: str
    home_runs: Optional[float]
    home_runs_stadium: Optional[float]
    home_runs_weather: Optional[float]


class BallparkPalProvider(ABC):
    @abstractmethod
    def get_hitter_park_factor(self, player: str, game_date: date) -> Optional[HitterParkFactor]:
        """This batter's real park+weather factor for the game on
        `game_date`, or `None` if Ballpark Pal has no data for them (not
        on today's slate, projection not generated yet, or no API key
        configured at all - see `NoBallparkPalProvider`). Callers should
        treat `None` exactly like any other missing signal: fall back to
        the existing static/Open-Meteo-derived value, never guess.
        """
        raise NotImplementedError


class LiveBallparkPalProvider(BallparkPalProvider):
    """One real HTTP call per date, cached - `/api/v1/parkfactors/hitters`
    returns every batter's factor for the whole slate in one response, so
    there's no per-player request cost regardless of how many candidates
    this report ends up scoring.
    """

    def __init__(self, api_key: str, session=None, timeout: float = 10.0):
        from odds_monitor.http_utils import build_retrying_session

        self.api_key = api_key
        # Retries transient connection failures (see that module's
        # docstring - confirmed live against The Odds API) instead of
        # losing every batter's park factor to one dropped connection.
        self.session = session or build_retrying_session()
        self.timeout = timeout
        self._cache: Dict[date, Dict[str, HitterParkFactor]] = {}

    def _fetch(self, game_date: date) -> Dict[str, HitterParkFactor]:
        if game_date in self._cache:
            return self._cache[game_date]

        result: Dict[str, HitterParkFactor] = {}
        try:
            resp = self.session.get(
                f"{BALLPARKPAL_API_BASE}/parkfactors/hitters",
                params={"date": game_date.isoformat()},
                headers={"X-API-Key": self.api_key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()

            # Response envelope is `{"meta": {...}, "data": ...}` (confirmed
            # live via Ballpark Pal's own docs). The exact shape of `data`
            # for this specific endpoint (a bare list of rows vs. something
            # nested) wasn't shown in the docs screenshots available during
            # development - accept either rather than guessing wrong and
            # silently returning nothing.
            rows = payload.get("data", payload)
            if isinstance(rows, dict):
                rows = rows.get("hitters", rows.get("rows", rows.get("data", [])))
            if not isinstance(rows, list):
                logger.warning(
                    "Unexpected Ballpark Pal parkfactors/hitters response shape for %s: %r", game_date, type(rows)
                )
                rows = []

            for row in rows:
                try:
                    name = str(row["playerName"]).strip().lower()
                    result[name] = HitterParkFactor(
                        player_name=row["playerName"],
                        home_runs=row.get("homeRuns"),
                        home_runs_stadium=row.get("homeRunsStadium"),
                        home_runs_weather=row.get("homeRunsWeather"),
                    )
                except (KeyError, TypeError):
                    logger.warning("Skipping unparsable Ballpark Pal park-factor row: %r", row)
        except Exception:
            logger.exception("Ballpark Pal parkfactors/hitters fetch failed for %s", game_date)

        self._cache[game_date] = result
        return result

    def get_hitter_park_factor(self, player: str, game_date: date) -> Optional[HitterParkFactor]:
        return self._fetch(game_date).get(player.strip().lower())


class NoBallparkPalProvider(BallparkPalProvider):
    """No API key configured - every lookup returns `None`, so callers fall
    back to this project's own static park-factor table + Open-Meteo
    wind/temp entirely unchanged. Same graceful-degradation pattern as
    `NoOddsProvider` in `mlb_props/market.py`.
    """

    def get_hitter_park_factor(self, player: str, game_date: date) -> Optional[HitterParkFactor]:
        return None


class MockBallparkPalProvider(BallparkPalProvider):
    """Synthetic per-hitter park factors - no network calls. Used by
    `--mock` mode and tests. Deterministic per (player, seed), not per
    date, since mock mode has no real slate to key a bulk fetch off of.
    """

    def __init__(self, seed=None):
        import random

        self._rng = random.Random(seed)

    def get_hitter_park_factor(self, player: str, game_date: date) -> Optional[HitterParkFactor]:
        stadium = round(self._rng.uniform(0.90, 1.15), 3)
        weather = round(self._rng.uniform(-0.08, 0.08), 3)
        return HitterParkFactor(
            player_name=player,
            home_runs=round(stadium + weather, 3),
            home_runs_stadium=stadium,
            home_runs_weather=weather,
        )
