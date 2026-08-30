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

Two things are wired into scoring:
1. `/api/v1/parkfactors/hitters` - a real, per-hitter park+weather
   multiplier that replaces this project's own static table + Open-Meteo
   estimate for that specific player, when Ballpark Pal has data for them.
2. `/api/v1/matchups` - Ballpark Pal's own independent HR/Hits model for
   every real batter-vs-starter matchup, surfaced as a genuine second
   opinion alongside this project's own `model_prob`. This endpoint's
   probability fields were confirmed live (2026-08-30, via
   `mlb_props_main.run_ballparkpal_matchups_check`) to be per-plate-
   appearance, not per-game - real strikeoutProbability values that run
   averaged 24.4% (mean) / 23.7% (median), matching real MLB's actual
   ~22% per-PA strikeout rate almost exactly, nowhere near the ~55-65%
   you'd see for "at least one strikeout across a full game". See
   `_per_pa_to_per_game` for the conversion this module applies before
   treating any of these numbers as comparable to this project's own
   per-game figures.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

BALLPARKPAL_API_BASE = "https://www.ballparkpal.com/api/v1"

# Modern-era average plate appearances a real everyday hitter gets across a
# full 9-inning game. Used to convert Ballpark Pal's real, confirmed-live
# per-plate-appearance probabilities (see this module's docstring) into
# this project's own per-game convention, so the two can sit side by side
# without a silent unit mismatch. Same conversion idea as statcast.py's
# `_PA_PER_9_INNINGS`, just for a batter's PA/game instead of a pitcher's
# PA/9-innings.
_PA_PER_GAME = 4.3


def _per_pa_to_per_game(per_pa_probability_pct: float) -> float:
    """Converts a real per-PA probability (as a %, e.g. 24.4 = 24.4%) to
    an estimated per-game probability (0-1 fraction, this project's own
    `model_prob` convention) via P(at least one in n independent trials)
    = 1 - (1-p)^n. Independence is an approximation - a batter's plate
    appearances in one real game aren't perfectly independent (lineup
    spot, game state, pitcher fatigue by the 4th PA, etc.) - but it's the
    standard way to make this conversion, and Ballpark Pal doesn't publish
    a per-game figure directly to use instead.
    """
    p = per_pa_probability_pct / 100.0
    return 1.0 - (1.0 - p) ** _PA_PER_GAME


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


@dataclass(frozen=True)
class MatchupProbability:
    """Ballpark Pal's own real, independent model for one batter-vs-
    starting-pitcher matchup, converted from their confirmed per-plate-
    appearance probabilities (see `_per_pa_to_per_game`'s docstring) -
    real numbers from a different model, so any agreement or disagreement
    with this project's own `model_prob` is a genuine second opinion, not
    the same model computed twice.

    `home_run_model_prob`/`hits_model_prob` are per-game, 0-1 fractions,
    directly comparable to `HRScoreResult.model_prob`/`HitsScoreResult.
    model_prob`. `hits_model_prob` sums Ballpark Pal's single +
    double/triple + home-run per-PA probabilities before converting (they
    represent mutually exclusive plate-appearance outcomes in their
    model). There is deliberately no total-bases figure here: this
    project's "2+ total bases" market can be satisfied across multiple
    plate appearances (e.g. two singles), which a single per-PA outcome
    breakdown can't represent - no honest analog exists from this
    endpoint, so none is offered.
    """

    batter_name: str
    pitcher_name: str
    home_run_model_prob: Optional[float]
    hits_model_prob: Optional[float]


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

    @abstractmethod
    def get_matchup_probability(self, batter: str, pitcher: str, game_date: date) -> Optional[MatchupProbability]:
        """Ballpark Pal's own real per-game HR/Hits probability for this
        exact batter-vs-starter matchup, or `None` if unavailable (not on
        today's slate, no model data for this pair, or no API key
        configured - see `NoBallparkPalProvider`)."""
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
        self._matchup_cache: Dict[date, Dict[Tuple[str, str], MatchupProbability]] = {}

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

            # Response envelope is `{"meta": {...}, "data": {"items": [...]}}`
            # - confirmed live 2026-08-30 (real response: {"meta": {...,
            # "count": 364}, "data": {"items": [{...one row per player per
            # game...}]}}). Docs screenshots available during development
            # didn't show this specific endpoint's nesting; "items" wasn't
            # among the guessed keys on the first attempt, which silently
            # produced zero rows despite a real HTTP 200 (see this
            # provider's earlier log for that). Other keys kept as a
            # fallback rather than hard-coding only "items", in case a
            # different endpoint or a future API version nests differently.
            rows = payload.get("data", payload)
            if isinstance(rows, dict):
                rows = rows.get("items", rows.get("hitters", rows.get("rows", rows.get("data", []))))
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

            # Confirmed live 2026-08-30: a real request against this
            # endpoint returned HTTP 200 with no parse warnings, yet ended
            # up with zero usable rows here (every score that run fell back
            # to the pre-existing static table unchanged). That's silent
            # unless logged explicitly - a raw sample of the actual payload
            # is the fastest way to see what shape/timing issue caused it
            # (e.g. rows keyed under a name this parser didn't try, or a
            # date where Ballpark Pal simply has no data yet) without
            # guessing again.
            if not result:
                logger.warning(
                    "Ballpark Pal parkfactors/hitters returned 0 usable rows for %s (HTTP 200, %d raw rows seen) - "
                    "raw payload sample: %r",
                    game_date,
                    len(rows),
                    str(payload)[:2000],
                )
            else:
                logger.info("Ballpark Pal parkfactors/hitters: %d real per-hitter factors for %s", len(result), game_date)
        except Exception:
            logger.exception("Ballpark Pal parkfactors/hitters fetch failed for %s", game_date)

        self._cache[game_date] = result
        return result

    def get_hitter_park_factor(self, player: str, game_date: date) -> Optional[HitterParkFactor]:
        return self._fetch(game_date).get(player.strip().lower())

    def _fetch_matchups(self, game_date: date) -> Dict[Tuple[str, str], MatchupProbability]:
        if game_date in self._matchup_cache:
            return self._matchup_cache[game_date]

        result: Dict[Tuple[str, str], MatchupProbability] = {}
        try:
            resp = self.session.get(
                f"{BALLPARKPAL_API_BASE}/matchups",
                # parkAdjusted=true applies hitter-specific park factors and
                # venue-level walk/strikeout adjustments - the more
                # realistic real-conditions estimate (matches the "Matchups
                # page with the park-adjusted checkbox enabled" per
                # Ballpark Pal's own docs), consistent with using their
                # real per-hitter park factor elsewhere in this module.
                params={"date": game_date.isoformat(), "parkAdjusted": "true"},
                headers={"X-API-Key": self.api_key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()

            # Same defensive nested-key handling as `_fetch()` above (see
            # its comment) - this is a different endpoint, so its real
            # shape isn't assumed to match /parkfactors/hitters's confirmed
            # `data.items` even though that's tried first.
            rows = payload.get("data", payload)
            if isinstance(rows, dict):
                rows = rows.get("items", rows.get("hitters", rows.get("rows", rows.get("data", []))))
            if not isinstance(rows, list):
                logger.warning("Unexpected Ballpark Pal matchups response shape for %s: %r", game_date, type(rows))
                rows = []

            for row in rows:
                try:
                    batter_key = str(row["batterName"]).strip().lower()
                    pitcher_key = str(row["pitcherName"]).strip().lower()
                    hr_pa = row.get("homeRunProbability")
                    single_pa = row.get("singleProbability")
                    xbh_pa = row.get("doubleTripleProbability")

                    hr_model_prob = round(_per_pa_to_per_game(hr_pa), 4) if hr_pa is not None else None
                    hits_model_prob = None
                    if hr_pa is not None and single_pa is not None and xbh_pa is not None:
                        hits_model_prob = round(_per_pa_to_per_game(hr_pa + single_pa + xbh_pa), 4)

                    result[(batter_key, pitcher_key)] = MatchupProbability(
                        batter_name=row["batterName"],
                        pitcher_name=row["pitcherName"],
                        home_run_model_prob=hr_model_prob,
                        hits_model_prob=hits_model_prob,
                    )
                except (KeyError, TypeError):
                    logger.warning("Skipping unparsable Ballpark Pal matchup row: %r", row)

            if not result:
                logger.warning(
                    "Ballpark Pal matchups returned 0 usable rows for %s (HTTP 200, %d raw rows seen) - "
                    "raw payload sample: %r",
                    game_date,
                    len(rows),
                    str(payload)[:2000],
                )
            else:
                logger.info(
                    "Ballpark Pal matchups: %d real batter-vs-pitcher probabilities for %s", len(result), game_date
                )
        except Exception:
            logger.exception("Ballpark Pal matchups fetch failed for %s", game_date)

        self._matchup_cache[game_date] = result
        return result

    def get_matchup_probability(self, batter: str, pitcher: str, game_date: date) -> Optional[MatchupProbability]:
        return self._fetch_matchups(game_date).get((batter.strip().lower(), pitcher.strip().lower()))


class NoBallparkPalProvider(BallparkPalProvider):
    """No API key configured - every lookup returns `None`, so callers fall
    back to this project's own static park-factor table + Open-Meteo
    wind/temp entirely unchanged. Same graceful-degradation pattern as
    `NoOddsProvider` in `mlb_props/market.py`.
    """

    def get_hitter_park_factor(self, player: str, game_date: date) -> Optional[HitterParkFactor]:
        return None

    def get_matchup_probability(self, batter: str, pitcher: str, game_date: date) -> Optional[MatchupProbability]:
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

    def get_matchup_probability(self, batter: str, pitcher: str, game_date: date) -> Optional[MatchupProbability]:
        # Synthesized directly in this project's own per-game convention -
        # skips the per-PA round trip the real provider does, since mock
        # mode only needs a plausible number, not a faithful re-derivation
        # of Ballpark Pal's real per-PA-to-per-game conversion.
        return MatchupProbability(
            batter_name=batter,
            pitcher_name=pitcher,
            home_run_model_prob=round(self._rng.uniform(0.04, 0.23), 4),
            hits_model_prob=round(self._rng.uniform(0.40, 0.85), 4),
        )
