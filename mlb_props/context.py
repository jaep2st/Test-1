"""Ballpark and weather context: how much a stadium and today's conditions
inflate or deflate home run probability.

Park factors are static, well-known 3-year-rolling HR park factors (100 =
league neutral; >100 = favors hitters). These change slowly and are safe to
hardcode; refresh yearly from a source like FanGraphs' park factor pages.

Weather uses Open-Meteo (https://open-meteo.com), a free API that needs no
key, keyed off each park's lat/lon. Wind blowing out to center field boosts
fly-ball carry; wind in suppresses it; both matter far more at open-air
parks than domes.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# 3-year rolling HR park factors, approximate, 100 = neutral. Refresh
# periodically from FanGraphs/Statcast park factor pages.
PARK_HR_FACTORS: Dict[str, float] = {
    "Coors Field": 118,
    "Great American Ball Park": 114,
    "Yankee Stadium": 112,
    "Citizens Bank Park": 111,
    "Chase Field": 108,
    "Camden Yards": 107,
    "Globe Life Field": 105,
    "Truist Park": 104,
    "American Family Field": 104,
    "loanDepot park": 92,
    "Comerica Park": 90,
    "Kauffman Stadium": 89,
    "T-Mobile Park": 88,
    "Oracle Park": 85,
    "Petco Park": 94,
    "Dodger Stadium": 101,
    "Wrigley Field": 103,
    "Fenway Park": 98,
    "Guaranteed Rate Field": 106,
    "Progressive Field": 97,
    "Target Field": 99,
    "Minute Maid Park": 100,
    "Busch Stadium": 93,
    "PNC Park": 91,
    "Nationals Park": 99,
    "Citi Field": 96,
    "Angel Stadium": 100,
    "Rogers Centre": 105,
    "George M. Steinbrenner Field": 95,
    # Confirmed live via web search (2026-08-27): the Rays are back at
    # Tropicana Field for the 2026 season after playing all of 2025 at
    # Steinbrenner Field while Hurricane Milton's roof damage was repaired
    # (ESPN: "Rays to return to Tropicana Field in '26 after hurricane
    # repairs", home opener April 6, 2026). Tropicana Field was missing
    # from this table entirely, so any Rays home game would have silently
    # fallen back to a neutral 100 park factor instead of the fixed dome's
    # real (pitcher-friendly) one. 96 matches its typical modern rolling
    # factor; refresh from FanGraphs/Statcast once 2026 games are in.
    "Tropicana Field": 96,
    # Confirmed live via web search (2026-08-27): the Athletics play their
    # 2025-2027 home games at Sutter Health Park in West Sacramento (a
    # ~14,000-seat Triple-A park with notably short fences), not Oakland
    # Coliseum (kept below only as a historical fallback - it's stale for
    # any current A's game). No established multi-year rolling HR factor
    # exists yet for this park; 128 reflects public reporting of its small
    # dimensions playing very hitter-friendly in 2025 - treat as a rougher
    # estimate than the other entries here and refresh once more seasons
    # of data exist.
    "Sutter Health Park": 128,
    "Oakland Coliseum": 89,  # historical - the Athletics no longer play here (see Sutter Health Park above)
}

# Roughly the home-plate lat/lon and orientation (degrees, home->CF compass
# bearing) for each park; used to resolve wind direction into "blowing out"
# vs "blowing in". Domes/retractable roofs (closed) neutralize wind - see
# DOME_PARKS.
PARK_COORDS: Dict[str, Tuple[float, float, float]] = {
    "Coors Field": (39.7559, -104.9942, 45),
    "Yankee Stadium": (40.8296, -73.9262, 75),
    "Dodger Stadium": (34.0739, -118.2400, 25),
    "Fenway Park": (42.3467, -71.0972, 40),
    "Wrigley Field": (41.9484, -87.6553, 30),
    "Oracle Park": (37.7786, -122.3893, 95),
    "Truist Park": (33.8908, -84.4678, 15),
    "Minute Maid Park": (29.7573, -95.3555, 0),
    "Citi Field": (40.7571, -73.8458, 30),
    "Camden Yards": (39.2839, -76.6218, 30),
    # Not exhaustive - falls back to no weather adjustment if a park is missing.
    # Sutter Health Park (the Athletics' real 2026 home - see
    # PARK_HR_FACTORS above) is deliberately NOT added here: its home-plate
    # coordinates are well-established (38.5804, -121.5138, confirmed live
    # via web search), but no verified center-field compass bearing was
    # found for it, and shipping a guessed bearing risks silently flipping
    # "wind out" to "wind in" for every game there - worse than the honest
    # zero-wind-effect fallback this park already gets by being absent here.
}

DOME_PARKS = {
    "Minute Maid Park",
    "Rogers Centre",
    "American Family Field",
    "Chase Field",
    "loanDepot park",
    "Globe Life Field",
    # Fixed (non-retractable) dome; confirmed live via web search
    # (2026-08-27) the Rays are back here for the 2026 season - see
    # PARK_HR_FACTORS above.
    "Tropicana Field",
}


@dataclass(frozen=True)
class ParkWeatherContext:
    park: str
    park_hr_factor: float  # 100 = neutral
    wind_out_mph: float  # positive = blowing out (helps HR), negative = blowing in
    temp_f: Optional[float]
    is_dome: bool
    weather_hr_boost_pct: float  # heuristic % change to HR odds from wind + temp, e.g. +6.0 or -4.0


class ParkWeatherProvider(ABC):
    @abstractmethod
    def get_context(self, park: str) -> ParkWeatherContext:
        raise NotImplementedError


class LiveParkWeatherProvider(ParkWeatherProvider):
    """Static park factor + live wind/temp from Open-Meteo (no API key).
    Not exercised live in this build environment (network access to
    `api.open-meteo.com` was blocked there) - verify the response shape
    with `--log-level DEBUG` before relying on it.
    """

    def __init__(self, session=None, timeout: float = 10.0):
        from odds_monitor.http_utils import build_retrying_session

        # Retries transient connection failures (see that module's
        # docstring - confirmed live against The Odds API) instead of
        # dropping weather data for a park on one dropped connection. Only
        # applied when no session is injected, so tests supplying a fake
        # session are unaffected.
        self.session = session or build_retrying_session()
        self.timeout = timeout

    def get_context(self, park: str) -> ParkWeatherContext:
        factor = PARK_HR_FACTORS.get(park, 100.0)
        is_dome = park in DOME_PARKS
        if is_dome or park not in PARK_COORDS:
            return ParkWeatherContext(park, factor, 0.0, None, is_dome, 0.0)

        lat, lon, cf_bearing = PARK_COORDS[park]
        try:
            resp = self.session.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,wind_speed_10m,wind_direction_10m",
                    "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            current = resp.json()["current"]
        except Exception:
            logger.exception("Open-Meteo fetch failed for %s", park)
            return ParkWeatherContext(park, factor, 0.0, None, is_dome, 0.0)

        wind_speed = float(current.get("wind_speed_10m", 0.0) or 0.0)
        wind_dir = float(current.get("wind_direction_10m", 0.0) or 0.0)
        temp_f = float(current.get("temperature_2m")) if current.get("temperature_2m") is not None else None

        # Wind direction is "from" compass bearing; component blowing toward
        # center field (i.e. "out") is positive when wind is roughly coming
        # from behind home plate toward CF, i.e. from-bearing ~= cf_bearing + 180.
        blowing_from_backstop = math.cos(math.radians(wind_dir - (cf_bearing + 180)))
        wind_out_mph = round(wind_speed * blowing_from_backstop, 1)

        boost = wind_out_mph * 1.1  # ~1.1% HR-odds shift per mph of sustained out/in wind, heuristic
        if temp_f is not None:
            boost += max(0.0, (temp_f - 70.0)) * 0.15  # warmer air carries fly balls further
        return ParkWeatherContext(park, factor, wind_out_mph, temp_f, is_dome, round(boost, 1))


class MockParkWeatherProvider(ParkWeatherProvider):
    """Synthetic wind/temp - no network calls. Park factors are the real
    static table above; only wind/temp are randomized.
    """

    def __init__(self, seed=None):
        import random

        self._rng = random.Random(seed)

    def get_context(self, park: str) -> ParkWeatherContext:
        factor = PARK_HR_FACTORS.get(park, 100.0)
        is_dome = park in DOME_PARKS
        if is_dome:
            return ParkWeatherContext(park, factor, 0.0, 72.0, True, 0.0)
        wind_out_mph = round(self._rng.uniform(-12, 12), 1)
        temp_f = round(self._rng.uniform(55, 95), 1)
        boost = wind_out_mph * 1.1 + max(0.0, (temp_f - 70.0)) * 0.15
        return ParkWeatherContext(park, factor, wind_out_mph, temp_f, False, round(boost, 1))
