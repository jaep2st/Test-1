"""Situational game context that shifts a matchup's predicted margin/total
away from a pure power-rating difference: home-field advantage (including a
real altitude effect at a handful of high-elevation venues), rest days
since each team's last game, the away team's travel distance, and - for
predicted totals only - live wind/precipitation at kickoff.

Real data: venue lat/lon/elevation/dome status from CFBD's `/venues`
endpoint (`cfbd.py`); wind/precipitation from Open-Meteo
(https://open-meteo.com), a free, no-key API, the same real source
`mlb_props/context.py` uses for MLB wind/temperature - keyed here off each
venue's real coordinates and the game's real kickoff time instead of the
current moment, using Open-Meteo's hourly forecast (available up to 16 days
out, comfortably covering this project's week-ahead cadence).

Every constant below is a disclosed, hand-set estimate from realistic,
publicly known effect sizes (documented next to each one) - not fit against
this project's own real results yet. See `scoring.py`'s module docstring
for the same caveat applied to the model as a whole.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Modern-era average real college-football home-field advantage, in points -
# consistently estimated across public CFB analytics work (Connelly's SP+,
# ESPN FPI, and independent studies) in the 2-2.5 point range, well below
# the ~3 points commonly cited for the NFL.
BASE_HOME_FIELD_ADVANTAGE = 2.3

# A real, well-documented effect: a sea-level team playing at meaningful
# altitude (thin air, more fatigue, especially for a team that hasn't
# practiced there) picks up a disclosed extra edge for the home team beyond
# a normal HFA - most associated with Wyoming (Laramie, ~7,220 ft), Air
# Force (Colorado Springs, ~6,621 ft), Colorado (Boulder, ~5,430 ft), Utah
# New Mexico (Albuquerque, ~5,312 ft), and BYU (Provo, ~4,551 ft). Applied
# on a sliding scale above ALTITUDE_THRESHOLD_FT, capped at
# MAX_ALTITUDE_BONUS - a real but modest effect, not a dominant one.
ALTITUDE_THRESHOLD_FT = 3500.0
MAX_ALTITUDE_BONUS = 2.0
ALTITUDE_BONUS_PER_1000FT = 0.35

# Extra rest matters (a bye week, a Thursday-then-Saturday short week for
# the opponent), but a normal 6-8 day week-to-week gap is noise - only the
# real edge beyond that is worth pricing, capped so a small schedule quirk
# never swings a prediction by more than a couple of points.
REST_EDGE_POINTS_PER_DAY = 0.12
MAX_REST_EDGE_POINTS = 2.5

# A long real away trip is a genuine, if modest, disadvantage (travel
# fatigue, time-zone shift, short week to prepare) - scaled by distance,
# capped well below the altitude/rest effects since travel's real impact on
# elite, well-resourced FBS programs (charter flights, days-ahead travel)
# is smaller than folk wisdom suggests.
TRAVEL_EDGE_POINTS_PER_1000_MILES = 0.4
MAX_TRAVEL_EDGE_POINTS = 1.5
TRAVEL_PENALTY_THRESHOLD_MILES = 300.0  # short in-region trips get no penalty at all

EARTH_RADIUS_MILES = 3958.8


@dataclass(frozen=True)
class VenueInfo:
    venue_id: Optional[int]
    name: str
    city: Optional[str]
    state: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    elevation_ft: Optional[float]
    is_dome: bool


class VenueProvider(ABC):
    @abstractmethod
    def get_venue(self, name: str, venue_id: Optional[int] = None) -> Optional[VenueInfo]:
        raise NotImplementedError

    @abstractmethod
    def home_venue_for_team(self, team: str) -> Optional[VenueInfo]:
        """The venue a team plays its home games at, for computing the
        away team's real travel distance to a given matchup's venue -
        `None` if this team's usual home venue can't be resolved (a team
        whose only games this provider has seen were all away/neutral).
        """
        raise NotImplementedError


class CfbdVenueProvider(VenueProvider):
    def __init__(self, client=None, schedule_client=None, season: Optional[int] = None):
        from .cfbd import CfbdClient

        self.client = client or CfbdClient()
        self._by_id: Dict[int, VenueInfo] = {}
        self._by_name: Dict[str, VenueInfo] = {}
        self._home_venue_by_team: Dict[str, VenueInfo] = {}
        self._loaded = False
        self._season = season

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        for raw in self.client.venues():
            try:
                info = VenueInfo(
                    venue_id=raw.get("id"),
                    name=raw.get("name") or "",
                    city=raw.get("city"),
                    state=raw.get("state"),
                    latitude=float(raw["latitude"]) if raw.get("latitude") is not None else None,
                    longitude=float(raw["longitude"]) if raw.get("longitude") is not None else None,
                    elevation_ft=float(raw["elevation"]) if raw.get("elevation") is not None else None,
                    is_dome=bool(raw.get("dome", False)),
                )
            except (TypeError, ValueError):
                logger.warning("Skipping unparsable CFBD venue entry: %r", raw)
                continue
            if info.venue_id is not None:
                self._by_id[info.venue_id] = info
            if info.name:
                self._by_name[info.name.strip().lower()] = info

        if self._season is not None:
            for raw in self.client.games(self._season, season_type="regular"):
                home_team = raw.get("homeTeam")
                venue_id = raw.get("venueId")
                if not home_team or raw.get("neutralSite") or home_team in self._home_venue_by_team:
                    continue
                venue = self._by_id.get(venue_id) if venue_id is not None else self._by_name.get(str(raw.get("venue") or "").strip().lower())
                if venue is not None:
                    self._home_venue_by_team[home_team] = venue

    def get_venue(self, name: str, venue_id: Optional[int] = None) -> Optional[VenueInfo]:
        self._load()
        if venue_id is not None and venue_id in self._by_id:
            return self._by_id[venue_id]
        return self._by_name.get((name or "").strip().lower())

    def home_venue_for_team(self, team: str) -> Optional[VenueInfo]:
        self._load()
        return self._home_venue_by_team.get(team)


class MockVenueProvider(VenueProvider):
    _VENUES = {
        "Ohio Stadium": VenueInfo(3891, "Ohio Stadium", "Columbus", "OH", 40.0017, -83.0197, 750, False),
        "Sanford Stadium": VenueInfo(3729, "Sanford Stadium", "Athens", "GA", 33.9497, -83.3733, 630, False),
        "Cotton Bowl": VenueInfo(3745, "Cotton Bowl", "Dallas", "TX", 32.7825, -96.7597, 470, False),
        "Husky Stadium": VenueInfo(3901, "Husky Stadium", "Seattle", "WA", 47.6503, -122.3017, 20, False),
        "Canvas Stadium": VenueInfo(4213, "Canvas Stadium", "Fort Collins", "CO", 40.5714, -105.0836, 5004, False),
        "War Memorial Stadium": VenueInfo(3801, "War Memorial Stadium", "Laramie", "WY", 41.3131, -105.5794, 7220, False),
    }
    _HOME_VENUE_BY_TEAM = {
        "Ohio State": _VENUES["Ohio Stadium"],
        "Michigan": VenueInfo(9999, "Michigan Stadium", "Ann Arbor", "MI", 42.2658, -83.7487, 840, False),
        "Georgia": _VENUES["Sanford Stadium"],
        "Alabama": VenueInfo(9998, "Bryant-Denny Stadium", "Tuscaloosa", "AL", 33.2083, -87.5503, 254, False),
        "Texas": VenueInfo(9997, "Darrell K Royal Stadium", "Austin", "TX", 30.2839, -97.7325, 505, False),
        "Oklahoma": VenueInfo(9996, "Gaylord Family Stadium", "Norman", "OK", 35.2059, -97.4425, 1180, False),
        "Oregon": VenueInfo(9995, "Autzen Stadium", "Eugene", "OR", 44.0582, -123.0685, 430, False),
        "Washington": _VENUES["Husky Stadium"],
        "Boise State": VenueInfo(9994, "Albertsons Stadium", "Boise", "ID", 43.6028, -116.1994, 2730, False),
        "Colorado State": _VENUES["Canvas Stadium"],
        "Air Force": VenueInfo(9993, "Falcon Stadium", "Colorado Springs", "CO", 38.9972, -104.8422, 6621, False),
        "Wyoming": _VENUES["War Memorial Stadium"],
    }

    def get_venue(self, name: str, venue_id: Optional[int] = None) -> Optional[VenueInfo]:
        return self._VENUES.get(name)

    def home_venue_for_team(self, team: str) -> Optional[VenueInfo]:
        return self._HOME_VENUE_BY_TEAM.get(team)


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * math.asin(min(1.0, math.sqrt(a)))


def altitude_bonus_points(venue_elevation_ft: Optional[float], neutral_site: bool) -> float:
    if neutral_site or venue_elevation_ft is None or venue_elevation_ft <= ALTITUDE_THRESHOLD_FT:
        return 0.0
    extra_thousands = (venue_elevation_ft - ALTITUDE_THRESHOLD_FT) / 1000.0
    return round(min(MAX_ALTITUDE_BONUS, extra_thousands * ALTITUDE_BONUS_PER_1000FT), 2)


def rest_edge_points(home_rest_days: Optional[int], away_rest_days: Optional[int]) -> float:
    if home_rest_days is None or away_rest_days is None:
        return 0.0
    diff = home_rest_days - away_rest_days
    return round(max(-MAX_REST_EDGE_POINTS, min(MAX_REST_EDGE_POINTS, diff * REST_EDGE_POINTS_PER_DAY)), 2)


def travel_edge_points(travel_miles: Optional[float]) -> float:
    if travel_miles is None or travel_miles <= TRAVEL_PENALTY_THRESHOLD_MILES:
        return 0.0
    return round(min(MAX_TRAVEL_EDGE_POINTS, (travel_miles / 1000.0) * TRAVEL_EDGE_POINTS_PER_1000_MILES), 2)


@dataclass(frozen=True)
class WeatherObservation:
    wind_mph: Optional[float]
    precip_probability_pct: Optional[float]
    temp_f: Optional[float]


class WeatherProvider(ABC):
    @abstractmethod
    def get_forecast(self, latitude: float, longitude: float, kickoff_utc_iso: Optional[str]) -> Optional[WeatherObservation]:
        raise NotImplementedError


class LiveOpenMeteoProvider(WeatherProvider):
    """Not exercised live in this build environment (no outbound network
    access here) - verify the response shape with `--log-level DEBUG`
    before relying on it, same disclosed-but-unverified posture as every
    other new real integration in this project.
    """

    def __init__(self, session=None, timeout: float = 10.0):
        from odds_monitor.http_utils import build_retrying_session

        self.session = session or build_retrying_session()
        self.timeout = timeout

    def get_forecast(self, latitude: float, longitude: float, kickoff_utc_iso: Optional[str]) -> Optional[WeatherObservation]:
        if kickoff_utc_iso is None:
            return None
        try:
            kickoff = datetime.fromisoformat(kickoff_utc_iso.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Could not parse kickoff time %r for weather lookup", kickoff_utc_iso)
            return None
        if (kickoff - datetime.now(timezone.utc)).days > 15:
            # Open-Meteo's hourly forecast only reaches ~16 days out - a
            # game scheduled further ahead than that has no real forecast
            # yet; "unknown" beats a stale/guessed one.
            return None
        try:
            resp = self.session.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
                    "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                    "forecast_days": 16,
                    "timezone": "UTC",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            hourly = resp.json()["hourly"]
            times = hourly["time"]
            target = kickoff.strftime("%Y-%m-%dT%H:00")
            idx = times.index(target) if target in times else min(range(len(times)), key=lambda i: abs(i))
        except Exception:
            logger.exception("Open-Meteo forecast fetch failed for (%s, %s)", latitude, longitude)
            return None
        try:
            return WeatherObservation(
                wind_mph=float(hourly["wind_speed_10m"][idx]) if hourly["wind_speed_10m"][idx] is not None else None,
                precip_probability_pct=float(hourly["precipitation_probability"][idx])
                if hourly["precipitation_probability"][idx] is not None
                else None,
                temp_f=float(hourly["temperature_2m"][idx]) if hourly["temperature_2m"][idx] is not None else None,
            )
        except (KeyError, IndexError, TypeError):
            logger.warning("Unexpected Open-Meteo hourly payload shape for (%s, %s)", latitude, longitude)
            return None


class MockWeatherProvider(WeatherProvider):
    def __init__(self, seed=None):
        import random

        self._rng = random.Random(seed)

    def get_forecast(self, latitude: float, longitude: float, kickoff_utc_iso: Optional[str]) -> Optional[WeatherObservation]:
        return WeatherObservation(
            wind_mph=round(self._rng.uniform(2, 18), 1),
            precip_probability_pct=round(self._rng.uniform(0, 40), 0),
            temp_f=round(self._rng.uniform(35, 85), 1),
        )


def weather_total_adjustment_pct(weather: Optional[WeatherObservation], is_dome: bool) -> float:
    """Heuristic %% shift to the predicted total from wind + precipitation -
    both suppress scoring (harder passing/kicking conditions), consistent
    with well-documented real NFL/CFB weather research, though this
    project's specific coefficients are hand-set, not fit to its own data.
    Always 0.0 at a dome or with no real forecast available.
    """
    if is_dome or weather is None:
        return 0.0
    adjustment = 0.0
    if weather.wind_mph is not None and weather.wind_mph > 15.0:
        adjustment -= min(6.0, (weather.wind_mph - 15.0) * 0.35)
    if weather.precip_probability_pct is not None and weather.precip_probability_pct > 50.0:
        adjustment -= min(4.0, (weather.precip_probability_pct - 50.0) * 0.06)
    return round(adjustment, 2)


@dataclass(frozen=True)
class GameContext:
    """Every situational adjustment for one matchup, all expressed as
    points added to the HOME team's predicted margin (positive favors
    home) except `weather_total_adjustment_pct`, which shifts the
    predicted total instead - see `scoring.py` for where each is applied.
    """

    venue: str
    neutral_site: bool
    is_dome: bool
    home_field_advantage_pts: float
    altitude_bonus_pts: float
    home_rest_days: Optional[int]
    away_rest_days: Optional[int]
    rest_edge_pts: float
    travel_miles: Optional[float]
    travel_edge_pts: float
    weather: Optional[WeatherObservation]
    weather_total_adjustment_pct: float

    @property
    def total_margin_adjustment_pts(self) -> float:
        return round(self.home_field_advantage_pts + self.altitude_bonus_pts + self.rest_edge_pts + self.travel_edge_pts, 2)


def build_game_context(
    matchup,
    venue_provider: VenueProvider,
    weather_provider: WeatherProvider,
    home_rest_days: Optional[int] = None,
    away_rest_days: Optional[int] = None,
) -> GameContext:
    venue_info = venue_provider.get_venue(matchup.venue, matchup.venue_id)
    is_dome = bool(venue_info and venue_info.is_dome)
    elevation = venue_info.elevation_ft if venue_info else None

    hfa = 0.0 if matchup.neutral_site else BASE_HOME_FIELD_ADVANTAGE
    altitude = altitude_bonus_points(elevation, matchup.neutral_site)
    rest_edge = rest_edge_points(home_rest_days, away_rest_days)

    travel_miles = None
    if venue_info and venue_info.latitude is not None and venue_info.longitude is not None:
        away_home_venue = venue_provider.home_venue_for_team(matchup.away_team)
        if away_home_venue and away_home_venue.latitude is not None and away_home_venue.longitude is not None:
            travel_miles = round(
                haversine_miles(venue_info.latitude, venue_info.longitude, away_home_venue.latitude, away_home_venue.longitude), 1
            )
    travel_edge = travel_edge_points(travel_miles)

    weather = None
    if venue_info and venue_info.latitude is not None and venue_info.longitude is not None and not is_dome:
        weather = weather_provider.get_forecast(venue_info.latitude, venue_info.longitude, matchup.start_date_utc)
    weather_pct = weather_total_adjustment_pct(weather, is_dome)

    return GameContext(
        venue=matchup.venue,
        neutral_site=matchup.neutral_site,
        is_dome=is_dome,
        home_field_advantage_pts=hfa,
        altitude_bonus_pts=altitude,
        home_rest_days=home_rest_days,
        away_rest_days=away_rest_days,
        rest_edge_pts=rest_edge,
        travel_miles=travel_miles,
        travel_edge_pts=travel_edge,
        weather=weather,
        weather_total_adjustment_pct=weather_pct,
    )
