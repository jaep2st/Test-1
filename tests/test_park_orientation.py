"""Covers LiveParkWeatherProvider's real center-field-bearing-aware wind
resolution (context.py) - this table/method already existed and already
correctly used each park's real compass orientation, not just raw wind mph,
but had no direct test coverage before this. See PARK_COORDS's docstring
for why coverage is currently 10 of the ~26 real open-air parks (a
disclosed data gap, not a logic gap) - this test exercises the logic itself
using parks that ARE in the table.
"""

from mlb_props.context import DOME_PARKS, PARK_COORDS, LiveParkWeatherProvider


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, wind_speed: float, wind_direction: float, temp_f: float = 70.0):
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.temp_f = temp_f

    def get(self, url, params=None, timeout=None):
        return _FakeResponse(
            {
                "current": {
                    "temperature_2m": self.temp_f,
                    "wind_speed_10m": self.wind_speed,
                    "wind_direction_10m": self.wind_direction,
                }
            }
        )


def test_wind_blowing_from_behind_home_plate_toward_center_reads_as_out():
    park = "Wrigley Field"
    _, _, cf_bearing = PARK_COORDS[park]
    # Wind direction is the compass bearing it's blowing FROM. Blowing
    # toward center field means it's coming from behind home plate, i.e.
    # from-bearing = cf_bearing + 180.
    from_bearing = (cf_bearing + 180) % 360
    provider = LiveParkWeatherProvider(session=_FakeSession(wind_speed=10.0, wind_direction=from_bearing))
    ctx = provider.get_context(park)
    assert ctx.wind_out_mph > 9.0  # ~full 10mph reads as "out"


def test_wind_blowing_from_center_field_toward_home_plate_reads_as_in():
    park = "Wrigley Field"
    _, _, cf_bearing = PARK_COORDS[park]
    provider = LiveParkWeatherProvider(session=_FakeSession(wind_speed=10.0, wind_direction=cf_bearing))
    ctx = provider.get_context(park)
    assert ctx.wind_out_mph < -9.0  # ~full 10mph reads as "in"


def test_crosswind_reads_as_roughly_zero():
    park = "Wrigley Field"
    _, _, cf_bearing = PARK_COORDS[park]
    from_bearing = (cf_bearing + 90) % 360  # perpendicular to the home-plate-to-CF line
    provider = LiveParkWeatherProvider(session=_FakeSession(wind_speed=10.0, wind_direction=from_bearing))
    ctx = provider.get_context(park)
    assert abs(ctx.wind_out_mph) < 1.0


def test_dome_ignores_real_wind_entirely():
    dome = next(iter(DOME_PARKS))
    provider = LiveParkWeatherProvider(session=_FakeSession(wind_speed=25.0, wind_direction=0.0))
    ctx = provider.get_context(dome)
    assert ctx.wind_out_mph == 0.0
    assert ctx.is_dome is True


def test_park_missing_from_the_coordinate_table_gets_the_honest_zero_wind_fallback():
    # See PARK_COORDS's docstring: a real, disclosed coverage gap - a park
    # not yet verified stays at the same neutral fallback as before this
    # feature existed, never a guessed bearing.
    missing_park = "Sutter Health Park"
    assert missing_park not in PARK_COORDS
    assert missing_park not in DOME_PARKS
    provider = LiveParkWeatherProvider(session=_FakeSession(wind_speed=25.0, wind_direction=0.0))
    ctx = provider.get_context(missing_park)
    assert ctx.wind_out_mph == 0.0
