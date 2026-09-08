from ncaaf.context import (
    altitude_bonus_points,
    haversine_miles,
    rest_edge_points,
    travel_edge_points,
    weather_total_adjustment_pct,
    WeatherObservation,
)


def test_haversine_zero_distance_for_same_point():
    assert haversine_miles(40.0, -83.0, 40.0, -83.0) == 0.0


def test_haversine_known_rough_distance_nyc_to_la():
    # Real great-circle distance NYC <-> LA is ~2,450 miles.
    miles = haversine_miles(40.7128, -74.0060, 34.0522, -118.2437)
    assert 2300 < miles < 2600


def test_altitude_bonus_zero_below_threshold_and_at_neutral_site():
    assert altitude_bonus_points(3000.0, neutral_site=False) == 0.0
    assert altitude_bonus_points(7220.0, neutral_site=True) == 0.0
    assert altitude_bonus_points(None, neutral_site=False) == 0.0


def test_altitude_bonus_increases_with_elevation_and_is_capped():
    low = altitude_bonus_points(4500.0, neutral_site=False)
    high = altitude_bonus_points(7220.0, neutral_site=False)  # Wyoming
    assert 0.0 < low < high
    assert high <= 2.0  # MAX_ALTITUDE_BONUS


def test_rest_edge_favors_the_more_rested_team_and_is_capped():
    assert rest_edge_points(None, None) == 0.0
    assert rest_edge_points(7, 7) == 0.0
    bye_week_home = rest_edge_points(14, 6)
    short_week_home = rest_edge_points(4, 7)
    assert bye_week_home > 0
    assert short_week_home < 0
    assert rest_edge_points(100, 6) <= 2.5  # MAX_REST_EDGE_POINTS


def test_travel_edge_zero_for_short_trips_positive_for_long_ones():
    assert travel_edge_points(None) == 0.0
    assert travel_edge_points(100.0) == 0.0
    assert travel_edge_points(2500.0) > 0.0
    assert travel_edge_points(10000.0) <= 1.5  # MAX_TRAVEL_EDGE_POINTS


def test_weather_adjustment_zero_at_dome_or_with_no_forecast():
    obs = WeatherObservation(wind_mph=30.0, precip_probability_pct=90.0, temp_f=40.0)
    assert weather_total_adjustment_pct(obs, is_dome=True) == 0.0
    assert weather_total_adjustment_pct(None, is_dome=False) == 0.0


def test_weather_adjustment_negative_for_wind_and_rain():
    calm = WeatherObservation(wind_mph=5.0, precip_probability_pct=5.0, temp_f=65.0)
    stormy = WeatherObservation(wind_mph=28.0, precip_probability_pct=85.0, temp_f=40.0)
    assert weather_total_adjustment_pct(calm, is_dome=False) == 0.0
    assert weather_total_adjustment_pct(stormy, is_dome=False) < 0.0
