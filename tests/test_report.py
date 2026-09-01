"""Covers mlb_props/report.py's clearance_cols/clearance_rates - the real
per-game L15/season clearance numbers shared by the console report,
html_report.py's prop tables, and pdf_report.py.
"""

from mlb_props.hot_streak import ClearanceWindow, HeatIndex
from mlb_props.report import clearance_cols, clearance_rates


def _heat(l15_games=15, hr_games=3, tb2_games=6, hit_games=9, season_games=100, season_hr=20, season_tb2=40, season_hit=65):
    return HeatIndex(
        player="Test Player", season_woba=0.330, last7_woba=0.330, last15_woba=0.330, last30_woba=0.330,
        last15_pa=60, z_score=0.0,
        clear_l15=ClearanceWindow(games=l15_games, hr_games=hr_games, tb2_games=tb2_games, hit_games=hit_games),
        clear_season=ClearanceWindow(games=season_games, hr_games=season_hr, tb2_games=season_tb2, hit_games=season_hit),
    )


def test_clearance_rates_matches_the_real_ratio_clearance_cols_formats():
    heat = _heat()
    l15_str, season_str = clearance_cols(heat, "hr")
    l15_rate, season_rate = clearance_rates(heat, "hr")
    assert l15_str == "3/15"
    assert l15_rate == 3 / 15
    assert season_str == "20%"
    assert round(season_rate * 100) == 20


def test_clearance_rates_is_none_with_no_heat_data():
    assert clearance_rates(None, "hr") == (None, None)


def test_clearance_rates_is_none_when_the_window_has_no_real_games():
    heat = _heat(l15_games=0, season_games=0)
    l15_rate, season_rate = clearance_rates(heat, "tb2")
    assert l15_rate is None
    assert season_rate is None
