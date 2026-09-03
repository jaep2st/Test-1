"""Covers hot_streak.heat_index_from_log - the pure z-score/clearance
computation extracted out of StatcastHotStreakProvider.get_heat_index so
mlb_props/historical_backtest.py can reuse it against one already-fetched,
possibly-wide-ranging log instead of a separate per-player network call
for every (player, as_of) pair. The one property that actually matters
here: it must stay genuinely lookahead-safe even when the log it's given
spans dates AFTER as_of.
"""

from datetime import date

import pandas as pd

from mlb_props.hot_streak import LEAGUE_AVG_WOBA, heat_index_from_log


def _log(rows):
    return pd.DataFrame(
        {
            "game_date": [r[0] for r in rows],
            "events": [r[1] for r in rows],
            "woba_value": [r[2] for r in rows],
            "woba_denom": [1] * len(rows),
        }
    )


def test_heat_index_from_log_excludes_games_after_as_of_even_when_the_log_includes_them():
    # A real home run AFTER as_of - if this leaked in, last15_woba (and
    # the z-score built from it) would be inflated by a game that hadn't
    # happened yet as of the cutoff. It must not show up at all.
    log = _log(
        [
            ("2026-08-10", "strikeout", 0.0),
            ("2026-08-25", "home_run", 2.0),  # after as_of - must be excluded
        ]
    )
    heat = heat_index_from_log("Test Player", log, as_of=date(2026, 8, 20))
    assert heat.last15_pa == 1  # only the 8/10 game counts
    assert heat.season_woba == 0.0  # the strikeout-only game, not the later HR


def test_heat_index_from_log_includes_a_game_exactly_on_as_of():
    log = _log([("2026-08-20", "home_run", 2.0)])
    heat = heat_index_from_log("Test Player", log, as_of=date(2026, 8, 20))
    assert heat.last15_pa == 1
    assert heat.season_woba == 2.0


def test_heat_index_from_log_is_neutral_for_an_empty_log():
    heat = heat_index_from_log("Test Player", pd.DataFrame({"game_date": [], "events": []}), as_of=date(2026, 8, 20))
    assert heat.season_woba == LEAGUE_AVG_WOBA
    assert heat.z_score == 0.0


def test_heat_index_from_log_is_neutral_when_every_real_row_is_after_as_of():
    log = _log([("2026-08-25", "home_run", 2.0)])
    heat = heat_index_from_log("Test Player", log, as_of=date(2026, 8, 20))
    assert heat.season_woba == LEAGUE_AVG_WOBA
    assert heat.z_score == 0.0
