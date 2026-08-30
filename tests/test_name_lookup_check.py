"""Covers mlb_props_main.run_name_lookup_check - the --name-lookup-check
diagnostic that checks one or more player names directly against the real
Baseball Savant leaderboard, isolated from the rest of the pipeline (real
schedule/lineups/odds) - see that function's docstring for why this exists
(a full run's own log is too large to inspect for one player's specific
lookup outcome after the fact).
"""

from unittest.mock import patch

import pandas as pd

from mlb_props_main import run_name_lookup_check


class _FakePyb:
    def __init__(self, barrels: pd.DataFrame, expected: pd.DataFrame):
        self._barrels = barrels
        self._expected = expected

    def statcast_batter_exitvelo_barrels(self, year, minBBE=25):
        return self._barrels

    def statcast_batter_expected_stats(self, year, minPA=25):
        return self._expected


def _patch_pyb(barrels: pd.DataFrame, expected: pd.DataFrame):
    return patch("mlb_props.statcast.PybaseballStatcastProvider._pyb", lambda self: _FakePyb(barrels, expected))


def test_found_name_reports_match_with_real_savant_spelling():
    barrels = pd.DataFrame({"last_name, first_name": ["Henderson, Gunnar", "Alonso, Pete"]})
    expected = pd.DataFrame({"last_name, first_name": ["Henderson, Gunnar"]})
    with _patch_pyb(barrels, expected):
        text = run_name_lookup_check(["Gunnar Henderson"], 2026)
    assert "MATCH: 'Gunnar Henderson'" in text
    assert "Henderson, Gunnar" in text
    assert "expected-stats row found" in text


def test_found_in_barrels_but_missing_from_expected_stats_is_flagged():
    barrels = pd.DataFrame({"last_name, first_name": ["Henderson, Gunnar"]})
    expected = pd.DataFrame({"last_name, first_name": []})
    with _patch_pyb(barrels, expected):
        text = run_name_lookup_check(["Gunnar Henderson"], 2026)
    assert "expected-stats row MISSING" in text


def test_no_match_suggests_close_real_spellings():
    # Real Savant row carries an extra middle name our query doesn't have,
    # so the exact-match lookup misses - the substring fallback on the last
    # name should still surface it as a likely real spelling.
    barrels = pd.DataFrame({"last_name, first_name": ["Suarez, Eugenio Jose"]})
    expected = pd.DataFrame({"last_name, first_name": []})
    with _patch_pyb(barrels, expected):
        text = run_name_lookup_check(["Eugenio Suarez"], 2026)
    assert "NO MATCH" in text
    assert "Suarez, Eugenio Jose" in text  # surfaced as a close-spelling candidate


def test_no_match_and_no_close_spelling_explains_likely_min_bbe_cause():
    barrels = pd.DataFrame({"last_name, first_name": ["Someone Else, Totally"]})
    expected = pd.DataFrame({"last_name, first_name": []})
    with _patch_pyb(barrels, expected):
        text = run_name_lookup_check(["Nobody Real"], 2026)
    assert "NO MATCH" in text
    assert "real absence, not a name-matching bug" in text


def test_checks_multiple_names_independently():
    barrels = pd.DataFrame({"last_name, first_name": ["Henderson, Gunnar"]})
    expected = pd.DataFrame({"last_name, first_name": ["Henderson, Gunnar"]})
    with _patch_pyb(barrels, expected):
        text = run_name_lookup_check(["Gunnar Henderson", "Pete Alonso"], 2026)
    assert "MATCH: 'Gunnar Henderson'" in text
    assert "NO MATCH: 'Pete Alonso'" in text
