"""Real player-name lookups against a Baseball Savant leaderboard
(`last_name, first_name` column format) - covers two confirmed-live
failure modes: a trailing generational suffix (Bobby Witt Jr., Fernando
Tatis Jr., ...) breaking the naive last-word split, and a diacritic our
own lineup source keeps but Savant's export doesn't (or vice versa).
"""

import pandas as pd
import pytest

from mlb_props.statcast import _find_player_row, _name_lookup_candidates


def test_simple_two_word_name_matches_last_first_format():
    df = pd.DataFrame({"last_name, first_name": ["Henderson, Gunnar", "Alonso, Pete"]})
    match = _find_player_row(df, "Gunnar Henderson")
    assert not match.empty
    assert match.iloc[0]["last_name, first_name"] == "Henderson, Gunnar"


def test_suffix_attaches_to_last_name_not_split_off():
    # Savant's real convention (confirmed live): the suffix rides with the
    # last name, not the first - "Witt Jr., Bobby", not "Witt, Bobby Jr."
    # or the broken "Jr., Bobby Witt" a naive last-word split produces.
    df = pd.DataFrame({"last_name, first_name": ["Witt Jr., Bobby", "Tatis Jr., Fernando"]})
    for name in ("Bobby Witt Jr.", "Fernando Tatis Jr."):
        match = _find_player_row(df, name)
        assert not match.empty, f"{name} should have matched"


def test_suffix_variants_without_trailing_period_also_match():
    df = pd.DataFrame({"last_name, first_name": ["Chisholm Jr., Jazz"]})
    match = _find_player_row(df, "Jazz Chisholm Jr")  # no trailing period
    assert not match.empty


def test_accented_name_matches_unaccented_leaderboard_row():
    df = pd.DataFrame({"last_name, first_name": ["Suarez, Eugenio"]})
    match = _find_player_row(df, "Eugenio Suárez")
    assert not match.empty


def test_unaccented_query_matches_accented_leaderboard_row():
    df = pd.DataFrame({"last_name, first_name": ["Suárez, Eugenio"]})
    match = _find_player_row(df, "Eugenio Suarez")
    assert not match.empty


def test_no_match_returns_empty_not_an_exception():
    df = pd.DataFrame({"last_name, first_name": ["Henderson, Gunnar"]})
    match = _find_player_row(df, "Nobody Real")
    assert match.empty


def test_missing_name_column_returns_empty():
    df = pd.DataFrame({"some_other_column": ["x", "y"]})
    match = _find_player_row(df, "Gunnar Henderson")
    assert match.empty


@pytest.mark.parametrize(
    "player,expected_in",
    [
        ("Gunnar Henderson", "henderson, gunnar"),
        ("Bobby Witt Jr.", "witt jr, bobby"),
        ("Eugenio Suárez", "suarez, eugenio"),
    ],
)
def test_name_lookup_candidates_include_the_expected_savant_format(player, expected_in):
    candidates = _name_lookup_candidates(player)
    assert expected_in in candidates


def test_name_lookup_candidates_never_duplicate():
    candidates = _name_lookup_candidates("Eugenio Suarez")  # already unaccented
    assert len(candidates) == len(set(candidates))
