"""Covers mlb_props/market.py's book_display_name(): confirmed live against
The Odds API's own bookmaker-apis docs page (2026-09-04) that its real API
key for Caesars Sportsbook is `williamhill_us` - Caesars still runs on the
old William Hill US platform after its 2021 acquisition, and the API key
was never renamed for the rebrand. This project had been showing that real,
freely-returned book under its unrecognizable legacy key in every price
and filter this whole time.
"""

from mlb_props.market import book_display_name


def test_williamhill_us_displays_as_caesars():
    assert book_display_name("williamhill_us") == "Caesars"


def test_known_books_get_their_real_brand_capitalization():
    # A blind .title() gets these wrong (Draftkings, Fanduel, Betmgm) -
    # confirmed real book keys this project has actually seen live.
    assert book_display_name("draftkings") == "DraftKings"
    assert book_display_name("fanduel") == "FanDuel"
    assert book_display_name("betmgm") == "BetMGM"
    assert book_display_name("espnbet") == "ESPN BET"


def test_is_case_and_whitespace_insensitive_on_input():
    assert book_display_name("WilliamHill_US") == "Caesars"
    assert book_display_name("  draftkings  ") == "DraftKings"


def test_unknown_book_falls_back_to_title_case_not_the_raw_key():
    assert book_display_name("some_new_book") == "Some New Book"


def test_none_or_empty_stays_empty_rather_than_guessing():
    assert book_display_name(None) == ""
    assert book_display_name("") == ""
