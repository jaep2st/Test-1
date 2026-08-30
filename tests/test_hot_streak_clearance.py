"""Real per-game HR/2+TB/1+Hits clearance counts (see hot_streak.py's
ClearanceWindow) - literal 'did this player clear the line in this exact
real game', computed from the same per-PA Statcast log already fetched for
the wOBA hot-streak windows, at no extra network cost.
"""

from mlb_props.hot_streak import (
    ClearanceWindow,
    MockHotStreakProvider,
    clearance_windows_from_outcomes,
    game_outcomes_from_events,
)


def test_game_outcomes_classifies_hr_2tb_and_hit_correctly():
    # One HR game, one 2-single (2 TB) game, one 1-single (1 TB, still a
    # hit but not 2+ TB) game, one strikeout-only game (no hit at all).
    games_events = [
        ["home_run"],
        ["single", "single"],
        ["single", "strikeout", "walk"],
        ["strikeout", "field_out"],
    ]
    outcomes = game_outcomes_from_events(games_events)
    assert outcomes == [
        (True, True, True),  # HR game: HR, 2+ TB (4 bases), and a hit
        (False, True, True),  # two singles: no HR, 2 TB, a hit
        (False, False, True),  # one single + K + BB: 1 TB only, still a hit
        (False, False, False),  # no hit at all
    ]


def test_clearance_windows_use_most_recent_games_first():
    # 20 games: alternating HR / no-HR, oldest first. Only the most recent
    # N should count toward each window, not the first N.
    games_events = []
    for i in range(20):
        games_events.append(["home_run"] if i % 2 == 0 else ["strikeout"])
    outcomes = game_outcomes_from_events(games_events)
    l5, l10, l15, season = clearance_windows_from_outcomes(outcomes)

    assert season.games == 20
    assert season.hr_games == 10  # every even-indexed game
    assert l15.games == 15
    assert l10.games == 10
    assert l5.games == 5
    # Windows are the *tail* of the list - confirm l5's HR count matches
    # manually counting the real last 5 entries, not the first 5.
    last5_expected = sum(1 for got_hr, _, _ in outcomes[-5:] if got_hr)
    assert l5.hr_games == last5_expected


def test_clearance_window_none_when_no_games_played():
    l5, l10, l15, season = clearance_windows_from_outcomes([])
    assert l5 is None and l10 is None and l15 is None and season is None


def test_clearance_window_rate_properties():
    w = ClearanceWindow(games=10, hr_games=2, tb2_games=4, hit_games=7)
    assert w.hr_rate == 0.2
    assert w.tb2_rate == 0.4
    assert w.hit_rate == 0.7


def test_clearance_window_rate_is_none_with_zero_games():
    w = ClearanceWindow(games=0, hr_games=0, tb2_games=0, hit_games=0)
    assert w.hr_rate is None and w.tb2_rate is None and w.hit_rate is None


def test_clearance_counts_are_internally_consistent():
    # A HR game always counts as a 2+ TB game, which always counts as a hit
    # game - true both for real per-game classification and for a window
    # rollup built from it.
    games_events = [
        ["home_run"],
        ["double"],
        ["single"],
        ["strikeout"],
        ["walk", "field_out"],
    ]
    outcomes = game_outcomes_from_events(games_events)
    for got_hr, got_2tb, got_hit in outcomes:
        if got_hr:
            assert got_2tb and got_hit
        if got_2tb:
            assert got_hit

    _, _, _, season = clearance_windows_from_outcomes(outcomes)
    assert season.hr_games <= season.tb2_games <= season.hit_games <= season.games


def test_mock_provider_populates_clearance_windows():
    provider = MockHotStreakProvider(seed=7)
    heat = provider.get_heat_index("Any Player")
    for window in (heat.clear_l5, heat.clear_l10, heat.clear_l15, heat.clear_season):
        assert window is not None
        assert window.hr_games <= window.tb2_games <= window.hit_games <= window.games
    assert heat.clear_l5.games == 5
    assert heat.clear_l10.games == 10
    assert heat.clear_l15.games == 15
