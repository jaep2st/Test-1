from odds_monitor.detector import find_discrepancies, find_odds_discrepancies
from odds_monitor.providers.mock import MockOddsProvider


def test_mock_provider_returns_lines_for_known_league():
    provider = MockOddsProvider(seed=1)
    lines = provider.fetch_player_props("nba")
    assert lines
    assert all(line.league == "nba" for line in lines)


def test_mock_provider_falls_back_to_nba_for_unknown_league():
    provider = MockOddsProvider(seed=1)
    lines = provider.fetch_player_props("some_unmapped_league")
    assert lines
    assert all(line.league == "some_unmapped_league" for line in lines)


def test_mock_provider_is_deterministic_with_seed():
    a = MockOddsProvider(seed=42).fetch_player_props("nba")
    b = MockOddsProvider(seed=42).fetch_player_props("nba")
    assert [(line.sportsbook, line.line) for line in a] == [(line.sportsbook, line.line) for line in b]


def test_mock_provider_can_trigger_discrepancies():
    provider = MockOddsProvider(seed=7, discrepancy_chance=1.0)
    lines = provider.fetch_player_props("nba")
    discrepancies = find_discrepancies(lines, min_spread=2.0)
    assert discrepancies


def test_mock_provider_mlb_returns_home_run_props():
    provider = MockOddsProvider(seed=3)
    lines = provider.fetch_player_props("mlb")
    assert lines
    assert all(line.league == "mlb" for line in lines)
    assert all(line.market == "player_home_runs" for line in lines)
    assert {line.side for line in lines} == {"yes", "no"}
    assert all(line.odds is not None for line in lines)


def test_mock_provider_mlb_can_trigger_odds_discrepancies():
    provider = MockOddsProvider(seed=11, discrepancy_chance=1.0)
    lines = provider.fetch_player_props("mlb")
    discrepancies = find_odds_discrepancies(lines, min_prob_spread=8.0)
    assert discrepancies
    # Home run props have no comparable line, so the line-based detector
    # should never flag anything here even with discrepancies forced.
    assert find_discrepancies(lines, min_spread=2.0) == []
