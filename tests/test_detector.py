from odds_monitor.detector import find_discrepancies
from odds_monitor.models import PropLine


def _line(**overrides):
    base = dict(
        player="Test Player",
        team="TST",
        league="nba",
        market="player_points",
        side="over",
        line=20.0,
        odds=-110,
        sportsbook="draftkings",
        event="TST @ OPP",
    )
    base.update(overrides)
    return PropLine(**base)


def test_flags_spread_at_or_above_threshold():
    lines = [_line(sportsbook="draftkings", line=20.0), _line(sportsbook="fanduel", line=22.0)]
    result = find_discrepancies(lines, min_spread=2.0)
    assert len(result) == 1
    assert result[0].spread == 2.0
    assert result[0].low.sportsbook == "draftkings"
    assert result[0].high.sportsbook == "fanduel"


def test_ignores_spread_below_threshold():
    lines = [_line(sportsbook="draftkings", line=20.0), _line(sportsbook="fanduel", line=21.5)]
    assert find_discrepancies(lines, min_spread=2.0) == []


def test_groups_are_independent_by_player_market_side_event():
    lines = [
        _line(player="A", line=20.0, sportsbook="dk"),
        _line(player="A", line=25.0, sportsbook="fd"),
        _line(player="B", line=10.0, sportsbook="dk"),
        _line(player="B", line=10.5, sportsbook="fd"),
    ]
    result = find_discrepancies(lines, min_spread=2.0)
    assert len(result) == 1
    assert result[0].player == "A"


def test_single_book_never_flagged():
    lines = [_line(sportsbook="draftkings", line=20.0)]
    assert find_discrepancies(lines, min_spread=2.0) == []


def test_picks_widest_spread_when_multiple_books():
    lines = [
        _line(sportsbook="dk", line=18.0),
        _line(sportsbook="fd", line=20.5),
        _line(sportsbook="mgm", line=22.0),
    ]
    result = find_discrepancies(lines, min_spread=2.0)
    assert len(result) == 1
    assert result[0].spread == 4.0


def test_different_sides_are_not_compared_to_each_other():
    lines = [_line(side="over", line=20.0), _line(side="under", line=25.0)]
    assert find_discrepancies(lines, min_spread=2.0) == []


def test_results_sorted_largest_spread_first():
    lines = [
        _line(player="A", line=20.0, sportsbook="dk"),
        _line(player="A", line=22.0, sportsbook="fd"),
        _line(player="B", line=20.0, sportsbook="dk"),
        _line(player="B", line=26.0, sportsbook="fd"),
    ]
    result = find_discrepancies(lines, min_spread=2.0)
    assert [d.player for d in result] == ["B", "A"]
