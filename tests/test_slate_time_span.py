"""Covers _log_slate_time_span: real per-game start times (already fetched
via MLB Stats API's gameDate, just not previously surfaced anywhere) are
needed to time things like "an hour before first pitch" or "the middle of
the slate," which vary daily and can't be expressed as a fixed cron time
without knowing them.
"""

import logging

from mlb_props.pipeline import _log_slate_time_span
from mlb_props.schedule import ProbableMatchup


def _game(game_time_utc=None, away="Away", home="Home"):
    return ProbableMatchup(
        away_team=away, home_team=home, venue="Some Park",
        away_pitcher="P1", home_pitcher="P2", game_time_utc=game_time_utc,
    )


def test_logs_earliest_and_latest_game_time(caplog):
    slate = [
        _game("2026-08-27T23:05:00Z", away="Late Away", home="Late Home"),
        _game("2026-08-27T17:10:00Z", away="Early Away", home="Early Home"),
        _game("2026-08-27T20:00:00Z"),
    ]
    with caplog.at_level(logging.INFO, logger="mlb_props.pipeline"):
        _log_slate_time_span(slate)

    [record] = [r for r in caplog.records if "SLATE_TIME_SPAN" in r.message]
    assert "2026-08-27T17:10:00+00:00" in record.message
    assert "2026-08-27T23:05:00+00:00" in record.message
    assert "games_with_times=3/3" in record.message


def test_missing_game_times_logs_a_warning_not_a_crash(caplog):
    slate = [_game(None), _game(None)]
    with caplog.at_level(logging.WARNING, logger="mlb_props.pipeline"):
        _log_slate_time_span(slate)
    assert any("no parseable game times" in r.message for r in caplog.records)


def test_unparsable_game_time_is_skipped_not_raised(caplog):
    slate = [_game("not-a-real-timestamp"), _game("2026-08-27T17:10:00Z")]
    with caplog.at_level(logging.INFO, logger="mlb_props.pipeline"):
        _log_slate_time_span(slate)
    [record] = [r for r in caplog.records if "SLATE_TIME_SPAN" in r.message]
    assert "games_with_times=1/2" in record.message
