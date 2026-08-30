"""Covers FallbackOddsProvider: added after a real run (2026-08-29) hit 401
on every event-odds request against The Odds API - likely a free-tier
credit quota exhausted by repeated same-day runs - and degraded the whole
report to model-only rankings even though a second, working odds provider
(Betstamp) was already configured and simply never tried.
"""

import pytest

from odds_monitor.models import PropLine
from odds_monitor.providers.fallback import FallbackOddsProvider
from odds_monitor.providers.theoddsapi import OddsFetchFailed


def _line(sportsbook: str) -> PropLine:
    return PropLine(
        player="Test Player",
        team=None,
        league="mlb",
        market="batter_total_bases",
        side="over",
        line=1.5,
        odds=-120,
        sportsbook=sportsbook,
        event="Away @ Home",
    )


class _FailingProvider:
    def __init__(self, exc=None):
        self.exc = exc or OddsFetchFailed("simulated systemic failure")
        self.calls = 0

    def fetch_player_props(self, league):
        self.calls += 1
        raise self.exc


class _WorkingProvider:
    def __init__(self, lines):
        self.lines = lines
        self.calls = 0

    def fetch_player_props(self, league):
        self.calls += 1
        return self.lines


def test_falls_back_to_secondary_when_primary_fails_systemically():
    secondary_lines = [_line("betstamp_book")]
    primary = _FailingProvider()
    secondary = _WorkingProvider(secondary_lines)

    lines = FallbackOddsProvider(primary, secondary).fetch_player_props("mlb")

    assert primary.calls == 1
    assert secondary.calls == 1
    assert lines is secondary_lines


def test_does_not_touch_secondary_when_primary_succeeds():
    primary_lines = [_line("draftkings")]
    primary = _WorkingProvider(primary_lines)
    secondary = _WorkingProvider([_line("betstamp_book")])

    lines = FallbackOddsProvider(primary, secondary).fetch_player_props("mlb")

    assert primary.calls == 1
    assert secondary.calls == 0
    assert lines is primary_lines


def test_does_not_touch_secondary_when_primary_legitimately_empty():
    # An empty slate (no games today) is not a failure - the primary
    # provider fetched successfully and found nothing, so the fallback
    # should not engage.
    primary = _WorkingProvider([])
    secondary = _WorkingProvider([_line("betstamp_book")])

    lines = FallbackOddsProvider(primary, secondary).fetch_player_props("mlb")

    assert primary.calls == 1
    assert secondary.calls == 0
    assert lines == []


def test_non_systemic_exception_from_primary_propagates_without_fallback():
    primary = _FailingProvider(exc=ValueError("some other kind of bug"))
    secondary = _WorkingProvider([_line("betstamp_book")])

    with pytest.raises(ValueError):
        FallbackOddsProvider(primary, secondary).fetch_player_props("mlb")

    assert secondary.calls == 0
