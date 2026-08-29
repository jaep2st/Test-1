"""Covers _parse_date's "today" resolution: confirmed live that a manual run
at 2026-08-28 20:22 ET (2026-08-29 00:22 UTC) pulled the next day's slate
instead of that evening's, because the naive `date.today()` it used to call
inherits the runner's OS timezone (UTC on GitHub Actions) - which rolls to
the next calendar date hours before a US Eastern evening slate is over.
"""

from datetime import date as date_cls
from datetime import datetime, timezone

import mlb_props_main as mm


def test_today_resolves_to_us_eastern_date_not_the_runner_utc_date(monkeypatch):
    # 2026-08-29 00:22 UTC == 2026-08-28 20:22 America/New_York (EDT, UTC-4).
    # A naive date.today() on a UTC-clocked machine at this instant returns
    # 2026-08-29 - the wrong slate, one day ahead of the actual US evening.
    fixed_utc = datetime(2026, 8, 29, 0, 22, tzinfo=timezone.utc)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_utc.astimezone(tz) if tz else fixed_utc

    monkeypatch.setattr(mm, "datetime", FrozenDateTime)

    assert mm._parse_date("today") == date_cls(2026, 8, 28)


def test_explicit_date_is_unaffected_by_timezone_handling():
    assert mm._parse_date("2026-08-29") == date_cls(2026, 8, 29)
