"""Covers MlbStatsApiScheduleProvider's real-time `status` field on
ProbableMatchup - added so the pipeline can cross-check odds against MLB's
own authoritative game state (see pipeline.py's
_filter_lines_to_confirmed_pregame_games).
"""

from datetime import date

from mlb_props.schedule import MlbStatsApiScheduleProvider


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, schedule_payload, roster_payload=None):
        self.schedule_payload = schedule_payload
        self.roster_payload = roster_payload or {"roster": []}

    def get(self, url, params=None, timeout=None):
        if url.endswith("/schedule"):
            return _FakeResponse(self.schedule_payload)
        return _FakeResponse(self.roster_payload)


def _game(away="A", home="B", status="Pre-Game"):
    return {
        "teams": {
            "away": {"team": {"id": 1, "name": away}},
            "home": {"team": {"id": 2, "name": home}},
        },
        "venue": {"name": "Some Park"},
        "status": {"detailedState": status},
        "gameDate": "2026-08-29T23:16:00Z",
    }


def test_status_is_populated_from_the_real_api_field():
    payload = {"dates": [{"games": [_game(status="In Progress")]}]}
    provider = MlbStatsApiScheduleProvider(session=_FakeSession(payload), include_rosters=False)

    slate = provider.get_slate(date(2026, 8, 29))

    assert len(slate) == 1
    assert slate[0].status == "In Progress"


def test_doubleheader_produces_two_matchups_with_different_statuses():
    # Confirmed live (2026-08-29): MLB's schedule endpoint returns a
    # doubleheader as two separate `games` entries for the same two teams,
    # each with its own real status - unlike The Odds API, which exposed
    # only one event for the same real-world doubleheader.
    payload = {
        "dates": [
            {
                "games": [
                    _game(away="Boston Red Sox", home="New York Yankees", status="Final"),
                    _game(away="Boston Red Sox", home="New York Yankees", status="In Progress"),
                ]
            }
        ]
    }
    provider = MlbStatsApiScheduleProvider(session=_FakeSession(payload), include_rosters=False)

    slate = provider.get_slate(date(2026, 8, 29))

    assert len(slate) == 2
    statuses = {m.status for m in slate}
    assert statuses == {"Final", "In Progress"}


def test_missing_status_field_defaults_to_none_not_a_crash():
    game = _game()
    del game["status"]
    payload = {"dates": [{"games": [game]}]}
    provider = MlbStatsApiScheduleProvider(session=_FakeSession(payload), include_rosters=False)

    slate = provider.get_slate(date(2026, 8, 29))

    assert slate[0].status is None
