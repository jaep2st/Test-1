"""Covers MlbStatsApiScheduleProvider's real-time `status` field on
ProbableMatchup - added so the pipeline can cross-check odds against MLB's
own authoritative game state (see pipeline.py's
_filter_lines_to_confirmed_pregame_games) - and its real confirmed-starting-
lineup fetch (see ProbableMatchup.lineup_source's docstring).
"""

from datetime import date, datetime, timedelta, timezone

from mlb_props.schedule import MlbStatsApiScheduleProvider


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, schedule_payload, roster_payload=None, boxscore_payload=None, boxscore_should_fail=False):
        self.schedule_payload = schedule_payload
        self.roster_payload = roster_payload or {"roster": []}
        self.boxscore_payload = boxscore_payload or {"teams": {"away": {"players": {}}, "home": {"players": {}}}}
        self.boxscore_should_fail = boxscore_should_fail
        self.boxscore_calls = 0

    def get(self, url, params=None, timeout=None):
        if url.endswith("/schedule"):
            return _FakeResponse(self.schedule_payload)
        if "/boxscore" in url:
            self.boxscore_calls += 1
            if self.boxscore_should_fail:
                raise RuntimeError("boxscore fetch failed")
            return _FakeResponse(self.boxscore_payload)
        return _FakeResponse(self.roster_payload)


def _game(away="A", home="B", status="Pre-Game", game_pk=100, game_time_utc="2026-08-29T23:16:00Z"):
    game = {
        "teams": {
            "away": {"team": {"id": 1, "name": away}},
            "home": {"team": {"id": 2, "name": home}},
        },
        "venue": {"name": "Some Park"},
        "status": {"detailedState": status},
        "gameDate": game_time_utc,
    }
    if game_pk is not None:
        game["gamePk"] = game_pk
    return game


def _soon_iso(hours=1.0):
    """A real ISO8601 UTC timestamp `hours` from now, "Z"-suffixed like
    MLB Stats API's own `gameDate` field."""
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


def _confirmed_boxscore_payload(away_starters, home_starters):
    """A real confirmed-lineup boxscore shape: each starter's entry has a
    real `battingOrder`; see MlbStatsApiScheduleProvider._confirmed_lineup_batters.
    """

    def _players(names):
        return {f"ID{i}": {"battingOrder": str(100 + i * 100), "person": {"fullName": name}} for i, name in enumerate(names)}

    return {"teams": {"away": {"players": _players(away_starters)}, "home": {"players": _players(home_starters)}}}


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


def test_confirmed_lineup_replaces_active_roster_when_posted_and_close_to_first_pitch():
    payload = {"dates": [{"games": [_game(game_time_utc=_soon_iso(1))]}]}
    boxscore = _confirmed_boxscore_payload(["Away Starter"], ["Home Starter"])
    provider = MlbStatsApiScheduleProvider(session=_FakeSession(payload, boxscore_payload=boxscore), include_rosters=True)

    slate = provider.get_slate(date(2026, 8, 29))

    assert len(slate) == 1
    assert slate[0].lineup_source == "confirmed"
    assert slate[0].away_batters == ["Away Starter"]
    assert slate[0].home_batters == ["Home Starter"]


def test_falls_back_to_active_roster_when_lineup_not_posted_yet():
    # Real boxscore shape, but no player anywhere has a real battingOrder
    # yet - the honest "not confirmed yet" case, not a fetch failure.
    payload = {"dates": [{"games": [_game(game_time_utc=_soon_iso(1))]}]}
    roster_payload = {"roster": [{"position": {"type": "Outfielder"}, "person": {"fullName": "Active Roster Guy"}}]}
    session = _FakeSession(payload, roster_payload=roster_payload)
    provider = MlbStatsApiScheduleProvider(session=session, include_rosters=True)

    slate = provider.get_slate(date(2026, 8, 29))

    assert slate[0].lineup_source == "active_roster"
    assert slate[0].away_batters == ["Active Roster Guy"]


def test_falls_back_to_active_roster_when_the_boxscore_fetch_fails():
    payload = {"dates": [{"games": [_game(game_time_utc=_soon_iso(1))]}]}
    roster_payload = {"roster": [{"position": {"type": "Outfielder"}, "person": {"fullName": "Active Roster Guy"}}]}
    session = _FakeSession(payload, roster_payload=roster_payload, boxscore_should_fail=True)
    provider = MlbStatsApiScheduleProvider(session=session, include_rosters=True)

    slate = provider.get_slate(date(2026, 8, 29))

    assert slate[0].lineup_source == "active_roster"
    assert slate[0].away_batters == ["Active Roster Guy"]


def test_never_tries_confirmed_lineup_when_the_game_is_far_from_first_pitch():
    # LINEUP_FETCH_WINDOW_HOURS is 5.0 - 12 hours out is well outside it,
    # so the boxscore endpoint should never even be called (MLB hasn't
    # posted a real lineup that early, so it'd be a wasted real request
    # every single run).
    payload = {"dates": [{"games": [_game(game_time_utc=_soon_iso(12))]}]}
    roster_payload = {"roster": [{"position": {"type": "Outfielder"}, "person": {"fullName": "Active Roster Guy"}}]}
    session = _FakeSession(payload, roster_payload=roster_payload)
    provider = MlbStatsApiScheduleProvider(session=session, include_rosters=True)

    slate = provider.get_slate(date(2026, 8, 29))

    assert session.boxscore_calls == 0
    assert slate[0].lineup_source == "active_roster"


def test_never_tries_confirmed_lineup_with_no_real_game_pk():
    game = _game(game_time_utc=_soon_iso(1), game_pk=None)
    payload = {"dates": [{"games": [game]}]}
    roster_payload = {"roster": [{"position": {"type": "Outfielder"}, "person": {"fullName": "Active Roster Guy"}}]}
    session = _FakeSession(payload, roster_payload=roster_payload)
    provider = MlbStatsApiScheduleProvider(session=session, include_rosters=True)

    slate = provider.get_slate(date(2026, 8, 29))

    assert session.boxscore_calls == 0
    assert slate[0].lineup_source == "active_roster"
