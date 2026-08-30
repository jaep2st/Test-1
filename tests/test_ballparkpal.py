"""Covers mlb_props/ballparkpal.py against the real response shape
documented on Ballpark Pal's own API docs page (read via screenshots,
2026-08-30 - see that module's docstring for why it wasn't fetched
programmatically)."""

from datetime import date

from mlb_props.ballparkpal import (
    LiveBallparkPalProvider,
    MockBallparkPalProvider,
    NoBallparkPalProvider,
)


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        return _FakeResponse(self.payload)


def _real_payload():
    # Real confirmed shape (live 2026-08-30): data.items, not a bare list.
    return {
        "meta": {"asOf": "2026-08-30T12:00:00Z", "count": 2},
        "data": {"items": [
            {
                "gameId": 776345,
                "gameTime": "19:10",
                "teamAway": "PHI",
                "teamHome": "LAA",
                "playerId": 545361,
                "playerName": "Mike Trout",
                "team": "LAA",
                "homeRuns": 1.08,
                "doublesTriples": 1.05,
                "singles": 1.02,
                "homeRunsStadium": 1.06,
                "doublesTriplesStadium": 1.04,
                "singlesStadium": 1.01,
                "homeRunsWeather": 0.02,
                "doublesTriplesWeather": 0.01,
                "singlesWeather": 0.01,
            },
            {
                "gameId": 776345,
                "gameTime": "19:10",
                "teamAway": "PHI",
                "teamHome": "LAA",
                "playerId": 592885,
                "playerName": "Bryce Harper",
                "team": "PHI",
                "homeRuns": 0.94,
                "doublesTriples": 0.97,
                "singles": 0.99,
                "homeRunsStadium": None,
                "doublesTriplesStadium": None,
                "singlesStadium": None,
                "homeRunsWeather": None,
                "doublesTriplesWeather": None,
                "singlesWeather": None,
            },
        ]},
    }


def test_live_provider_parses_a_real_response_and_sends_the_api_key_header():
    session = _FakeSession(_real_payload())
    provider = LiveBallparkPalProvider(api_key="bpp_live_test", session=session)

    factor = provider.get_hitter_park_factor("Mike Trout", date(2026, 8, 30))

    assert factor is not None
    assert factor.home_runs == 1.08
    assert factor.home_runs_stadium == 1.06
    assert factor.home_runs_weather == 0.02

    url, params, headers = session.calls[0]
    assert url.endswith("/parkfactors/hitters")
    assert params == {"date": "2026-08-30"}
    assert headers == {"X-API-Key": "bpp_live_test"}


def test_live_provider_lookup_is_case_and_whitespace_insensitive():
    session = _FakeSession(_real_payload())
    provider = LiveBallparkPalProvider(api_key="k", session=session)

    factor = provider.get_hitter_park_factor("  mike TROUT  ", date(2026, 8, 30))

    assert factor is not None
    assert factor.player_name == "Mike Trout"


def test_live_provider_returns_none_fields_as_none_not_a_crash():
    session = _FakeSession(_real_payload())
    provider = LiveBallparkPalProvider(api_key="k", session=session)

    factor = provider.get_hitter_park_factor("Bryce Harper", date(2026, 8, 30))

    assert factor is not None
    assert factor.home_runs == 0.94
    assert factor.home_runs_stadium is None
    assert factor.home_runs_weather is None


def test_live_provider_caches_one_fetch_per_date():
    session = _FakeSession(_real_payload())
    provider = LiveBallparkPalProvider(api_key="k", session=session)

    provider.get_hitter_park_factor("Mike Trout", date(2026, 8, 30))
    provider.get_hitter_park_factor("Bryce Harper", date(2026, 8, 30))
    provider.get_hitter_park_factor("Mike Trout", date(2026, 8, 30))

    assert len(session.calls) == 1


def test_live_provider_unknown_player_returns_none():
    session = _FakeSession(_real_payload())
    provider = LiveBallparkPalProvider(api_key="k", session=session)

    assert provider.get_hitter_park_factor("Nobody Real", date(2026, 8, 30)) is None


def test_live_provider_fetch_failure_returns_none_not_raises():
    class _RaisingSession:
        def get(self, *a, **k):
            raise ConnectionError("boom")

    provider = LiveBallparkPalProvider(api_key="k", session=_RaisingSession())

    assert provider.get_hitter_park_factor("Mike Trout", date(2026, 8, 30)) is None


def test_live_provider_unexpected_response_shape_returns_none_not_raises():
    session = _FakeSession({"meta": {}, "data": "not a list"})
    provider = LiveBallparkPalProvider(api_key="k", session=session)

    assert provider.get_hitter_park_factor("Mike Trout", date(2026, 8, 30)) is None


def test_live_provider_also_accepts_a_bare_list_under_data():
    # Not the real confirmed shape (see _real_payload's comment - real is
    # data.items), but kept as a fallback in case a different endpoint or a
    # future API version nests differently - this proves that fallback
    # path still works, not just the "items" one.
    payload = {
        "meta": {},
        "data": [{"playerName": "Mike Trout", "homeRuns": 1.1, "homeRunsStadium": 1.05, "homeRunsWeather": 0.05}],
    }
    session = _FakeSession(payload)
    provider = LiveBallparkPalProvider(api_key="k", session=session)

    factor = provider.get_hitter_park_factor("Mike Trout", date(2026, 8, 30))
    assert factor is not None
    assert factor.home_runs == 1.1


def test_live_provider_skips_a_row_missing_playername_without_dropping_the_rest():
    payload = {
        "meta": {},
        "data": [
            {"playerName": "Mike Trout", "homeRuns": 1.1, "homeRunsStadium": 1.1, "homeRunsWeather": 0.0},
            {"homeRuns": 1.2},  # missing playerName
        ],
    }
    session = _FakeSession(payload)
    provider = LiveBallparkPalProvider(api_key="k", session=session)

    assert provider.get_hitter_park_factor("Mike Trout", date(2026, 8, 30)) is not None


def test_no_provider_always_returns_none():
    provider = NoBallparkPalProvider()
    assert provider.get_hitter_park_factor("Mike Trout", date(2026, 8, 30)) is None


def test_mock_provider_is_deterministic_per_seed():
    a = MockBallparkPalProvider(seed=1).get_hitter_park_factor("Mike Trout", date(2026, 8, 30))
    b = MockBallparkPalProvider(seed=1).get_hitter_park_factor("Mike Trout", date(2026, 8, 30))
    assert a == b


def test_mock_provider_home_runs_equals_stadium_plus_weather():
    factor = MockBallparkPalProvider(seed=2).get_hitter_park_factor("Mike Trout", date(2026, 8, 30))
    assert round(factor.home_runs_stadium + factor.home_runs_weather, 3) == factor.home_runs
