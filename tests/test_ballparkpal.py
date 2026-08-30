"""Covers mlb_props/ballparkpal.py against the real response shape
documented on Ballpark Pal's own API docs page (read via screenshots,
2026-08-30 - see that module's docstring for why it wasn't fetched
programmatically)."""

from datetime import date

from mlb_props.ballparkpal import (
    LiveBallparkPalProvider,
    MockBallparkPalProvider,
    NoBallparkPalProvider,
    _per_pa_to_per_game,
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


# --- /api/v1/matchups (confirmed live 2026-08-30 to be per-plate-appearance,
# not per-game - see ballparkpal.py's module docstring) -----------------


def test_per_pa_to_per_game_matches_the_documented_formula():
    # P(at least one in n independent trials) = 1 - (1-p)^n, n=4.3.
    assert round(_per_pa_to_per_game(24.4), 4) == round(1 - (1 - 0.244) ** 4.3, 4)


def test_per_pa_to_per_game_zero_stays_zero():
    assert _per_pa_to_per_game(0.0) == 0.0


def test_per_pa_to_per_game_is_larger_than_the_input_per_pa_rate():
    # The whole point: a per-game probability (multiple chances) must be
    # >= the single-PA rate it's built from.
    assert _per_pa_to_per_game(22.0) > 0.22


def _matchups_payload():
    # Real confirmed shape and field names (live 2026-08-30, via
    # mlb_props_main.run_ballparkpal_matchups_check): strikeoutProbability
    # averaged 24.4% that run - matches real MLB's ~22% per-PA K rate.
    return {
        "meta": {"asOf": "2026-08-30T13:54:00Z", "count": 2},
        "data": [
            {
                "gameId": 822700,
                "batterId": 545361,
                "batterName": "Mike Trout",
                "batterTeam": "LAA",
                "pitcherId": 123456,
                "pitcherName": "Cristopher Sanchez",
                "pitcherTeam": "PHI",
                "homeRunProbability": 3.2,
                "doubleTripleProbability": 5.1,
                "singleProbability": 14.6,
                "walkProbability": 9.8,
                "strikeoutProbability": 22.1,
            },
            {
                "gameId": 822700,
                "batterId": 592885,
                "batterName": "Bryce Harper",
                "batterTeam": "PHI",
                "pitcherId": 654321,
                "pitcherName": "Someone Else",
                "pitcherTeam": "LAA",
                "homeRunProbability": None,
                "doubleTripleProbability": 4.0,
                "singleProbability": None,
                "walkProbability": 8.0,
                "strikeoutProbability": 20.0,
            },
        ],
    }


def test_get_matchup_probability_converts_real_per_pa_values_to_per_game():
    session = _FakeSession(_matchups_payload())
    provider = LiveBallparkPalProvider(api_key="k", session=session)

    result = provider.get_matchup_probability("Mike Trout", "Cristopher Sanchez", date(2026, 8, 30))

    assert result is not None
    expected_hr = _per_pa_to_per_game(3.2)
    expected_hits = _per_pa_to_per_game(3.2 + 14.6 + 5.1)
    assert result.home_run_model_prob == round(expected_hr, 4)
    assert result.hits_model_prob == round(expected_hits, 4)
    # Sanity: per-game must exceed the raw per-PA fraction, and stay a
    # real probability.
    assert 0.032 < result.home_run_model_prob < 1.0
    assert 0.0 < result.hits_model_prob < 1.0


def test_get_matchup_probability_lookup_is_case_and_whitespace_insensitive():
    session = _FakeSession(_matchups_payload())
    provider = LiveBallparkPalProvider(api_key="k", session=session)

    result = provider.get_matchup_probability("  MIKE trout ", " cristopher SANCHEZ  ", date(2026, 8, 30))

    assert result is not None
    assert result.batter_name == "Mike Trout"


def test_get_matchup_probability_handles_partial_nulls_without_crashing():
    # Bryce Harper's row: homeRunProbability and singleProbability are both
    # None - home_run_model_prob must be None (can't convert None), and
    # hits_model_prob must also be None (can't sum with a missing term)
    # rather than silently treating a missing field as 0%.
    session = _FakeSession(_matchups_payload())
    provider = LiveBallparkPalProvider(api_key="k", session=session)

    result = provider.get_matchup_probability("Bryce Harper", "Someone Else", date(2026, 8, 30))

    assert result is not None
    assert result.home_run_model_prob is None
    assert result.hits_model_prob is None


def test_get_matchup_probability_unknown_pair_returns_none():
    session = _FakeSession(_matchups_payload())
    provider = LiveBallparkPalProvider(api_key="k", session=session)

    assert provider.get_matchup_probability("Nobody Real", "Also Nobody", date(2026, 8, 30)) is None


def test_get_matchup_probability_caches_one_fetch_per_date():
    session = _FakeSession(_matchups_payload())
    provider = LiveBallparkPalProvider(api_key="k", session=session)

    provider.get_matchup_probability("Mike Trout", "Cristopher Sanchez", date(2026, 8, 30))
    provider.get_matchup_probability("Bryce Harper", "Someone Else", date(2026, 8, 30))

    assert len(session.calls) == 1
    _, params, headers = session.calls[0]
    assert params == {"date": "2026-08-30", "parkAdjusted": "true"}
    assert headers == {"X-API-Key": "k"}


def test_get_matchup_probability_fetch_failure_returns_none_not_raises():
    class _RaisingSession:
        def get(self, *a, **k):
            raise ConnectionError("boom")

    provider = LiveBallparkPalProvider(api_key="k", session=_RaisingSession())

    assert provider.get_matchup_probability("Mike Trout", "Cristopher Sanchez", date(2026, 8, 30)) is None


def test_no_provider_matchup_probability_returns_none():
    assert NoBallparkPalProvider().get_matchup_probability("Mike Trout", "X", date(2026, 8, 30)) is None


def test_mock_provider_matchup_probability_is_deterministic_and_in_range():
    a = MockBallparkPalProvider(seed=3).get_matchup_probability("Mike Trout", "X", date(2026, 8, 30))
    b = MockBallparkPalProvider(seed=3).get_matchup_probability("Mike Trout", "X", date(2026, 8, 30))
    assert a == b
    assert 0.0 < a.home_run_model_prob < 1.0
    assert 0.0 < a.hits_model_prob < 1.0
