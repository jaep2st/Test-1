import pytest

from odds_monitor.providers.theoddsapi import OddsFetchFailed, TheOddsApiProvider


class _FakeResponse:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _UnauthorizedResponse:
    """Stands in for a real 401 from The Odds API - raise_for_status()
    raises, same as requests.Response does for a 4xx/5xx status.
    """

    def raise_for_status(self):
        raise Exception("401 Client Error: Unauthorized")

    def json(self):
        raise AssertionError("json() should never be reached - raise_for_status() should raise first")


class _FakeSession:
    """Stubs the two real-world calls TheOddsApiProvider makes: the events
    list, then one per-event odds fetch. Keyed by the request path so a
    single fake session can serve a whole `fetch_player_props` call. A
    response value of None (instead of a payload dict) serves an
    `_UnauthorizedResponse` for that path, to simulate a real API failure.
    """

    def __init__(self, responses):
        self.responses = responses
        self.requested_params = []

    def get(self, url, params=None, timeout=None):
        self.requested_params.append((url, params))
        for path_suffix, payload in self.responses.items():
            if url.endswith(path_suffix):
                return _UnauthorizedResponse() if payload is None else _FakeResponse(payload)
        raise AssertionError(f"Unexpected URL requested: {url}")


def _provider(responses):
    return TheOddsApiProvider(api_key="test-key", session=_FakeSession(responses))


def test_parses_home_run_and_total_bases_outcomes():
    events_payload = [{"id": "evt1", "home_team": "New York Yankees", "away_team": "Houston Astros"}]
    odds_payload = {
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "batter_home_runs",
                        "outcomes": [
                            {"name": "Yes", "description": "Aaron Judge", "price": 350},
                            {"name": "No", "description": "Aaron Judge", "price": -450},
                        ],
                    },
                    {
                        "key": "batter_total_bases",
                        "outcomes": [
                            {"name": "Over", "description": "Aaron Judge", "point": 1.5, "price": -130},
                            {"name": "Under", "description": "Aaron Judge", "point": 1.5, "price": 110},
                        ],
                    },
                ],
            }
        ]
    }
    provider = _provider({"/events": events_payload, "/evt1/odds": odds_payload})

    lines = provider.fetch_player_props("mlb")

    assert len(lines) == 4
    hr_yes = next(l for l in lines if l.market == "batter_home_runs" and l.side == "yes")
    assert hr_yes.player == "Aaron Judge"
    assert hr_yes.odds == 350
    assert hr_yes.event == "Houston Astros @ New York Yankees"
    assert hr_yes.sportsbook == "draftkings"

    tb_over = next(l for l in lines if l.market == "batter_total_bases" and l.side == "over")
    assert tb_over.line == 1.5
    assert tb_over.odds == -130


def test_over_under_home_run_outcomes_normalize_to_yes_no():
    events_payload = [{"id": "evt1", "home_team": "A", "away_team": "B"}]
    odds_payload = {
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "batter_home_runs",
                        "outcomes": [
                            {"name": "Over", "description": "Some Player", "point": 0.5, "price": 400},
                            {"name": "Under", "description": "Some Player", "point": 0.5, "price": -550},
                        ],
                    }
                ],
            }
        ]
    }
    provider = _provider({"/events": events_payload, "/evt1/odds": odds_payload})

    lines = provider.fetch_player_props("mlb")

    sides = {l.side for l in lines}
    assert sides == {"yes", "no"}


def test_unparsable_outcomes_are_skipped_not_raised():
    events_payload = [{"id": "evt1", "home_team": "A", "away_team": "B"}]
    odds_payload = {
        "bookmakers": [
            {
                "key": "betmgm",
                "markets": [
                    {
                        "key": "batter_home_runs",
                        "outcomes": [
                            {"name": "Yes", "price": 300},  # missing "description" (player name)
                            {"name": "Weird", "description": "Player X", "price": 200},  # unrecognized side
                            {"name": "Yes", "description": "Player Y", "price": 250},  # valid
                        ],
                    }
                ],
            }
        ]
    }
    provider = _provider({"/events": events_payload, "/evt1/odds": odds_payload})

    lines = provider.fetch_player_props("mlb")

    assert len(lines) == 1
    assert lines[0].player == "Player Y"


def test_unknown_league_returns_no_lines_without_network_call():
    provider = _provider({})
    assert provider.fetch_player_props("nba") == []


def test_missing_api_key_raises_value_error(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    try:
        TheOddsApiProvider(session=_FakeSession({}))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_all_events_failing_raises_odds_fetch_failed():
    # Confirmed live (2026-08-29): a real run hit 401 on all 17 per-event
    # odds calls in a row while /events itself succeeded - a systemic
    # problem (quota/auth), not an empty market. That distinction is what
    # lets a caller fall back to a second odds provider instead of quietly
    # reporting "no props" - see odds_monitor/providers/fallback.py.
    events_payload = [
        {"id": "evt1", "home_team": "A", "away_team": "B"},
        {"id": "evt2", "home_team": "C", "away_team": "D"},
    ]
    provider = _provider({"/events": events_payload, "/evt1/odds": None, "/evt2/odds": None})

    with pytest.raises(OddsFetchFailed):
        provider.fetch_player_props("mlb")


def test_events_list_itself_failing_raises_odds_fetch_failed():
    provider = _provider({"/events": None})

    with pytest.raises(OddsFetchFailed):
        provider.fetch_player_props("mlb")


def test_a_genuinely_empty_slate_does_not_raise():
    # No games today is not a failure - distinct from every event's odds
    # call failing.
    provider = _provider({"/events": []})
    assert provider.fetch_player_props("mlb") == []


def test_partial_per_event_failure_still_returns_what_succeeded():
    events_payload = [
        {"id": "evt1", "home_team": "A", "away_team": "B"},
        {"id": "evt2", "home_team": "C", "away_team": "D"},
    ]
    odds_payload = {
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "batter_home_runs",
                        "outcomes": [{"name": "Yes", "description": "Player X", "price": 300}],
                    }
                ],
            }
        ]
    }
    # evt1 succeeds, evt2 fails - a mix, not total failure, should not raise.
    provider = _provider({"/events": events_payload, "/evt1/odds": odds_payload, "/evt2/odds": None})

    lines = provider.fetch_player_props("mlb")

    assert len(lines) == 1
    assert lines[0].player == "Player X"


def test_api_key_whitespace_and_newlines_are_stripped():
    # Same footgun confirmed live on BETSTAMP_API_KEY (2026-08-29): a
    # secret pasted with stray surrounding whitespace/newlines. This
    # provider sends the key as a query param rather than a header, so it
    # wouldn't raise the same InvalidHeader error, but an un-stripped
    # trailing newline would still corrupt the key value.
    provider = TheOddsApiProvider(api_key="  test-key\n\n", session=_FakeSession({}))
    assert provider.api_key == "test-key"


def test_api_key_from_env_is_also_stripped(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "env-key\n")
    provider = TheOddsApiProvider(session=_FakeSession({}))
    assert provider.api_key == "env-key"


class _QuotaHeaderSession:
    """Stubs just the free /events call, with the quota headers The Odds
    API attaches to every response (success or failure).
    """

    def get(self, url, params=None, timeout=None):
        return _FakeResponse([], headers={"x-requests-remaining": "437", "x-requests-used": "63"})


def test_quota_headers_are_logged_on_the_free_events_call(caplog):
    # Confirmed live (2026-08-29): after a run exhausted the free tier, the
    # only way to see the actual remaining-credit count was logging into
    # the dashboard - the free /events call gets hit every run regardless,
    # so logging these headers here turns it into a built-in quota check.
    import logging

    provider = TheOddsApiProvider(api_key="test-key", session=_QuotaHeaderSession())

    with caplog.at_level(logging.INFO, logger="odds_monitor.providers.theoddsapi"):
        provider.fetch_player_props("mlb")

    assert any("used=63 remaining=437" in r.message for r in caplog.records)
