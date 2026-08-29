"""Covers BetstampProvider, including the base URL confirmed live
(2026-08-29) against Betstamp's own published API docs page:
`https://api.pro.betstamp.com` (a `.pro` subdomain, not the previously
guessed `https://api.betstamp.com`), and the confirmed top-level response
shape `{"markets": [...]}`.
"""

from odds_monitor.providers.betstamp import DEFAULT_BASE_URL, BetstampProvider


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.requested = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.requested.append((url, headers, params))
        return _FakeResponse(self.payload)


def test_default_base_url_matches_betstamps_published_docs():
    # Confirmed live via Betstamp's own API docs page - see this module's
    # docstring. A regression here would silently point requests at the
    # wrong host again.
    assert DEFAULT_BASE_URL == "https://api.pro.betstamp.com"


def test_requests_confirmed_endpoint_with_api_key_header():
    session = _FakeSession({"markets": []})
    provider = BetstampProvider(api_key="test-key", session=session)

    provider.fetch_player_props("mlb")

    url, headers, params = session.requested[0]
    assert url == "https://api.pro.betstamp.com/api/markets"
    assert headers["X-API-KEY"] == "test-key"
    assert params["league"] == "mlb"


def test_parses_markets_from_confirmed_response_shape():
    payload = {
        "markets": [
            {
                "player": "Aaron Judge",
                "team": "NYY",
                "bet_type": "batter_home_runs",
                "side": "Yes",
                "line": 0.5,
                "odds": 350,
                "book": "draftkings",
                "event": "Houston Astros @ New York Yankees",
            }
        ]
    }
    provider = BetstampProvider(api_key="test-key", session=_FakeSession(payload))

    lines = provider.fetch_player_props("mlb")

    assert len(lines) == 1
    line = lines[0]
    assert line.player == "Aaron Judge"
    assert line.market == "batter_home_runs"
    assert line.side == "yes"
    assert line.odds == 350
    assert line.sportsbook == "draftkings"
    assert line.event == "Houston Astros @ New York Yankees"


def test_market_entry_missing_required_fields_is_skipped_not_raised():
    payload = {
        "markets": [
            {"player": "Player X", "bet_type": "batter_home_runs"},  # missing side/line/book/event
            {
                "player": "Player Y",
                "bet_type": "batter_home_runs",
                "side": "Yes",
                "line": 0.5,
                "odds": 250,
                "book": "fanduel",
                "event": "A @ B",
            },
        ]
    }
    provider = BetstampProvider(api_key="test-key", session=_FakeSession(payload))

    lines = provider.fetch_player_props("mlb")

    assert len(lines) == 1
    assert lines[0].player == "Player Y"


def test_missing_api_key_raises_value_error(monkeypatch):
    monkeypatch.delenv("BETSTAMP_API_KEY", raising=False)
    try:
        BetstampProvider(session=_FakeSession({"markets": []}))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_api_key_whitespace_and_newlines_are_stripped():
    # Confirmed live (2026-08-29): a BETSTAMP_API_KEY secret pasted with
    # stray surrounding whitespace/newlines produces a header value
    # `requests` rejects outright (InvalidHeader), so the request never
    # even gets sent. Guard against that footgun rather than just the
    # cleanest possible input.
    session = _FakeSession({"markets": []})
    provider = BetstampProvider(api_key="  test-key\n\n", session=session)

    assert provider.api_key == "test-key"

    provider.fetch_player_props("mlb")
    _, headers, _ = session.requested[0]
    assert headers["X-API-KEY"] == "test-key"


def test_api_key_from_env_is_also_stripped(monkeypatch):
    monkeypatch.setenv("BETSTAMP_API_KEY", "env-key\n")
    provider = BetstampProvider(session=_FakeSession({"markets": []}))
    assert provider.api_key == "env-key"
