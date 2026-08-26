from odds_monitor.providers.theoddsapi import TheOddsApiProvider


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Stubs the two real-world calls TheOddsApiProvider makes: the events
    list, then one per-event odds fetch. Keyed by the request path so a
    single fake session can serve a whole `fetch_player_props` call.
    """

    def __init__(self, responses):
        self.responses = responses
        self.requested_params = []

    def get(self, url, params=None, timeout=None):
        self.requested_params.append((url, params))
        for path_suffix, payload in self.responses.items():
            if url.endswith(path_suffix):
                return _FakeResponse(payload)
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
