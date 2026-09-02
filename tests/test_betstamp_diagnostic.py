"""Covers mlb_props_main.run_betstamp_diagnostic - the --betstamp-diagnostic
that calls Betstamp's real /api/markets directly and reports how many real
PropLines actually parsed, since odds_monitor/providers/betstamp.py's field
names and required-param values were never confirmed against a real
response (see that module's docstring).
"""

from mlb_props_main import run_betstamp_diagnostic
from odds_monitor.providers.betstamp import BetstampProvider


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)
        self.url = "https://api.pro.betstamp.com/api/markets"

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def get(self, url, headers=None, params=None, timeout=None):
        return _FakeResponse(self.payload, self.status_code)


def _patch_provider(monkeypatch, payload, status_code=200):
    real_init = BetstampProvider.__init__

    def fake_init(self, api_key=None, **kwargs):
        real_init(self, api_key=api_key, session=_FakeSession(payload, status_code))

    monkeypatch.setattr(BetstampProvider, "__init__", fake_init)


def test_no_markets_reports_zero_lines(monkeypatch):
    _patch_provider(monkeypatch, {"markets": []})

    text = run_betstamp_diagnostic("test-key", books=None)

    assert "0 real PropLine(s) successfully parsed" in text


def test_a_real_parseable_market_shows_up_in_the_sample(monkeypatch):
    _patch_provider(
        monkeypatch,
        {
            "markets": [
                {
                    "player": "Aaron Judge",
                    "market_type": "batter_home_runs",
                    "side": "yes",
                    "line": 0.5,
                    "odds": 450,
                    "book": "draftkings",
                    "event": "Yankees @ Red Sox",
                }
            ]
        },
    )

    text = run_betstamp_diagnostic("test-key", books=None)

    assert "1 real PropLine(s) successfully parsed" in text
    assert "Aaron Judge" in text
    assert "draftkings" in text


def test_unparsable_markets_are_reported_as_zero_not_a_crash(monkeypatch):
    # A market entry missing a required field (per _FIELD_ALIASES) is
    # skipped with a logged warning, not a crash - the diagnostic still
    # returns a real (zero) count rather than raising.
    _patch_provider(monkeypatch, {"markets": [{"totally": "unrecognized shape"}]})

    text = run_betstamp_diagnostic("test-key", books=None)

    assert "0 real PropLine(s) successfully parsed" in text
