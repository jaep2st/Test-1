"""Covers build_retrying_session: added after a real GitHub Actions run
(2026-08-27T03:29 UTC) hit a bare ConnectionResetError on the very first
call to The Odds API, with no retry, silently degrading that run to
model-only rankings despite a working API key. See odds_monitor/http_utils.py.
"""

import requests

from odds_monitor.http_utils import build_retrying_session


def test_returns_a_real_requests_session():
    session = build_retrying_session()
    assert isinstance(session, requests.Session)


def test_https_and_http_adapters_are_configured_to_retry():
    session = build_retrying_session(total=3, backoff_factor=0.5)
    for scheme in ("https://", "http://"):
        retry = session.adapters[scheme].max_retries
        assert retry.total == 3
        assert retry.connect == 3
        assert retry.read == 3
        assert retry.backoff_factor == 0.5
        assert 429 in retry.status_forcelist
        assert 503 in retry.status_forcelist


def test_does_not_retry_client_errors_other_than_429():
    # 4xx other than 429 (e.g. a bad API key -> 401, a malformed request ->
    # 400) should fail fast, not retry three times first.
    retry = build_retrying_session().adapters["https://"].max_retries
    assert 401 not in retry.status_forcelist
    assert 400 not in retry.status_forcelist


def test_only_get_is_retried():
    retry = build_retrying_session().adapters["https://"].max_retries
    assert retry.allowed_methods == frozenset(["GET"])
