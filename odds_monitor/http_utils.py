"""Shared HTTP session helper for the live (non-mock) data providers.

Confirmed live: a real GitHub Actions run (2026-08-27T03:29 UTC) hit a bare
`ConnectionResetError` on the very first call to The Odds API
(`GET /sports/mlb/events`), with no retry - `TheOddsApiProvider` caught the
exception and returned no lines, which silently degraded that entire run to
model-only rankings (no market price/EV% for any prop) despite a working
API key and despite every other live data source (MLB Stats API, Baseball
Savant, Open-Meteo) succeeding in the same run. A single dropped TCP
connection shouldn't cost a whole day's market data.

`build_retrying_session()` gives every live HTTP provider in this project
(`odds_monitor.providers.theoddsapi`, `odds_monitor.providers.betstamp`,
`mlb_props.schedule`'s `MlbStatsApiScheduleProvider`, `mlb_props.context`'s
`LiveParkWeatherProvider`) a `requests.Session` that retries connection-level
failures (resets, timeouts) and 429/5xx responses with exponential backoff,
instead of failing the whole fetch on the first hiccup. GET-only (nothing
here does a non-idempotent write), and deliberately does NOT retry on 4xx
(other than 429) - a bad API key or malformed request should fail fast and
loud, not retry three times first.
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover - very old urllib3 vendored under requests
    from requests.packages.urllib3.util.retry import Retry  # type: ignore


def build_retrying_session(total: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """A `requests.Session` pre-configured to retry transient failures.

    3 retries with a 0.5s backoff factor means retry delays of roughly
    0.5s, 1s, 2s (urllib3's `{backoff_factor} * (2 ** (retry_number - 1))`)
    - a handful of seconds of extra worst-case latency per request in
    exchange for surviving exactly the kind of single dropped connection
    that cost a full run its market data above.
    """
    session = requests.Session()
    retry = Retry(
        total=total,
        connect=total,
        read=total,
        status=total,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
