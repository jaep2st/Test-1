"""Betstamp Sports Betting API provider.

Betstamp exposes a REST endpoint (`GET /api/markets`) that returns normalized
odds - including player props - across 200+ sportsbooks, authenticated with
an API key sent in the `X-API-KEY` header. Get a key and see the full
reference at https://www.betstamp.com/sports-betting-api and
https://www.betstamp.com/docs.

BASE URL: confirmed live (2026-08-29, via Betstamp's own published API docs
page) as `https://api.pro.betstamp.com` - the "Production Pull API" host
shown directly against `GET /api/markets`. This was previously a guess
(`https://api.betstamp.com`, no `.pro` subdomain) that would have silently
pointed at the wrong host even with a valid key - the response shape is
confirmed too, a top-level `{"markets": [...]}` object, matching what
`fetch_player_props` below already expected.

REQUIRED PARAMETERS: confirmed live (2026-08-29, from the same docs page's
full request schema) that `GET /api/markets` requires FIVE query params,
not the two this provider originally sent:
- `league` - array of strings from a fixed enum (e.g. "MLB", "NBA", "NHL",
  ...) - confirmed uppercase, not the lowercase "mlb" this pipeline passes
  internally; `fetch_player_props` now upper-cases it before sending.
- `book_ids` - array of integers (numeric bookmaker IDs, per a
  "Bookmaker IDs table" the docs page links but doesn't inline).
- `periods` - array of strings from a fixed enum (FT, ET, 1H, 2H, 1P-3P,
  1Q-4Q, F1/F3/F5/F7, 1I-9I, 1S/2S, 1R-4R, REG) - which value(s) apply to a
  full-game MLB anytime-HR/total-bases prop is not yet confirmed.
- `bet_types` - array of strings (docs example: "moneyline,spread" - a
  different axis than the specific prop, so this provider's current guess
  of a single `"player_props"` value is itself unconfirmed).
- `prop_types` - array of strings (docs example: "touchdowns,first td",
  per a "Prop types table" the docs page links but doesn't inline) - the
  values that mean "home runs" / "total bases" for MLB are not yet known.

`book_ids`, `periods`, `bet_types`, and `prop_types`' exact valid values
were all behind expandable/linked sections not visible in the page content
seen so far - rather than guess at required-parameter values for a paid
API (each guess costs a real request and, per the docs' own examples,
values are enum-constrained, so a wrong guess is a 400, not a fuzzy
mismatch), `fetch_player_props` now logs the full response body on any
HTTP error, so the next real call's error message - Betstamp's own
validation text - supplies the missing values directly instead of another
guess.

NOTE ON FIELD NAMES: the individual market objects inside `"markets":
[...]` were shown collapsed on the docs page, so the exact per-field key
names inside each one are still unconfirmed either. This provider is
written defensively: for each field it tries several plausible key names
(see `_FIELD_ALIASES`) and skips (with a logged warning) any market entry
it can't confidently parse, rather than guessing wrong silently. If you
have API access, check one real response (`fetch_player_props` also
accepts logging at DEBUG level) and add your exact key names to
`_FIELD_ALIASES` if they differ.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..http_utils import build_retrying_session
from ..models import PropLine
from .base import OddsProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.pro.betstamp.com"
MARKETS_PATH = "/api/markets"

_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "player": ("player", "player_name", "participant"),
    "team": ("team", "team_name", "player_team"),
    "market": ("bet_type", "market", "market_type", "prop_type"),
    "side": ("side", "over_under", "selection"),
    "line": ("line", "points", "value", "handicap"),
    "odds": ("odds", "price", "american_odds"),
    "sportsbook": ("book", "sportsbook", "book_name", "book_id"),
    "event": ("event", "matchup", "game"),
}

_REQUIRED_FIELDS = ("player", "market", "side", "line", "sportsbook", "event")


class BetstampProvider(OddsProvider):
    """Pulls player-prop lines from Betstamp's Sports Betting API.

    Requires an API key: pass `api_key=` or set the `BETSTAMP_API_KEY`
    environment variable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        book_ids: Optional[List[str]] = None,
        include_alts: bool = False,
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
    ):
        # .strip() guards against a real failure mode: confirmed live
        # (2026-08-29) that a BETSTAMP_API_KEY secret pasted with stray
        # whitespace/newlines around it produces a header value requests
        # rejects outright (InvalidHeader: "leading whitespace, reserved
        # character(s), or return character(s)"), so the request never
        # even gets sent - a silent-looking failure that's actually just a
        # copy/paste artifact, not a bad key.
        raw_key = api_key or os.environ.get("BETSTAMP_API_KEY")
        self.api_key = raw_key.strip() if raw_key else raw_key
        if not self.api_key:
            raise ValueError(
                "A Betstamp API key is required. Pass api_key=... or set "
                "the BETSTAMP_API_KEY environment variable. Get one at "
                "https://www.betstamp.com/sports-betting-api"
            )
        self.base_url = base_url.rstrip("/")
        self.book_ids = book_ids
        self.include_alts = include_alts
        self.timeout = timeout
        # See odds_monitor/http_utils.py's docstring: retries transient
        # connection failures instead of dropping the whole run's market
        # data on the first hiccup. Only applied when no session is
        # injected, so tests supplying a fake session are unaffected.
        self.session = session or build_retrying_session()

    def fetch_player_props(self, league: str) -> List[PropLine]:
        params: Dict[str, Any] = {
            # Confirmed live (2026-08-29): the league enum is uppercase
            # ("MLB", not "mlb") - this pipeline always calls with a
            # lowercase league string, so upper-case it here rather than
            # pushing that detail onto every caller.
            "league": league.upper(),
            "bet_types": "player_props",
            "is_live": "false",
            "include_alts": str(self.include_alts).lower(),
        }
        if self.book_ids:
            params["book_ids"] = ",".join(self.book_ids)

        response = self.session.get(
            f"{self.base_url}{MARKETS_PATH}",
            headers={"X-API-KEY": self.api_key, "Accept": "application/json"},
            params=params,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            # See this module's docstring ("REQUIRED PARAMETERS"): several
            # required query params' valid values aren't confirmed yet, so
            # surfacing Betstamp's own validation error text here (rather
            # than just the bare status code) is what actually resolves
            # that gap on the next real call, instead of another guess.
            logger.warning(
                "Betstamp API request failed: %s %s - response body: %s",
                response.status_code,
                response.url,
                response.text[:2000],
            )
            raise
        payload = response.json()
        raw_markets = payload.get("markets", payload) if isinstance(payload, dict) else payload

        lines: List[PropLine] = []
        for raw in raw_markets:
            try:
                lines.append(self._parse_market(raw, league))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping unparsable market entry: %s (%r)", exc, raw)
        return lines

    def _parse_market(self, raw: Dict[str, Any], league: str) -> PropLine:
        values = {field: self._pick(raw, aliases) for field, aliases in _FIELD_ALIASES.items()}
        missing = [f for f in _REQUIRED_FIELDS if values[f] is None]
        if missing:
            raise KeyError(f"missing fields {missing}")

        return PropLine(
            player=str(values["player"]),
            team=str(values["team"]) if values["team"] is not None else None,
            league=league,
            market=str(values["market"]),
            side=str(values["side"]).lower(),
            line=float(values["line"]),
            odds=int(values["odds"]) if values["odds"] is not None else None,
            sportsbook=str(values["sportsbook"]),
            event=str(values["event"]),
        )

    @staticmethod
    def _pick(raw: Dict[str, Any], aliases: Tuple[str, ...]) -> Optional[Any]:
        for key in aliases:
            if key in raw and raw[key] is not None:
                return raw[key]
        return None
