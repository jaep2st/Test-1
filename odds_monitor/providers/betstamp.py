"""Betstamp Sports Betting API provider.

Betstamp exposes a REST endpoint (`GET /api/markets`) that returns normalized
odds - including player props - across 200+ sportsbooks, authenticated with
an API key sent in the `X-API-KEY` header. Get a key and see the full
reference at https://www.betstamp.com/sports-betting-api and
https://www.betstamp.com/docs.

NOTE ON FIELD NAMES: Betstamp's public marketing/docs pages describe the
endpoint, its query parameters (`league`, `bet_types`, `is_live`,
`include_alts`, `book_ids`), and the auth header, but not the exact JSON
response schema per plan/version. This provider is written defensively: for
each field it tries several plausible key names (see `_FIELD_ALIASES`) and
skips (with a logged warning) any market entry it can't confidently parse,
rather than guessing wrong silently. If you have API access, check one real
response (`fetch_player_props` also accepts logging at DEBUG level) and add
your exact key names to `_FIELD_ALIASES` if they differ.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..http_utils import build_retrying_session
from ..models import PropLine
from .base import OddsProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.betstamp.com"
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
        self.api_key = api_key or os.environ.get("BETSTAMP_API_KEY")
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
            "league": league,
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
        response.raise_for_status()
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
