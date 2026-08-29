"""The Odds API (the-odds-api.com) provider.

The Odds API exposes real cross-sportsbook odds, including MLB player
props, over a simple REST API authenticated with an `apiKey` query
parameter. Free tier: 500 credits/month, no credit card, sign up at
https://the-odds-api.com and an API key is emailed instantly.

Two calls per league fetch:
1. `GET /v4/sports/{sport_key}/events` - free (doesn't cost credits), lists
   today's/upcoming games (id, home_team, away_team).
2. `GET /v4/sports/{sport_key}/events/{event_id}/odds` per event - the
   actual player-prop odds. This one costs credits (roughly
   `markets_requested x regions_requested` per call per their pricing docs),
   so this provider deliberately requests only the two markets this pipeline
   uses and a single region ("us") to keep a full MLB slate (~15 games/day)
   well within the free tier.

NOTE ON FIELD NAMES: this is written against the schema documented at
https://the-odds-api.com/liveapi/guides/v4/#get-event-odds - each
bookmaker's market has an `outcomes` list where, for player-prop markets,
`description` carries the player's name (`name` is the selection label:
"Over"/"Under" or "Yes"/"No") and `point` carries the line. That schema
hasn't been verified against a real response from this dev sandbox (no
network access here - see the mlb_props/statcast.py module docstring for
why). Parsing is defensive throughout: any market/outcome that doesn't
match the expected shape is skipped with a logged warning rather than
crashing the whole fetch, and both "Yes"/"No" and "Over"/"Under" outcome
labels are accepted for the home-run market in case books differ. If a
live run's `--log-level DEBUG` output shows different field names, fix the
constants/parsing below and this docstring.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from ..http_utils import build_retrying_session
from ..models import PropLine
from .base import OddsProvider

logger = logging.getLogger(__name__)


class OddsFetchFailed(Exception):
    """Raised when every per-event odds request in a fetch failed - a
    systemic problem (auth/quota exhaustion, an outage) as opposed to a
    legitimately empty market (a game with no props posted yet, which
    returns an empty list, not this exception). Callers that want to fall
    back to a different odds provider on a real failure (see
    odds_monitor/providers/fallback.py) should catch this specifically;
    anything else (a genuinely empty slate, a per-game parsing hiccup)
    still degrades to an empty list exactly as before.
    """


DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4"

# Maps this pipeline's generic league string to The Odds API's sport key.
_SPORT_KEYS: Dict[str, str] = {"mlb": "baseball_mlb"}

# The Odds API's real, documented market keys for these two props (see
# https://the-odds-api.com/sports-odds-data/betting-markets.html).
MARKET_KEY_HOME_RUN = "batter_home_runs"
MARKET_KEY_TOTAL_BASES = "batter_total_bases"

# Normalizes whatever outcome label a book uses onto the "yes"/"no" or
# "over"/"under" sides this pipeline's edges/ev code expects (see
# mlb_props/market.py and mlb_props/edges.py) - some books may present the
# home-run prop as Yes/No, others as Over/Under 0.5.
_HR_SIDE_ALIASES = {"yes": "yes", "over": "yes", "no": "no", "under": "no"}
_TB_SIDE_ALIASES = {"over": "over", "under": "under"}

_DEFAULT_HR_LINE = 0.5
_DEFAULT_TB_LINE = 1.5


class TheOddsApiProvider(OddsProvider):
    """Pulls real MLB home-run and 2+ total-bases prop odds from
    The Odds API.

    Requires an API key: pass `api_key=` or set the `ODDS_API_KEY`
    environment variable. Get one free at https://the-odds-api.com.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        regions: str = "us",
        books: Optional[List[str]] = None,
        timeout: float = 15.0,
        session: Optional[requests.Session] = None,
    ):
        # .strip() guards against the same real footgun confirmed live on
        # BETSTAMP_API_KEY (2026-08-29): a secret pasted with stray
        # surrounding whitespace/newlines. Here it's sent as a query
        # param rather than a header, so it wouldn't raise the same
        # InvalidHeader error, but an un-stripped trailing newline would
        # still corrupt the key value and fail auth in a confusing way.
        raw_key = api_key or os.environ.get("ODDS_API_KEY")
        self.api_key = raw_key.strip() if raw_key else raw_key
        if not self.api_key:
            raise ValueError(
                "An Odds API key is required. Pass api_key=... or set the "
                "ODDS_API_KEY environment variable. Get one free at "
                "https://the-odds-api.com"
            )
        self.base_url = base_url.rstrip("/")
        self.regions = regions
        self.books = set(b.lower() for b in books) if books else None
        self.timeout = timeout
        # See odds_monitor/http_utils.py's docstring: retries transient
        # connection failures instead of dropping the whole run's market
        # data on the first hiccup. Only applied when no session is
        # injected, so tests supplying a fake session are unaffected.
        self.session = session or build_retrying_session()

    def fetch_player_props(self, league: str) -> List[PropLine]:
        sport_key = _SPORT_KEYS.get(league.lower())
        if sport_key is None:
            logger.warning("TheOddsApiProvider has no sport-key mapping for league %r - returning no lines", league)
            return []

        try:
            events = self._get(f"/sports/{sport_key}/events", {})
        except Exception as exc:
            logger.exception("Failed to fetch %s event list from The Odds API", league)
            raise OddsFetchFailed(f"Failed to fetch {league} event list: {exc}") from exc

        lines: List[PropLine] = []
        attempted = 0
        failed = 0
        for event in events or []:
            event_id = event.get("id")
            home_team = event.get("home_team")
            away_team = event.get("away_team")
            if not event_id or not home_team or not away_team:
                continue
            event_label = f"{away_team} @ {home_team}"
            attempted += 1
            try:
                payload = self._get(
                    f"/sports/{sport_key}/events/{event_id}/odds",
                    {
                        "regions": self.regions,
                        "markets": f"{MARKET_KEY_HOME_RUN},{MARKET_KEY_TOTAL_BASES}",
                        "oddsFormat": "american",
                    },
                )
            except Exception:
                failed += 1
                logger.warning("Failed to fetch odds for event %r (%s) - skipping this game", event_id, event_label, exc_info=True)
                continue
            lines.extend(self._parse_event_odds(payload, league, event_label))

        # Every single per-event request failing (with at least one
        # attempted) is a systemic problem - confirmed live: a real run hit
        # 401 Unauthorized on all 17 event-odds calls in a row while the
        # free /events call itself succeeded, consistent with a free-tier
        # credit quota running out mid-day. A slate with 0 games (an
        # off-day) or a handful of per-game hiccups among mostly-successful
        # calls both still degrade to whatever partial lines were gathered,
        # same as before - only total failure raises, so a caller wanting a
        # fallback provider (see fallback.py) can catch this specifically.
        if attempted and failed == attempted:
            raise OddsFetchFailed(
                f"All {attempted} event-odds requests failed for {league} - likely an API auth/quota "
                "problem, not an empty market. See the per-event WARNING logs above for the underlying errors."
            )
        return lines

    def _get(self, path: str, params: Dict[str, Any]) -> Any:
        full_params = dict(params)
        full_params["apiKey"] = self.api_key
        response = self.session.get(f"{self.base_url}{path}", params=full_params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _parse_event_odds(self, payload: Dict[str, Any], league: str, event_label: str) -> List[PropLine]:
        lines: List[PropLine] = []
        bookmakers = (payload or {}).get("bookmakers", []) or []
        if not bookmakers:
            logger.debug("%s: no bookmakers in response yet (empty bookmakers list)", event_label)
        for bookmaker in bookmakers:
            book_key = str(bookmaker.get("key", "")).lower()
            if self.books and book_key not in self.books:
                continue
            for market in bookmaker.get("markets", []) or []:
                market_key = market.get("key")
                logger.debug(
                    "%s: bookmaker %s carries market %r with %d outcome(s)",
                    event_label,
                    book_key,
                    market_key,
                    len(market.get("outcomes", []) or []),
                )
                if market_key == MARKET_KEY_HOME_RUN:
                    side_aliases, our_market, default_line = _HR_SIDE_ALIASES, MARKET_KEY_HOME_RUN, _DEFAULT_HR_LINE
                    # Extra diagnostics just for this market: figure out
                    # whether it's genuinely two-sided (Yes+No per player) -
                    # find_fair_prices() needs both sides quoted by at least
                    # one book to compute a no-vig price, and a single-sided
                    # "anytime" market (only "Yes" quoted) can't be de-vigged
                    # that way even though every outcome parses fine.
                    outcomes = market.get("outcomes", []) or []
                    names_seen = sorted({str(o.get("name")) for o in outcomes})
                    per_player_sides: Dict[str, set] = {}
                    per_player_points: Dict[str, set] = {}
                    for o in outcomes:
                        player_name = str(o.get("description"))
                        per_player_sides.setdefault(player_name, set()).add(str(o.get("name")))
                        per_player_points.setdefault(player_name, set()).add(o.get("point"))
                    two_sided_players = sum(1 for sides in per_player_sides.values() if len(sides) >= 2)
                    multi_line_players = sum(1 for points in per_player_points.values() if len(points) >= 2)
                    logger.debug(
                        "%s/%s batter_home_runs: outcome name labels seen=%s | %d/%d players have 2+ distinct "
                        "sides quoted | %d/%d players have 2+ distinct point/line values quoted",
                        event_label,
                        book_key,
                        names_seen,
                        two_sided_players,
                        len(per_player_sides),
                        multi_line_players,
                        len(per_player_points),
                    )
                    if multi_line_players:
                        sample = next((k, v) for k, v in per_player_points.items() if len(v) >= 2)
                        logger.debug("%s/%s batter_home_runs: sample multi-line player %r has points=%s", event_label, book_key, *sample)
                    # Exact (point -> price) per player, so a specific
                    # candidate's real quoted price can be verified directly
                    # rather than inferred from the final report number -
                    # this is the actual money figure users will see.
                    per_player_point_price: Dict[str, Dict[str, int]] = {}
                    for o in outcomes:
                        player_name = str(o.get("description"))
                        per_player_point_price.setdefault(player_name, {})[str(o.get("point"))] = o.get("price")
                    logger.debug("%s/%s batter_home_runs: point->price per player=%s", event_label, book_key, per_player_point_price)
                elif market_key == MARKET_KEY_TOTAL_BASES:
                    side_aliases, our_market, default_line = _TB_SIDE_ALIASES, MARKET_KEY_TOTAL_BASES, _DEFAULT_TB_LINE
                else:
                    continue
                for outcome in market.get("outcomes", []) or []:
                    try:
                        lines.append(
                            self._parse_outcome(outcome, side_aliases, our_market, default_line, book_key, league, event_label)
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        logger.warning("Skipping unparsable outcome on %s/%s: %s (%r)", book_key, market_key, exc, outcome)
        return lines

    @staticmethod
    def _parse_outcome(
        outcome: Dict[str, Any],
        side_aliases: Dict[str, str],
        market: str,
        default_line: float,
        book_key: str,
        league: str,
        event_label: str,
    ) -> PropLine:
        player = outcome.get("description")
        if not player:
            raise KeyError("outcome has no player 'description'")
        raw_side = str(outcome.get("name", "")).strip().lower()
        side = side_aliases.get(raw_side)
        if side is None:
            raise ValueError(f"unrecognized outcome side {raw_side!r}")
        price = outcome.get("price")
        if price is None:
            raise KeyError("outcome has no 'price'")
        return PropLine(
            player=str(player),
            team=None,
            league=league,
            market=market,
            side=side,
            line=float(outcome.get("point", default_line) or default_line),
            odds=int(price),
            sportsbook=book_key,
            event=event_label,
        )
