"""MLB home run and total-bases prop markets: constants, plus a mock odds
provider so the whole pipeline is demoable with `--mock` and no API key.

Real odds default to `odds_monitor.providers.theoddsapi.TheOddsApiProvider`
(needs `ODDS_API_KEY` - free tier, no card, sign up at the-odds-api.com) -
`fetch_player_props("mlb")` pulls real cross-book player-prop odds and hands
back `PropLine`s. `MARKET_HOME_RUN`/`MARKET_TOTAL_BASES` below are set to
The Odds API's own real, documented market keys
(https://the-odds-api.com/sports-odds-data/betting-markets.html), so no
translation is needed between this module and that provider.

`odds_monitor.providers.betstamp.BetstampProvider` (needs
`BETSTAMP_API_KEY`) is also still available as an alternative - see
`mlb_props_main.build_providers()` for which one gets picked. Betstamp's
exact market-name string for each prop wasn't verified live (see that
provider's docstring), so these constants may not match its real field
values if you use it instead of The Odds API.
"""

from __future__ import annotations

from typing import List, Optional

from odds_monitor.models import PropLine
from odds_monitor.providers.base import OddsProvider

MARKET_HOME_RUN = "batter_home_runs"  # two-way: "yes" / "no", to hit a home run
MARKET_TOTAL_BASES = "batter_total_bases"  # two-way: "over" / "under" a line (2+ bases = line 1.5)
MARKET_HITS = "batter_hits"  # two-way: "over" / "under" a line (1+ hit = line 0.5)
TOTAL_BASES_LINE_FOR_2PLUS = 1.5
HOME_RUN_LINE_FOR_1PLUS = 0.5  # the standard "hits a HR" line, as opposed to a longer-shot "2+ HRs" (1.5) etc.
HITS_LINE_FOR_1PLUS = 0.5  # the standard "gets a hit" line, as opposed to a longer-shot "2+ hits" (1.5) etc.

# The recommended/scored side for each market this project ever bets on -
# shared by results.py (matching a recorded pick's price against a fresh
# odds fetch for CLV) and betting.py (finding real cross-book value for
# live lines). This project never scores or recommends the other side.
RECOMMENDED_SIDE_FOR_MARKET = {MARKET_HOME_RUN: "yes", MARKET_TOTAL_BASES: "over", MARKET_HITS: "over"}

# The Odds API's real per-book `key` field (what PropLine.sportsbook actually
# holds - see theoddsapi.py's _parse_outcome) doesn't always match the real
# sportsbook's own current brand name. Confirmed live against The Odds API's
# own bookmaker-apis docs page (2026-09-04): `williamhill_us` is Caesars
# Sportsbook - Caesars still runs on the old William Hill US platform after
# its 2021 acquisition, and the API key was never renamed for the rebrand.
# This is a real, live, freely-returned book in every fetch this project has
# ever made (confirmed across many real runs, going back to this project's
# earliest live data) - it's simply been showing up under its old platform's
# name this whole time, which reads as an unrecognizable book to anyone
# checking this site against a real sportsbook app, and silently defeated
# any attempt to filter/label picks by "caesars" (the name this project's
# own mock data and everyone's mental model actually use).
#
# Every entry here is a real book key this project has actually seen in a
# live fetch - not a guessed/speculative list - with its real current brand
# name; several (DraftKings, FanDuel, BetMGM, ESPN BET) have capitalization
# a blind .title() would get wrong.
BOOK_DISPLAY_NAMES = {
    "williamhill_us": "Caesars",
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "betmgm": "BetMGM",
    "betrivers": "BetRivers",
    "espnbet": "ESPN BET",
    "hardrockbet": "Hard Rock Bet",
    "betonlineag": "BetOnline.ag",
    "betparx": "betPARX",
    "ballybet": "Bally Bet",
    "rebet": "ReBet",
    "mybookieag": "MyBookie.ag",
    "fanatics": "Fanatics",
    "fliff": "Fliff",
    "bovada": "Bovada",
}


def book_display_name(book_key: Optional[str]) -> str:
    """The real, human-recognizable sportsbook name for a book key that may
    be a rebranded/legacy API key, or have brand capitalization a blind
    title-case would get wrong (see BOOK_DISPLAY_NAMES for every book this
    project has actually seen live). Falls back to title-casing (with
    underscores as spaces) for a real book this project hasn't seen yet,
    rather than showing a raw snake_case API key - still better than
    nothing, just not guaranteed to match that book's real branding.
    `None`/empty stays empty rather than guessing.
    """
    if not book_key:
        return book_key or ""
    key = book_key.strip().lower()
    return BOOK_DISPLAY_NAMES.get(key, book_key.replace("_", " ").title())


_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "espnbet", "fanatics"]


class NoOddsProvider(OddsProvider):
    """Returns no lines at all. Used when no odds API key is configured, so
    the rest of the pipeline (Statcast, matchup, weather, hot-streak scoring)
    still runs and produces model-only rankings instead of hard-failing -
    every `EdgeCandidate` just comes back with `has_market_data=False`.
    """

    def fetch_player_props(self, league: str) -> List[PropLine]:
        return []


class MockMlbPropsOddsProvider(OddsProvider):
    """Synthetic HR/total-bases prop odds for a given list of batters - no
    network calls, no API key. Randomizes each book's price around a
    per-player "true" implied probability (with a realistic vig), and
    occasionally makes one book's price a deliberate outlier so the
    line-shopping / +EV detector in `odds_monitor.ev` has something to find.
    """

    def __init__(self, batters: List[str], events_by_batter: Optional[dict] = None, seed: Optional[int] = None, outlier_chance: float = 0.3):
        import random

        self._rng = random.Random(seed)
        self.batters = batters
        self.events_by_batter = events_by_batter or {}
        self.outlier_chance = outlier_chance

    def _price_pair(self, true_prob: float) -> "tuple[int, int]":
        """Given a 'true' probability of the 'yes'/'over' side, generate a
        realistic vig-included American-odds pair for both sides.
        """
        vig = self._rng.uniform(0.03, 0.07)
        yes_prob = min(0.95, max(0.05, true_prob + vig / 2))
        no_prob = min(0.95, max(0.05, (1 - true_prob) + vig / 2))
        return self._prob_to_american(yes_prob), self._prob_to_american(no_prob)

    @staticmethod
    def _prob_to_american(prob: float) -> int:
        prob = min(0.95, max(0.05, prob))
        if prob >= 0.5:
            return -round(prob / (1 - prob) * 100)
        return round((1 - prob) / prob * 100)

    def fetch_player_props(self, league: str) -> List[PropLine]:
        if league.lower() != "mlb":
            return []
        lines: List[PropLine] = []
        for batter in self.batters:
            event = self.events_by_batter.get(batter, "MLB Game")
            true_hr_prob = self._rng.uniform(0.05, 0.20)
            true_tb_prob = self._rng.uniform(0.32, 0.58)
            true_hits_prob = self._rng.uniform(0.55, 0.78)

            for market, true_prob, side_names in (
                (MARKET_HOME_RUN, true_hr_prob, ("yes", "no")),
                (MARKET_TOTAL_BASES, true_tb_prob, ("over", "under")),
                (MARKET_HITS, true_hits_prob, ("over", "under")),
            ):
                books = self._rng.sample(_BOOKS, k=self._rng.randint(4, len(_BOOKS)))
                outlier_book = self._rng.choice(books) if self._rng.random() < self.outlier_chance else None
                for book in books:
                    prob = true_prob
                    if book == outlier_book:
                        # Simulate a book that's slow to move, offering
                        # meaningfully better value on the "yes"/"over" side.
                        prob = max(0.04, true_prob - self._rng.uniform(0.03, 0.07))
                    side_odds = self._price_pair(prob)
                    for side, odds in zip(side_names, side_odds):
                        line_value = TOTAL_BASES_LINE_FOR_2PLUS if market == MARKET_TOTAL_BASES else 0.5
                        lines.append(
                            PropLine(
                                player=batter,
                                team=None,
                                league="mlb",
                                market=market,
                                side=side,
                                line=line_value,
                                odds=odds,
                                sportsbook=book,
                                event=event,
                            )
                        )
        return lines
