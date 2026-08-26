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
TOTAL_BASES_LINE_FOR_2PLUS = 1.5

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

            for market, true_prob, side_names in (
                (MARKET_HOME_RUN, true_hr_prob, ("yes", "no")),
                (MARKET_TOTAL_BASES, true_tb_prob, ("over", "under")),
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
