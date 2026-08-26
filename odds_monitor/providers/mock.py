"""Synthetic provider for local development, demos, and tests.

Requires no API key. Produces plausible player-prop lines across a handful
of sportsbooks and occasionally injects a deliberate cross-book gap so the
detectors have something to find - useful for exercising the full pipeline
(fetch -> detect -> notify) end to end before wiring up real credentials.

Covers both prop shapes:
  - line props (over/under a point value), e.g. player_points
  - binary props (Yes/No, no line - only a price), e.g. player_home_runs
"""

import random
from typing import List, Optional

from ..models import PropLine
from ..odds_math import probability_pct_to_american_odds
from .base import OddsProvider

_SPORTSBOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet"]

# Markets with no comparable point line - just a Yes/No price.
_BINARY_MARKETS = {"player_home_runs"}

# (event, [(player, team, market), ...])
_SAMPLE_EVENTS = {
    "nba": [
        (
            "LAL @ BOS",
            [
                ("LeBron James", "LAL", "player_points"),
                ("Jayson Tatum", "BOS", "player_points"),
                ("Anthony Davis", "LAL", "player_rebounds"),
            ],
        ),
        (
            "GSW @ PHX",
            [
                ("Stephen Curry", "GSW", "player_points"),
                ("Devin Booker", "PHX", "player_points"),
            ],
        ),
    ],
    "nfl": [
        (
            "KC @ BUF",
            [
                ("Patrick Mahomes", "KC", "player_passing_yards"),
                ("Josh Allen", "BUF", "player_passing_yards"),
            ],
        ),
    ],
    "mlb": [
        (
            "NYY @ BOS",
            [
                ("Aaron Judge", "NYY", "player_home_runs"),
                ("Rafael Devers", "BOS", "player_home_runs"),
            ],
        ),
        (
            "LAD @ SD",
            [
                ("Shohei Ohtani", "LAD", "player_home_runs"),
                ("Fernando Tatis Jr.", "SD", "player_home_runs"),
            ],
        ),
    ],
}


class MockOddsProvider(OddsProvider):
    """Generates synthetic player-prop lines. No network calls, no API key."""

    def __init__(self, seed: Optional[int] = None, discrepancy_chance: float = 0.35):
        self._rng = random.Random(seed)
        self.discrepancy_chance = discrepancy_chance

    def fetch_player_props(self, league: str) -> List[PropLine]:
        league_key = league.lower()
        events = _SAMPLE_EVENTS.get(league_key, _SAMPLE_EVENTS["nba"])
        lines: List[PropLine] = []
        for event, players in events:
            for player, team, market in players:
                if market in _BINARY_MARKETS:
                    lines.extend(self._binary_prop_lines(player, team, league_key, market, event))
                else:
                    lines.extend(self._line_prop_lines(player, team, league_key, market, event))
        return lines

    def _pick_books(self) -> List[str]:
        return self._rng.sample(_SPORTSBOOKS, k=self._rng.randint(3, len(_SPORTSBOOKS)))

    def _line_prop_lines(self, player: str, team: str, league_key: str, market: str, event: str) -> List[PropLine]:
        lines: List[PropLine] = []
        base_line = round(self._rng.uniform(15, 32) * 2) / 2  # nearest 0.5
        for side in ("over", "under"):
            for book in self._pick_books():
                jitter = 0.0
                if self._rng.random() < self.discrepancy_chance:
                    jitter = self._rng.choice([-3, -2.5, -2, 2, 2.5, 3])
                lines.append(
                    PropLine(
                        player=player,
                        team=team,
                        league=league_key,
                        market=market,
                        side=side,
                        line=base_line + jitter,
                        odds=self._rng.choice([-140, -125, -115, -110, 100, 105, 120]),
                        sportsbook=book,
                        event=event,
                    )
                )
        return lines

    def _binary_prop_lines(self, player: str, team: str, league_key: str, market: str, event: str) -> List[PropLine]:
        lines: List[PropLine] = []
        # A plausible "chance to hit a home run this game" base rate.
        base_prob_pct = self._rng.uniform(15, 45)
        for side, side_prob_pct in (("yes", base_prob_pct), ("no", 100 - base_prob_pct)):
            for book in self._pick_books():
                jitter = 0.0
                if self._rng.random() < self.discrepancy_chance:
                    jitter = self._rng.choice([-15, -12, -10, 10, 12, 15])
                priced_prob = min(max(side_prob_pct + jitter, 3.0), 92.0)
                lines.append(
                    PropLine(
                        player=player,
                        team=team,
                        league=league_key,
                        market=market,
                        side=side,
                        line=0.5,  # no real line for a Yes/No prop; kept as a placeholder
                        odds=probability_pct_to_american_odds(priced_prob),
                        sportsbook=book,
                        event=event,
                    )
                )
        return lines
