"""Synthetic provider for local development, demos, and tests.

Requires no API key. Produces plausible player-prop lines across a handful
of sportsbooks and occasionally injects a deliberate cross-book gap so the
detector has something to find - useful for exercising the full pipeline
(fetch -> detect -> notify) end to end before wiring up real credentials.
"""

import random
from typing import List, Optional

from ..models import PropLine
from .base import OddsProvider

_SPORTSBOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet"]

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
                base_line = round(self._rng.uniform(15, 32) * 2) / 2  # nearest 0.5
                for side in ("over", "under"):
                    books = self._rng.sample(_SPORTSBOOKS, k=self._rng.randint(3, len(_SPORTSBOOKS)))
                    for book in books:
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
