"""Data structures shared across providers, the detector, and notifiers."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Tuple


@dataclass(frozen=True)
class PropLine:
    """A single sportsbook's line for one player-prop market."""

    player: str
    team: Optional[str]
    league: str
    market: str  # e.g. "player_points", "player_assists", "player_rebounds"
    side: str  # "over" or "under"
    line: float  # the point value being offered, e.g. 24.5
    odds: Optional[int]  # American odds for this side, if available
    sportsbook: str  # e.g. "draftkings", "fanduel"
    event: str  # e.g. "LAL @ BOS"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def key(self) -> Tuple[str, str, str, str, str]:
        """Groups lines that should be compared against each other."""
        return (
            self.player.strip().lower(),
            self.league.lower(),
            self.market.lower(),
            self.side.lower(),
            self.event.lower(),
        )


@dataclass(frozen=True)
class Discrepancy:
    """The result of comparing one prop across sportsbooks: the widest gap found."""

    player: str
    league: str
    market: str
    side: str
    event: str
    spread: float
    low: PropLine
    high: PropLine
    all_lines: Tuple[PropLine, ...]

    def describe(self) -> str:
        books = ", ".join(
            f"{line.sportsbook}={line.line:g}" for line in sorted(self.all_lines, key=lambda l: l.line)
        )
        return (
            f"[{self.spread:g} pt gap] {self.player} ({self.event}) - {self.market} {self.side}: "
            f"{self.low.sportsbook} {self.low.line:g} vs {self.high.sportsbook} {self.high.line:g} "
            f"| all books: {books}"
        )
