"""Data structures shared across providers, the detector, and notifiers."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol, Tuple, runtime_checkable

from .odds_math import implied_probability_pct

# Sides that carry a comparable point line, e.g. "over 24.5".
LINE_SIDES = ("over", "under")
# Sides for binary props with no line, only a price, e.g. "Yes" to hit a home run.
BINARY_SIDES = ("yes", "no")


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


@dataclass(frozen=True)
class OddsDiscrepancy:
    """The result of comparing a binary (Yes/No) prop's price across books -
    used for markets like "to hit a home run" that have no point line, only
    odds. The signal here is a gap in implied win probability, not points.
    """

    player: str
    league: str
    market: str
    side: str
    event: str
    prob_spread: float  # percentage points between the best and worst price
    best: PropLine  # lowest implied probability = best price for the bettor
    worst: PropLine  # highest implied probability = worst price
    best_prob_pct: float
    worst_prob_pct: float
    all_lines: Tuple[PropLine, ...]

    def describe(self) -> str:
        def fmt_odds(odds: Optional[int]) -> str:
            if odds is None:
                return "n/a"
            return f"+{odds}" if odds > 0 else str(odds)

        books = ", ".join(
            f"{line.sportsbook}={fmt_odds(line.odds)} ({implied_probability_pct(line.odds):.1f}%)"
            for line in sorted(self.all_lines, key=lambda l: l.odds if l.odds is not None else 0)
            if line.odds is not None
        )
        return (
            f"[{self.prob_spread:.1f} pt implied-prob gap] {self.player} ({self.event}) - "
            f"{self.market} {self.side}: {self.best.sportsbook} {fmt_odds(self.best.odds)} "
            f"({self.best_prob_pct:.1f}%) vs {self.worst.sportsbook} {fmt_odds(self.worst.odds)} "
            f"({self.worst_prob_pct:.1f}%) | all books: {books}"
        )


@runtime_checkable
class Alertable(Protocol):
    """Anything a Notifier can render: Discrepancy and OddsDiscrepancy both
    satisfy this without a shared base class.
    """

    def describe(self) -> str: ...
