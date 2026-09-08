"""A real, independently-computed Elo rating for every FBS team, built
purely from final scores of games that have already happened - point-in-
time by construction (a team's rating after week N only ever reflects
games through week N), which is exactly what makes it safe to use in a real
backtest without lookahead risk (see `historical_backtest.py`'s module
docstring - CFBD's own SP+ rating is season-level, not point-in-time, so it
can leak future information into an early-season backtest; this can't).

This is the second half of this project's power rating (see `ratings.py`,
which blends this with SP+) - a genuine second opinion built on a
completely different methodology (results + margin of victory, vs. SP+'s
opponent-adjusted play-by-play efficiency), the same "two independent
signals, blended and cross-checked" posture `mlb_props` uses for its
market-edge/model-edge split and its Ballpark Pal second opinion.

Standard margin-of-victory-weighted Elo, the same family of formula
FiveThirtyEight publicly documented for NFL/college football Elo: a plain
win/loss Elo update, scaled by a log-of-margin multiplier so a 45-point
blowout moves ratings more than a 3-point nail-biter, and damped as the
pre-game rating gap widens (a huge favorite's easy win says less than a
close underdog's win would).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

logger = logging.getLogger(__name__)

BASE_RATING = 1500.0
K_FACTOR = 24.0
HOME_FIELD_ELO = 65.0  # applied only inside the update's expected-score calc, never exposed as a scoring.py HFA (see ratings.py's docstring on avoiding double-counting home field)
SEASON_REGRESSION = 1.0 / 3.0  # fraction each team's rating regresses toward BASE_RATING at the start of a new season


@dataclass(frozen=True)
class GameResult:
    """The minimal real inputs an Elo update needs - deliberately decoupled
    from `schedule.Matchup`/CFBD's own field names so this module has no
    dependency on the rest of this package and can be unit-tested (or
    reused for a different sport) on its own.
    """

    season: int
    week: int
    home_team: str
    away_team: str
    home_points: int
    away_points: int
    neutral_site: bool = False


def _expected_home_win_prob(elo_home: float, elo_away: float, neutral_site: bool) -> float:
    hfa = 0.0 if neutral_site else HOME_FIELD_ELO
    return 1.0 / (1.0 + 10 ** (-((elo_home + hfa) - elo_away) / 400.0))


def _mov_multiplier(margin: int, elo_diff_winner_perspective: float) -> float:
    """538's public college-football Elo margin-of-victory multiplier:
    log-scaled by the actual margin, damped as the pre-game favorite's
    rating edge grows (a 40-point favorite winning by 3 says much more
    than a pick'em team winning by 3 does).
    """
    return math.log(max(abs(margin), 1) + 1) * (2.2 / (0.001 * abs(elo_diff_winner_perspective) + 2.2))


def compute_elo_ratings(
    games: Sequence[GameResult],
    base_rating: float = BASE_RATING,
    k: float = K_FACTOR,
    season_regression: float = SEASON_REGRESSION,
) -> Dict[str, float]:
    """Processes `games` in real chronological order (sorted here by
    `(season, week)` - callers don't need to pre-sort) and returns each
    team's rating after the last game given. A team never seen before
    starts at `base_rating` - a disclosed, neutral default, not a guess
    about that team's real strength.

    At the start of a new season (the first game whose `season` differs
    from the previous game processed), every team's current rating
    regresses `season_regression` of the way back toward `base_rating` -
    standard practice for multi-season Elo: this year's team isn't last
    year's team, but it isn't a blank slate either (returning starters,
    same coaching staff, recruiting continuity all carry real signal
    forward).

    Purely a function of `games` - call this with games only through the
    real point in time a caller wants a rating "as of" (see
    `historical_backtest.py`) for a genuinely leak-free rating, or with
    every completed game so far this season (see `ratings.py`) for the
    current live rating.
    """
    ratings: Dict[str, float] = {}
    ordered = sorted(games, key=lambda g: (g.season, g.week))
    current_season: Optional[int] = None

    for game in ordered:
        if current_season is not None and game.season != current_season:
            for team in ratings:
                ratings[team] += (base_rating - ratings[team]) * season_regression
        current_season = game.season

        elo_home = ratings.setdefault(game.home_team, base_rating)
        elo_away = ratings.setdefault(game.away_team, base_rating)

        if game.home_points == game.away_points:
            # No real ties in modern FBS football (overtime resolves every
            # game) - treat an equal-score entry as bad/incomplete data
            # rather than silently applying a 0.5 result that shouldn't be
            # possible.
            logger.warning("Skipping games with equal scores in Elo update (unexpected for FBS): %r", game)
            continue

        home_won = game.home_points > game.away_points
        actual_home = 1.0 if home_won else 0.0
        expected_home = _expected_home_win_prob(elo_home, elo_away, game.neutral_site)

        margin = abs(game.home_points - game.away_points)
        hfa = 0.0 if game.neutral_site else HOME_FIELD_ELO
        elo_diff_winner_perspective = (elo_home + hfa - elo_away) if home_won else (elo_away - hfa - elo_home)
        multiplier = _mov_multiplier(margin, elo_diff_winner_perspective)

        delta = k * multiplier * (actual_home - expected_home)
        ratings[game.home_team] = elo_home + delta
        ratings[game.away_team] = elo_away - delta

    return ratings


def elo_diff_to_points(elo_diff: float) -> float:
    """Converts an Elo rating gap into an expected point-spread margin -
    the same rough conversion (~25 Elo points per expected point of
    margin) FiveThirtyEight publicly used for its own NFL/college Elo
    models. An approximation, not a fitted constant specific to this
    project's own real data yet - `refit.py` can recalibrate it once
    enough real resolved picks exist.
    """
    return elo_diff / 25.0
