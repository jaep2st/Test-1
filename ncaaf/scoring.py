"""Predicts a game's margin, total, and every market's win/cover
probability from the two blended power ratings (`ratings.py`) plus
situational context (`context.py`).

This is a transparent, hand-built statistical model, not a trained/
calibrated one - the same posture `mlb_props/scoring.py` discloses for its
own heuristic scores. Two deliberately separate estimates feed the final
numbers, kept as a real, visible cross-check on each other rather than
silently reconciled:

1. **Margin / win probability** comes from the blended overall
   `power_rating` (`ratings.TeamRating.power_rating`) plus every
   situational adjustment in `context.GameContext` - this is the
   authoritative number for moneyline and against-the-spread probabilities.
2. **Total** comes from the separate offense/defense rating split - each
   team's implied points scored is `league-average + own offense rating -
   opponent's defense rating`, then nudged by live weather. This can (and
   sometimes will) imply a slightly different margin than #1's power-rating
   diff, since offense/defense are their own independent blend of SP+ and
   PPA - a real, disclosed inconsistency rather than one silently forced to
   agree, exactly analogous to `mlb_props`' model-vs-Ballpark-Pal "genuine
   second opinion, never blended in" posture.

Probabilities are computed by treating the real final-score margin/total as
approximately normally distributed around the predicted value -
`MARGIN_STD_DEV`/`TOTAL_STD_DEV` are realistic, publicly-known real-world
spreads for FBS games, not fit to this project's own results yet (see
`refit.py`, once enough real resolved picks exist). Two known, disclosed
simplifications this approximation makes: (1) it ignores the real
"key number" clustering around 3 and 7 points that a true discrete margin
distribution has (a market spread sitting exactly on 3 or 7 carries more
real push risk and different true cover odds than the continuous normal
curve implies - less pronounced in college football than the NFL, where
this effect is best documented, but still real); (2) win/cover probability
both use the *same* std dev, when a true outcome distribution's tails
likely aren't identical for "did team A win" vs "did team A cover a
specific number." Treat every probability here as a genuine directional
estimate to cross-check against the market's own no-vig price (see
`game_odds.py`), not as ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from .context import GameContext
from .ratings import TeamRating
from .schedule import Matchup

# Realistic modern-FBS average points scored per team per game (national
# scoring average has hovered in the high-20s to low-30s in recent
# seasons) - the anchor `offense_rating`/`defense_rating` (both expressed
# as "points above average") are added to/subtracted from.
LEAGUE_AVG_POINTS_PER_TEAM = 27.5

# Real-world full-game margin/total std devs, from publicly documented CFB
# scoring variance - see module docstring for why these aren't yet fit to
# this project's own data.
MARGIN_STD_DEV = 16.5
TOTAL_STD_DEV = 14.0


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


@dataclass(frozen=True)
class GameScore:
    matchup: Matchup
    predicted_margin: float  # home perspective: positive = home favored by this many points
    predicted_home_points: float
    predicted_away_points: float
    predicted_total: float
    home_win_prob: float  # P(home wins outright), from predicted_margin
    context: GameContext
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def event(self) -> str:
        return self.matchup.event

    def home_cover_prob(self, home_spread: float) -> float:
        """P(home covers a market spread quoted from the home team's own
        perspective - e.g. `home_spread=-6.5` means home must win by more
        than 6.5 to cover, `home_spread=+3.5` means home covers by losing
        by less than 3.5 or winning outright). See module docstring for the
        normal-approximation caveat, including the real key-number push
        risk this doesn't separately model.
        """
        z = (self.predicted_margin + home_spread) / MARGIN_STD_DEV
        return round(_normal_cdf(z), 4)

    def away_cover_prob(self, home_spread: float) -> float:
        """The away side of the same spread market - NOT exactly
        `1 - home_cover_prob` once real push probability is accounted for,
        but this project's continuous-normal approximation doesn't model
        pushes separately (see module docstring), so it is here."""
        return round(1.0 - self.home_cover_prob(home_spread), 4)

    def over_prob(self, total_line: float) -> float:
        z = (self.predicted_total - total_line) / TOTAL_STD_DEV
        return round(_normal_cdf(z), 4)

    def under_prob(self, total_line: float) -> float:
        return round(1.0 - self.over_prob(total_line), 4)


def compute_game_score(
    matchup: Matchup,
    home_rating: Optional[TeamRating],
    away_rating: Optional[TeamRating],
    context: GameContext,
) -> Optional[GameScore]:
    """`None` (never a guessed number) when either team has no real power
    rating - most commonly an FCS/non-FBS opponent CFBD's SP+/PPA/Elo have
    no rating for at all, a normal occurrence in early-season "buy games."
    Callers should treat a `None` result as "no pick for this game," same
    "unknown stays unknown" convention as the rest of this project.
    """
    if home_rating is None or away_rating is None:
        return None

    context_adjustment = context.total_margin_adjustment_pts
    power_diff = home_rating.power_rating - away_rating.power_rating
    predicted_margin = round(power_diff + context_adjustment, 2)

    # Half the situational adjustment is folded into each side's own
    # implied score (rather than all of it onto one side) purely so the
    # two per-team point projections read naturally (e.g. a big home-
    # field edge nudges the home score up AND the away score down a
    # little, instead of one lopsided number) - this is a display/
    # bookkeeping choice, not a claim that HFA specifically boosts
    # offense; predicted_margin above (the number probabilities are
    # actually computed from) already carries the adjustment in full.
    home_expected = LEAGUE_AVG_POINTS_PER_TEAM + home_rating.offense_rating - away_rating.defense_rating + context_adjustment / 2.0
    away_expected = LEAGUE_AVG_POINTS_PER_TEAM + away_rating.offense_rating - home_rating.defense_rating - context_adjustment / 2.0
    home_expected = max(0.0, home_expected)
    away_expected = max(0.0, away_expected)

    weather_multiplier = 1.0 + context.weather_total_adjustment_pct / 100.0
    predicted_home_points = round(home_expected * weather_multiplier, 2)
    predicted_away_points = round(away_expected * weather_multiplier, 2)
    predicted_total = round(predicted_home_points + predicted_away_points, 2)

    home_win_prob = round(_normal_cdf(predicted_margin / MARGIN_STD_DEV), 4)

    components = {
        "home_power_rating": home_rating.power_rating,
        "away_power_rating": away_rating.power_rating,
        "power_diff": round(power_diff, 2),
        "home_field_advantage_pts": context.home_field_advantage_pts,
        "altitude_bonus_pts": context.altitude_bonus_pts,
        "rest_edge_pts": context.rest_edge_pts,
        "travel_edge_pts": context.travel_edge_pts,
        "weather_total_adjustment_pct": context.weather_total_adjustment_pct,
        "home_offense_vs_away_defense": round(home_rating.offense_rating - away_rating.defense_rating, 2),
        "away_offense_vs_home_defense": round(away_rating.offense_rating - home_rating.defense_rating, 2),
    }

    return GameScore(
        matchup=matchup,
        predicted_margin=predicted_margin,
        predicted_home_points=predicted_home_points,
        predicted_away_points=predicted_away_points,
        predicted_total=predicted_total,
        home_win_prob=home_win_prob,
        context=context,
        components=components,
    )
