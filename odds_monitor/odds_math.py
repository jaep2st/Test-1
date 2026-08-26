"""American odds <-> implied win probability conversions.

Used for binary (Yes/No) props like "to hit a home run", where the actual
signal worth flagging is a gap in *price* between books, not a point line -
those markets don't carry a line to compare in the first place.
"""


def implied_probability_pct(american_odds: int) -> float:
    """Convert American odds to an implied win probability, as a percentage (0-100)."""
    if american_odds >= 0:
        return 100.0 * 100.0 / (american_odds + 100.0)
    return 100.0 * (-american_odds) / (-american_odds + 100.0)


def probability_pct_to_american_odds(probability_pct: float) -> int:
    """Convert an implied win probability (0-100) to American odds. Inverse of
    implied_probability_pct, used by the mock provider to generate plausible odds
    from a target probability.
    """
    p = min(max(probability_pct / 100.0, 0.01), 0.99)
    if p >= 0.5:
        odds = -100.0 * p / (1.0 - p)
    else:
        odds = 100.0 * (1.0 - p) / p
    return int(round(odds))
