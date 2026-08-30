"""Composite scoring: blend batted-ball quality, matchup edges, recent form,
and park/weather into a single 0-100 score per player per prop type, plus a
heuristic model probability used to cross-check market prices for +EV.

This is a transparent, hand-weighted heuristic - not a trained/calibrated
model. Treat the resulting "model probability" as a directional estimate to
compare against the market's own no-vig consensus (see `odds_monitor.ev`),
not as ground truth. Component weights and normalization bounds are called
out explicitly below so they're easy to inspect and adjust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .context import ParkWeatherContext
from .hot_streak import HeatIndex
from .matchup import MatchupProfile, LEAGUE_AVG_WOBA
from .statcast import BatterProfile, PitcherProfile


def _normalize(value: float, lo: float, hi: float) -> float:
    """Map value into 0-100 given an expected realistic [lo, hi] range,
    clipped at the ends. Not a true population percentile - see module
    docstring.
    """
    if hi == lo:
        return 50.0
    pct = (value - lo) / (hi - lo) * 100.0
    return max(0.0, min(100.0, pct))


def _lerp3(score: float, at0: float, at50: float, at100: float) -> float:
    """Piecewise-linear calibration curve from a 0-100 score to a
    probability, anchored at three realistic base-rate points."""
    if score <= 50:
        return at0 + (at50 - at0) * (score / 50.0)
    return at50 + (at100 - at50) * ((score - 50) / 50.0)


# ---------------------------------------------------------------------------
# Home run score
# ---------------------------------------------------------------------------

# (weight, normalization range) for each HR-relevant component. Ranges are
# realistic full-season MLB bounds for qualified hitters/pitchers.
HR_WEIGHTS: Dict[str, float] = {
    "barrel_pct": 0.18,
    "hard_hit_pct": 0.13,
    "avg_exit_velo": 0.08,
    "hr_fb_pct": 0.12,
    "pull_air_pct": 0.06,
    "platoon_edge": 0.10,
    "pitcher_allowed": 0.12,
    "pitch_mix_edge": 0.05,
    "park_factor": 0.06,
    "weather_boost": 0.08,
    "hot_streak": 0.02,
}
assert abs(sum(HR_WEIGHTS.values()) - 1.0) < 1e-9


@dataclass(frozen=True)
class HRScoreResult:
    player: str
    score: float  # 0-100 composite
    model_prob: float  # heuristic per-game HR probability implied by the score
    components: Dict[str, float]  # each component's 0-100 contribution, for transparency
    # Raw weather inputs, surfaced separately from `components` so callers
    # can show *why* weather moved the score, not just that it did.
    park: str
    wind_out_mph: float  # positive = blowing out (helps HR), negative = blowing in
    temp_f: Optional[float]
    is_dome: bool
    weather_boost_pct: float  # heuristic % shift to HR odds from wind + temp


def compute_hr_score(
    batter: BatterProfile,
    pitcher: PitcherProfile,
    matchup: MatchupProfile,
    park_weather: ParkWeatherContext,
    heat: HeatIndex,
) -> HRScoreResult:
    platoon_diff = matchup.platoon_woba - LEAGUE_AVG_WOBA
    components = {
        "barrel_pct": _normalize(batter.barrel_pct, 3.0, 22.0),
        "hard_hit_pct": _normalize(batter.hard_hit_pct, 28.0, 58.0),
        "avg_exit_velo": _normalize(batter.avg_exit_velo, 85.0, 95.5),
        "hr_fb_pct": _normalize(batter.hr_fb_pct, 6.0, 34.0),
        "pull_air_pct": _normalize(batter.pull_air_pct, 12.0, 48.0),
        "platoon_edge": _normalize(platoon_diff, -0.06, 0.06),
        # A "bad" pitcher (allows lots of hard/barreled contact and HR/9) is
        # good for the batter, so this is intentionally *not* inverted vs.
        # the pitcher's own quality - it's the batter's opportunity.
        "pitcher_allowed": _normalize(
            (pitcher.barrel_pct_allowed / 12.0 + pitcher.hard_hit_pct_allowed / 46.0 + pitcher.hr_per_9 / 2.1) / 3.0,
            0.15,
            1.0,
        ),
        "pitch_mix_edge": _normalize(matchup.pitch_mix_edge, -0.045, 0.045),
        "park_factor": _normalize(park_weather.park_hr_factor, 85.0, 118.0),
        "weather_boost": _normalize(park_weather.weather_hr_boost_pct, -12.0, 12.0),
        "hot_streak": _normalize(heat.z_score, -2.0, 2.0),
    }
    score = sum(components[k] * w for k, w in HR_WEIGHTS.items())
    # Calibration anchors: ~4% HR probability for a replacement-level bat in
    # a tough spot, ~10% for a league-average everyday hitter in a neutral
    # spot, ~23% for an elite power bat in a great matchup/park/weather spot.
    model_prob = _lerp3(score, at0=0.04, at50=0.10, at100=0.23) / 1.0
    return HRScoreResult(
        player=batter.player,
        score=round(score, 1),
        model_prob=round(model_prob, 4),
        components=components,
        park=park_weather.park,
        wind_out_mph=park_weather.wind_out_mph,
        temp_f=park_weather.temp_f,
        is_dome=park_weather.is_dome,
        weather_boost_pct=park_weather.weather_hr_boost_pct,
    )


# ---------------------------------------------------------------------------
# 2+ total bases score
# ---------------------------------------------------------------------------

TB_WEIGHTS: Dict[str, float] = {
    "iso": 0.18,
    "xslg": 0.17,
    "hard_hit_pct": 0.14,
    "sweet_spot_pct": 0.08,
    "barrel_pct": 0.10,
    "platoon_edge": 0.10,
    "pitcher_allowed": 0.12,
    "park_factor": 0.04,
    "weather_boost": 0.05,
    "hot_streak": 0.02,
}
assert abs(sum(TB_WEIGHTS.values()) - 1.0) < 1e-9


@dataclass(frozen=True)
class TotalBasesScoreResult:
    player: str
    score: float
    model_prob: float  # heuristic per-game P(2+ total bases)
    components: Dict[str, float]
    park: str
    wind_out_mph: float
    temp_f: Optional[float]
    is_dome: bool
    weather_boost_pct: float


def compute_total_bases_score(
    batter: BatterProfile,
    pitcher: PitcherProfile,
    matchup: MatchupProfile,
    park_weather: ParkWeatherContext,
    heat: HeatIndex,
) -> TotalBasesScoreResult:
    platoon_diff = matchup.platoon_woba - LEAGUE_AVG_WOBA
    components = {
        "iso": _normalize(batter.iso, 0.100, 0.320),
        "xslg": _normalize(batter.xslg, 0.360, 0.650),
        "hard_hit_pct": _normalize(batter.hard_hit_pct, 28.0, 58.0),
        "sweet_spot_pct": _normalize(batter.sweet_spot_pct, 22.0, 45.0),
        "barrel_pct": _normalize(batter.barrel_pct, 3.0, 22.0),
        "platoon_edge": _normalize(platoon_diff, -0.06, 0.06),
        "pitcher_allowed": _normalize(
            (pitcher.hard_hit_pct_allowed / 46.0 + pitcher.xslg_allowed / 0.47) / 2.0, 0.5, 1.15
        ),
        "park_factor": _normalize(park_weather.park_hr_factor, 85.0, 118.0),
        "weather_boost": _normalize(park_weather.weather_hr_boost_pct, -12.0, 12.0),
        "hot_streak": _normalize(heat.z_score, -2.0, 2.0),
    }
    score = sum(components[k] * w for k, w in TB_WEIGHTS.items())
    # Calibration anchors: a typical everyday hitter reaches 2+ total bases
    # in roughly 38-42% of games; elite power/contact bats in great spots
    # can push well past 60%.
    model_prob = _lerp3(score, at0=0.24, at50=0.40, at100=0.63)
    return TotalBasesScoreResult(
        player=batter.player,
        score=round(score, 1),
        model_prob=round(model_prob, 4),
        components=components,
        park=park_weather.park,
        wind_out_mph=park_weather.wind_out_mph,
        temp_f=park_weather.temp_f,
        is_dome=park_weather.is_dome,
        weather_boost_pct=park_weather.weather_hr_boost_pct,
    )


# ---------------------------------------------------------------------------
# 1+ hits score
# ---------------------------------------------------------------------------

# NOTE ON A REAL DATA GAP: getting at least one hit is, in real scouting
# terms, driven heavily by a batter's strikeout/contact rate (fewer strikeouts
# = more balls in play = more chances for a hit) and by the pitcher's own
# strikeout rate. Neither is available from the Statcast leaderboards this
# project pulls (see statcast.py's module docstring - `BatterProfile`/
# `PitcherProfile` carry batted-ball *quality* once contact is made, not
# swing-and-miss/strikeout tendency at all). That's a real, acknowledged
# blind spot: a high-power, high-strikeout slugger with elite exit velocity
# will score well here despite being a genuinely worse "gets a hit" bet than
# a high-contact, lower-power hitter this model can't distinguish from a
# below-average one on that axis. Weighted to lean on `xwoba` (a contact+
# power blend less skewed toward pure power than `iso`/barrel%) as the
# primary signal for that reason, with hot-streak form given a bigger role
# than in the HR/TB scores to partially compensate. Park/weather are
# deliberately excluded entirely - both this project's park factor and
# weather-boost inputs are specifically HR-oriented (see context.py) and
# have no real relationship to a batter simply making contact for a hit.
HITS_WEIGHTS: Dict[str, float] = {
    "xwoba": 0.22,
    "hard_hit_pct": 0.14,
    "sweet_spot_pct": 0.10,
    "barrel_pct": 0.08,
    "avg_exit_velo": 0.06,
    "platoon_edge": 0.14,
    "pitcher_allowed": 0.18,
    "hot_streak": 0.08,
}
assert abs(sum(HITS_WEIGHTS.values()) - 1.0) < 1e-9


@dataclass(frozen=True)
class HitsScoreResult:
    player: str
    score: float
    model_prob: float  # heuristic per-game P(1+ hits)
    components: Dict[str, float]
    # Park/weather aren't scoring inputs for this market (see the module
    # note above) - carried through only so `EdgeCandidate`/the report's
    # weather column, shared with HR/TB, still has something to show.
    park: str
    wind_out_mph: float
    temp_f: Optional[float]
    is_dome: bool
    weather_boost_pct: float


def compute_hits_score(
    batter: BatterProfile,
    pitcher: PitcherProfile,
    matchup: MatchupProfile,
    park_weather: ParkWeatherContext,
    heat: HeatIndex,
) -> HitsScoreResult:
    platoon_diff = matchup.platoon_woba - LEAGUE_AVG_WOBA
    components = {
        "xwoba": _normalize(batter.xwoba, 0.290, 0.430),
        "hard_hit_pct": _normalize(batter.hard_hit_pct, 28.0, 58.0),
        "sweet_spot_pct": _normalize(batter.sweet_spot_pct, 22.0, 45.0),
        "barrel_pct": _normalize(batter.barrel_pct, 3.0, 22.0),
        "avg_exit_velo": _normalize(batter.avg_exit_velo, 85.0, 95.5),
        "platoon_edge": _normalize(platoon_diff, -0.06, 0.06),
        "pitcher_allowed": _normalize(
            (pitcher.xwoba_allowed / 0.36 + pitcher.hard_hit_pct_allowed / 46.0) / 2.0, 0.55, 1.15
        ),
        "hot_streak": _normalize(heat.z_score, -2.0, 2.0),
    }
    score = sum(components[k] * w for k, w in HITS_WEIGHTS.items())
    # Calibration anchors: a real MLB everyday hitter gets 1+ hits in
    # roughly 65-70% of games (4ish PA/game adds up even for a modest
    # hitter); a cold bat in a tough matchup can fall to ~45-50%, and an
    # elite-contact bat in a great spot can push past 80%.
    model_prob = _lerp3(score, at0=0.48, at50=0.66, at100=0.82)
    return HitsScoreResult(
        player=batter.player,
        score=round(score, 1),
        model_prob=round(model_prob, 4),
        components=components,
        park=park_weather.park,
        wind_out_mph=park_weather.wind_out_mph,
        temp_f=park_weather.temp_f,
        is_dome=park_weather.is_dome,
        weather_boost_pct=park_weather.weather_hr_boost_pct,
    )
