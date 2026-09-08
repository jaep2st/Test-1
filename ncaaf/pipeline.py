"""Orchestrates the full run: this week's slate -> blended power ratings
-> per-game situational context -> predicted margin/total/probabilities
-> real cross-book odds -> ranked +EV game-line picks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

from .context import build_game_context, VenueProvider, WeatherProvider
from .edges import EdgeCandidate, build_game_edges, rank_candidates
from .market import GameOddsProvider
from .ratings import RatingsProvider
from .schedule import Matchup, ScheduleProvider
from .scoring import GameScore, compute_game_score

logger = logging.getLogger(__name__)

DEFAULT_REST_LOOKBACK_WEEKS = 4


@dataclass(frozen=True)
class WeekReport:
    season: int
    week: int
    season_type: str
    slate: List[Matchup]
    scores: Dict[str, GameScore]  # keyed by "Away @ Home"
    candidates: List[EdgeCandidate]  # ranked


def _team_last_game_dates(schedule: ScheduleProvider, season: int, week: int, season_type: str) -> Dict[str, datetime]:
    """Best-effort real "last played" date per team, from the previous few
    real weeks' slates - used only to compute a rest-days edge (see
    `context.rest_edge_points`). A team not found (bye-week gaps longer
    than the lookback window, a mock provider with no real week-to-week
    history, or this being their season opener) simply gets no rest-days
    signal - `context.py` treats that as a real 0.0 adjustment, not a
    guess.
    """
    last_game: Dict[str, datetime] = {}
    for wk in range(max(1, week - DEFAULT_REST_LOOKBACK_WEEKS), week):
        try:
            games = schedule.get_week_slate(season, wk, season_type)
        except Exception:
            logger.exception("Failed to fetch week %s slate for rest-days lookback", wk)
            continue
        for m in games:
            if not m.start_date_utc:
                continue
            try:
                dt = datetime.fromisoformat(m.start_date_utc.replace("Z", "+00:00"))
            except ValueError:
                continue
            for team in (m.home_team, m.away_team):
                if team not in last_game or dt > last_game[team]:
                    last_game[team] = dt
    return last_game


def run_pipeline(
    season: int,
    week: int,
    schedule: ScheduleProvider,
    ratings_provider: RatingsProvider,
    venue_provider: VenueProvider,
    weather_provider: WeatherProvider,
    odds: GameOddsProvider,
    season_type: str = "regular",
    min_ev_percent: float = 0.0,
    compute_rest_days: bool = True,
) -> WeekReport:
    slate = schedule.get_week_slate(season, week, season_type)
    if not slate:
        logger.warning("No games found for %s week %s (%s)", season, week, season_type)

    ratings = ratings_provider.get_ratings(season)
    last_game_dates = _team_last_game_dates(schedule, season, week, season_type) if compute_rest_days else {}

    scores: Dict[str, GameScore] = {}
    unrated_games = 0
    for m in slate:
        home_rating = ratings.get(m.home_team)
        away_rating = ratings.get(m.away_team)
        if home_rating is None or away_rating is None:
            unrated_games += 1
            logger.info(
                "No power rating for %s - skipping (common for an FCS/non-FBS opponent)", m.event
            )
            continue

        home_rest = away_rest = None
        if m.start_date_utc:
            try:
                kickoff = datetime.fromisoformat(m.start_date_utc.replace("Z", "+00:00"))
                if m.home_team in last_game_dates:
                    home_rest = (kickoff - last_game_dates[m.home_team]).days
                if m.away_team in last_game_dates:
                    away_rest = (kickoff - last_game_dates[m.away_team]).days
            except ValueError:
                pass

        context = build_game_context(m, venue_provider, weather_provider, home_rest, away_rest)
        score = compute_game_score(m, home_rating, away_rating, context)
        if score is not None:
            scores[score.event] = score

    if unrated_games:
        logger.info("%d of %d games skipped for missing power ratings this week", unrated_games, len(slate))

    try:
        lines = odds.fetch_game_odds()
    except Exception:
        logger.exception("Failed to fetch NCAAF game odds")
        lines = []

    candidates = rank_candidates(build_game_edges(scores, lines), min_ev_percent)

    return WeekReport(season=season, week=week, season_type=season_type, slate=slate, scores=scores, candidates=candidates)
