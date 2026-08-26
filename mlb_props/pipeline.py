"""Orchestrates the full run: slate -> per-player Statcast/matchup/hot-streak/
park-weather data -> composite scores -> odds -> ranked +EV report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from odds_monitor.ev import find_fair_prices
from odds_monitor.providers.base import OddsProvider

from .context import ParkWeatherProvider
from .edges import EdgeCandidate, build_hr_edges, build_total_bases_edges, rank_candidates
from .hot_streak import HeatIndex, HotStreakProvider
from .hot_streak import LEAGUE_AVG_WOBA as HOT_STREAK_LEAGUE_AVG_WOBA
from .matchup import LEAGUE_AVG_WOBA as MATCHUP_LEAGUE_AVG_WOBA
from .matchup import MatchupProfile, MatchupProvider
from .schedule import ProbableMatchup, ScheduleProvider
from .scoring import HRScoreResult, TotalBasesScoreResult, compute_hr_score, compute_total_bases_score
from .statcast import StatcastProvider

logger = logging.getLogger(__name__)

DEFAULT_MAX_CANDIDATES = 30


@dataclass(frozen=True)
class MatchupEnvironment:
    """A game's overall HR-friendliness, for the "best matchups" leaderboard -
    independent of any specific player prop.
    """

    matchup: ProbableMatchup
    park_hr_factor: float
    weather_boost_pct: float
    away_pitcher_vulnerability: Optional[float]  # 0-100, higher = easier to hit HR off
    home_pitcher_vulnerability: Optional[float]
    environment_score: float  # 0-100 composite


@dataclass(frozen=True)
class SlateReport:
    game_date: date
    slate: List[ProbableMatchup]
    matchup_environments: List[MatchupEnvironment]
    hot_batters: List
    hr_edges: List[EdgeCandidate]
    tb_edges: List[EdgeCandidate]


def _resolve_batters(slate: List[ProbableMatchup], extra_batters: Optional[List[str]]) -> Dict[str, dict]:
    """Map batter name -> {event, opposing_pitcher, opposing_throws, park, bats_side}.

    MLB Stats API only posts real lineups shortly before first pitch, so
    well ahead of game time `away_batters`/`home_batters` on each
    `ProbableMatchup` may be empty. `extra_batters` lets a caller (the CLI's
    `--batters`) supply names explicitly; they're matched to the first game
    on the slate if no per-team info is available. Prefer real posted
    lineups when present.
    """
    out: Dict[str, dict] = {}
    for game in slate:
        event = f"{game.away_team} @ {game.home_team}"
        for batter in game.away_batters:
            out[batter] = dict(event=event, opposing_pitcher=game.home_pitcher, park=game.venue)
        for batter in game.home_batters:
            out[batter] = dict(event=event, opposing_pitcher=game.away_pitcher, park=game.venue)

    if extra_batters:
        fallback_game = slate[0] if slate else None
        for batter in extra_batters:
            if batter in out:
                continue
            if fallback_game is None:
                continue
            out[batter] = dict(
                event=f"{fallback_game.away_team} @ {fallback_game.home_team}",
                opposing_pitcher=fallback_game.home_pitcher,
                park=fallback_game.venue,
            )
    return out


def _score_matchup_environments(
    slate: List[ProbableMatchup],
    statcast: StatcastProvider,
    park_weather: ParkWeatherProvider,
) -> List[MatchupEnvironment]:
    from .scoring import _normalize  # local import: internal helper, not part of the public scoring API

    environments: List[MatchupEnvironment] = []
    for game in slate:
        pw = park_weather.get_context(game.venue)
        away_vuln = home_vuln = None
        if game.away_pitcher:
            p = statcast.pitcher_profile(game.away_pitcher)
            if p:
                away_vuln = _normalize(
                    (p.barrel_pct_allowed / 12.0 + p.hard_hit_pct_allowed / 46.0 + p.hr_per_9 / 2.1) / 3.0, 0.15, 1.0
                )
        if game.home_pitcher:
            p = statcast.pitcher_profile(game.home_pitcher)
            if p:
                home_vuln = _normalize(
                    (p.barrel_pct_allowed / 12.0 + p.hard_hit_pct_allowed / 46.0 + p.hr_per_9 / 2.1) / 3.0, 0.15, 1.0
                )
        vulns = [v for v in (away_vuln, home_vuln) if v is not None]
        pitcher_component = sum(vulns) / len(vulns) if vulns else 50.0
        park_component = _normalize(pw.park_hr_factor, 85.0, 118.0)
        weather_component = _normalize(pw.weather_hr_boost_pct, -12.0, 12.0)
        environment_score = round(pitcher_component * 0.5 + park_component * 0.3 + weather_component * 0.2, 1)
        environments.append(
            MatchupEnvironment(
                matchup=game,
                park_hr_factor=pw.park_hr_factor,
                weather_boost_pct=pw.weather_hr_boost_pct,
                away_pitcher_vulnerability=away_vuln,
                home_pitcher_vulnerability=home_vuln,
                environment_score=environment_score,
            )
        )
    environments.sort(key=lambda e: e.environment_score, reverse=True)
    return environments


def _neutral_matchup(batter_name: str, pitcher_name: str) -> MatchupProfile:
    """A matchup with no real signal - used as a stand-in during the cheap
    prefilter pass so `compute_hr_score`/`compute_total_bases_score` can run
    without yet paying for a real (network-bound) matchup lookup.
    """
    return MatchupProfile(
        batter=batter_name,
        pitcher=pitcher_name,
        platoon_woba=MATCHUP_LEAGUE_AVG_WOBA,
        platoon_pa=0,
        bvp_pa=0,
        bvp_hr=0,
        bvp_avg=0.0,
        bvp_slg=0.0,
        pitch_mix_edge=0.0,
    )


def _neutral_heat(batter_name: str) -> HeatIndex:
    """A 'perfectly average, not hot or cold' HeatIndex - see `_neutral_matchup`."""
    return HeatIndex(
        player=batter_name,
        season_woba=HOT_STREAK_LEAGUE_AVG_WOBA,
        last7_woba=HOT_STREAK_LEAGUE_AVG_WOBA,
        last15_woba=HOT_STREAK_LEAGUE_AVG_WOBA,
        last30_woba=HOT_STREAK_LEAGUE_AVG_WOBA,
        last15_pa=0,
        z_score=0.0,
    )


def run_pipeline(
    game_date: date,
    schedule: ScheduleProvider,
    statcast: StatcastProvider,
    matchup_provider: MatchupProvider,
    hot_streak: HotStreakProvider,
    park_weather: ParkWeatherProvider,
    odds: OddsProvider,
    extra_batters: Optional[List[str]] = None,
    min_ev_percent: float = 0.0,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> SlateReport:
    slate = schedule.get_slate(game_date)
    if not slate:
        logger.warning("No games found on the slate for %s", game_date)

    batter_context = _resolve_batters(slate, extra_batters)
    environments = _score_matchup_environments(slate, statcast, park_weather)

    # Phase 1 - cheap prefilter: Statcast batter/pitcher profiles and park/
    # weather all come from a season leaderboard fetched *once* and cached
    # (see PybaseballStatcastProvider), so scoring every roster batter with
    # neutral matchup/hot-streak inputs is fast even across a full slate.
    # Real matchup and hot-streak data cost a network round trip *per
    # player* on the real providers though, so we only pay for those on the
    # `max_candidates` most promising batters by this cheap pass - not
    # every batter on every active roster (which can be 150-250+ players).
    # Each entry: (prefilter_score, batter_name, ctx, BatterProfile, PitcherProfile, park_ctx)
    prelim: List[tuple] = []
    for batter_name, ctx in batter_context.items():
        batter = statcast.batter_profile(batter_name)
        if batter is None:
            logger.warning("Skipping %s - no Statcast batter profile available", batter_name)
            continue

        opposing_pitcher_name = ctx.get("opposing_pitcher")
        pitcher = statcast.pitcher_profile(opposing_pitcher_name) if opposing_pitcher_name else None
        if pitcher is None:
            logger.warning("Skipping %s - no opposing probable pitcher/profile available", batter_name)
            continue

        park_ctx = park_weather.get_context(ctx["park"])
        neutral_matchup = _neutral_matchup(batter_name, pitcher.player)
        neutral_heat = _neutral_heat(batter_name)
        prelim_hr = compute_hr_score(batter, pitcher, neutral_matchup, park_ctx, neutral_heat)
        prelim_tb = compute_total_bases_score(batter, pitcher, neutral_matchup, park_ctx, neutral_heat)
        prelim.append((max(prelim_hr.score, prelim_tb.score), batter_name, ctx, batter, pitcher, park_ctx))

    prelim.sort(key=lambda c: c[0], reverse=True)
    candidates = prelim[:max_candidates]
    if len(prelim) > max_candidates:
        logger.info(
            "Prefiltered %d batters down to the top %d by cheap Statcast-only score before running "
            "real matchup/hot-streak lookups",
            len(prelim),
            max_candidates,
        )

    # Phase 2 - full score: only for the prefiltered candidates.
    hr_scores: List[HRScoreResult] = []
    tb_scores: List[TotalBasesScoreResult] = []
    heat_indices: List[HeatIndex] = []
    event_lookup: Dict[str, str] = {}

    for _, batter_name, ctx, batter, pitcher, park_ctx in candidates:
        event_lookup[batter_name] = ctx["event"]
        # Real HR/fly-ball rate is worth fetching per-player here (only ~30
        # candidates, not the full roster) - see StatcastProvider.
        # enrich_batted_ball's docstring for why it isn't in phase 1.
        batter = statcast.enrich_batted_ball(batter)
        heat = hot_streak.get_heat_index(batter_name, as_of=game_date)
        heat_indices.append(heat)
        matchup = matchup_provider.get_matchup(
            batter_name, batter.bats, pitcher.player, pitcher.throws, pitcher.pitch_mix
        )

        hr_result = compute_hr_score(batter, pitcher, matchup, park_ctx, heat)
        tb_result = compute_total_bases_score(batter, pitcher, matchup, park_ctx, heat)
        hr_scores.append(hr_result)
        tb_scores.append(tb_result)

        # Full component breakdown for every scored candidate - not shown
        # in the report itself (too dense for a ranked table), but useful
        # for writing up *why* a specific player ranked where they did.
        # Logged at INFO (not DEBUG) so it survives even without
        # --log-level DEBUG, and printed with a stable "CANDIDATE_DETAIL"
        # prefix so it's easy to grep out of a run's logs.
        logger.info(
            "CANDIDATE_DETAIL %s vs %s (%s) | %s | %s | %s | HR components=%s | TB components=%s",
            batter_name,
            pitcher.player,
            ctx["event"],
            batter,
            pitcher,
            matchup,
            hr_result.components,
            tb_result.components,
        )
        logger.info("CANDIDATE_DETAIL %s heat=%s", batter_name, heat)

    try:
        odds_lines = odds.fetch_player_props("mlb")
    except Exception:
        logger.exception("Failed to fetch MLB player-prop odds")
        odds_lines = []
    fair_prices = find_fair_prices(odds_lines)

    hr_edges = rank_candidates(build_hr_edges(hr_scores, fair_prices, odds_lines, event_lookup), min_ev_percent)
    tb_edges = rank_candidates(build_total_bases_edges(tb_scores, fair_prices, odds_lines, event_lookup), min_ev_percent)

    heat_indices.sort(key=lambda h: h.z_score, reverse=True)

    return SlateReport(
        game_date=game_date,
        slate=slate,
        matchup_environments=environments,
        hot_batters=heat_indices,
        hr_edges=hr_edges,
        tb_edges=tb_edges,
    )
