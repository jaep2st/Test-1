"""Orchestrates the full run: slate -> per-player Statcast/matchup/hot-streak/
park-weather data -> composite scores -> odds -> ranked +EV report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Dict, List, Optional

from odds_monitor.ev import find_fair_prices
from odds_monitor.models import PropLine
from odds_monitor.providers.base import OddsProvider

from .ballparkpal import BallparkPalProvider, NoBallparkPalProvider
from .context import ParkWeatherProvider
from .edges import EdgeCandidate, build_hits_edges, build_hr_edges, build_total_bases_edges, rank_candidates
from .hot_streak import HeatIndex, HotStreakProvider
from .hot_streak import LEAGUE_AVG_WOBA as HOT_STREAK_LEAGUE_AVG_WOBA
from .matchup import LEAGUE_AVG_WOBA as MATCHUP_LEAGUE_AVG_WOBA
from .matchup import MatchupProfile, MatchupProvider
from .schedule import ProbableMatchup, ScheduleProvider
from .scoring import (
    HitsScoreResult,
    HRScoreResult,
    TotalBasesScoreResult,
    compute_hits_score,
    compute_hr_score,
    compute_total_bases_score,
)
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
    hits_edges: List[EdgeCandidate]


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


def _log_slate_time_span(slate: List[ProbableMatchup]) -> None:
    """Logs the day's earliest and latest first-pitch times (UTC), parsed
    from each game's real `game_time_utc` (MLB Stats API's `gameDate`
    field - an ISO8601 UTC timestamp, e.g. '2026-08-27T23:05:00Z').

    Not surfaced in the rendered report itself (a table of scored props has
    no natural place for a single slate-wide fact) - this is specifically
    for external schedulers/log-readers that need to know "when does today's
    slate start/end" without re-deriving it from the full matchup list, e.g.
    timing a check for "an hour before first pitch" or "the middle of the
    slate" (which both vary daily and can't be expressed as a fixed cron
    time without this).
    """
    times: List[datetime] = []
    for game in slate:
        if not game.game_time_utc:
            continue
        try:
            times.append(datetime.fromisoformat(game.game_time_utc.replace("Z", "+00:00")))
        except ValueError:
            logger.warning("Could not parse game_time_utc %r for %s @ %s", game.game_time_utc, game.away_team, game.home_team)
    if not times:
        logger.warning("SLATE_TIME_SPAN: no parseable game times on today's slate (%d games)", len(slate))
        return
    first, last = min(times), max(times)
    logger.info(
        "SLATE_TIME_SPAN first_pitch_utc=%s last_pitch_utc=%s games_with_times=%d/%d",
        first.isoformat(),
        last.isoformat(),
        len(times),
        len(slate),
    )


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


# MLB Stats API's real, observed status strings for a game that hasn't
# started yet (confirmed live 2026-08-29 via mlb_props_main.py's
# --game-status-check). Anything else ("In Progress", "Final", "Game Over",
# "Postponed", "Suspended", etc.) is treated as not-pregame.
#
# "delayed start" confirmed live 2026-08-30 (real game: Cincinnati Reds @
# Chicago Cubs) as MLB's own detailedState for a game held before first
# pitch (rain, etc.) - genuinely still pregame, not "started then paused".
# Missing it here was a real bug: it silently dropped 207 real, currently-
# posted draftkings/betmgm/betrivers odds lines for that exact game,
# because the pair never made it into pregame_pairs below even though the
# game hadn't started. Deliberately NOT matching a bare "delayed" - MLB
# uses statuses like "In Progress - Delayed" for a mid-game hold, which is
# correctly not-pregame and shouldn't be swept in by a broader substring
# match.
_PREGAME_STATUSES = frozenset({"scheduled", "pre-game", "preview", "warmup", "delayed start"})


def _filter_lines_to_confirmed_pregame_games(lines: List[PropLine], slate: List[ProbableMatchup]) -> List[PropLine]:
    """Second, independent safety net on top of the odds provider's own
    commence_time-based live-game filter (see theoddsapi.py's
    `include_live` docstring). Confirmed live (2026-08-29), two real gaps
    that filter alone can't catch:

    1. A real event's `commence_time` can just be wrong - a genuinely
       Final game (Kansas City @ Cleveland) whose odds-provider event
       still looked pregame, so a real price for it slipped through.
    2. A doubleheader's two real games can be exposed by the odds
       provider as a *single* event with no way to tell which specific
       game a price is actually for (confirmed: The Odds API returned
       exactly one event for Boston @ NY Yankees and one for Arizona @
       San Francisco on a day MLB's own schedule shows two real games for
       each - one Final/In Progress, one still Pre-Game/upcoming).

    This cross-checks every priced line against MLB's own authoritative
    per-game status instead of trusting the odds provider's belief about
    game state at all: a matchup with *no* game still pregame (checking
    every real game between those two teams today, so a doubleheader's
    still-upcoming nightcap correctly keeps its price even though the
    earlier game already finished) has its lines dropped entirely.

    A matchup with unknown status (`status=None` - e.g. `--mock` mode, or
    a schedule fetch that didn't populate it) is treated as pregame, same
    as before this check existed - strictly additive, never a new way to
    lose real data on incomplete schedule info. A matchup absent from
    today's schedule entirely (shouldn't happen, but not impossible on a
    date mismatch) is likewise kept rather than dropped on missing info.
    """
    pregame_pairs = set()
    all_pairs = set()
    for game in slate:
        pair = (game.away_team, game.home_team)
        all_pairs.add(pair)
        status = (game.status or "").strip().lower()
        if not status or status in _PREGAME_STATUSES:
            pregame_pairs.add(pair)

    kept: List[PropLine] = []
    dropped = 0
    for line in lines:
        # PropLine.event is "away @ home" - see theoddsapi.py's event_label.
        parts = line.event.split(" @ ", 1)
        pair = (parts[0], parts[1]) if len(parts) == 2 else None
        if pair is None or pair not in all_pairs or pair in pregame_pairs:
            kept.append(line)
        else:
            dropped += 1
    if dropped:
        logger.warning(
            "Dropped %d odds line(s) for game(s) MLB's own schedule confirms are no longer pregame "
            "(the odds provider's own start-time data missed this) - see _filter_lines_to_confirmed_"
            "pregame_games's docstring",
            dropped,
        )
    return kept


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
    ballparkpal: Optional[BallparkPalProvider] = None,
) -> SlateReport:
    ballparkpal = ballparkpal or NoBallparkPalProvider()
    slate = schedule.get_slate(game_date)
    if not slate:
        logger.warning("No games found on the slate for %s", game_date)
    else:
        _log_slate_time_span(slate)

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
    hits_scores: List[HitsScoreResult] = []
    heat_indices: List[HeatIndex] = []
    event_lookup: Dict[str, str] = {}

    for _, batter_name, ctx, batter, pitcher, park_ctx in candidates:
        event_lookup[batter_name] = ctx["event"]
        # Real HR/fly-ball rate is worth fetching per-player here (only ~30
        # candidates, not the full roster) - see StatcastProvider.
        # enrich_batted_ball's docstring for why it isn't in phase 1.
        batter = statcast.enrich_batted_ball(batter)

        # Real, per-hitter park+weather factor from Ballpark Pal, when
        # configured, replaces this project's own static-table + rough
        # Open-Meteo wind/temp estimate for this specific batter - a
        # domain-specific model beats a game-level heuristic shared across
        # every batter in the park. `None` (no key configured, or Ballpark
        # Pal has no projection for this player/game yet) leaves park_ctx
        # untouched, same missing-signal-falls-back-to-existing-behavior
        # posture as every other optional data source in this project.
        bp_factor = ballparkpal.get_hitter_park_factor(batter_name, game_date)
        if bp_factor is not None and bp_factor.home_runs is not None:
            park_ctx = replace(
                park_ctx,
                park_hr_factor=round(bp_factor.home_runs * 100, 1),
                weather_hr_boost_pct=round((bp_factor.home_runs_weather or 0.0) * 100, 1),
            )

        heat = hot_streak.get_heat_index(batter_name, as_of=game_date)
        heat_indices.append(heat)
        matchup = matchup_provider.get_matchup(
            batter_name, batter.bats, pitcher.player, pitcher.throws, pitcher.pitch_mix
        )

        hr_result = compute_hr_score(batter, pitcher, matchup, park_ctx, heat)
        tb_result = compute_total_bases_score(batter, pitcher, matchup, park_ctx, heat)
        hits_result = compute_hits_score(batter, pitcher, matchup, park_ctx, heat)

        # Ballpark Pal's own independent HR/Hits model for this exact
        # matchup, when configured - a genuine second opinion surfaced
        # alongside our own model_prob, never blended into it. See
        # ballparkpal.py's MatchupProbability docstring for why no
        # total-bases figure exists to attach to tb_result.
        bp_matchup = ballparkpal.get_matchup_probability(batter_name, pitcher.player, game_date)
        if bp_matchup is not None:
            if bp_matchup.home_run_model_prob is not None:
                hr_result = replace(hr_result, bp_model_prob=bp_matchup.home_run_model_prob)
            if bp_matchup.hits_model_prob is not None:
                hits_result = replace(hits_result, bp_model_prob=bp_matchup.hits_model_prob)

        hr_scores.append(hr_result)
        tb_scores.append(tb_result)
        hits_scores.append(hits_result)

        # Full component breakdown for every scored candidate - not shown
        # in the report itself (too dense for a ranked table), but useful
        # for writing up *why* a specific player ranked where they did.
        # Logged at INFO (not DEBUG) so it survives even without
        # --log-level DEBUG, and printed with a stable "CANDIDATE_DETAIL"
        # prefix so it's easy to grep out of a run's logs.
        logger.info(
            "CANDIDATE_DETAIL %s vs %s (%s) | %s | %s | %s | HR components=%s | TB components=%s | Hits components=%s",
            batter_name,
            pitcher.player,
            ctx["event"],
            batter,
            pitcher,
            matchup,
            hr_result.components,
            tb_result.components,
            hits_result.components,
        )
        logger.info("CANDIDATE_DETAIL %s heat=%s", batter_name, heat)

    try:
        odds_lines = odds.fetch_player_props("mlb")
    except Exception:
        logger.exception("Failed to fetch MLB player-prop odds")
        odds_lines = []
    odds_lines = _filter_lines_to_confirmed_pregame_games(odds_lines, slate)
    fair_prices = find_fair_prices(odds_lines)

    hr_edges = rank_candidates(build_hr_edges(hr_scores, fair_prices, odds_lines, event_lookup), min_ev_percent)
    tb_edges = rank_candidates(build_total_bases_edges(tb_scores, fair_prices, odds_lines, event_lookup), min_ev_percent)
    hits_edges = rank_candidates(build_hits_edges(hits_scores, fair_prices, odds_lines, event_lookup), min_ev_percent)

    heat_indices.sort(key=lambda h: h.z_score, reverse=True)

    return SlateReport(
        game_date=game_date,
        slate=slate,
        matchup_environments=environments,
        hot_batters=heat_indices,
        hr_edges=hr_edges,
        tb_edges=tb_edges,
        hits_edges=hits_edges,
    )
