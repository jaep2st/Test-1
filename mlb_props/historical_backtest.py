"""Backtests the one piece of this project's model that can be checked
against real MLB history without any risk of lookahead bias:
`hot_streak.StatcastHotStreakProvider.get_heat_index`'s `as_of` date - it
was already built to only ever query Statcast data through a given date,
never after it.

Everything else in scoring.py (barrel%, hard-hit%, xwOBA, ISO, platoon/
pitch-mix - most of the model's actual weight) comes from Baseball
Savant's SEASON leaderboard endpoints (`pybaseball.statcast_batter_
exitvelo_barrels`, `statcast_batter_expected_stats`), which take only a
`year`, no date range - confirmed directly against pybaseball's own real
function signatures (no `start_dt`/`end_dt` parameter exists on either).
Reusing those for a date mid-season would silently let the rest of that
season's games leak into a "prediction" for one game in the middle of it -
real lookahead bias, not something a parameter fixes. This module
deliberately limits itself to the piece that's already safe; backtesting
the rest of the model needs a real as-of-aware batted-ball-quality
computation from raw per-pitch data first (a separate, larger follow-up -
see this project's Performance page for the live-forward version of that
same "is this real signal" question, which doesn't have this constraint).

Player discovery for a past date: this project's real `ScheduleProvider`
builds `away_batters`/`home_batters` from each team's CURRENT active
roster (see schedule.py's `ProbableMatchup` docstring) - fine for "today,"
wrong for a past date (a roster changes all season via trades/call-ups/
injuries). This module instead asks MLB Stats API's real per-game boxscore
(`GET /api/v1/game/{game_pk}/boxscore`) for exactly who batted in that
specific historical game - ground truth, not a roster approximation. The
exact field names below (`stats.batting.plateAppearances`, `person.
fullName`) are a reasonable reading of MLB Stats API's known shape, not
yet confirmed against a real response from this sandbox (no network
access here - see statcast.py's module docstring for why); parsing is
defensive throughout, skipping an unparsable entry with a logged warning
rather than crashing the whole fetch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from .hot_streak import HotStreakProvider
from .report import clearance_rates
from .results import resolve_player_game_outcome
from .schedule import MLB_STATS_API_BASE, ScheduleProvider

logger = logging.getLogger(__name__)


def fetch_boxscore_batters(session, game_pk: int, timeout: float = 10.0, base_url: str = MLB_STATS_API_BASE) -> List[str]:
    """Real batters who actually appeared in one specific, already-played
    MLB game - ground truth for a historical backtest, not a roster
    projection (see this module's docstring). A player counted here has a
    real recorded plate appearance for this exact game, not just a spot
    on the active roster that day.
    """
    resp = session.get(f"{base_url}/game/{game_pk}/boxscore", timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    batters: List[str] = []
    for side in ("away", "home"):
        players = payload.get("teams", {}).get(side, {}).get("players", {})
        if not isinstance(players, dict):
            continue
        for _player_id, entry in players.items():
            try:
                batting = entry.get("stats", {}).get("batting", {})
                # "plateAppearances" is the most direct real field; a
                # missing/zero value there but a real "atBats" still
                # counts (a walk-only or sac-only line can leave PA at 0
                # in some response shapes) - either being real and
                # positive is what "actually batted" means here.
                pa = batting.get("plateAppearances") or batting.get("atBats")
                if not pa:
                    continue
                name = entry.get("person", {}).get("fullName")
                if name:
                    batters.append(name)
            except (KeyError, TypeError, AttributeError):
                logger.warning("Skipping unparsable boxscore player entry for game_pk=%s: %r", game_pk, entry)
    return batters


@dataclass(frozen=True)
class HotStreakObservation:
    """One real (player, game) pair: the hot-streak signal exactly as it
    would have been computed the day before this real game (no lookahead -
    see this module's docstring), and what actually happened. Real data
    only - a market's clearance rate is `None`, never a guessed 0, when
    `clearance_rates` itself would have nothing real to report.
    """

    game_date: str
    player: str
    z_score: float
    l15_clear_hr_rate: Optional[float]
    l15_clear_tb2_rate: Optional[float]
    l15_clear_hit_rate: Optional[float]
    season_clear_hr_rate: Optional[float]
    season_clear_tb2_rate: Optional[float]
    season_clear_hit_rate: Optional[float]
    got_hr: bool
    got_2plus_tb: bool
    got_hit: bool


def collect_hot_streak_observations(
    schedule: ScheduleProvider,
    hot_streak: HotStreakProvider,
    pyb,
    session,
    game_dates: List[date],
    timeout: float = 10.0,
) -> List[HotStreakObservation]:
    """The real collection loop: for every real game on every date in
    `game_dates`, every real batter who actually appeared (via the
    boxscore, not a roster guess), compute the hot-streak signal as of
    the day before (`as_of=game_date - 1 day` - excludes this exact game
    from its own "recent form," the one detail that would otherwise make
    this circular) and resolve what they actually did, reusing
    `results.resolve_player_game_outcome` (already tested, already the
    real production path for nightly result resolution) rather than new
    outcome logic. A player the boxscore lists but whose outcome can't be
    resolved (shouldn't happen, but real APIs are real APIs) is skipped,
    not guessed.
    """
    id_cache: Dict[str, Optional[int]] = {}
    observations: List[HotStreakObservation] = []
    for game_date in game_dates:
        matchups = schedule.get_slate(game_date)
        for matchup in matchups:
            if matchup.game_pk is None:
                continue
            try:
                batters = fetch_boxscore_batters(session, matchup.game_pk, timeout=timeout)
            except Exception:
                logger.exception("Boxscore fetch failed for game_pk=%s (%s @ %s, %s)", matchup.game_pk, matchup.away_team, matchup.home_team, game_date)
                continue
            for player in batters:
                heat = hot_streak.get_heat_index(player, as_of=game_date - timedelta(days=1))
                outcome = resolve_player_game_outcome(pyb, id_cache, player, game_date)
                if outcome is None:
                    continue
                l15_hr, season_hr = clearance_rates(heat, "hr")
                l15_tb2, season_tb2 = clearance_rates(heat, "tb2")
                l15_hit, season_hit = clearance_rates(heat, "hit")
                observations.append(
                    HotStreakObservation(
                        game_date=game_date.isoformat(),
                        player=player,
                        z_score=heat.z_score,
                        l15_clear_hr_rate=l15_hr,
                        l15_clear_tb2_rate=l15_tb2,
                        l15_clear_hit_rate=l15_hit,
                        season_clear_hr_rate=season_hr,
                        season_clear_tb2_rate=season_tb2,
                        season_clear_hit_rate=season_hit,
                        got_hr=outcome.got_hr,
                        got_2plus_tb=outcome.got_2plus_tb,
                        got_hit=outcome.got_hit,
                    )
                )
    return observations


def _real_rate(hits: int, n: int) -> str:
    return f"{hits}/{n} ({hits / n * 100:.1f}%)" if n else "n/a (0 real observations)"


def _bucket_by_z(observations: List[HotStreakObservation]) -> List[Tuple[str, List[HotStreakObservation]]]:
    hot = [o for o in observations if o.z_score >= 1.0]
    cold = [o for o in observations if o.z_score <= -1.0]
    neutral = [o for o in observations if -1.0 < o.z_score < 1.0]
    return [("Hot (z >= +1.0)", hot), ("Neutral (-1.0 < z < +1.0)", neutral), ("Cold (z <= -1.0)", cold)]


def summarize_hot_streak_backtest(observations: List[HotStreakObservation]) -> str:
    """Human-readable report: does the real hot/cold z-score signal (and
    a player's own recent real clearance rate) actually predict what
    happened, or is it noise? Every bucket boundary is disclosed, every
    bucket's real n is shown - a small bucket is flagged, not hidden,
    same posture as backtest.py's calibration_buckets() for live picks.
    """
    out = [f"HISTORICAL HOT-STREAK SIGNAL BACKTEST - {len(observations)} real (player, game) observation(s)", ""]
    if not observations:
        out.append("No real observations collected - nothing to report.")
        return "\n".join(out)

    out.append("Overall real base rates (no-skill baseline this signal has to beat):")
    n = len(observations)
    out.append(f"  1+ HR:      {_real_rate(sum(o.got_hr for o in observations), n)}")
    out.append(f"  2+ TB:      {_real_rate(sum(o.got_2plus_tb for o in observations), n)}")
    out.append(f"  1+ Hit:     {_real_rate(sum(o.got_hit for o in observations), n)}")
    out.append("")

    out.append("By real hot/cold z-score bucket (computed as of the day BEFORE each game - no lookahead):")
    for label, group in _bucket_by_z(observations):
        gn = len(group)
        out.append(f"  {label} - n={gn}")
        if gn:
            out.append(f"    1+ HR:  {_real_rate(sum(o.got_hr for o in group), gn)}")
            out.append(f"    2+ TB:  {_real_rate(sum(o.got_2plus_tb for o in group), gn)}")
            out.append(f"    1+ Hit: {_real_rate(sum(o.got_hit for o in group), gn)}")
    out.append("")

    out.append("Read: if this signal is real, 'Hot' rates should sit above the overall baseline and 'Cold' below it,")
    out.append("in every market, not just by chance in one. Small buckets (roughly n<30) are noisy - read the split")
    out.append("as suggestive, not conclusive, until this has run across more real dates.")
    return "\n".join(out)
