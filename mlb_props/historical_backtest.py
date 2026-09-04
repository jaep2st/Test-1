"""Backtests the one piece of this project's model that can be checked
against real MLB history without any risk of lookahead bias:
`hot_streak.StatcastHotStreakProvider.get_heat_index`'s `as_of` date - it
was already built to only ever query Statcast data through a given date,
never after it. This module reuses that same real computation
(`hot_streak.heat_index_from_log`, extracted specifically so it could be
reused here) against each real player's own Statcast log, fetched once
per player for the whole backtest window rather than once per (player,
game) observation - the fix for a real, measured problem: the first
version of this re-fetched a player's entire season-to-date log
separately for every date they appeared, which took ~2 hours for a
5-day window. See `collect_hot_streak_observations`'s docstring for the
exact fix.

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
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from ._ids import lookup_mlbam_id
from .hot_streak import game_outcomes_from_events, heat_index_from_log
from .report import clearance_rates
from .results import _append_jsonl, _load_jsonl
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


def _outcome_from_log(log, game_date: date) -> Optional["Tuple[bool, bool, bool]"]:
    """The real (got_hr, got_2plus_tb, got_hit) outcome for one exact date,
    read directly out of a player's already-fetched multi-date log - the
    same real per-game outcome logic `results.resolve_player_game_outcome`
    uses, just sliced locally instead of a second network fetch for a
    date this log already covers. `None` (never a guessed False) if the
    log has no real plate-appearance rows for this exact date - "unknown
    stays unknown," same convention as the rest of this project.
    """
    if log is None or len(log) == 0 or "events" not in log.columns or "game_date" not in log.columns:
        return None
    day_rows = log[(log["game_date"] == game_date.isoformat()) & log["events"].notna()]
    if day_rows.empty:
        return None
    return game_outcomes_from_events([list(day_rows["events"])])[0]


def collect_hot_streak_observations(
    schedule: ScheduleProvider,
    pyb,
    game_dates: List[date],
    season_start: date,
    session,
    timeout: float = 10.0,
) -> List[HotStreakObservation]:
    """The real collection loop: for every real game on every date in
    `game_dates`, every real batter who actually appeared (via the
    boxscore, not a roster guess), compute the hot-streak signal as of
    the day before (`as_of=game_date - 1 day` - excludes this exact game
    from its own "recent form," the one detail that would otherwise make
    this circular) and resolve what they actually did.

    The one thing that made the first version of this impractically slow
    (~2 hours for a 5-day window): it re-fetched a player's ENTIRE
    season-to-date Statcast log separately for every single date they
    appeared in `game_dates` - a player who shows up in 5 of this
    window's games got the same multi-month history re-downloaded 5
    times, each just a day later than the last. Each real player's log
    is now fetched exactly ONCE here (`season_start` through the latest
    date in `game_dates`), cached, and reused both for that player's
    hot-streak signal (via `hot_streak.heat_index_from_log`, which
    correctly filters to `game_date <= as_of` on its own - see that
    function's docstring for why that's still lookahead-safe against a
    log that spans dates after any individual as_of) and for every one
    of their real per-game outcomes (via `_outcome_from_log` above) -
    zero extra network calls either way. A player whose log can't be
    fetched, or who the boxscore lists but the log has no real outcome
    for on that exact date (shouldn't happen, but real APIs are real
    APIs), is skipped for that one observation, not guessed.
    """
    id_cache: Dict[str, Optional[int]] = {}
    log_cache: Dict[str, object] = {}
    end_date = max(game_dates)

    def player_log(player: str):
        if player in log_cache:
            return log_cache[player]
        player_id = lookup_mlbam_id(pyb, player, id_cache)
        if player_id is None:
            log_cache[player] = None
            return None
        try:
            log = pyb.statcast_batter(season_start.isoformat(), end_date.isoformat(), player_id)
        except Exception:
            logger.exception("statcast_batter fetch failed for %r", player)
            log = None
        log_cache[player] = log
        return log

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
                log = player_log(player)
                outcome = _outcome_from_log(log, game_date)
                if outcome is None:
                    continue
                got_hr, got_2plus_tb, got_hit = outcome
                heat = heat_index_from_log(player, log, as_of=game_date - timedelta(days=1))
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
                        got_hr=got_hr,
                        got_2plus_tb=got_2plus_tb,
                        got_hit=got_hit,
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


@dataclass(frozen=True)
class HistoricalBacktestRun:
    """One full `--historical-backtest-hot-streak` run's real summary
    stats, persisted to `data/historical_backtest/runs.jsonl` (see
    `record_historical_backtest_run`) so this evidence accumulates across
    runs instead of scrolling away the moment the workflow step's log
    disappears - the same "the git repo is the database" convention
    `results.py` already uses for picks/results/CLV. Every rate here is
    the exact real number `summarize_hot_streak_backtest` already prints;
    this is that same computation, structured for storage and re-display
    (see `performance_report.py`'s historical-backtest section) instead
    of only ever appearing as one run's console text.
    """

    run_at: str  # ISO 8601 UTC, when this backtest was executed
    start_date: str  # inclusive
    end_date: str  # inclusive
    n_observations: int
    # {"hr": 0.10, "tb2": 0.32, "hit": 0.57} - the real overall base rates
    # every bucket below has to beat/undershoot for the signal to be real.
    overall_rates: Dict[str, float]
    # {"Hot (z >= +1.0)": {"n": 627, "hr": 0.129, "tb2": 0.344, "hit": 0.603}, ...}
    # - same three buckets/order as _bucket_by_z, real n disclosed per
    # bucket so a reader can judge for themselves whether it's noise.
    buckets: Dict[str, Dict[str, float]]


def build_historical_backtest_run(
    observations: List[HotStreakObservation], start_date: date, end_date: date, run_at: Optional[datetime] = None
) -> Optional[HistoricalBacktestRun]:
    """The same real computation `summarize_hot_streak_backtest` prints,
    structured into a `HistoricalBacktestRun` instead of formatted text -
    see that function for the exact same bucket logic/thresholds. `None`
    for zero real observations (nothing real to persist), same "don't
    write a phantom record" convention as `results.record_picks` et al.
    """
    if not observations:
        return None
    run_at = run_at or datetime.now(timezone.utc)
    n = len(observations)

    def rates(group: List[HotStreakObservation]) -> Dict[str, float]:
        gn = len(group)
        return {
            "hr": round(sum(o.got_hr for o in group) / gn, 4),
            "tb2": round(sum(o.got_2plus_tb for o in group) / gn, 4),
            "hit": round(sum(o.got_hit for o in group) / gn, 4),
        }

    buckets = {}
    for label, group in _bucket_by_z(observations):
        bucket_rates = {"n": len(group)}
        if group:
            bucket_rates.update(rates(group))
        buckets[label] = bucket_rates

    return HistoricalBacktestRun(
        run_at=run_at.isoformat(),
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        n_observations=n,
        overall_rates=rates(observations),
        buckets=buckets,
    )


def record_historical_backtest_run(run: Optional[HistoricalBacktestRun], out_path: str) -> int:
    """Appends one real backtest run's summary to `out_path` (a JSONL
    file, same convention as results.py's picks/results/CLV history).
    `None` (the zero-observations case) writes nothing and returns 0 -
    never a phantom record for a run that found nothing real."""
    if run is None:
        return 0
    return _append_jsonl([run], out_path)


def load_historical_backtest_runs(path: str) -> List[HistoricalBacktestRun]:
    return _load_jsonl(path, HistoricalBacktestRun)
