"""Turns the real recorded history in `mlb_props/results.py` into an "is
this actually working" readout: calibration (do our probabilities mean what
they say), closing-line value (are our picks beating the market's own
closing price), and hit rate broken down by market/tier. Every number here
is computed from real recorded picks and real resolved outcomes - never
recomputed model output, never a guess.

Small samples are the norm early on, especially right after this shipped -
every stat here reports its own sample size alongside the number, never
hides it. See `performance_report.py` for how this gets rendered.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from .results import ClvRecord, GameOutcome, PickRecord, latest_pick_per_key, load_clv, load_picks, load_results

# Same US-Eastern convention this project already anchors "today" to (see
# mlb_props_main.py's _MLB_TZ) - MLB is a US league, so "what time did this
# run" should mean the real Eastern hour a person would recognize, not
# whatever timezone the runner happens to be in (always UTC on GitHub
# Actions).
_ET = ZoneInfo("America/New_York")


def _load_all(data_dir: str, subdir: str, loader) -> list:
    pattern = str(Path(data_dir) / subdir / "*.jsonl")
    out = []
    for path in sorted(glob.glob(pattern)):
        out.extend(loader(path))
    return out


def load_all_picks(data_dir: str) -> List[PickRecord]:
    return _load_all(data_dir, "picks", load_picks)


def load_all_results(data_dir: str) -> List[GameOutcome]:
    return _load_all(data_dir, "results", load_results)


def load_all_clv(data_dir: str) -> List[ClvRecord]:
    return _load_all(data_dir, "clv", load_clv)


def latest_results_by_key(results: List[GameOutcome]) -> Dict[Tuple[str, str], GameOutcome]:
    """Collapses repeated resolutions of the same (player, game_date) down
    to the most recently written one. Append order in the JSONL file is
    chronological, so "last wins" is enough without a separate timestamp
    field - unlike a `PickRecord`, a resolved real outcome doesn't change
    between resolutions barring a correction upstream.
    """
    by_key: Dict[Tuple[str, str], GameOutcome] = {}
    for o in results:
        by_key[(o.player.strip().lower(), o.game_date)] = o
    return by_key


@dataclass(frozen=True)
class ResolvedPick:
    """One recorded pick joined with its real resolved outcome - the unit
    every stat below is built from."""

    pick: PickRecord
    won: bool


def resolve_picks(picks: List[PickRecord], results: List[GameOutcome]) -> List[ResolvedPick]:
    """Joins the latest snapshot of every recorded pick against its real
    resolved outcome. Picks with no resolved outcome yet (the game hasn't
    been resolved) or an unrecognized market are silently excluded - every
    stat downstream only ever counts picks we actually know the real answer
    for, never an assumed loss.
    """
    outcomes = latest_results_by_key(results)
    resolved = []
    for pick in latest_pick_per_key(picks).values():
        outcome = outcomes.get((pick.player.strip().lower(), pick.game_date))
        if outcome is None:
            continue
        won = outcome.hit_for(pick.market)
        if won is None:
            continue
        resolved.append(ResolvedPick(pick=pick, won=won))
    return resolved


@dataclass(frozen=True)
class CalibrationBucket:
    """One probability decile: what the model said vs. what really
    happened, for every resolved pick (any tier, with or without a real
    market price - this checks the model itself, not just the +EV picks)
    whose `model_prob` fell in `[lo, hi)`. `n == 0` buckets are kept (not
    dropped) so an empty range in the chart is visibly empty, not missing.
    """

    lo: float
    hi: float
    n: int
    predicted_mean: Optional[float]
    actual_rate: Optional[float]


def calibration_buckets(resolved: List[ResolvedPick], n_buckets: int = 10) -> List[CalibrationBucket]:
    edges = [i / n_buckets for i in range(n_buckets + 1)]
    buckets = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        in_bucket = [r for r in resolved if lo <= r.pick.model_prob < hi or (hi >= 1.0 and r.pick.model_prob >= hi)]
        if not in_bucket:
            buckets.append(CalibrationBucket(lo=lo, hi=hi, n=0, predicted_mean=None, actual_rate=None))
            continue
        predicted_mean = sum(r.pick.model_prob for r in in_bucket) / len(in_bucket)
        actual_rate = sum(1 for r in in_bucket if r.won) / len(in_bucket)
        buckets.append(
            CalibrationBucket(
                lo=lo, hi=hi, n=len(in_bucket), predicted_mean=round(predicted_mean, 4), actual_rate=round(actual_rate, 4)
            )
        )
    return buckets


@dataclass(frozen=True)
class ClvSummary:
    n: int
    mean_clv_percent: Optional[float]
    beat_close_percent: Optional[float]  # 0-100: share of picks with clv_percent > 0


def clv_summary(clv_records: List[ClvRecord]) -> ClvSummary:
    if not clv_records:
        return ClvSummary(n=0, mean_clv_percent=None, beat_close_percent=None)
    n = len(clv_records)
    mean_clv = sum(r.clv_percent for r in clv_records) / n
    beat = sum(1 for r in clv_records if r.clv_percent > 0) / n * 100.0
    return ClvSummary(n=n, mean_clv_percent=round(mean_clv, 2), beat_close_percent=round(beat, 1))


@dataclass(frozen=True)
class HitRateGroup:
    key: str
    n: int
    hit_rate: float


def _group_hit_rate(resolved: List[ResolvedPick], key_fn: Callable[[ResolvedPick], str]) -> List[HitRateGroup]:
    groups: Dict[str, List[ResolvedPick]] = {}
    for r in resolved:
        groups.setdefault(key_fn(r), []).append(r)
    out = []
    for key, items in sorted(groups.items()):
        rate = sum(1 for r in items if r.won) / len(items)
        out.append(HitRateGroup(key=key, n=len(items), hit_rate=round(rate, 4)))
    return out


def hit_rate_by_market(resolved: List[ResolvedPick]) -> List[HitRateGroup]:
    return _group_hit_rate(resolved, lambda r: r.pick.market)


def hit_rate_by_tier(resolved: List[ResolvedPick]) -> List[HitRateGroup]:
    return _group_hit_rate(resolved, lambda r: r.pick.tier)


def hit_rate_by_lineup_source(resolved: List[ResolvedPick]) -> List[HitRateGroup]:
    """Real hit rate grouped by whether each pick was scored against MLB's
    real, posted starting lineup ("confirmed") or the active-roster proxy
    ("active_roster") at pick time - see schedule.py's
    ProbableMatchup.lineup_source docstring for why this exists. Once
    enough real data accumulates, this is the real, testable version of
    "does confirming the lineup first actually help": does the confirmed
    group's real hit rate come in above the active-roster group's.
    """
    return _group_hit_rate(resolved, lambda r: r.pick.lineup_source)


def recorded_at_et(pick: PickRecord) -> datetime:
    """`pick.recorded_at` (real ISO 8601 UTC, see PickRecord's docstring)
    converted to real US-Eastern wall-clock time - the actual hour a
    person watching this run would have seen it fire.
    """
    return datetime.fromisoformat(pick.recorded_at).astimezone(_ET)


def _run_hour_bucket(pick: PickRecord) -> str:
    # 24-hour, zero-padded so it both reads unambiguously and sorts
    # correctly as a plain string - no AM/PM comparison bugs.
    return f"{recorded_at_et(pick).hour:02d}:00 ET"


def hit_rate_by_run_hour(resolved: List[ResolvedPick]) -> List[HitRateGroup]:
    """Real hit rate grouped by the actual US-Eastern hour each pick was
    recorded (PickRecord.recorded_at), not by this project's own four
    scheduled cron times - a manual workflow_dispatch run at any hour
    still counts honestly here, never silently excluded. Lets a real
    pattern in when this model performs best emerge from actual results,
    rather than a guess about which of the daily runs is "the good one."
    """
    return _group_hit_rate(resolved, lambda r: _run_hour_bucket(r.pick))
