"""Turns the real recorded history in `results.py` into an "is this
actually working" readout: calibration, closing-line value, and hit rate by
market/tier - the `ncaaf` counterpart to `mlb_props/backtest.py`. Every
number here is computed from real recorded picks and real resolved
outcomes, never recomputed model output or a guess.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from odds_monitor.ev import american_to_decimal

from .betting import MIN_EV_PERCENT_TO_RECOMMEND, recommend_units
from .edges import effective_tier
from .results import ClvRecord, GameOutcome, PickRecord, hit_for, load_clv, load_picks, load_results


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


def latest_results_by_key(results: List[GameOutcome]) -> Dict[Tuple[str, int, int], GameOutcome]:
    """Keyed by `(event, season, week)`, not just `event` - the same two
    teams meet again in a later season (a real annual rivalry) or, rarely,
    twice in one season (a conference championship rematch), and without
    season/week in the key a later result would silently overwrite an
    earlier real one, exactly the collision `mlb_props/backtest.py`'s
    `_latest_pick_per_game` docstring documents fixing for MLB's repeated-
    series case.
    """
    by_key: Dict[Tuple[str, int, int], GameOutcome] = {}
    for o in results:
        by_key[(o.key, o.season, o.week)] = o
    return by_key


@dataclass(frozen=True)
class ResolvedPick:
    pick: PickRecord
    won: bool


def _latest_pick_per_game(picks: List[PickRecord]) -> Dict[Tuple[str, str, str, str, int, int], PickRecord]:
    """Same idea as `results.latest_pick_per_key`, but keyed with
    `(season, week)` added - a team can face the same opponent in
    different seasons (or, rarely, twice in one season via a conference
    championship rematch), and without the season/week in the key those
    would collide and silently drop one occurrence's real history, the
    same real bug `mlb_props/backtest.py`'s `_latest_pick_per_game`
    docstring documents fixing for MLB's own repeated-series case.
    """
    latest: Dict[Tuple[str, str, str, str, int, int], PickRecord] = {}
    for p in picks:
        key = p.key + (p.season, p.week)
        prev = latest.get(key)
        if prev is None or p.recorded_at > prev.recorded_at:
            latest[key] = p
    return latest


def resolve_picks(picks: List[PickRecord], results: List[GameOutcome]) -> List[ResolvedPick]:
    outcomes = latest_results_by_key(results)
    resolved = []
    for pick in _latest_pick_per_game(picks).values():
        outcome = outcomes.get((pick.event.strip().lower(), pick.season, pick.week))
        if outcome is None:
            continue
        won = hit_for(pick, outcome)
        if won is None:  # push - a real, no-money-changes-hands outcome, never counted as a loss
            continue
        resolved.append(ResolvedPick(pick=pick, won=won))
    return resolved


@dataclass(frozen=True)
class CalibrationBucket:
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
        buckets.append(CalibrationBucket(lo=lo, hi=hi, n=len(in_bucket), predicted_mean=round(predicted_mean, 4), actual_rate=round(actual_rate, 4)))
    return buckets


@dataclass(frozen=True)
class ClvSummary:
    n: int
    mean_clv_percent: Optional[float]
    beat_close_percent: Optional[float]


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
    return _group_hit_rate(resolved, lambda r: effective_tier(r.pick.tier, r.pick.books_quoting))


@dataclass(frozen=True)
class UnitsRecord:
    season: int
    week: int
    event: str
    market: str
    selection: str
    tier: str
    best_price: int
    units: float
    won: bool
    net_units: float


def units_ledger(resolved: List[ResolvedPick]) -> List[UnitsRecord]:
    ledger = []
    for r in resolved:
        p = r.pick
        if p.best_price is None or p.ev_percent_model is None or p.ev_percent_model < MIN_EV_PERCENT_TO_RECOMMEND:
            continue
        tier = effective_tier(p.tier, p.books_quoting)
        units = recommend_units(p.model_prob, p.best_price, tier)
        if units is None:
            continue
        net_units = units * (american_to_decimal(p.best_price) - 1.0) if r.won else -units
        selection = f"{p.team or p.side} {p.point:+g}" if p.point is not None else (p.team or p.side)
        ledger.append(
            UnitsRecord(
                season=p.season, week=p.week, event=p.event, market=p.market, selection=selection, tier=tier,
                best_price=p.best_price, units=units, won=r.won, net_units=round(net_units, 4),
            )
        )
    return ledger


@dataclass(frozen=True)
class UnitsSummary:
    n_bets: int
    total_units_staked: float
    net_units: float
    roi_percent: Optional[float]
    strong_n_bets: int
    strong_net_units: float
    speculative_n_bets: int
    speculative_net_units: float


def units_summary(ledger: List[UnitsRecord]) -> UnitsSummary:
    if not ledger:
        return UnitsSummary(n_bets=0, total_units_staked=0.0, net_units=0.0, roi_percent=None, strong_n_bets=0, strong_net_units=0.0, speculative_n_bets=0, speculative_net_units=0.0)
    total_staked = sum(r.units for r in ledger)
    net = sum(r.net_units for r in ledger)
    strong = [r for r in ledger if r.tier == "agree"]
    speculative = [r for r in ledger if r.tier != "agree"]
    return UnitsSummary(
        n_bets=len(ledger), total_units_staked=round(total_staked, 2), net_units=round(net, 2),
        roi_percent=round(net / total_staked * 100.0, 1) if total_staked > 0 else None,
        strong_n_bets=len(strong), strong_net_units=round(sum(r.net_units for r in strong), 2),
        speculative_n_bets=len(speculative), speculative_net_units=round(sum(r.net_units for r in speculative), 2),
    )


@dataclass(frozen=True)
class WeeklyUnits:
    season: int
    week: int
    net_units: float
    cumulative_units: float


def units_by_week(ledger: List[UnitsRecord]) -> List[WeeklyUnits]:
    by_week: Dict[Tuple[int, int], float] = {}
    for r in ledger:
        key = (r.season, r.week)
        by_week[key] = by_week.get(key, 0.0) + r.net_units
    out = []
    running = 0.0
    for key in sorted(by_week):
        running += by_week[key]
        out.append(WeeklyUnits(season=key[0], week=key[1], net_units=round(by_week[key], 2), cumulative_units=round(running, 2)))
    return out
