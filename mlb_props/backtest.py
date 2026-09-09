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
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from odds_monitor.ev import american_to_decimal

from .betting import MIN_EV_PERCENT_TO_RECOMMEND, recommend_units
from .edges import effective_tier
from .results import ClvRecord, GameOutcome, PickRecord, load_clv, load_picks, load_results

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


def _latest_pick_per_game(picks: List[PickRecord]) -> Dict[Tuple[str, str, str, str], PickRecord]:
    """Same idea as `results.latest_pick_per_key`, but keyed by
    `(player, market, event, game_date)` instead of just `PickRecord.key`'s
    `(player, market, event)`.

    That 3-tuple key is right for its actual job (collapsing repeated
    same-day snapshots of the same pick, within one day's picks file - see
    `record_closing_odds`, which always loads exactly one day's file). It's
    wrong here: `resolve_picks` joins picks across *every* recorded day at
    once, and MLB teams routinely play the same opponent on consecutive
    days (a series) - `event` is just "Team A @ Team B" with no date in it,
    so a player's pick from day 1 of a series and day 3 of the same series
    share the exact same 3-tuple key. Deduping across all days with that
    key silently collapses the whole series down to its last recorded day,
    discarding every earlier day's real result.

    Confirmed against this project's own real recorded history
    (2026-09-07): 540 of 801 real `(player, market, event)` keys spanned
    2+ real game_dates, and 89% of all recorded pick snapshots belonged to
    one of those collisions - `resolve_picks` was silently dropping most
    of a series' earlier days from every stat on the Performance page
    (hit rate by market/tier/lineup source, calibration, and the weight-
    refit/market-blend training data in refit.py, which all consume this
    function's output). Adding `game_date` to the key fixes it without
    touching `PickRecord.key`/`latest_pick_per_key`, which are still
    correct for their own single-day callers.

    Plain "latest `recorded_at` wins" has a second, separate bug: books
    routinely pull a player prop entirely once a game nears first pitch
    (confirmed live 2026-09-08 - Coby Mayo, Guardians @ Orioles: a real
    confirmed agree-tier pick at -150 recorded 4pm ET had its market
    pulled by 6:36pm ET once the game reached "Warmup", recorded as
    `best_price=None`/`tier="no_market"`). That later snapshot is real
    and chronologically newer, but it carries no price - if it wins the
    tie-break outright, a real, bettable price silently vanishes from
    every stat on the Performance page in favor of a snapshot that was
    never actually bettable. `_is_more_authoritative` keeps "later wins"
    as the default, except it never lets a priceless snapshot replace an
    already-priced one.
    """
    latest: Dict[Tuple[str, str, str, str], PickRecord] = {}
    for p in picks:
        key = p.key + (p.game_date,)
        prev = latest.get(key)
        if prev is None or _is_more_authoritative(p, prev):
            latest[key] = p
    return latest


def _is_more_authoritative(candidate: PickRecord, current: PickRecord) -> bool:
    """True if `candidate` should replace `current` as the recorded pick for
    a given `(player, market, event, game_date)` key. Newer always wins,
    unless the newer snapshot has no real price while the current one does -
    see `_latest_pick_per_game`'s docstring for why that specific case must
    never win on recency alone.
    """
    if candidate.recorded_at <= current.recorded_at:
        return False
    if candidate.best_price is None and current.best_price is not None:
        return False
    return True


def last_priced_pick_by_key(picks: List[PickRecord]) -> Dict[Tuple[str, str, str], PickRecord]:
    """Same "later wins, unless later is unpriced" rule as
    `_latest_pick_per_game`, but keyed by plain `PickRecord.key` (player,
    market, event) rather than `_latest_pick_per_game`'s four-tuple - for a
    single day's picks file (every row already shares one `game_date`, so
    it doesn't need to be part of the key; same convention as
    `record_closing_odds`/`latest_pick_per_key` in results.py).

    Built for html_report.py's stale-pregame-price fallback: once a game
    has started, the live pipeline has no real price left for most of its
    candidates (books pull props near first pitch - see
    `_latest_pick_per_game`'s docstring). A person checking the site for
    the first time after that point should still be able to see the last
    real price this project actually recorded for that game today, clearly
    marked as pregame/stale rather than live.
    """
    latest: Dict[Tuple[str, str, str], PickRecord] = {}
    for p in picks:
        key = p.key
        prev = latest.get(key)
        if prev is None or _is_more_authoritative(p, prev):
            latest[key] = p
    return latest


def resolve_picks(picks: List[PickRecord], results: List[GameOutcome]) -> List[ResolvedPick]:
    """Joins the latest same-day snapshot of every recorded pick against its
    real resolved outcome. Picks with no resolved outcome yet (the game
    hasn't been resolved) or an unrecognized market are silently excluded -
    every stat downstream only ever counts picks we actually know the real
    answer for, never an assumed loss.
    """
    outcomes = latest_results_by_key(results)
    resolved = []
    for pick in _latest_pick_per_game(picks).values():
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


@dataclass(frozen=True)
class UnitsRecord:
    """One resolved pick's real would-have-bet outcome, in units (1 unit =
    1% of bankroll - see `betting.py`'s module docstring). Only a
    resolved pick that would have actually cleared this project's own
    real recommendation bar (`betting.MIN_EV_PERCENT_TO_RECOMMEND`, same
    Kelly sizing as the live Recommended Bets section) gets a record here -
    this is never every resolved pick, only the ones a real bettor
    following this site would have staked money on.
    """

    game_date: str
    player: str
    market: str
    tier: str  # effective_tier-corrected, not the possibly-stale raw field
    best_price: int
    units: float  # staked, always positive
    won: bool
    net_units: float  # +profit if won, -units if lost


def units_ledger(resolved: List[ResolvedPick]) -> List[UnitsRecord]:
    """Every resolved pick that would have been a real recommended bet,
    re-sized with the exact same `betting.recommend_units` Kelly math the
    live site uses - so "up/down X units" always means the same thing here
    as it does on the Recommended Bets page a real bettor already trusts.

    Deliberately recomputed from each pick's own recorded `model_prob`/
    `best_price` rather than looked up from a separately-stored units
    figure (this project didn't record one until now) - and deliberately
    reads `tier` through `edges.effective_tier` first, since a stale
    "agree" would otherwise get sized at quarter-Kelly (the strong-tier
    multiplier) instead of the correct, more conservative speculative
    fraction. A pick with no real price, below the EV bar, or with no
    positive Kelly edge is silently excluded, same as the live page - it
    was never a real bet, so it was never money won or lost.
    """
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
        ledger.append(
            UnitsRecord(
                game_date=p.game_date,
                player=p.player,
                market=p.market,
                tier=tier,
                best_price=p.best_price,
                units=units,
                won=r.won,
                net_units=round(net_units, 4),
            )
        )
    return ledger


@dataclass(frozen=True)
class UnitsSummary:
    """The real, honest "are we up or down" headline for the Performance
    page - see `units_ledger`'s docstring for exactly which resolved picks
    count. Split by tier the same way the live Recommended Bets section
    is (strong == tier "agree", speculative == everything else that still
    cleared the bar), so the summary answers not just "up or down" but
    "which kind of pick is actually carrying that number."
    """

    n_bets: int
    total_units_staked: float
    net_units: float
    roi_percent: Optional[float]  # net_units / total_units_staked * 100; None if n_bets == 0
    strong_n_bets: int
    strong_net_units: float
    speculative_n_bets: int
    speculative_net_units: float


def units_summary(ledger: List[UnitsRecord]) -> UnitsSummary:
    if not ledger:
        return UnitsSummary(
            n_bets=0,
            total_units_staked=0.0,
            net_units=0.0,
            roi_percent=None,
            strong_n_bets=0,
            strong_net_units=0.0,
            speculative_n_bets=0,
            speculative_net_units=0.0,
        )
    total_staked = sum(r.units for r in ledger)
    net = sum(r.net_units for r in ledger)
    strong = [r for r in ledger if r.tier == "agree"]
    speculative = [r for r in ledger if r.tier != "agree"]
    return UnitsSummary(
        n_bets=len(ledger),
        total_units_staked=round(total_staked, 2),
        net_units=round(net, 2),
        roi_percent=round(net / total_staked * 100.0, 1) if total_staked > 0 else None,
        strong_n_bets=len(strong),
        strong_net_units=round(sum(r.net_units for r in strong), 2),
        speculative_n_bets=len(speculative),
        speculative_net_units=round(sum(r.net_units for r in speculative), 2),
    )


@dataclass(frozen=True)
class DailyUnits:
    game_date: str
    net_units: float  # that day's own net units, not cumulative
    cumulative_units: float  # running total through this date, inclusive


def units_by_date(ledger: List[UnitsRecord]) -> List[DailyUnits]:
    """Real net units per real game date, plus the running cumulative total
    through that date - the actual trend line behind the headline number,
    sorted chronologically so a reader can see whether "up 4 units" is a
    steady climb or one big day carrying a losing streak.
    """
    by_date: Dict[str, float] = {}
    for r in ledger:
        by_date[r.game_date] = by_date.get(r.game_date, 0.0) + r.net_units
    out = []
    running = 0.0
    for game_date in sorted(by_date):
        running += by_date[game_date]
        out.append(DailyUnits(game_date=game_date, net_units=round(by_date[game_date], 2), cumulative_units=round(running, 2)))
    return out


_STRONG_LABEL = "Strong (agree)"
_SPECULATIVE_LABEL = "Speculative (model only)"


@dataclass(frozen=True)
class EdgeConfidence:
    """Whether a tier's real per-bet edge (net_units per bet) is
    statistically distinguishable from zero *yet*, given how many real
    bets have actually resolved - a different, more careful question than
    `units_summary`'s plain "up or down". A small real sample can show a
    big-looking net_units number that's still well within noise; this is
    the honest read on whether that number has actually earned trust.

    Uses a plain normal-approximation 95% CI on the mean net units per
    bet (no numpy/scipy - same zero-dependency, pure-Python convention as
    refit.py's own hand-rolled logistic fit). Not exact for a very small
    n, but real and honest, never a fabricated precision.
    """

    tier_label: str
    n_bets: int
    mean_net_units_per_bet: float
    ci_lo_95: float
    ci_hi_95: float
    # True only if the entire 95% CI sits above 0 (a real, statistically
    # distinguishable positive edge); False only if it sits entirely
    # below 0 (a real, statistically distinguishable negative edge); None
    # if 0 is still inside the interval - "not enough data to say yet,"
    # never a forced yes/no this sample can't actually support.
    significant: Optional[bool]


def edge_confidence(ledger: List[UnitsRecord]) -> List[EdgeConfidence]:
    """Same tier split as `units_summary` (Strong == effective "agree",
    Speculative == everything else that cleared the real bet bar), but
    for the question "is this tier's real edge proven yet" instead of
    "up or down". Always returns both tiers, even with zero or one real
    bet resolved so far - `significant` is simply `None` (unknown, not
    "no edge") until there's enough real data to say anything at all.
    """
    groups: Dict[str, List[float]] = {_STRONG_LABEL: [], _SPECULATIVE_LABEL: []}
    for r in ledger:
        groups[_STRONG_LABEL if r.tier == "agree" else _SPECULATIVE_LABEL].append(r.net_units)

    out = []
    for label in (_STRONG_LABEL, _SPECULATIVE_LABEL):
        values = groups[label]
        n = len(values)
        if n == 0:
            out.append(EdgeConfidence(label, n, 0.0, 0.0, 0.0, None))
            continue
        mean = sum(values) / n
        if n < 2:
            # A real mean is well-defined at n=1; a variance/CI isn't -
            # report the mean honestly, leave the interval at that same
            # point (zero-width, not a claim of confidence) and the
            # verdict at None.
            out.append(EdgeConfidence(label, n, round(mean, 4), round(mean, 4), round(mean, 4), None))
            continue
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        stderr = math.sqrt(variance / n)
        ci_lo, ci_hi = mean - 1.96 * stderr, mean + 1.96 * stderr
        significant = True if ci_lo > 0 else (False if ci_hi < 0 else None)
        out.append(EdgeConfidence(label, n, round(mean, 4), round(ci_lo, 4), round(ci_hi, 4), significant))
    return out
