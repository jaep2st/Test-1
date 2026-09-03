"""Real, permanent history of every pick the model has ever made - the
foundation everything in `mlb_props/backtest.py` is computed from.

Nothing else in this project persists across runs: a `SlateReport` is built
fresh, rendered, and discarded every time `run_pipeline` finishes. This
module is what turns that into a real track record:

- `record_picks` snapshots every scored candidate to a permanent JSONL file
  the day it was picked.
- `resolve_player_game_outcome`/`resolve_results_for_date` look up what a
  picked player actually did once the game is final - reusing the same
  real per-game Statcast outcome logic `hot_streak.py` already has and has
  already been tested there, not new stat-computation logic.
- `record_closing_odds` snapshots the market again right before lock to
  compute closing-line value (CLV) - the standard sharp-bettor metric for
  whether a price was genuinely good, independent of whether any single
  bet actually hit.

Storage: one JSON-lines file per calendar day per kind (picks/results/clv),
committed straight back to the repo by the `mlb-props-report` GitHub
Actions workflow. Plain JSONL (not a database) so every file is readable
with `cat`/`jq` and diffs cleanly in git - see that workflow's comments.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from odds_monitor.ev import american_to_decimal
from odds_monitor.models import PropLine

from .edges import EdgeCandidate
from .hot_streak import game_outcomes_from_events
from ._ids import lookup_mlbam_id
from .market import MARKET_HITS, MARKET_HOME_RUN, MARKET_TOTAL_BASES
from .market import RECOMMENDED_SIDE_FOR_MARKET as _SIDE_FOR_MARKET
from .pipeline import SlateReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PickRecord:
    """One scored candidate, snapshotted exactly as it was shown in that
    run's report - never recomputed later, so a recorded pick's model_prob/
    price always matches what a real reader actually saw that day.
    """

    game_date: str  # ISO YYYY-MM-DD
    # ISO 8601 UTC. A slate can be recorded more than once a day (the
    # workflow runs twice) - this is what orders multiple snapshots of the
    # same (player, market); backtest.py and record_closing_odds both keep
    # only the latest.
    recorded_at: str
    player: str
    market: str
    event: str
    tier: str  # EdgeCandidate.tier - "agree" / "model_only" / "model_only_single_sided" / "no_market"
    model_score: float
    model_prob: float
    bp_model_prob: Optional[float]
    market_fair_prob: Optional[float]
    best_price: Optional[int]
    best_book: Optional[str]
    ev_percent_model: Optional[float]
    ev_percent_market: Optional[float]
    edge_vs_market: Optional[float]
    books_quoting: int
    # Each scoring component's raw 0-100-normalized value at pick time -
    # see EdgeCandidate.components' docstring. `{}` for any pick recorded
    # before this field existed (an old JSONL row deserializes fine via
    # this default, it just carries no real features to fit from) - this
    # is what mlb_props/refit.py needs to fit real weights against real
    # outcomes; before this field, REFIT_READY_DAYS could never actually
    # trigger anything, no matter how many days passed.
    components: Dict[str, float] = field(default_factory=dict)
    # "confirmed" if this pick was scored against MLB's real, posted
    # starting lineup at pick time; "active_roster" (the honest default)
    # if it was scored against the active-roster proxy instead - see
    # EdgeCandidate.lineup_source's docstring. "active_roster" for any
    # pick recorded before this field existed too - an honest default,
    # not a claim it was actually confirmed.
    lineup_source: str = "active_roster"

    @property
    def key(self) -> Tuple[str, str, str]:
        """(player, market, event), lowercased - matches a pick across
        recording, result-resolution, and closing-odds snapshots the same
        way `PropLine.key` matches lines across books."""
        return (self.player.strip().lower(), self.market.lower(), self.event.lower())


@dataclass(frozen=True)
class GameOutcome:
    """What a player actually did in one real game, resolved after the
    fact - literal booleans, same convention as `hot_streak.ClearanceWindow`'s
    per-game counts, not a probability.
    """

    game_date: str
    player: str
    got_hr: bool
    got_2plus_tb: bool
    got_hit: bool

    def hit_for(self, market: str) -> Optional[bool]:
        """Whether this outcome counts as a "win" for `market` (see
        market.py's MARKET_* constants). `None` for an unrecognized market -
        never a silent guess.
        """
        if market == MARKET_HOME_RUN:
            return self.got_hr
        if market == MARKET_TOTAL_BASES:
            return self.got_2plus_tb
        if market == MARKET_HITS:
            return self.got_hit
        return None


@dataclass(frozen=True)
class ClvRecord:
    """Closing-line value for one recorded pick: the price at pick time vs.
    the best matching price right before lock, for the exact same (player,
    market, event, side). Positive `clv_percent` means the recorded price
    paid out more than the price available at closing - beating the close
    consistently is the standard sharp-bettor signal of a real edge,
    independent of whether any single pick actually hit (see this module's
    docstring).
    """

    game_date: str
    recorded_at: str
    player: str
    market: str
    event: str
    pick_price: int
    pick_book: str
    closing_price: int
    closing_book: str
    clv_percent: float


def _append_jsonl(records, out_path: str) -> int:
    records = list(records)
    if not records:
        # No empty phantom file for a day/run with nothing to record - the
        # workflow's git-commit step only ever sees real, non-empty history
        # files (see that workflow's comments).
        return 0
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a") as f:
        for r in records:
            f.write(json.dumps(asdict(r), sort_keys=True) + "\n")
            n += 1
    return n


def _load_jsonl(path: str, cls):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(cls(**json.loads(line)))
    return out


def _pick_record(game_date: date, recorded_at: datetime, e: EdgeCandidate) -> PickRecord:
    return PickRecord(
        game_date=game_date.isoformat(),
        recorded_at=recorded_at.isoformat(),
        player=e.player,
        market=e.market,
        event=e.event,
        tier=e.tier,
        model_score=e.model_score,
        model_prob=e.model_prob,
        bp_model_prob=e.bp_model_prob,
        market_fair_prob=e.market_fair_prob,
        best_price=e.best_line.odds if e.best_line else None,
        best_book=e.best_line.sportsbook if e.best_line else None,
        ev_percent_model=e.ev_percent_model,
        ev_percent_market=e.ev_percent_market,
        edge_vs_market=e.edge_vs_market,
        books_quoting=e.books_quoting,
        components=e.components,
        lineup_source=e.lineup_source,
    )


def record_picks(report: SlateReport, out_path: str, recorded_at: Optional[datetime] = None) -> int:
    """Appends every scored candidate across all three markets to
    `out_path` as one `PickRecord` per line - the permanent record
    `backtest.py` reads. Appends, never overwrites or dedupes: a second
    same-day run genuinely is a second, later snapshot of the model's
    opinion (lineups/odds can move during a slate) - see
    `PickRecord.recorded_at`. Returns the count written.
    """
    recorded_at = recorded_at or datetime.now(timezone.utc)
    records = [
        _pick_record(report.game_date, recorded_at, e)
        for edges in (report.hr_edges, report.tb_edges, report.hits_edges)
        for e in edges
    ]
    return _append_jsonl(records, out_path)


def load_picks(path: str) -> List[PickRecord]:
    return _load_jsonl(path, PickRecord)


def latest_pick_per_key(picks: List[PickRecord]) -> Dict[Tuple[str, str, str], PickRecord]:
    """Collapses possibly-multiple same-day snapshots of the same (player,
    market, event) down to the latest one - see `PickRecord.recorded_at`'s
    docstring. Shared by `record_closing_odds` below and `backtest.py`, so
    "which snapshot counts as the pick" is answered in exactly one place.
    """
    latest: Dict[Tuple[str, str, str], PickRecord] = {}
    for p in picks:
        prev = latest.get(p.key)
        if prev is None or p.recorded_at > prev.recorded_at:
            latest[p.key] = p
    return latest


def resolve_player_game_outcome(
    pyb, id_cache: Dict[str, Optional[int]], player: str, game_date: date
) -> Optional[GameOutcome]:
    """Fetches `player`'s real Statcast log for the single real date
    `game_date` and resolves what actually happened, reusing the same
    per-game outcome logic `hot_streak.game_outcomes_from_events` already
    has (tested there) - this is wiring for a single-day fetch, not new
    stat-computation logic.

    Returns `None` (never a guessed False) when the player has no MLBAM id,
    the fetch fails, or the log has no real plate appearances for that date
    (the player didn't play, got scratched, etc.) - "unknown" stays
    unknown, same convention as the rest of this project.
    """
    player_id = lookup_mlbam_id(pyb, player, id_cache)
    if player_id is None:
        return None
    try:
        log = pyb.statcast_batter(game_date.isoformat(), game_date.isoformat(), player_id)
    except Exception:
        logger.exception("statcast_batter fetch failed resolving %r on %s", player, game_date)
        return None
    if log is None or log.empty or "events" not in log.columns:
        return None
    pa_rows = log[log["events"].notna()]
    if pa_rows.empty:
        return None
    events = list(pa_rows["events"])
    got_hr, got_2plus_tb, got_hit = game_outcomes_from_events([events])[0]
    return GameOutcome(
        game_date=game_date.isoformat(), player=player, got_hr=got_hr, got_2plus_tb=got_2plus_tb, got_hit=got_hit
    )


def resolve_results_for_date(pyb, picks_path: str, out_path: str, game_date: date) -> int:
    """Reads every pick recorded for `game_date` (see `record_picks`),
    resolves each distinct player's real outcome once (a player can appear
    across multiple markets/snapshots the same day - only one real
    Statcast fetch per player, not one per pick), and appends one
    `GameOutcome` per distinct player to `out_path`. Returns the count
    written; 0 (no fetch attempted) if no picks were recorded for that
    date. Safe to call more than once for the same date - `backtest.py`
    keeps the latest resolution per player/date. The caller (the workflow)
    only invokes this once real games are safely final.
    """
    picks = load_picks(picks_path)
    players = sorted({p.player for p in picks if p.game_date == game_date.isoformat()})
    if not players:
        return 0
    id_cache: Dict[str, Optional[int]] = {}
    outcomes = []
    for player in players:
        outcome = resolve_player_game_outcome(pyb, id_cache, player, game_date)
        if outcome is not None:
            outcomes.append(outcome)
        else:
            logger.warning("Could not resolve a real outcome for %r on %s", player, game_date)
    return _append_jsonl(outcomes, out_path)


def load_results(path: str) -> List[GameOutcome]:
    return _load_jsonl(path, GameOutcome)


def record_closing_odds(
    picks_path: str, odds_lines: List[PropLine], out_path: str, recorded_at: Optional[datetime] = None
) -> int:
    """For every distinct (player, market, event) with a real recorded
    price in `picks_path`, finds the best currently-quoted price for that
    same recommended side among `odds_lines` (a fresh odds fetch the
    caller already made - see `mlb_props_main.py`) and records the
    closing-line-value comparison. A pick with no matching closing line
    (the market disappeared, the game already started, etc.) is skipped
    entirely, never recorded with a guessed value. Returns the count
    written.
    """
    recorded_at = recorded_at or datetime.now(timezone.utc)
    picks = [p for p in load_picks(picks_path) if p.best_price is not None and p.best_book is not None]
    latest = latest_pick_per_key(picks)

    best_current: Dict[Tuple[str, str, str], PropLine] = {}
    for line in odds_lines:
        if line.odds is None:
            continue
        side = _SIDE_FOR_MARKET.get(line.market)
        if side is None or line.side.lower() != side:
            continue
        k = (line.player.strip().lower(), line.market.lower(), line.event.lower())
        cur = best_current.get(k)
        if cur is None or american_to_decimal(line.odds) > american_to_decimal(cur.odds):
            best_current[k] = line

    records = []
    for key, pick in latest.items():
        closing = best_current.get(key)
        if closing is None:
            continue
        pick_dec = american_to_decimal(pick.best_price)
        close_dec = american_to_decimal(closing.odds)
        clv_percent = (pick_dec / close_dec - 1.0) * 100.0
        records.append(
            ClvRecord(
                game_date=pick.game_date,
                recorded_at=recorded_at.isoformat(),
                player=pick.player,
                market=pick.market,
                event=pick.event,
                pick_price=pick.best_price,
                pick_book=pick.best_book,
                closing_price=closing.odds,
                closing_book=closing.sportsbook,
                clv_percent=round(clv_percent, 2),
            )
        )
    return _append_jsonl(records, out_path)


def load_clv(path: str) -> List[ClvRecord]:
    return _load_jsonl(path, ClvRecord)
