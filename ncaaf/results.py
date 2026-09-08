"""Real, permanent history of every pick this model has ever made - the
`ncaaf` counterpart to `mlb_props/results.py`, same JSONL-per-file,
git-committed-by-the-workflow storage convention (see that module's
docstring for the full rationale).

Filed per real season/week (`<data_dir>/picks/<season>-wk<week>.jsonl`)
rather than per calendar day, since college football's real unit of
"a slate" is a week, not a day.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from odds_monitor.ev import american_to_decimal

from .edges import EdgeCandidate
from .market import GameLine
from .pipeline import WeekReport

logger = logging.getLogger(__name__)


def week_key(season: int, week: int) -> str:
    return f"{season}-wk{week:02d}"


@dataclass(frozen=True)
class PickRecord:
    season: int
    week: int
    recorded_at: str  # ISO 8601 UTC
    event: str
    market: str
    side: str
    team: Optional[str]
    point: Optional[float]
    tier: str
    model_prob: float
    market_fair_prob: Optional[float]
    best_price: Optional[int]
    best_book: Optional[str]
    ev_percent_model: Optional[float]
    ev_percent_market: Optional[float]
    edge_vs_market: Optional[float]
    books_quoting: int
    predicted_margin: float
    predicted_total: float
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, str, str, str]:
        """(event, market, side, point) - matches a pick across recording,
        result-resolution, and closing-odds snapshots, same convention as
        `mlb_props.results.PickRecord.key`."""
        return (
            self.event.strip().lower(),
            self.market,
            self.side,
            str(round(self.point, 2)) if self.point is not None else "",
        )


@dataclass(frozen=True)
class GameOutcome:
    season: int
    week: int
    event: str
    home_points: int
    away_points: int

    @property
    def key(self) -> str:
        return self.event.strip().lower()


@dataclass(frozen=True)
class ClvRecord:
    season: int
    week: int
    recorded_at: str
    event: str
    market: str
    side: str
    point: Optional[float]
    pick_price: int
    pick_book: str
    closing_price: int
    closing_book: str
    clv_percent: float


def _append_jsonl(records, out_path: str) -> int:
    records = list(records)
    if not records:
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


def _pick_record(season: int, week: int, recorded_at: datetime, c: EdgeCandidate) -> PickRecord:
    return PickRecord(
        season=season,
        week=week,
        recorded_at=recorded_at.isoformat(),
        event=c.event,
        market=c.market,
        side=c.side,
        team=c.team,
        point=c.point,
        tier=c.tier,
        model_prob=c.model_prob,
        market_fair_prob=c.market_fair_prob,
        best_price=c.best_line.odds if c.best_line else None,
        best_book=c.best_line.sportsbook if c.best_line else None,
        ev_percent_model=c.ev_percent_model,
        ev_percent_market=c.ev_percent_market,
        edge_vs_market=c.edge_vs_market,
        books_quoting=c.books_quoting,
        predicted_margin=c.predicted_margin,
        predicted_total=c.predicted_total,
        components=c.components,
    )


def record_picks(report: WeekReport, out_path: str, recorded_at: Optional[datetime] = None) -> int:
    """Appends every scored, priced candidate this run to `out_path`.
    Appends, never dedupes - a later-in-the-week run genuinely is a newer
    snapshot as lines move (see `latest_pick_per_key`).
    """
    recorded_at = recorded_at or datetime.now(timezone.utc)
    records = [_pick_record(report.season, report.week, recorded_at, c) for c in report.candidates if c.has_market_data]
    return _append_jsonl(records, out_path)


def load_picks(path: str) -> List[PickRecord]:
    return _load_jsonl(path, PickRecord)


def latest_pick_per_key(picks: List[PickRecord]) -> Dict[Tuple[str, str, str, str], PickRecord]:
    latest: Dict[Tuple[str, str, str, str], PickRecord] = {}
    for p in picks:
        prev = latest.get(p.key)
        if prev is None or p.recorded_at > prev.recorded_at:
            latest[p.key] = p
    return latest


def resolve_results_for_week(client, picks_path: str, out_path: str, season: int, week: int) -> int:
    """Fetches CFBD's real final scores for `season`/`week` and appends one
    `GameOutcome` per distinct event that appears in `picks_path` and is
    marked `completed` by CFBD. Safe to call more than once (backtest.py
    keeps only the latest game outcome per event/week via append order).
    Only call this once `week`'s real games are safely final.
    """
    picks = [p for p in load_picks(picks_path) if p.season == season and p.week == week]
    events = {p.event.strip().lower() for p in picks}
    if not events:
        return 0
    outcomes: List[GameOutcome] = []
    for raw in client.games(season, week=week, season_type="regular"):
        if not raw.get("completed"):
            continue
        home = raw.get("homeTeam")
        away = raw.get("awayTeam")
        if not home or not away:
            continue
        event = f"{away} @ {home}"
        if event.strip().lower() not in events:
            continue
        try:
            outcomes.append(
                GameOutcome(season=season, week=week, event=event, home_points=int(raw["homePoints"]), away_points=int(raw["awayPoints"]))
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping unparsable final score for %s: %r", event, raw)
    return _append_jsonl(outcomes, out_path)


def load_results(path: str) -> List[GameOutcome]:
    return _load_jsonl(path, GameOutcome)


def hit_for(pick: PickRecord, outcome: GameOutcome) -> Optional[bool]:
    """Whether `pick` won against `outcome`'s real final score. `None`
    means a push (a real, no-money-changes-hands outcome) - excluded from
    hit-rate stats, never counted as a loss.
    """
    margin = outcome.home_points - outcome.away_points
    total = outcome.home_points + outcome.away_points
    if pick.market == "h2h":
        if margin == 0:
            return None
        home_won = margin > 0
        return home_won if pick.side == "home" else not home_won
    if pick.market == "spreads":
        if pick.point is None:
            return None
        home_spread = pick.point if pick.side == "home" else -pick.point
        result = margin + home_spread
        if result == 0:
            return None
        home_covered = result > 0
        return home_covered if pick.side == "home" else not home_covered
    if pick.market == "totals":
        if pick.point is None:
            return None
        if total == pick.point:
            return None
        over_hit = total > pick.point
        return over_hit if pick.side == "over" else not over_hit
    return None


def record_closing_odds(picks_path: str, lines: List[GameLine], out_path: str, recorded_at: Optional[datetime] = None) -> int:
    """For every distinct pick in `picks_path`, finds the best currently-
    quoted price for the exact same (event, market, side, point) among
    `lines` (a fresh odds fetch the caller already made) and records
    closing-line value. A pick with no matching current line (the market's
    gone, the game started) is skipped entirely.
    """
    recorded_at = recorded_at or datetime.now(timezone.utc)
    picks = [p for p in load_picks(picks_path) if p.best_price is not None and p.best_book is not None]
    latest = latest_pick_per_key(picks)

    best_current: Dict[Tuple[str, str, str, str], GameLine] = {}
    for line in lines:
        if line.odds is None:
            continue
        key = (line.event.strip().lower(), line.market, line.side, str(round(line.point, 2)) if line.point is not None else "")
        cur = best_current.get(key)
        if cur is None or american_to_decimal(line.odds) > american_to_decimal(cur.odds):
            best_current[key] = line

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
                season=pick.season,
                week=pick.week,
                recorded_at=recorded_at.isoformat(),
                event=pick.event,
                market=pick.market,
                side=pick.side,
                point=pick.point,
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
