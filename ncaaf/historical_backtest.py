"""A real backtest against CFBD's real historical betting lines
(`GET /lines`) across past seasons - the one part of this model that can be
checked against real history with minimal lookahead risk, using this
project's own point-in-time Elo rather than the live pipeline's blended
SP+-plus-Elo power rating.

**Why Elo alone, not the full blended model**: `elo.py`'s rating after
week N is purely a function of games played through week N - genuinely
point-in-time. CFBD's SP+ rating (`ratings_sp`), by contrast, is a
season-level number - fetching "2023 SP+" returns a rating computed from
(and reflecting) the *entire* 2023 season, including games that happened
after the early-season week being backtested. Using it for a week-3
backtest would silently leak information about how those teams' whole
seasons played out - exactly the kind of lookahead bias that makes a
backtest look better than real deployment ever would (the same concern
`mlb_props/historical_backtest.py`'s module docstring raises about its own
model's non-backtestable components). This backtest is real evidence for
the Elo half of the model only - a genuine, if partial, validation, not a
claim about the full blended model's real historical performance.

**Why spreads only, not totals**: Elo alone has no offense/defense split
(see `ratings.py`), so it has no real basis for a total-points prediction.
A point-in-time backtest of the live model's total-prediction logic would
need historical week-by-week offense/defense splits this project doesn't
compute yet - a disclosed gap, not a silent one.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .context import BASE_HOME_FIELD_ADVANTAGE
from .elo import BASE_RATING, GameResult, compute_elo_ratings, elo_diff_to_points

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoricalGame:
    season: int
    week: int
    event: str
    home_team: str
    away_team: str
    actual_margin: int  # home_points - away_points
    market_home_spread: float  # real historical consensus spread, home perspective (negative = home favored)
    elo_predicted_margin: float  # point-in-time Elo + HFA, computed using only games strictly before this one
    picked_home: bool  # which side the point-in-time Elo model would have picked against this real market number
    covered: Optional[bool]  # True/False/None(push) - whether the picked side actually covered


def _median_spread(raw_lines: List[dict]) -> Optional[float]:
    """CFBD's `/lines` real per-provider list for one game - takes the real
    median across whichever providers quoted a real spread (some games
    have several, some just one, some none pre-modern-era). `None` (never
    a guess) if no provider quoted one.
    """
    values = []
    for line in raw_lines or []:
        spread = line.get("spread")
        if spread is not None:
            try:
                values.append(float(spread))
            except (TypeError, ValueError):
                continue
    return statistics.median(values) if values else None


def build_historical_games(
    client, start_season: int, end_season: int, elo_warmup_seasons: int = 2
) -> List[HistoricalGame]:
    """Fetches every real completed FBS regular-season game with a real
    historical spread from CFBD between `start_season` and `end_season`
    (inclusive), computes each one's point-in-time Elo prediction (using
    `elo_warmup_seasons` of real prior-season games to seed ratings before
    the backtest window itself starts scoring anything - not counted in
    the real results), and records whether the side the Elo model would
    have picked against the real market number actually covered.
    """
    all_games: List[GameResult] = []
    lines_by_season: Dict[int, Dict[int, List[dict]]] = {}
    for season in range(start_season - elo_warmup_seasons, end_season + 1):
        for raw in client.games(season, season_type="regular"):
            if not raw.get("completed"):
                continue
            try:
                all_games.append(
                    GameResult(
                        season=season,
                        week=raw["week"],
                        home_team=raw["homeTeam"],
                        away_team=raw["awayTeam"],
                        home_points=int(raw["homePoints"]),
                        away_points=int(raw["awayPoints"]),
                        neutral_site=bool(raw.get("neutralSite", False)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping unparsable completed game: %r", raw)
        if season >= start_season:
            for raw in client.lines(season, season_type="regular"):
                week = raw.get("week")
                if week is None:
                    continue
                lines_by_season.setdefault(season, {}).setdefault(week, []).append(raw)

    results: List[HistoricalGame] = []
    for season in range(start_season, end_season + 1):
        season_weeks = sorted({g.week for g in all_games if g.season == season})
        for week in season_weeks:
            elo_ratings = compute_elo_ratings([g for g in all_games if (g.season, g.week) < (season, week)])
            week_games = [g for g in all_games if g.season == season and g.week == week]
            games_by_teams = {(g.home_team, g.away_team): g for g in week_games}
            for raw in lines_by_season.get(season, {}).get(week, []):
                home_team = raw.get("homeTeam")
                away_team = raw.get("awayTeam")
                if not home_team or not away_team:
                    continue
                game = games_by_teams.get((home_team, away_team))
                if game is None:
                    continue
                market_spread = _median_spread(raw.get("lines"))
                if market_spread is None:
                    continue

                elo_home = elo_ratings.get(home_team, BASE_RATING)
                elo_away = elo_ratings.get(away_team, BASE_RATING)
                hfa = 0.0 if game.neutral_site else BASE_HOME_FIELD_ADVANTAGE
                elo_predicted_margin = elo_diff_to_points(elo_home - elo_away) + hfa

                edge = elo_predicted_margin + market_spread  # > 0 means the model likes home more than the market's own number implies
                picked_home = edge > 0
                actual_margin = game.home_points - game.away_points
                result_value = actual_margin + market_spread
                covered: Optional[bool]
                if result_value == 0:
                    covered = None
                else:
                    home_covered = result_value > 0
                    covered = home_covered if picked_home else not home_covered

                results.append(
                    HistoricalGame(
                        season=season, week=week, event=f"{away_team} @ {home_team}", home_team=home_team,
                        away_team=away_team, actual_margin=actual_margin, market_home_spread=market_spread,
                        elo_predicted_margin=round(elo_predicted_margin, 2), picked_home=picked_home, covered=covered,
                    )
                )
    return results


@dataclass(frozen=True)
class BacktestSummary:
    n_games: int
    n_decided: int  # excludes pushes
    wins: int
    losses: int
    pushes: int
    ats_win_pct: Optional[float]
    mean_edge_points: float
    by_season: Dict[int, Tuple[int, int, int]]  # season -> (wins, losses, pushes)


def summarize_backtest(games: List[HistoricalGame]) -> BacktestSummary:
    wins = sum(1 for g in games if g.covered is True)
    losses = sum(1 for g in games if g.covered is False)
    pushes = sum(1 for g in games if g.covered is None)
    decided = wins + losses
    by_season: Dict[int, List[int]] = {}
    for g in games:
        bucket = by_season.setdefault(g.season, [0, 0, 0])
        if g.covered is True:
            bucket[0] += 1
        elif g.covered is False:
            bucket[1] += 1
        else:
            bucket[2] += 1
    edges = [abs(g.elo_predicted_margin + g.market_home_spread) for g in games]
    return BacktestSummary(
        n_games=len(games),
        n_decided=decided,
        wins=wins,
        losses=losses,
        pushes=pushes,
        ats_win_pct=round(wins / decided, 4) if decided else None,
        mean_edge_points=round(statistics.mean(edges), 2) if edges else 0.0,
        by_season={s: tuple(v) for s, v in sorted(by_season.items())},
    )


def render_backtest_summary(summary: BacktestSummary, start_season: int, end_season: int) -> str:
    lines = [
        f"NCAAF HISTORICAL ATS BACKTEST (Elo-only, point-in-time) - {start_season}-{end_season}",
        "=" * 70,
        "",
        f"{summary.n_games} real games with a real historical spread found.",
        f"ATS record picking the point-in-time-Elo-favored side vs. the real market number: "
        f"{summary.wins}-{summary.losses}-{summary.pushes} "
        f"({f'{summary.ats_win_pct:.1%}' if summary.ats_win_pct is not None else 'n/a'} against {summary.n_decided} decided games)",
        f"Mean |model edge| vs. the market number: {summary.mean_edge_points:.2f} points",
        "",
        "By season:",
    ]
    for season, (w, l, p) in summary.by_season.items():
        decided = w + l
        pct = f"{w / decided:.1%}" if decided else "n/a"
        lines.append(f"  {season}: {w}-{l}-{p} ({pct})")
    lines.append("")
    lines.append(
        "A real, break-even ATS win rate against -110 juice is ~52.4% - meaningfully above that, over a real "
        "multi-season sample, is the honest bar for 'this signal has a real edge.' This backtests the point-in-time "
        "Elo component only, not the live model's full SP+-blended power rating (season-level SP+ can't be "
        "backtested point-in-time without leaking future information - see this module's docstring) or its "
        "situational context adjustments beyond home-field advantage."
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class BacktestRun:
    run_at: str
    start_season: int
    end_season: int
    summary: dict


def record_backtest_run(summary: BacktestSummary, start_season: int, end_season: int, out_path: str) -> int:
    from datetime import datetime, timezone

    run = BacktestRun(
        run_at=datetime.now(timezone.utc).isoformat(), start_season=start_season, end_season=end_season, summary=asdict(summary)
    )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(run), sort_keys=True) + "\n")
    return 1
