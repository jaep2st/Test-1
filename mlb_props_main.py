#!/usr/bin/env python3
"""Entry point for the MLB home run / 2+ total bases prop finder.

Try it with no setup at all (synthetic data, no API key, no network):
    python mlb_props_main.py --mock --once

Real data needs:
- `pip install pybaseball pandas` for Statcast batted-ball quality,
  matchup/platoon splits, and recent-form data (all free, no key, but
  outbound access to baseballsavant.mlb.com/FanGraphs is required).
- Outbound access to statsapi.mlb.com (free, no key) for the day's slate
  and probable pitchers, and api.open-meteo.com (free, no key) for wind/
  temperature.
- An Odds API key (ODDS_API_KEY) for real cross-book odds - free tier, no
  card, sign up at the-odds-api.com and see odds_monitor/providers/
  theoddsapi.py. (A Betstamp key via BETSTAMP_API_KEY also works as an
  alternative - see odds_monitor/providers/betstamp.py.) Optional: without
  either, props still get scored, just with no market price/EV% attached.

If this machine can't reach those hosts (some sandboxed environments
can't), run this via `.github/workflows/mlb-props-report.yml` instead - a
GitHub Actions runner has normal internet access and publishes a fresh
HTML report to GitHub Pages on every run. See the main README.

    python mlb_props_main.py --date 2026-08-26 --min-ev 2 --html-out report.html
"""

import argparse
import logging
import os
import sys
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

from collections import defaultdict

from odds_monitor.http_utils import build_retrying_session
from odds_monitor.providers.base import OddsProvider
from odds_monitor.providers.betstamp import BetstampProvider
from odds_monitor.providers.fallback import FallbackOddsProvider
from odds_monitor.providers.theoddsapi import TheOddsApiProvider

from mlb_props.ballparkpal import (
    BALLPARKPAL_API_BASE,
    BallparkPalProvider,
    LiveBallparkPalProvider,
    MockBallparkPalProvider,
    NoBallparkPalProvider,
)
from mlb_props.context import LiveParkWeatherProvider, MockParkWeatherProvider, ParkWeatherProvider
from mlb_props.hot_streak import HotStreakProvider, MockHotStreakProvider, StatcastHotStreakProvider
from mlb_props.market import MockMlbPropsOddsProvider, NoOddsProvider
from mlb_props.matchup import MatchupProvider, MockMatchupProvider, PybaseballMatchupProvider
from mlb_props.html_report import render_html_report
from mlb_props.pipeline import run_pipeline
from mlb_props.report import render_report
from mlb_props.schedule import MLB_STATS_API_BASE, MlbStatsApiScheduleProvider, MockScheduleProvider, ScheduleProvider
from mlb_props.statcast import MockStatcastProvider, PybaseballStatcastProvider, StatcastProvider

logger = logging.getLogger(__name__)


# MLB is a US league; the slate's "today" is the US Eastern calendar day,
# not whatever timezone the machine running this happens to be in. Confirmed
# live: a manual run at 2026-08-28 20:22 ET (2026-08-29 00:22 UTC) against
# the naive `date.today()` pulled the *next* day's slate instead of that
# evening's - the GitHub Actions runner's OS clock is UTC, which rolls to
# the next calendar date at 8pm ET (EDT) / 7pm ET (EST), hours before that
# evening's games are anywhere near over. Anchoring to America/New_York
# keeps `--date today` meaning what a bettor means by "today" regardless of
# the runner's own timezone.
_MLB_TZ = ZoneInfo("America/New_York")


def _parse_date(value: str) -> date:
    if value == "today":
        return datetime.now(_MLB_TZ).date()
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_providers(args: argparse.Namespace):
    if args.mock:
        schedule: ScheduleProvider = MockScheduleProvider()
        statcast: StatcastProvider = MockStatcastProvider(seed=args.mock_seed)
        matchup: MatchupProvider = MockMatchupProvider(seed=args.mock_seed)
        hot_streak: HotStreakProvider = MockHotStreakProvider(seed=args.mock_seed)
        park_weather: ParkWeatherProvider = MockParkWeatherProvider(seed=args.mock_seed)

        # The mock odds provider needs to know which batters/events to
        # price; build that from the mock slate up front.
        mock_slate = schedule.get_slate(args.game_date)
        events_by_batter = {}
        all_batters: List[str] = list(args.batters or [])
        for game in mock_slate:
            event = f"{game.away_team} @ {game.home_team}"
            for b in game.away_batters + game.home_batters:
                events_by_batter[b] = event
                if b not in all_batters:
                    all_batters.append(b)
        odds: OddsProvider = MockMlbPropsOddsProvider(
            batters=all_batters, events_by_batter=events_by_batter, seed=args.mock_seed
        )
        ballparkpal: BallparkPalProvider = MockBallparkPalProvider(seed=args.mock_seed)
        return schedule, statcast, matchup, hot_streak, park_weather, odds, ballparkpal

    year = args.year or args.game_date.year
    schedule = MlbStatsApiScheduleProvider()
    statcast = PybaseballStatcastProvider(year=year)
    matchup = PybaseballMatchupProvider(year=year)
    hot_streak = StatcastHotStreakProvider(season_start=date(year, 3, 1))
    park_weather = LiveParkWeatherProvider()

    # The Odds API is the default real-odds source (free tier, self-serve
    # signup, no account approval needed - see odds_monitor/providers/
    # theoddsapi.py). Betstamp stays available as an alternative for anyone
    # who already has a key for it. When both keys are configured, The Odds
    # API is tried first but Betstamp now backstops it: confirmed live
    # (2026-08-29) that a free-tier ODDS_API_KEY can run out of credits
    # mid-day and return 401 on every event, which used to mean the whole
    # run silently degraded to model-only rankings even with a second,
    # working key sitting right there unused. FallbackOddsProvider only
    # engages on that kind of systemic failure (see its docstring) - a
    # slate with genuinely no props posted yet still returns empty, same
    # as always.
    odds_api_key = args.odds_api_key or os.environ.get("ODDS_API_KEY")
    betstamp_key = args.api_key or os.environ.get("BETSTAMP_API_KEY")
    if odds_api_key and betstamp_key:
        odds: OddsProvider = FallbackOddsProvider(
            primary=TheOddsApiProvider(api_key=odds_api_key, books=args.books),
            secondary=BetstampProvider(api_key=betstamp_key, book_ids=args.books),
        )
    elif odds_api_key:
        odds = TheOddsApiProvider(api_key=odds_api_key, books=args.books)
    elif betstamp_key:
        odds = BetstampProvider(api_key=betstamp_key, book_ids=args.books)
    else:
        # Degrade gracefully rather than hard-failing: Statcast/matchup/
        # weather/hot-streak scoring is independently useful even with no
        # odds API key configured yet - props just show model-only rankings
        # (no market price, no EV%) until one is set.
        logger.warning(
            "No odds API key configured (--odds-api-key/ODDS_API_KEY or --api-key/"
            "BETSTAMP_API_KEY) - running without live odds. Props will show model scores "
            "only, with no market price or EV%%, until a key is set."
        )
        odds = NoOddsProvider()

    ballparkpal_key = args.ballparkpal_api_key or os.environ.get("BALLPARKPAL_API_KEY")
    if ballparkpal_key:
        ballparkpal = LiveBallparkPalProvider(api_key=ballparkpal_key)
    else:
        # Optional enhancement, not a required data source (unlike odds):
        # degrades to this project's own existing static park-factor table
        # + Open-Meteo estimate, unchanged - no warning needed, since most
        # runs simply won't have this key configured.
        ballparkpal = NoBallparkPalProvider()
    return schedule, statcast, matchup, hot_streak, park_weather, odds, ballparkpal


def run_live_odds_scan(odds_api_key: str, books: Optional[List[str]]) -> str:
    """Standalone diagnostic for `--live-odds-scan`: fetches odds for
    already-started games too (normally excluded - see theoddsapi.py's
    module docstring on why comparing them to this model is invalid) and
    looks for the one kind of "value" that's still checkable without any
    model at all: the *same exact bet* (player, market, side, line, event)
    priced differently across books, i.e. real line-shopping. Deliberately
    never computes an EV%% or "edge" against this model's score for live
    lines - only real cross-book price comparison.
    """
    provider = TheOddsApiProvider(api_key=odds_api_key, books=books)
    lines = provider.fetch_player_props("mlb", include_live=True)
    live_lines = [line for line in lines if line.is_live]

    out: List[str] = []
    out.append(f"LIVE ODDS SCAN - {len(live_lines)} live-game prop line(s) found "
               f"across {len({l.event for l in live_lines})} in-progress game(s)")
    out.append("")

    if not live_lines:
        out.append("No live-game odds available right now.")
        return "\n".join(out)

    groups: Dict[tuple, List] = defaultdict(list)
    for line in live_lines:
        groups[line.key].append(line)

    multi_book = {k: v for k, v in groups.items() if len({l.sportsbook for l in v}) >= 2}
    out.append(f"{len(groups)} distinct live prop(s) total, {len(multi_book)} quoted by 2+ books "
               "(the only ones where a real cross-book price comparison is possible).")
    out.append("")

    if multi_book:
        out.append("Cross-book price comparison (same player/market/side/line, different books):")
        for key, group in sorted(multi_book.items(), key=lambda kv: -max(l.odds for l in kv[1] if l.odds is not None)):
            best = max(group, key=lambda l: l.odds if l.odds is not None else -10**9)
            worst = min(group, key=lambda l: l.odds if l.odds is not None else 10**9)
            spread = (best.odds or 0) - (worst.odds or 0)
            out.append(
                f"  - {best.player} ({best.event}) {best.market} {best.side} {best.line:g}: "
                f"best {best.sportsbook}={best.odds:+d}, worst {worst.sportsbook}={worst.odds:+d} "
                f"(spread {spread:+d}) | all: " + ", ".join(f"{l.sportsbook}={l.odds:+d}" for l in group)
            )
    else:
        out.append(
            "No live prop is currently quoted by more than one book, so there's no real cross-book value "
            "to find right now - every live line here is a single book's price with nothing to compare it "
            "against. That's not the same as \"good value\"; it's \"unverifiable,\" which is why the main "
            "report excludes these entirely rather than guessing."
        )
        out.append("")
        out.append("Single-book live lines, for reference only (NOT compared to this model - see above):")
        for key, group in sorted(groups.items(), key=lambda kv: kv[1][0].event):
            line = group[0]
            out.append(f"  - {line.player} ({line.event}) {line.market} {line.side} {line.line:g}: {line.sportsbook}={line.odds:+d}")

    return "\n".join(out)


def run_ballparkpal_matchups_check(game_date: date, api_key: str) -> str:
    """Standalone diagnostic for `--ballparkpal-matchups-check`: resolves a
    real, unstated question about Ballpark Pal's `/api/v1/matchups`
    endpoint - are `homeRunProbability`/`strikeoutProbability`/etc. per-
    plate-appearance or per-game? Not stated anywhere in Ballpark Pal's own
    API docs (read manually - see mlb_props/ballparkpal.py's module
    docstring for why), and that integration deliberately doesn't use this
    endpoint yet because guessing wrong here would silently make two
    models that actually agree look like they wildly disagree.

    `strikeoutProbability` is the cleanest real-world yardstick available:
    MLB's actual league-average strikeout rate is famously ~22% per plate
    appearance vs. ~60-65% per game (P(at least one K) across a real
    ~4 PA/game). Whichever range the real numbers cluster around settles
    the question decisively - no guessing required.
    """
    import statistics

    session = build_retrying_session()
    resp = session.get(
        f"{BALLPARKPAL_API_BASE}/matchups",
        params={"date": game_date.isoformat(), "parkAdjusted": "true"},
        headers={"X-API-Key": api_key},
        timeout=15.0,
    )
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("data", payload)
    if isinstance(rows, dict):
        rows = rows.get("items", rows.get("hitters", rows.get("rows", rows.get("data", []))))

    out: List[str] = [f"BALLPARK PAL MATCHUPS CHECK (probability basis) - {game_date.isoformat()}", ""]
    if not isinstance(rows, list) or not rows:
        out.append(f"No usable rows (raw payload sample): {str(payload)[:1500]}")
        return "\n".join(out)

    out.append(f"{len(rows)} real batter-vs-starter matchup rows returned.")
    out.append("")
    for field in ("homeRunProbability", "strikeoutProbability", "walkProbability", "singleProbability"):
        values = [row[field] for row in rows if row.get(field) is not None]
        if not values:
            out.append(f"{field}: no non-null values in this response")
            continue
        out.append(
            f"{field}: n={len(values)} min={min(values):.1f} max={max(values):.1f} "
            f"mean={statistics.mean(values):.1f} median={statistics.median(values):.1f}"
        )
    out.append("")
    out.append(
        "Read: strikeoutProbability mean near ~20-25% => per-plate-appearance basis "
        "(matches real MLB's ~22% league-average K rate per PA). Mean near ~55-70% => "
        "per-game basis (matches P(at least one K) across a real ~4 PA/game)."
    )
    out.append("")
    out.append("Sample rows, highest homeRunProbability first:")
    for row in sorted(rows, key=lambda r: r.get("homeRunProbability") or -1, reverse=True)[:8]:
        out.append(
            f"  - {row.get('batterName')} vs {row.get('pitcherName')}: HR={row.get('homeRunProbability')} "
            f"K={row.get('strikeoutProbability')} BB={row.get('walkProbability')} 1B={row.get('singleProbability')}"
        )
    return "\n".join(out)


def run_game_status_check(game_date: date) -> str:
    """Standalone diagnostic for `--game-status-check`: pulls MLB's own
    authoritative real-time game status (Scheduled/In Progress/Final, with
    inning for in-progress games) directly from the MLB Stats API's
    schedule endpoint - the same host `MlbStatsApiScheduleProvider` already
    uses for the slate/probable-pitchers, just reading its `status` field
    too (unused elsewhere in this project).

    This exists specifically because `TheOddsApiProvider`'s live-game
    filter (see that module's `include_live` docstring) relies on The Odds
    API's own `commence_time` per event, and that has been confirmed wrong
    for at least one real game (2026-08-29: Kansas City @ Cleveland showed
    as not-yet-started there while it had actually started) - MLB's own
    schedule endpoint is the actual source of truth for game state, not a
    third-party odds aggregator's copy of it.
    """
    session = build_retrying_session()
    resp = session.get(
        f"{MLB_STATS_API_BASE}/schedule",
        params={"sportId": 1, "date": game_date.isoformat()},
        timeout=10.0,
    )
    resp.raise_for_status()
    payload = resp.json()

    out: List[str] = [f"GAME STATUS CHECK (MLB Stats API, authoritative) - {game_date.isoformat()}", ""]
    games = [g for block in payload.get("dates", []) for g in block.get("games", [])]
    if not games:
        out.append("No games found for this date.")
        return "\n".join(out)

    for game in games:
        try:
            teams = game["teams"]
            away = teams["away"]["team"]["name"]
            home = teams["home"]["team"]["name"]
            status = game.get("status", {}).get("detailedState", "Unknown")
            inning = game.get("linescore", {}).get("currentInningOrdinal")
            half = game.get("linescore", {}).get("inningState")
            when = f" ({half} {inning})" if inning and half else ""
            out.append(f"  {away} @ {home}: {status}{when}")
        except (KeyError, TypeError):
            out.append(f"  (unparsable game entry: {game.get('gamePk', '?')})")
    return "\n".join(out)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find +EV MLB home run, 2+ total bases, and 1+ hits props using Statcast quality-of-contact, "
        "matchup/platoon/pitch-mix edges, recent form, ballpark/weather, and cross-book odds."
    )
    parser.add_argument("--date", dest="game_date", default="today", type=_parse_date, help="Slate date, YYYY-MM-DD (default: today).")
    parser.add_argument("--year", type=int, default=None, help="Season year for Statcast lookups (default: the slate date's year).")
    parser.add_argument("--mock", action="store_true", help="Use synthetic sample data end-to-end. No API key or network required.")
    parser.add_argument("--mock-seed", type=int, default=None, help="Seed for reproducible --mock output.")
    parser.add_argument(
        "--batters",
        action="append",
        default=None,
        help="Extra batter full name to include, matched to the first slate game if no real lineup is posted yet (repeatable).",
    )
    parser.add_argument("--min-ev", type=float, default=0.0, help="Minimum EV%% (by our model) required to show a prop (default: 0, i.e. show all).")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=30,
        help="Cap on how many batters get real (network-bound) matchup/hot-streak lookups, chosen by a cheap "
        "Statcast-only prefilter pass (default: 30). Real providers cost a request per player for those two "
        "signals, so on a full slate with the active-roster fallback (~150-250+ batters) this is what keeps a "
        "run from taking a very long time; raise it for more thorough (but slower) coverage.",
    )
    parser.add_argument("--top", type=int, default=15, help="Max rows to show per section (default: 15).")
    parser.add_argument(
        "--odds-api-key", default=None, help="The Odds API key (or set ODDS_API_KEY) - free signup at the-odds-api.com. Not needed with --mock."
    )
    parser.add_argument("--api-key", default=None, help="Betstamp API key (or set BETSTAMP_API_KEY), used if no Odds API key is configured. Not needed with --mock.")
    parser.add_argument("--books", action="append", default=None, help="Restrict to specific sportsbook IDs (repeatable).")
    parser.add_argument(
        "--ballparkpal-api-key",
        default=None,
        help="Ballpark Pal API key (or set BALLPARKPAL_API_KEY) - optional real per-hitter park+weather "
        "factor upgrade over the built-in static table/Open-Meteo estimate. Not needed with --mock; "
        "without it, scoring falls back to the existing behavior unchanged.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    parser.add_argument("--out", default=None, help="Write the console-text report to this file instead of (or in addition to) stdout.")
    parser.add_argument("--html-out", default=None, help="Also write a self-contained styled HTML report to this file (see mlb_props/html_report.py).")
    parser.add_argument(
        "--live-odds-scan",
        action="store_true",
        help="Skip the normal model/report pipeline entirely and instead scan already-started games' live odds "
        "for real cross-book price value (never compared to this model's pregame-only score - see "
        "odds_monitor/providers/theoddsapi.py). Requires --odds-api-key/ODDS_API_KEY.",
    )
    parser.add_argument(
        "--game-status-check",
        action="store_true",
        help="Skip the normal model/report pipeline entirely and instead print MLB's own authoritative "
        "real-time game status (Scheduled/In Progress/Final) for every game on --date, straight from the "
        "MLB Stats API - useful for cross-checking whether a game the odds pipeline treated as pregame has "
        "actually already started (see theoddsapi.py's commence_time-based filter, which relies on a "
        "third-party copy of this same information and has been confirmed wrong for at least one real game).",
    )
    parser.add_argument(
        "--ballparkpal-matchups-check",
        action="store_true",
        help="Skip the normal model/report pipeline entirely and instead fetch Ballpark Pal's "
        "/api/v1/matchups for --date and print summary statistics (min/max/mean) for its probability "
        "fields - resolves whether they're per-plate-appearance or per-game (undocumented) by comparing "
        "strikeoutProbability against real MLB base rates. Requires --ballparkpal-api-key/BALLPARKPAL_API_KEY. "
        "Diagnostic only - see mlb_props/ballparkpal.py for why this endpoint isn't used in scoring yet.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.game_status_check:
        text = run_game_status_check(args.game_date)
        print(text)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text + "\n")
            logger.info("Wrote game status check to %s", args.out)
        return 0

    if args.ballparkpal_matchups_check:
        ballparkpal_key = args.ballparkpal_api_key or os.environ.get("BALLPARKPAL_API_KEY")
        if not ballparkpal_key:
            print(
                "Configuration error: --ballparkpal-matchups-check requires --ballparkpal-api-key or "
                "BALLPARKPAL_API_KEY.",
                file=sys.stderr,
            )
            return 2
        text = run_ballparkpal_matchups_check(args.game_date, ballparkpal_key)
        print(text)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text + "\n")
            logger.info("Wrote Ballpark Pal matchups check to %s", args.out)
        return 0

    if args.live_odds_scan:
        odds_api_key = args.odds_api_key or os.environ.get("ODDS_API_KEY")
        if not odds_api_key:
            print("Configuration error: --live-odds-scan requires --odds-api-key or ODDS_API_KEY.", file=sys.stderr)
            return 2
        text = run_live_odds_scan(odds_api_key, args.books)
        print(text)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text + "\n")
            logger.info("Wrote live odds scan to %s", args.out)
        return 0

    try:
        schedule, statcast, matchup, hot_streak, park_weather, odds, ballparkpal = build_providers(args)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    report = run_pipeline(
        game_date=args.game_date,
        schedule=schedule,
        statcast=statcast,
        matchup_provider=matchup,
        hot_streak=hot_streak,
        park_weather=park_weather,
        odds=odds,
        extra_batters=args.batters,
        min_ev_percent=args.min_ev,
        max_candidates=args.max_candidates,
        ballparkpal=ballparkpal,
    )
    text = render_report(report, top=args.top)

    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        logger.info("Wrote report to %s", args.out)
    if args.html_out:
        html_text = render_html_report(report, top=args.top, is_mock=args.mock)
        with open(args.html_out, "w") as f:
            f.write(html_text)
        logger.info("Wrote HTML report to %s", args.html_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
