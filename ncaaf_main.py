#!/usr/bin/env python3
"""Entry point for the NCAA football (FBS) game-line betting model.

Try it with no setup at all (synthetic data, no API key, no network):
    python ncaaf_main.py --mock

Real data needs:
- A College Football Data API key (CFBD_API_KEY) for schedule/ratings/
  venues/historical lines - free signup at https://collegefootballdata.com/key.
  See ncaaf/cfbd.py.
- An Odds API key (ODDS_API_KEY) for real live cross-book spread/moneyline/
  total odds - free tier, no card, sign up at https://the-odds-api.com. See
  ncaaf/market.py. Without it, games still get scored (predicted score,
  margin, win probability), just with no market price or EV%% attached.
- Outbound access to api.open-meteo.com (free, no key) for kickoff weather.

If this machine can't reach those hosts (some sandboxed environments
can't), run this via `.github/workflows/mlb-props-report.yml` instead - a
GitHub Actions runner has normal internet access and publishes a fresh HTML
report to GitHub Pages (public/ncaaf/index.html) on every run.

    python ncaaf_main.py --season 2026 --week 6 --min-ev 2 --html-out report.html
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional, Sequence

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

from ncaaf.context import CfbdVenueProvider, LiveOpenMeteoProvider, MockVenueProvider, MockWeatherProvider, VenueProvider, WeatherProvider
from ncaaf.html_report import render_html_report
from ncaaf.market import GameOddsProvider, MockGameOddsProvider, NoGameOddsProvider, TheOddsApiGameOddsProvider
from ncaaf.performance_report import render_performance_report
from ncaaf.pipeline import run_pipeline
from ncaaf.ratings import CfbdRatingsProvider, MockRatingsProvider, RatingsProvider
from ncaaf.report import render_report
from ncaaf.results import record_closing_odds, record_picks, resolve_results_for_week, week_key
from ncaaf.schedule import CfbdScheduleProvider, MockScheduleProvider, ScheduleProvider, current_season_week

logger = logging.getLogger(__name__)


def build_providers(args: argparse.Namespace):
    if args.mock:
        schedule: ScheduleProvider = MockScheduleProvider()
        ratings: RatingsProvider = MockRatingsProvider()
        venues: VenueProvider = MockVenueProvider()
        weather: WeatherProvider = MockWeatherProvider(seed=args.mock_seed)
        mock_slate = schedule.get_week_slate(args.season, args.week, args.season_type)
        odds: GameOddsProvider = MockGameOddsProvider(mock_slate, seed=args.mock_seed)
        return schedule, ratings, venues, weather, odds

    from ncaaf.cfbd import CfbdClient

    cfbd_key = args.cfbd_api_key or os.environ.get("CFBD_API_KEY")
    if not cfbd_key:
        raise ValueError(
            "A College Football Data API key is required for real (non --mock) data. Pass --cfbd-api-key or set "
            "CFBD_API_KEY. Get one free at https://collegefootballdata.com/key"
        )
    client = CfbdClient(api_key=cfbd_key)
    schedule = CfbdScheduleProvider(client=client)
    ratings = CfbdRatingsProvider(client=client)
    venues = CfbdVenueProvider(client=client, season=args.season)
    weather = LiveOpenMeteoProvider()

    odds_api_key = args.odds_api_key or os.environ.get("ODDS_API_KEY")
    if odds_api_key:
        odds = TheOddsApiGameOddsProvider(api_key=odds_api_key, books=args.books)
    else:
        logger.warning(
            "No odds API key configured (--odds-api-key/ODDS_API_KEY) - running without live odds. Games will "
            "show predicted scores/probabilities only, with no market price or EV%, until a key is set."
        )
        odds = NoGameOddsProvider()
    return schedule, ratings, venues, weather, odds


def run_resolve_results(cfbd_key: str, data_dir: str, season: int, week: int) -> str:
    from ncaaf.cfbd import CfbdClient

    client = CfbdClient(api_key=cfbd_key)
    picks_path = os.path.join(data_dir, "picks", f"{week_key(season, week)}.jsonl")
    out_path = os.path.join(data_dir, "results", f"{week_key(season, week)}.jsonl")
    n = resolve_results_for_week(client, picks_path, out_path, season, week)
    return f"Resolved {n} real game outcome(s) for {week_key(season, week)}: {picks_path} -> {out_path}"


def run_record_clv(odds: GameOddsProvider, data_dir: str, season: int, week: int) -> str:
    picks_path = os.path.join(data_dir, "picks", f"{week_key(season, week)}.jsonl")
    out_path = os.path.join(data_dir, "clv", f"{week_key(season, week)}.jsonl")
    lines = odds.fetch_game_odds()
    n = record_closing_odds(picks_path, lines, out_path)
    return f"Recorded closing-line value for {n} pick(s) on {week_key(season, week)}: {picks_path} -> {out_path}"


def run_historical_backtest(cfbd_key: str, start_season: int, end_season: int, data_dir: Optional[str] = None) -> str:
    from ncaaf.cfbd import CfbdClient
    from ncaaf.historical_backtest import build_historical_games, record_backtest_run, render_backtest_summary, summarize_backtest

    client = CfbdClient(api_key=cfbd_key)
    games = build_historical_games(client, start_season, end_season)
    summary = summarize_backtest(games)
    text = render_backtest_summary(summary, start_season, end_season)
    if data_dir:
        record_backtest_run(summary, start_season, end_season, os.path.join(data_dir, "historical_backtest", "runs.jsonl"))
        text += f"\n\nPersisted this run's summary to {data_dir}/historical_backtest/runs.jsonl"
    return text


def _resolve_season_week(args: argparse.Namespace) -> None:
    """Fills in `args.season`/`args.week` from CFBD's real calendar when
    not explicitly given - never guesses when `--mock` is set (mock mode
    has no real calendar to resolve against, so an explicit or default
    week is used instead).
    """
    if args.season is not None and args.week is not None:
        return
    if args.mock:
        args.season = args.season or datetime.now(timezone.utc).year
        args.week = args.week or 1
        return
    from ncaaf.cfbd import CfbdClient

    cfbd_key = args.cfbd_api_key or os.environ.get("CFBD_API_KEY")
    if not cfbd_key:
        raise ValueError("--season/--week were not both given, and resolving the current week needs --cfbd-api-key/CFBD_API_KEY.")
    client = CfbdClient(api_key=cfbd_key)
    season, week, season_type = current_season_week(client)
    args.season = args.season or season
    args.week = args.week or week
    if args.season_type == "regular":  # only override the default, never an explicit --season-type
        args.season_type = season_type


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ranked +EV NCAA football (FBS) spread/moneyline/total picks from blended SP+ + Elo power "
        "ratings, situational context, and real cross-book odds."
    )
    parser.add_argument("--season", type=int, default=None, help="Season year (default: resolved from CFBD's real calendar).")
    parser.add_argument("--week", type=int, default=None, help="Week number (default: resolved from CFBD's real calendar).")
    parser.add_argument("--season-type", default="regular", choices=["regular", "postseason"], help="Season type (default: regular).")
    parser.add_argument("--mock", action="store_true", help="Use synthetic sample data end-to-end. No API key or network required.")
    parser.add_argument("--mock-seed", type=int, default=None, help="Seed for reproducible --mock output.")
    parser.add_argument("--min-ev", type=float, default=0.0, help="Minimum EV%% (by our model) required to show a pick (default: 0, i.e. show all).")
    parser.add_argument("--top", type=int, default=40, help="Max rows to show in the ranked picks table (default: 40).")
    parser.add_argument("--cfbd-api-key", default=None, help="College Football Data API key (or set CFBD_API_KEY) - free at collegefootballdata.com/key. Not needed with --mock.")
    parser.add_argument("--odds-api-key", default=None, help="The Odds API key (or set ODDS_API_KEY) - free signup at the-odds-api.com. Not needed with --mock.")
    parser.add_argument("--books", action="append", default=None, help="Restrict to specific sportsbook IDs (repeatable).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    parser.add_argument("--out", default=None, help="Write the console-text report to this file instead of (or in addition to) stdout.")
    parser.add_argument("--html-out", default=None, help="Also write a self-contained styled HTML report to this file.")
    parser.add_argument(
        "--data-dir", default="data/ncaaf",
        help="Root directory for this project's permanent real-pick/result/CLV history: <data-dir>/picks/<season>-wk<week>.jsonl, "
        "<data-dir>/results/..., <data-dir>/clv/..., <data-dir>/historical_backtest/runs.jsonl. Default: 'data/ncaaf'.",
    )
    parser.add_argument("--record-picks", action="store_true", help="Also append every priced candidate this run to <data-dir>/picks/<week>.jsonl.")
    parser.add_argument("--performance-out", default=None, help="Also render the Performance dashboard (from <data-dir>) to this file.")
    parser.add_argument("--resolve-results", action="store_true", help="Skip the normal pipeline and instead resolve --season/--week's already-recorded picks against CFBD's real final scores. Requires --season and --week.")
    parser.add_argument("--record-clv", action="store_true", help="Skip the normal pipeline and instead snapshot current odds for --season/--week's already-recorded picks to compute closing-line value.")
    parser.add_argument("--performance-only", action="store_true", help="Skip the normal pipeline (no network calls) and only re-render the Performance dashboard from <data-dir>. Requires --performance-out.")
    parser.add_argument("--historical-backtest", action="store_true", help="Skip the normal pipeline and instead backtest the point-in-time Elo component against CFBD's real historical betting lines. Requires --backtest-start-season/--backtest-end-season.")
    parser.add_argument("--backtest-start-season", type=int, default=None, help="First real season (inclusive) for --historical-backtest.")
    parser.add_argument("--backtest-end-season", type=int, default=None, help="Last real season (inclusive) for --historical-backtest.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.historical_backtest:
        if not args.backtest_start_season or not args.backtest_end_season:
            print("Configuration error: --historical-backtest requires --backtest-start-season and --backtest-end-season.", file=sys.stderr)
            return 2
        cfbd_key = args.cfbd_api_key or os.environ.get("CFBD_API_KEY")
        if not cfbd_key:
            print("Configuration error: --historical-backtest requires --cfbd-api-key or CFBD_API_KEY.", file=sys.stderr)
            return 2
        text = run_historical_backtest(cfbd_key, args.backtest_start_season, args.backtest_end_season, data_dir=args.data_dir)
        print(text)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text + "\n")
        return 0

    if args.performance_only:
        if not args.performance_out:
            print("Configuration error: --performance-only requires --performance-out.", file=sys.stderr)
            return 2
        perf_html = render_performance_report(args.data_dir)
        with open(args.performance_out, "w") as f:
            f.write(perf_html)
        print(f"Wrote performance report to {args.performance_out}")
        return 0

    try:
        _resolve_season_week(args)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.resolve_results:
        cfbd_key = args.cfbd_api_key or os.environ.get("CFBD_API_KEY")
        if not cfbd_key:
            print("Configuration error: --resolve-results requires --cfbd-api-key or CFBD_API_KEY.", file=sys.stderr)
            return 2
        text = run_resolve_results(cfbd_key, args.data_dir, args.season, args.week)
        print(text)
        return 0

    if args.record_clv:
        try:
            _schedule, _ratings, _venues, _weather, odds = build_providers(args)
        except ValueError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        text = run_record_clv(odds, args.data_dir, args.season, args.week)
        print(text)
        return 0

    try:
        schedule, ratings, venues, weather, odds = build_providers(args)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    report = run_pipeline(
        season=args.season, week=args.week, schedule=schedule, ratings_provider=ratings, venue_provider=venues,
        weather_provider=weather, odds=odds, season_type=args.season_type, min_ev_percent=args.min_ev,
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
    if args.record_picks:
        picks_path = os.path.join(args.data_dir, "picks", f"{week_key(args.season, args.week)}.jsonl")
        n = record_picks(report, picks_path)
        logger.info("Recorded %d pick(s) to %s", n, picks_path)
    if args.performance_out:
        perf_html = render_performance_report(args.data_dir)
        with open(args.performance_out, "w") as f:
            f.write(perf_html)
        logger.info("Wrote performance report to %s", args.performance_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
