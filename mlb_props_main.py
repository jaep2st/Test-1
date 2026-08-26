#!/usr/bin/env python3
"""Entry point for the MLB home run / 2+ total bases prop finder.

Try it with no setup at all (synthetic data, no API key, no network):
    python mlb_props_main.py --mock --once

Real data needs:
- `pip install pybaseball pandas` for Statcast batted-ball quality,
  matchup/platoon splits, and recent-form data (all free, no key, but
  outbound access to baseballsavant.mlb.com/FanGraphs is required).
- A Betstamp API key (BETSTAMP_API_KEY) for real cross-book odds - see
  odds_monitor/providers/betstamp.py and the main README.
- Outbound access to statsapi.mlb.com (free, no key) for the day's slate
  and probable pitchers, and api.open-meteo.com (free, no key) for wind/
  temperature.

    python mlb_props_main.py --date 2026-08-26 --min-ev 2
"""

import argparse
import logging
import sys
from datetime import date, datetime
from typing import List, Optional, Sequence

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

from odds_monitor.providers.base import OddsProvider
from odds_monitor.providers.betstamp import BetstampProvider

from mlb_props.context import LiveParkWeatherProvider, MockParkWeatherProvider, ParkWeatherProvider
from mlb_props.hot_streak import HotStreakProvider, MockHotStreakProvider, PybaseballHotStreakProvider
from mlb_props.market import MockMlbPropsOddsProvider
from mlb_props.matchup import MatchupProvider, MockMatchupProvider, PybaseballMatchupProvider
from mlb_props.pipeline import run_pipeline
from mlb_props.report import render_report
from mlb_props.schedule import MlbStatsApiScheduleProvider, MockScheduleProvider, ScheduleProvider
from mlb_props.statcast import MockStatcastProvider, PybaseballStatcastProvider, StatcastProvider

logger = logging.getLogger(__name__)


def _parse_date(value: str) -> date:
    if value == "today":
        return date.today()
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
        return schedule, statcast, matchup, hot_streak, park_weather, odds

    year = args.year or args.game_date.year
    schedule = MlbStatsApiScheduleProvider()
    statcast = PybaseballStatcastProvider(year=year)
    matchup = PybaseballMatchupProvider(year=year)
    hot_streak = PybaseballHotStreakProvider(season_start=date(year, 3, 1))
    park_weather = LiveParkWeatherProvider()
    odds = BetstampProvider(api_key=args.api_key, book_ids=args.books)
    return schedule, statcast, matchup, hot_streak, park_weather, odds


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find +EV MLB home run and 2+ total bases props using Statcast quality-of-contact, "
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
    parser.add_argument("--top", type=int, default=15, help="Max rows to show per section (default: 15).")
    parser.add_argument("--api-key", default=None, help="Betstamp API key (or set BETSTAMP_API_KEY). Not needed with --mock.")
    parser.add_argument("--books", action="append", default=None, help="Restrict to specific sportsbook IDs (repeatable).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    parser.add_argument("--out", default=None, help="Write the report to this file instead of (or in addition to) stdout.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        schedule, statcast, matchup, hot_streak, park_weather, odds = build_providers(args)
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
    )
    text = render_report(report, top=args.top)

    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        logger.info("Wrote report to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
