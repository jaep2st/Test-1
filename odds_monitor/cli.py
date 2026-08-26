"""Command-line entry point.

Examples:
    # Try it out with no API key or credentials:
    python main.py --mock --once

    # Run for real against Betstamp, checking every 5 minutes, alerting to
    # both console and Discord:
    python main.py --min-spread 2 --interval 300 --notify discord
"""

import argparse
import logging
import sys
from typing import List, Optional, Sequence

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars can be set another way

from .notifiers.base import Notifier
from .notifiers.console import ConsoleNotifier
from .notifiers.discord_notifier import DiscordNotifier
from .notifiers.email_notifier import EmailNotifier
from .providers.base import OddsProvider
from .providers.betstamp import BetstampProvider
from .providers.mock import MockOddsProvider
from .scheduler import run_forever, run_once


def build_provider(args: argparse.Namespace) -> OddsProvider:
    if args.mock:
        return MockOddsProvider(seed=args.mock_seed)
    return BetstampProvider(api_key=args.api_key, book_ids=args.books)


def build_notifiers(args: argparse.Namespace) -> List[Notifier]:
    notifiers: List[Notifier] = [ConsoleNotifier()]
    channels = set(args.notify)
    if "discord" in channels:
        notifiers.append(DiscordNotifier(webhook_url=args.discord_webhook))
    if "email" in channels:
        notifiers.append(EmailNotifier())
    return notifiers


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor sportsbook player-prop lines for cross-book point discrepancies."
    )
    parser.add_argument(
        "--league",
        action="append",
        dest="leagues",
        default=None,
        help="League to monitor (repeatable, e.g. --league nba --league mlb). Default: nba, nfl, mlb.",
    )
    parser.add_argument(
        "--min-spread",
        type=float,
        default=2.0,
        help="Minimum point gap between books required to flag a line-based prop discrepancy "
        "(e.g. player_points) (default: 2.0).",
    )
    parser.add_argument(
        "--min-prob-spread",
        type=float,
        default=8.0,
        help="Minimum implied win-probability gap, in percentage points, required to flag a "
        "binary Yes/No prop discrepancy (e.g. player_home_runs) (default: 8.0).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=300.0,
        help="Seconds between checks when running continuously (default: 300).",
    )
    parser.add_argument("--once", action="store_true", help="Run a single check and exit instead of looping.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic sample data instead of the Betstamp API. No API key required.",
    )
    parser.add_argument("--mock-seed", type=int, default=None, help="Seed for reproducible --mock output.")
    parser.add_argument("--api-key", default=None, help="Betstamp API key (or set BETSTAMP_API_KEY).")
    parser.add_argument(
        "--books",
        action="append",
        default=None,
        help="Restrict to specific sportsbook IDs (repeatable). Default: all books Betstamp returns.",
    )
    parser.add_argument(
        "--notify",
        action="append",
        default=[],
        choices=["discord", "email"],
        help="Additional notification channel(s) to enable (repeatable). Console output always runs.",
    )
    parser.add_argument("--discord-webhook", default=None, help="Discord webhook URL (or set DISCORD_WEBHOOK_URL).")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    leagues = args.leagues or ["nba", "nfl", "mlb"]

    try:
        provider = build_provider(args)
        notifiers = build_notifiers(args)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.once:
        run_once(provider, leagues, args.min_spread, args.min_prob_spread, notifiers)
    else:
        run_forever(provider, leagues, args.min_spread, args.min_prob_spread, notifiers, args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
