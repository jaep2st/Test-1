"""Runs the fetch -> detect -> notify pipeline once, or on a loop."""

import logging
import time
from typing import Iterable, List

from .detector import find_discrepancies, find_odds_discrepancies
from .models import Alertable
from .notifiers.base import Notifier
from .providers.base import OddsProvider

logger = logging.getLogger(__name__)


def run_once(
    provider: OddsProvider,
    leagues: Iterable[str],
    min_spread: float,
    min_prob_spread: float,
    notifiers: List[Notifier],
) -> List[Alertable]:
    all_discrepancies: List[Alertable] = []

    for league in leagues:
        try:
            lines = provider.fetch_player_props(league)
        except Exception:
            logger.exception("Failed to fetch player props for league=%s", league)
            continue

        line_discrepancies = find_discrepancies(lines, min_spread=min_spread)
        odds_discrepancies = find_odds_discrepancies(lines, min_prob_spread=min_prob_spread)
        found = len(line_discrepancies) + len(odds_discrepancies)
        if found:
            logger.info(
                "Found %d discrepancie(s) for %s (%d line, %d odds)",
                found,
                league,
                len(line_discrepancies),
                len(odds_discrepancies),
            )
        all_discrepancies.extend(line_discrepancies)
        all_discrepancies.extend(odds_discrepancies)

    if all_discrepancies:
        for notifier in notifiers:
            try:
                notifier.notify(all_discrepancies)
            except Exception:
                logger.exception("Notifier %s failed", type(notifier).__name__)
    else:
        logger.info("No discrepancies found this check.")

    return all_discrepancies


def run_forever(
    provider: OddsProvider,
    leagues: Iterable[str],
    min_spread: float,
    min_prob_spread: float,
    notifiers: List[Notifier],
    interval_seconds: float,
) -> None:
    leagues = list(leagues)
    logger.info(
        "Starting odds monitor: leagues=%s min_spread=%.1f min_prob_spread=%.1f interval=%ss",
        leagues,
        min_spread,
        min_prob_spread,
        interval_seconds,
    )
    while True:
        start = time.monotonic()
        run_once(provider, leagues, min_spread, min_prob_spread, notifiers)
        elapsed = time.monotonic() - start
        time.sleep(max(0.0, interval_seconds - elapsed))
