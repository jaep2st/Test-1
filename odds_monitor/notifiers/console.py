"""Log-based notifier - always safe to run alongside others as a record of
what fired, and sufficient on its own if you're watching the terminal.
"""

import logging
from typing import Iterable

from ..models import Discrepancy
from .base import Notifier

logger = logging.getLogger(__name__)


class ConsoleNotifier(Notifier):
    def notify(self, discrepancies: Iterable[Discrepancy]) -> None:
        for d in discrepancies:
            logger.warning(d.describe())
