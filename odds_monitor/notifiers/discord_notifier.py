"""Posts discrepancies to a Discord channel via an incoming webhook.

Create one under Channel Settings -> Integrations -> Webhooks in Discord,
then set its URL via `webhook_url=` or the DISCORD_WEBHOOK_URL environment
variable.
"""

import logging
import os
from typing import Iterable, Optional

import requests

from ..models import Discrepancy
from .base import Notifier

logger = logging.getLogger(__name__)

_DISCORD_MESSAGE_LIMIT = 2000
_MAX_LINES_SHOWN = 20


class DiscordNotifier(Notifier):
    def __init__(self, webhook_url: Optional[str] = None, timeout: float = 10.0):
        self.webhook_url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
        if not self.webhook_url:
            raise ValueError("Set webhook_url=... or the DISCORD_WEBHOOK_URL env var to use DiscordNotifier.")
        self.timeout = timeout

    def notify(self, discrepancies: Iterable[Discrepancy]) -> None:
        discrepancies = list(discrepancies)
        if not discrepancies:
            return

        shown = discrepancies[:_MAX_LINES_SHOWN]
        lines = [f"- {d.describe()}" for d in shown]
        content = "**Odds discrepancy alert**\n" + "\n".join(lines)
        if len(discrepancies) > _MAX_LINES_SHOWN:
            content += f"\n...and {len(discrepancies) - _MAX_LINES_SHOWN} more."
        content = content[:_DISCORD_MESSAGE_LIMIT]

        response = requests.post(self.webhook_url, json={"content": content}, timeout=self.timeout)
        if response.status_code >= 300:
            logger.error("Discord webhook failed (%s): %s", response.status_code, response.text)
