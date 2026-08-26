"""Sends discrepancy alerts over SMTP (e.g. Gmail with an app password, or
any other SMTP relay).

Configure via constructor arguments or these environment variables:
  SMTP_HOST, SMTP_PORT (default 587), SMTP_USERNAME, SMTP_PASSWORD,
  ALERT_EMAIL_FROM (defaults to SMTP_USERNAME), ALERT_EMAIL_TO
  (comma-separated for multiple recipients)
"""

import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Iterable, List, Optional

from ..models import Alertable
from .base import Notifier

logger = logging.getLogger(__name__)


class EmailNotifier(Notifier):
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        sender: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        use_tls: bool = True,
    ):
        self.host = host or os.environ.get("SMTP_HOST")
        self.port = port or int(os.environ.get("SMTP_PORT", "587"))
        self.username = username or os.environ.get("SMTP_USERNAME")
        self.password = password or os.environ.get("SMTP_PASSWORD")
        self.sender = sender or os.environ.get("ALERT_EMAIL_FROM") or self.username
        self.recipients = recipients or [
            r.strip() for r in os.environ.get("ALERT_EMAIL_TO", "").split(",") if r.strip()
        ]
        self.use_tls = use_tls

        missing = [
            name
            for name, value in (
                ("host", self.host),
                ("username", self.username),
                ("password", self.password),
                ("sender", self.sender),
                ("recipients", self.recipients),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"EmailNotifier is missing required config: {', '.join(missing)}")

    def notify(self, discrepancies: Iterable[Alertable]) -> None:
        discrepancies = list(discrepancies)
        if not discrepancies:
            return

        message = EmailMessage()
        message["Subject"] = f"[Odds Monitor] {len(discrepancies)} player prop discrepancy(ies) found"
        message["From"] = self.sender
        message["To"] = ", ".join(self.recipients)
        message.set_content("\n".join(d.describe() for d in discrepancies))

        with smtplib.SMTP(self.host, self.port, timeout=15) as server:
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(message)
        logger.info("Emailed %d discrepancies to %s", len(discrepancies), self.recipients)
