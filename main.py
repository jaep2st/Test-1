#!/usr/bin/env python3
"""Entry point for the odds discrepancy monitor.

Try it with no setup at all:
    python main.py --mock --once

See README.md for real Betstamp/Discord/email configuration.
"""

import sys

from odds_monitor.cli import main

if __name__ == "__main__":
    sys.exit(main())
