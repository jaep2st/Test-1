# Odds Discrepancy Monitor

Watches player-prop lines (points, assists, rebounds, etc.) across
sportsbooks and alerts you whenever the same prop's line differs by 2 or
more points between books - a signal worth a closer look for +EV or
middling opportunities.

Pipeline: **fetch** lines from a provider -> **detect** cross-book gaps ->
**notify** you (console, Discord, and/or email).

## Why Betstamp via its API, not scraping

Betstamp publishes a real REST API for this exact use case - normalized
odds and player props from 200+ sportsbooks, authenticated with an API key
- documented at
[betstamp.com/sports-betting-api](https://www.betstamp.com/sports-betting-api)
and [betstamp.com/docs](https://www.betstamp.com/docs). That's what
`odds_monitor/providers/betstamp.py` talks to. Scraping their site instead
would be slower, more fragile, and likely against their terms of service,
so this project doesn't do that.

One caveat: the exact JSON field names Betstamp's API returns for a given
plan/version weren't accessible from this environment while building the
provider (the docs site itself was reachable via search but not directly
fetchable here). The provider is written defensively - it tries several
plausible key names per field (see `_FIELD_ALIASES` in
`odds_monitor/providers/betstamp.py`) and logs+skips any entry it can't
parse instead of guessing wrong. **Before relying on live alerts**, run
once with `--log-level DEBUG`, inspect a real response, and adjust
`_FIELD_ALIASES` if your account's field names differ.

## Quick start (no API key needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generates synthetic sample data and prints any discrepancies found:
python main.py --mock --once
```

## Running for real

1. Get a Betstamp API key: https://www.betstamp.com/sports-betting-api
2. `cp .env.example .env` and fill in `BETSTAMP_API_KEY` (and any
   notification settings you want).
3. Run a single check:
   ```bash
   python main.py --once
   ```
4. Or run continuously (checks every 5 minutes by default):
   ```bash
   python main.py
   ```

## Notifications

Console output always runs. Add more channels with `--notify`:

```bash
# Discord (needs DISCORD_WEBHOOK_URL in .env or --discord-webhook)
python main.py --notify discord

# Email over SMTP (needs SMTP_* / ALERT_EMAIL_* in .env)
python main.py --notify email

# Both
python main.py --notify discord --notify email
```

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--league` | `nba`, `nfl` | League to monitor; repeatable |
| `--min-spread` | `2.0` | Minimum point gap to flag |
| `--interval` | `300` | Seconds between checks (continuous mode) |
| `--once` | off | Run a single check and exit |
| `--mock` | off | Use synthetic data, no API key needed |
| `--mock-seed` | random | Seed for reproducible `--mock` runs |
| `--api-key` | `$BETSTAMP_API_KEY` | Betstamp API key |
| `--books` | all | Restrict to specific sportsbook IDs; repeatable |
| `--notify` | none | `discord`, `email`; repeatable |
| `--log-level` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |

## Running it continuously

For always-on monitoring, either:

- Leave `python main.py` running in a terminal/`tmux`/`screen` session or
  under a process supervisor (`systemd`, `supervisord`, `pm2`).
- Or use `--once` on a schedule via `cron` or a scheduled GitHub Action,
  since each run is stateless (it just does one fetch/detect/notify pass).

Example crontab entry, checking every 5 minutes:

```
*/5 * * * * cd /path/to/repo && .venv/bin/python main.py --once >> odds_monitor.log 2>&1
```

## Project layout

```
odds_monitor/
  models.py               PropLine / Discrepancy data classes
  detector.py              groups lines and flags >= min-spread gaps
  providers/
    base.py                 OddsProvider interface
    betstamp.py              real Betstamp API client
    mock.py                  synthetic data, no API key needed
  notifiers/
    base.py                 Notifier interface
    console.py               logs to stdout
    discord_notifier.py      posts to a Discord webhook
    email_notifier.py        sends via SMTP
  scheduler.py              run_once / run_forever loop
  cli.py                    argument parsing and wiring
main.py                    entry point
tests/                     pytest suite (detector, mock provider, CLI)
```

## Adding another data source or alert channel

- New data source: implement `OddsProvider.fetch_player_props(league) ->
  List[PropLine]` (see `providers/mock.py` for the simplest example) and
  wire it up in `cli.py`.
- New alert channel: implement `Notifier.notify(discrepancies)` (see
  `notifiers/console.py`) and add it to `build_notifiers` in `cli.py`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
