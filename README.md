# Odds Discrepancy Monitor + MLB Home Run / 2+ Total Bases Finder

Two tools sharing one odds pipeline:

- **`odds_monitor`** (`main.py`) - watches player-prop lines (points,
  assists, rebounds, etc.) across sportsbooks and alerts you whenever the
  same prop's line differs by 2+ points between books.
- **`mlb_props`** (`mlb_props_main.py`) - a daily MLB report that ranks the
  best home run and 2+ total bases props on the slate. It combines Statcast
  batted-ball quality (barrel%, hard-hit%, exit velocity, launch angle,
  xwOBA/xSLG), platoon splits, batter-vs-pitcher history, pitch-mix fit,
  recent hot/cold form, ballpark factors and live wind/temperature into a
  composite score per player, then cross-checks that score against real
  cross-book odds (via the same no-vig EV math) to surface +EV spots and
  flag cross-book price discrepancies worth line-shopping.

Pipeline: **fetch** lines from a provider -> **detect** cross-book gaps (or,
for `mlb_props`, **compute** a no-vig fair price and **score** every batter)
-> **notify**/**report**.

## MLB props quick start (no API key needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Synthetic slate, Statcast profiles, matchups, and odds end-to-end:
python mlb_props_main.py --mock --mock-seed 1
```

This prints four sections: the slate's best HR-friendly matchups (park +
weather + opposing pitcher vulnerability), who's hot right now, the
top-ranked home run props, and the top-ranked 2+ total bases props - each
ranked by expected value against the best price actually on the market.

### Running it for real

Real mode needs several free-but-separate data sources wired together:

| Data | Source | Needs a key? |
|---|---|---|
| Today's slate + probable pitchers | MLB Stats API (`statsapi.mlb.com`) | No |
| Barrel%, hard-hit%, exit velo, launch angle, xwOBA/xSLG | Baseball Savant, via `pybaseball` | No |
| Platoon splits, batter-vs-pitcher history, pitch-mix fit | Statcast pitch logs, via `pybaseball` | No |
| Recent form (last 7/15/30 days) | Baseball Savant pitch logs, via `pybaseball` | No |
| Ballpark factors + live wind/temperature | Static table + Open-Meteo | No |
| Cross-book player-prop odds | The Odds API (free tier, self-serve key) | Yes (props run model-only without it) |

```bash
pip install pybaseball pandas   # only needed for real (non --mock) Statcast/matchup/form data
cp .env.example .env            # fill in ODDS_API_KEY (optional - see below)
python mlb_props_main.py --date today --min-ev 2 --html-out report.html
```

This needs a machine with normal outbound internet access to those five
hosts - a plain sandboxed environment (including some Claude Code sessions)
may not have it. **The `mlb-props-report` GitHub Actions workflow below is
the recommended way to run this for real, on every run, with zero local
setup** - a GitHub-hosted runner has that access by default. Running it
locally or in CI both work the same way; pick whichever you'll actually use.

No `ODDS_API_KEY` (and no `BETSTAMP_API_KEY` either)? The pipeline still
runs - `NoOddsProvider` returns no lines and every prop shows a model score
with no market price or EV% (`odds_monitor.ev` only produces EV% when there
are books to compare against). Statcast/matchup/weather/hot-streak scoring
is independently useful, so this degrades gracefully instead of failing.

**Getting an odds API key:** [the-odds-api.com](https://the-odds-api.com) -
click "Get API Key", enter an email, done. Free tier (500 credits/month), no
card required. `odds_monitor/providers/theoddsapi.py` is the client;
`ODDS_API_KEY` is checked before `BETSTAMP_API_KEY`, so set whichever one you
have (both work, The Odds API is just easier to sign up for).

Real lineups aren't posted by MLB until close to first pitch, so well
ahead of game time you'll likely want to pass specific hitters explicitly:

```bash
python mlb_props_main.py --batters "Aaron Judge" --batters "Juan Soto" --min-ev 0
```

**Caveat on `_COLUMN_ALIASES`/`_FIELD_ALIASES`:** each real provider is
written defensively against the *documented* shape of its data source and
logs+skips anything it can't parse rather than guessing silently - but
field names occasionally drift from a library/API's docs. The first time
you run for real (locally or via the workflow, with `--log-level DEBUG`),
check the logs for parse warnings and adjust the alias dicts in
`mlb_props/statcast.py`, `mlb_props/matchup.py`, `mlb_props/hot_streak.py`,
and `odds_monitor/providers/theoddsapi.py` (or `betstamp.py`, if using that
instead) if needed.

### Always-current live report via GitHub Actions + Pages

`.github/workflows/mlb-props-report.yml` runs the real pipeline on a
GitHub-hosted runner (normal internet access, no sandbox restrictions),
regenerates `mlb_props/html_report.py`'s HTML report from scratch every
time, and publishes it to GitHub Pages - nothing is cached between runs, so
the published page always reflects that run's live fetch.

One-time setup:

1. **Settings -> Secrets and variables -> Actions -> New repository
   secret** -> name it `ODDS_API_KEY` -> paste your free key from
   https://the-odds-api.com. (Skip this to publish a model-only report with
   no odds/EV columns. A `BETSTAMP_API_KEY` secret works too, as an
   alternative.)
2. **Settings -> Pages -> Build and deployment -> Source: "GitHub
   Actions"**.
3. **Actions tab -> "MLB props report" -> Run workflow** to publish the
   first version immediately. The daily `schedule:` trigger in the workflow
   only fires once this file is on the repo's default branch - trigger it
   manually from a feature branch in the meantime.

After that, the page at your repo's Pages URL always shows the most recent
run - open it any time, or click "Run workflow" again whenever you want an
on-demand refresh with brand-new data.

### How the composite score and +EV flag work

`mlb_props/scoring.py` weights each factor (see `HR_WEIGHTS` /
`TB_WEIGHTS` there for exact numbers) into a transparent 0-100 score, then
maps that score onto a heuristic model probability calibrated to realistic
MLB base rates (~10% average HR-per-game, ~42% average 2+ total-bases
game). That's **not** a trained/calibrated model - it's a directional
estimate you cross-check against the market. Two independent signals drive
the ranking:

1. **Model edge**: does our score say this player's HR/2+TB probability is
   higher than what the best available price actually pays for?
2. **Market edge**: regardless of our model, is one book's price
   meaningfully better than the no-vig consensus price across all books
   quoting it (classic line-shopping value, via `odds_monitor/ev.py`)?

A prop flagged by both is the strongest kind of spot. Every row in the
report shows both EV%s plus the number of books used for the consensus, so
you can judge how much to trust the edge yourself.

### MLB props CLI options

| Flag | Default | Description |
|---|---|---|
| `--date` | today | Slate date, `YYYY-MM-DD` |
| `--year` | slate date's year | Season year for Statcast lookups |
| `--mock` | off | Synthetic data end-to-end, no API key/network |
| `--mock-seed` | random | Seed for reproducible `--mock` output |
| `--batters` | none | Extra batter name to include (repeatable) - useful before lineups post |
| `--min-ev` | `0` | Minimum EV% (by our model) required to show a prop |
| `--top` | `15` | Max rows per section |
| `--odds-api-key` | `$ODDS_API_KEY` | The Odds API key, checked first (omit both this and `--api-key` for a model-only report, no odds) |
| `--api-key` | `$BETSTAMP_API_KEY` | Betstamp API key, used if no Odds API key is set |
| `--books` | all | Restrict to specific sportsbook IDs (repeatable) |
| `--out` | none | Also write the console-text report to this file |
| `--html-out` | none | Also write the styled HTML report to this file |
| `--log-level` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |

---

# Odds Discrepancy Monitor (`odds_monitor`)

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
    theoddsapi.py            real The Odds API client (mlb_props default)
    betstamp.py              real Betstamp API client (alternative)
    mock.py                  synthetic data, no API key needed
  notifiers/
    base.py                 Notifier interface
    console.py               logs to stdout
    discord_notifier.py      posts to a Discord webhook
    email_notifier.py        sends via SMTP
  scheduler.py              run_once / run_forever loop
  cli.py                    argument parsing and wiring
  ev.py                     American<->decimal/prob odds math + no-vig fair pricing (shared with mlb_props)
main.py                    odds_monitor entry point

mlb_props/
  schedule.py               today's slate + probable pitchers (MLB Stats API / mock)
  statcast.py                batter/pitcher batted-ball quality: barrel%, hard-hit%, exit velo,
                              launch angle, xwOBA/xSLG (pybaseball / mock)
  matchup.py                  platoon splits, batter-vs-pitcher history, pitch-mix edge (pybaseball / mock)
  hot_streak.py                rolling 7/15/30-day form vs. season baseline, as a z-score (pybaseball / mock)
  context.py                    ballpark HR factors + live wind/temperature (Open-Meteo / mock)
  market.py                      HR/total-bases market constants + mock/no-op odds providers
  scoring.py                      composite 0-100 score -> heuristic model probability
  edges.py                         combines model score + market no-vig consensus into ranked +EV candidates
  pipeline.py                       orchestrates the full run
  report.py                          renders the console-text report
  html_report.py                     renders the styled, self-contained HTML report
mlb_props_main.py          mlb_props entry point

.github/workflows/
  mlb-props-report.yml     runs mlb_props for real on a schedule/on-demand, publishes to GitHub Pages

tests/                     pytest suite (odds_monitor detector/mock/CLI/EV math, mlb_props scoring/pipeline/HTML/CLI)
```

## Adding another data source or alert channel

- New data source: implement `OddsProvider.fetch_player_props(league) ->
  List[PropLine]` (see `providers/mock.py` for the simplest example) and
  wire it up in `cli.py` (or `mlb_props_main.py`'s `build_providers`).
- New alert channel: implement `Notifier.notify(discrepancies)` (see
  `notifiers/console.py`) and add it to `build_notifiers` in `cli.py`.
- New `mlb_props` signal: each factor (Statcast, matchup, hot streak,
  park/weather) is its own small provider interface with a `Pybaseball*`/
  `Live*` implementation and a `Mock*` implementation - follow that pattern,
  then fold it into `scoring.py`'s weights.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
