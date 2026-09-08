# Odds Discrepancy Monitor + MLB Home Run / 2+ Total Bases Finder + NCAAF Game-Line Model

Three tools sharing one odds pipeline:

- **`odds_monitor`** (`main.py`) - watches player-prop lines (points,
  assists, rebounds, etc.) across sportsbooks and alerts you whenever the
  same prop's line differs by 2+ points between books.
- **`mlb_props`** (`mlb_props_main.py`) - a daily MLB report that ranks the
  best home run, 2+ total bases, and 1+ hits props on the slate. It combines Statcast
  batted-ball quality (barrel%, hard-hit%, exit velocity, launch angle,
  xwOBA/xSLG), platoon splits, batter-vs-pitcher history, pitch-mix fit,
  recent hot/cold form, ballpark factors and live wind/temperature into a
  composite score per player, then cross-checks that score against real
  cross-book odds (via the same no-vig EV math) to surface +EV spots and
  flag cross-book price discrepancies worth line-shopping.
- **`ncaaf`** (`ncaaf_main.py`) - a weekly NCAA football (FBS) report that
  ranks the best spread, moneyline, and total picks across the slate. It
  blends opponent-adjusted SP+ ratings with this project's own point-in-time
  Elo, layers on situational context (home field, altitude, rest, travel,
  live kickoff weather), and cross-checks the resulting predicted score
  against real cross-book odds to surface +EV spots - see "NCAAF quick
  start" below.

Pipeline: **fetch** lines from a provider -> **detect** cross-book gaps (or,
for `mlb_props`/`ncaaf`, **compute** a no-vig fair price and **score** every
batter/game) -> **notify**/**report**.

## MLB props quick start (no API key needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Synthetic slate, Statcast profiles, matchups, and odds end-to-end:
python mlb_props_main.py --mock --mock-seed 1
```

This prints five sections: the slate's best HR-friendly matchups (park +
weather + opposing pitcher vulnerability), who's hot right now, the
top-ranked home run props, the top-ranked 2+ total bases props, and the
top-ranked 1+ hits props - each ranked by expected value against the best
price actually on the market.

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
| Real per-hitter park + weather factor (optional upgrade) | Ballpark Pal API | Optional - falls back to the static table + Open-Meteo above without it |
| "BP Model" cross-check column, HR/Hits (optional) | Ballpark Pal API, `/api/v1/matchups` | Optional - Ballpark Pal's own independent model shown alongside ours, not blended into it. Their real numbers are per-plate-appearance; converted to per-game via P(at least 1 in ~4.3 PA) - see `mlb_props/ballparkpal.py` |

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
   alternative.) Optionally, also add a `BALLPARKPAL_API_KEY` secret (from
   ballparkpal.com's own API Access page) to upgrade park/weather scoring
   with real per-hitter modeled factors - the report runs the same without
   it, just with the built-in static table + Open-Meteo estimate instead.
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

`mlb_props/scoring.py` weights each factor (see `HR_WEIGHTS` / `TB_WEIGHTS` /
`HITS_WEIGHTS` there for exact numbers) into a transparent 0-100 score, then
maps that score onto a heuristic model probability calibrated to realistic
MLB base rates (~10% average HR-per-game, ~42% average 2+ total-bases game,
~66% average 1+ hit game). That's **not** a trained/calibrated model - it's a
directional estimate you cross-check against the market. The 1+ hits score
uses real per-batter and per-pitcher strikeout rate (`batter_k_pct` /
`pitcher_k_pct_allowed` in `HITS_WEIGHTS` - see `compute_hits_score`'s
docstring), derived from the same pitch-level Statcast log already fetched
for `hr_fb_pct`/pitch-mix, at no extra network cost; it's a season-long rate,
not a whiff rate specific to that night's exact pitch-mix matchup, but it's a
real number, not a blind spot. Two independent signals drive the ranking:

1. **Model edge**: does our score say this player's probability is higher
   than what the best available price actually pays for?
2. **Market edge**: regardless of our model, is one book's price
   meaningfully better than the no-vig consensus price across all books
   quoting it (classic line-shopping value, via `odds_monitor/ev.py`)?

A prop flagged by both is the strongest kind of spot. Every row in the
report shows both EV%s plus the number of books used for the consensus, so
you can judge how much to trust the edge yourself.

### Data quality notes (permanent, applies to every run)

These two are printed at the bottom of every generated report (text and
HTML) as well, so they travel with the output itself, not just this doc:

- **"EV%" means model vs. market, not "the market is wrong."** Every EV%
  figure is our model's probability compared against the book's own no-vig
  fair price. A positive EV% means our model disagrees with the market in
  the bettor's favor - it is not proof the market is mispriced. The market
  could just as easily be right and the model wrong. Treat it as one
  informed opinion set against another, not a guaranteed edge.
- **Pull-air% is permanently unavailable for the HR score (6% of its
  weight).** Neither Baseball Savant leaderboard this project pulls carries
  a pull-rate column, and FanGraphs (which does) returns 403 to every
  request from this environment's hosting provider (GitHub Actions). That
  component defaults to 0 for every player, every run - a disclosed gap,
  not a hidden zero.

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
| `--ballparkpal-api-key` | `$BALLPARKPAL_API_KEY` | Optional Ballpark Pal API key - real per-hitter park+weather factor upgrade (see `mlb_props/ballparkpal.py`); scoring is unchanged without it |
| `--books` | all | Restrict to specific sportsbook IDs (repeatable) |
| `--out` | none | Also write the console-text report to this file |
| `--html-out` | none | Also write the styled HTML report to this file |
| `--log-level` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |

---

# NCAAF Game-Line Model (`ncaaf`)

A weekly NCAA football (FBS) report that ranks the best point-spread,
moneyline, and total picks across the slate.

## How the model works

1. **Power ratings** (`ncaaf/ratings.py`) - each team gets a blended power
   rating from two independent sources, z-scored and combined (60/40):
   - **SP+** (`ncaaf/cfbd.py`, via the College Football Data API) - Bill
     Connelly's public, opponent-adjusted efficiency rating (overall +
     offense/defense/special-teams splits).
   - **Elo** (`ncaaf/elo.py`) - this project's own rating, computed purely
     from real final scores with a margin-of-victory multiplier - genuinely
     point-in-time (a team's rating after week N only reflects games
     through week N), unlike SP+'s season-level number. A real second
     opinion, not a duplicate of the same signal.

   A third signal, **PPA** (predicted points added, also from CFBD), blends
   into the offense/defense split specifically - a different methodology
   (play-by-play efficiency) than SP+'s own, so agreement between the two
   is a real cross-check.
2. **Situational context** (`ncaaf/context.py`) - home-field advantage
   (~2.3 points, the real modern-era average, 0 at a neutral site), a real
   altitude bonus at high-elevation venues (Wyoming, Air Force, Colorado,
   etc. - via CFBD's real venue elevation data), a rest-days edge, the away
   team's real travel distance, and live wind/precipitation at kickoff
   (totals only, via Open-Meteo).
3. **Scoring** (`ncaaf/scoring.py`) - predicted margin comes from the
   blended power-rating difference plus every context adjustment; predicted
   total comes from the separate offense/defense split. Win/cover/over
   probabilities use a normal approximation around those predictions - see
   that module's docstring for the real, disclosed caveats (it doesn't
   separately model "key number" push risk around 3/7 points, for example).
4. **Market cross-check** (`ncaaf/game_odds.py`, `ncaaf/edges.py`) - real
   cross-book spread/moneyline/total odds get de-vigged into a no-vig fair
   price (reusing the same odds math `odds_monitor`/`mlb_props` use) and
   compared against the model's own probability, the same **model edge**/
   **market edge** dual-signal design as `mlb_props` - see
   `mlb_props/edges.py`'s docstring for the shared philosophy.

This is a transparent, hand-built statistical model - not a trained/
calibrated one. Every weight and std-dev constant is disclosed in the
module that uses it, and `ncaaf/refit.py` can fit real replacement weights
against this project's own resolved picks once enough real history exists
(a proposal to review, never an automatic live change - same posture
`mlb_props/refit.py` uses).

## Quick start (no API key needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Synthetic slate, ratings, and odds end-to-end:
python ncaaf_main.py --mock --season 2026 --week 3 --mock-seed 1
```

## Running it for real

| Data | Source | Needs a key? |
|---|---|---|
| Schedule, venues, SP+/PPA ratings, historical betting lines | College Football Data API (`api.collegefootballdata.com`) | Yes (free, email signup) |
| Elo ratings | Computed by this project from CFBD's real game results | No (rides on the CFBD key above) |
| Kickoff wind/precipitation | Open-Meteo | No |
| Cross-book spread/moneyline/total odds | The Odds API (free tier, self-serve key) | Yes (report runs model-only, no odds/EV, without it) |

```bash
cp .env.example .env   # fill in CFBD_API_KEY and ODDS_API_KEY
python ncaaf_main.py --season 2026 --week 6 --min-ev 2 --html-out report.html
```

Omit `--season`/`--week` in real mode and this project resolves the actual
current week from CFBD's own `/calendar` endpoint - no manual tracking of
"what week is it" needed.

**Getting a College Football Data API key:**
[collegefootballdata.com/key](https://collegefootballdata.com/key) - email
signup, key emailed instantly, no card. Required for anything beyond
`--mock` (schedule, ratings, venues, and historical-line backtesting all
come from here).

**Getting an Odds API key:** same as `mlb_props` above -
[the-odds-api.com](https://the-odds-api.com), free tier, no card. Used for
`americanfootball_ncaaf` spread/moneyline/total odds - a single bulk
request per run covers the whole slate (see `ncaaf/market.py`), unlike
`mlb_props`' player-prop odds which need one request per game.

### Always-current live report via GitHub Actions + Pages

The same `.github/workflows/mlb-props-report.yml` workflow that publishes
the MLB report also builds and publishes this one, to `public/ncaaf/` on
the same GitHub Pages site (one workflow, one Pages deployment - see that
file's top comment for why both products have to be built together). Add a
`CFBD_API_KEY` repository secret (Settings -> Secrets and variables ->
Actions) alongside the existing `ODDS_API_KEY` and the NCAAF steps run
automatically; leave it unset and only the MLB report publishes.

### Real backtesting against real historical lines

Unlike `mlb_props` (which has no access to real historical odds),
`ncaaf/historical_backtest.py` backtests this project's own point-in-time
Elo rating against CFBD's real historical betting lines
(`GET /lines`, aggregated from real sportsbooks going back many seasons):

```bash
python ncaaf_main.py --historical-backtest \
  --backtest-start-season 2021 --backtest-end-season 2024 \
  --cfbd-api-key "$CFBD_API_KEY"
```

This validates the Elo half of the model only, not the full SP+-blended
power rating - see that module's docstring for exactly why (CFBD's SP+ is
a season-level number, not point-in-time, so using it for an early-season
backtest would leak information about how the rest of that season played
out). A real, disclosed partial validation, not a claim about the full
model's historical performance.

### Real track record (Performance page)

Same real, permanently-recorded pick/result/CLV history design as
`mlb_props` (see `ncaaf/results.py`/`ncaaf/backtest.py`), filed per real
season/week instead of per day
(`data/ncaaf/picks/<season>-wk<week>.jsonl`, etc.) since a college football
"slate" is a week, not a day. `public/ncaaf/performance.html` shows real
calibration, closing-line value, hit rate by market/tier, and a real units
ledger - computed from resolved picks, never a synthetic backtest.

### NCAAF CLI options

| Flag | Default | Description |
|---|---|---|
| `--season` / `--week` | resolved from CFBD's calendar | Real season year / week number |
| `--season-type` | `regular` | `regular` or `postseason` |
| `--mock` | off | Synthetic data end-to-end, no API key/network |
| `--mock-seed` | random | Seed for reproducible `--mock` output |
| `--min-ev` | `0` | Minimum EV% (by our model) required to show a pick |
| `--top` | `40` | Max rows shown in the ranked picks table |
| `--cfbd-api-key` | `$CFBD_API_KEY` | College Football Data API key - required for real (non-`--mock`) data |
| `--odds-api-key` | `$ODDS_API_KEY` | The Odds API key - omit for a model-only report (no odds/EV) |
| `--books` | all | Restrict to specific sportsbook IDs (repeatable) |
| `--out` / `--html-out` | none | Console-text / styled HTML report output |
| `--data-dir` | `data/ncaaf` | Root for the real pick/result/CLV history |
| `--record-picks` | off | Append this run's picks to the real history |
| `--performance-out` | none | Also render the Performance dashboard |
| `--resolve-results` | off | Resolve `--season`/`--week`'s recorded picks against CFBD's real final scores |
| `--record-clv` | off | Snapshot current odds to compute closing-line value |
| `--historical-backtest` | off | Backtest point-in-time Elo vs. CFBD's real historical lines (`--backtest-start-season`/`--backtest-end-season`) |
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

ncaaf/
  cfbd.py                    College Football Data API client (schedule, SP+/PPA ratings, venues, historical lines)
  schedule.py                 real weekly FBS slate + current-week resolution
  elo.py                       this project's own point-in-time Elo rating
  ratings.py                    blends SP+ + PPA + Elo into each team's power rating
  context.py                     home field/altitude/rest/travel/weather situational adjustments
  scoring.py                      predicted margin/total/win-cover-over probabilities
  market.py                        real cross-book spread/moneyline/total odds (The Odds API) + mock
  game_odds.py                      no-vig fair-price devig for game-line markets
  edges.py                           combines model + market into ranked +EV candidates
  betting.py                          fractional-Kelly recommended-bet sizing
  pipeline.py                          orchestrates the full weekly run
  report.py                             renders the console-text report
  html_report.py                         renders the styled HTML report
  site_style.py                           shared visual system (reuses mlb_props/site_style.py)
  results.py                               real pick/result/CLV history (per real season/week)
  backtest.py                               calibration/hit-rate/CLV/units aggregation from real history
  refit.py                                   real logistic-regression weight refit + market-blend fit (proposal only)
  historical_backtest.py                      real backtest vs. CFBD's historical betting lines (Elo only)
  performance_report.py                        renders the Performance dashboard
ncaaf_main.py               ncaaf entry point

.github/workflows/
  mlb-props-report.yml     runs mlb_props AND ncaaf for real on a schedule/on-demand, publishes both to GitHub Pages

tests/                     pytest suite (odds_monitor detector/mock/CLI/EV math, mlb_props scoring/pipeline/HTML/CLI,
                            ncaaf elo/scoring/game_odds/edges/pipeline/CLI/HTML)
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
