"""Renders a `SlateReport` as a self-contained, styled HTML page - the same
report as `report.py`'s console text, laid out for scanning: summary tiles,
matchup environments, who's-hot leaderboard, and the two ranked prop tables,
every column included. Used by `mlb_props_main.py --html-out` and by the
`mlb-props-report` GitHub Actions workflow to publish a live page via GitHub
Pages on every run.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from .betting import MIN_EV_PERCENT_TO_RECOMMEND, RecommendedBet, build_recommended_bets
from .edges import EdgeCandidate
from .hot_streak import HeatIndex
from .market import MARKET_HITS, MARKET_HOME_RUN, MARKET_TOTAL_BASES
from .pipeline import MatchupEnvironment, SlateReport
from .report import clearance_cols, clearance_rates, heat_lookup
from .scoring import HITS_WEIGHTS, HR_WEIGHTS, TB_WEIGHTS
from .site_style import STYLE as _STYLE
from .site_style import nav_html

_WEIGHTS_BY_MARKET = {MARKET_HOME_RUN: HR_WEIGHTS, MARKET_TOTAL_BASES: TB_WEIGHTS, MARKET_HITS: HITS_WEIGHTS}
# Same US-Eastern convention this project already anchors "today" to (see
# mlb_props_main.py's _MLB_TZ) - a real MLB start time should read in the
# US Eastern hour a person would recognize, not raw UTC.
_ET = ZoneInfo("America/New_York")


def _fmt_start_time_et(game_time_utc: Optional[str]) -> str:
    """`ProbableMatchup.game_time_utc` (real MLB Stats API `gameDate`,
    already fetched for every game every run - just never shown) as a
    real US-Eastern start time. "TBD" for a genuinely missing/malformed
    value - never a guessed time.
    """
    if not game_time_utc:
        return "TBD"
    try:
        dt = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00")).astimezone(_ET)
    except ValueError:
        return "TBD"
    hour12 = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{hour12}:{dt.minute:02d} {ampm} ET"


def _esc(s: object) -> str:
    return html.escape(str(s))


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_opt_pct(x) -> str:
    return _fmt_pct(x) if x is not None else "n/a"


def _wind_chip(env: MatchupEnvironment) -> str:
    if env.matchup.venue and env.weather_boost_pct == 0.0 and env.away_pitcher_vulnerability is None:
        pass  # not used - dome detection handled by weather_boost_pct alone isn't reliable; keep chip generic below
    direction = "wind-out" if env.weather_boost_pct > 0 else ("wind-in" if env.weather_boost_pct < 0 else "")
    return direction


_MARKET_SHORT_LABELS = {MARKET_HOME_RUN: "1+ HR", MARKET_TOTAL_BASES: "2+ TB", MARKET_HITS: "1+ Hits"}
_VERDICT_RANK = {"STRONG BET": 3, "SPECULATIVE": 2, "PASS": 1, "NO PRICE YET": 0}


def _game_roster_html(candidates: List[EdgeCandidate]) -> str:
    """Every real scored candidate for one specific game, across all three
    markets - "click a game, see everything helpful about it," using data
    this project has already computed every run, nothing new fetched.
    Sorted so the best real plays for this game lead: Strong verdicts
    first, then Speculative, then Pass/No price, each group by EV%
    descending within itself.
    """
    if not candidates:
        return '<div class="empty" style="padding:10px 0;">No candidates scored for this game.</div>'

    def sort_key(c: EdgeCandidate):
        label, _cls = _verdict(c.has_market_data, c.tier, c.ev_percent_model)
        return (_VERDICT_RANK[label], c.ev_percent_model if c.ev_percent_model is not None else -999.0)

    ordered = sorted(candidates, key=sort_key, reverse=True)
    rows = []
    for c in ordered:
        label, css_class = _verdict(c.has_market_data, c.tier, c.ev_percent_model)
        price_text = f"{c.best_line.odds:+d} {c.best_line.sportsbook}" if c.best_line else "no price yet"
        rows.append(
            f'<div class="game-roster-row"><span class="verdict {css_class}">{_esc(label)}</span>'
            f'<span class="grp"><b>{_esc(c.player)}</b><span class="grm">{_esc(_MARKET_SHORT_LABELS.get(c.market, c.market))}</span></span>'
            f'<span class="grz num">{_esc(price_text)}</span></div>'
        )
    return "".join(rows)


def _env_card(env: MatchupEnvironment, rank: int, candidates: List[EdgeCandidate]) -> str:
    m = env.matchup
    wind_class = "wind-out" if env.weather_boost_pct > 0 else ("wind-in" if env.weather_boost_pct < 0 else "")
    pitchers = " vs ".join(p for p in (m.away_pitcher, m.home_pitcher) if p) or "Probable pitchers TBA"
    start_time = _fmt_start_time_et(m.game_time_utc)
    # m.status: MLB Stats API's own real-time game status (Pre-Game/In
    # Progress/Final/etc, see ProbableMatchup's docstring) - already
    # fetched every run, shown here for the first time.
    status_suffix = f" &middot; {_esc(m.status)}" if m.status else ""
    return f"""
      <div class="env-card">
        <div class="rank">#{rank} ENVIRONMENT</div>
        <div class="matchup">{_esc(m.away_team)} @ {_esc(m.home_team)}</div>
        <div class="start-time">{_esc(start_time)}{status_suffix}</div>
        <div class="pitchers">{_esc(pitchers)}</div>
        <div class="park">{_esc(m.venue)} &middot; park HR factor {env.park_hr_factor:.0f} (neutral = 100)</div>
        <div class="env-bar-row"><div class="env-bar"><i style="width:{max(0, min(100, env.environment_score)):.1f}%"></i></div><div class="env-score num">{env.environment_score:.1f}</div></div>
        <div class="env-chips">
          <span class="chip {wind_class}">{env.weather_boost_pct:+.1f}% weather</span>
        </div>
        <span class="expand-toggle" data-role="expand">view this game's props &#9662;</span>
        <div class="detail-panel game-roster">{_game_roster_html(candidates)}</div>
      </div>"""


def _heat_label_class(label: str) -> str:
    return "label-" + label.replace(" ", "-")


def _hot_row(rank: int, h) -> str:
    bar_pct = max(2, min(100, abs(h.z_score) / 3 * 100))
    return f"""
      <div class="hot-row"><div class="rank num">{rank}</div><div class="name">{_esc(h.player)}</div>
        <div><span class="label-pill {_heat_label_class(h.label)}">{_esc(h.label)}</span></div>
        <div class="heat-bar"><i style="width:{bar_pct:.0f}%"></i></div>
        <div class="z num {'pos' if h.z_score >= 0 else 'neg'}">{h.z_score:+.2f}</div>
      </div>"""


def _verdict(has_market_data: bool, tier: str, ev_percent_model: Optional[float]) -> "tuple[str, str]":
    """A single, plain-language read on whether a candidate is actually
    worth betting - not a new judgment call, just making this project's
    own already-computed tier/EV classification (the exact same numbers
    that decide what shows up in "Tonight's Recommended Bets" - see
    betting.py's MIN_EV_PERCENT_TO_RECOMMEND) visible at a glance on
    every row, not just the ones that happen to clear Kelly sizing too.
    Returns (label, css class).

    - "NO PRICE YET": has_market_data is False - nothing to check the
      model against yet.
    - "PASS": a real price exists, but the edge is at or below noise
      level (< MIN_EV_PERCENT_TO_RECOMMEND) - the same bar this project
      uses everywhere else to call an edge real vs. noise.
    - "SPECULATIVE": a real, above-bar edge, but only this project's own
      model sees it (tier != "agree") - same meaning as the Speculative
      section of Tonight's Recommended Bets.
    - "STRONG BET": a real, above-bar edge AND the market's own cross-book
      consensus agrees (tier == "agree") - same meaning as the Strong
      section there.
    """
    if not has_market_data:
        return ("NO PRICE YET", "verdict-none")
    if ev_percent_model is None or ev_percent_model < MIN_EV_PERCENT_TO_RECOMMEND:
        return ("PASS", "verdict-pass")
    if tier == "agree":
        return ("STRONG BET", "verdict-strong")
    return ("SPECULATIVE", "verdict-speculative")


def _verdict_badge(has_market_data: bool, tier: str, ev_percent_model: Optional[float]) -> str:
    label, css_class = _verdict(has_market_data, tier, ev_percent_model)
    return f'<span class="verdict {css_class}">{_esc(label)}</span>'


def _lineup_source_note(lineup_source: str) -> str:
    """A small, honest per-pick signal for real information-freshness -
    see schedule.py's ProbableMatchup.lineup_source docstring for exactly
    why this exists: player-prop edge depends heavily on batting-order/
    starting status, which MLB only confirms 1-4 real hours before first
    pitch, so "was this scored against the real lineup or a same-day
    guess" matters more here than any generic bet-timing rule of thumb.
    """
    if lineup_source == "confirmed":
        return '<div class="lineup-note lineup-confirmed">&#10003; Lineup confirmed</div>'
    return '<div class="lineup-note lineup-projected">Active roster &mdash; lineup not posted yet</div>'


# --- Ballpark-Pal-style colored rating chips -------------------------------
# A quick "how good is this number" visual signal on the columns that matter
# most for spotting a real bet (green = best, red = worst, 5 steps) - the
# same colored-badge idea used by prop-rating sites, applied to this
# project's own already-computed numbers. Every bucket boundary below is
# either a real threshold this project already uses elsewhere
# (MIN_EV_PERCENT_TO_RECOMMEND) or this table's own real spread of values
# (for the clearance-rate columns, since a 20% clearance rate means
# something very different for 2+ TB than for 1+ Hits, so a fixed scale
# would mislead across markets) - never an arbitrary made-up cutoff.


def _ev_bucket(ev: Optional[float]) -> Optional[int]:
    if ev is None:
        return None
    if ev >= 10.0:
        return 4
    if ev >= MIN_EV_PERCENT_TO_RECOMMEND:
        return 3
    if ev >= 0.0:
        return 2
    if ev >= -10.0:
        return 1
    return 0


def _edge_bucket(edge: Optional[float]) -> Optional[int]:
    if edge is None:
        return None
    pct = edge * 100
    if pct >= 5.0:
        return 4
    if pct >= 0.0:
        return 3
    if pct >= -3.0:
        return 2
    if pct >= -8.0:
        return 1
    return 0


def _quantile_cuts(values: List[float]) -> List[float]:
    """4 cut points splitting `values` into quintiles - this table's own
    real spread of values, computed fresh per table/market rather than a
    fixed guess. Degrades gracefully (fewer distinct real cuts) for a
    short list; `[]` for an empty one.
    """
    if not values:
        return []
    s = sorted(values)
    n = len(s)
    return [s[min(n - 1, int(p * n))] for p in (0.2, 0.4, 0.6, 0.8)]


def _rate_bucket(value: Optional[float], cuts: List[float]) -> Optional[int]:
    if value is None or not cuts:
        return None
    bucket = 0
    for c in cuts:
        if value > c:
            bucket += 1
    return min(bucket, 4)


def _chip(text: str, bucket: Optional[int]) -> str:
    if bucket is None:
        return _esc(text)
    return f'<span class="rate-chip rate-chip-{bucket}">{_esc(text)}</span>'


def _component_label(name: str) -> str:
    return name.replace("_", " ").replace("pct", "%").title().replace("Hr", "HR").replace("Fb", "FB").replace("Iso", "ISO").replace("Xslg", "xSLG")


def _other_props_html(player: str, market: str, event: str, all_props_by_player: Optional[Dict[str, List[EdgeCandidate]]]) -> str:
    """"Click a player, see their props": every other real scored
    candidate for this exact player tonight (a different market, or the
    same market in a different game - a doubleheader) - real data this
    project has already computed for every candidate every run, just
    never linked across markets before. Excludes the row's own (market,
    event) so a candidate never lists itself. Empty when the player has
    no other real candidate tonight - never a fabricated "nothing else"
    placeholder, just nothing rendered.
    """
    if not all_props_by_player:
        return ""
    others = [
        c for c in all_props_by_player.get(player.strip().lower(), []) if not (c.market == market and c.event == event)
    ]
    if not others:
        return ""
    ordered = sorted(others, key=lambda c: (c.event, c.market))
    rows = []
    for c in ordered:
        label, css_class = _verdict(c.has_market_data, c.tier, c.ev_percent_model)
        price_text = f"{c.best_line.odds:+d} {c.best_line.sportsbook}" if c.best_line else "no price yet"
        rows.append(
            f'<div class="game-roster-row"><span class="verdict {css_class}">{_esc(label)}</span>'
            f'<span class="grp"><b>{_esc(_MARKET_SHORT_LABELS.get(c.market, c.market))}</b><span class="grm">{_esc(c.event)}</span></span>'
            f'<span class="grz num">{_esc(price_text)}</span></div>'
        )
    return f'<div class="other-props-head">Also scored tonight</div>{"".join(rows)}'


def _component_detail_html(
    market: str,
    components: Dict[str, float],
    player: Optional[str] = None,
    event: Optional[str] = None,
    all_props_by_player: Optional[Dict[str, List[EdgeCandidate]]] = None,
) -> str:
    """A click-to-expand panel with two real, already-computed pieces:

    1. "Why" - the real per-component 0-100 value scoring.py computed for
       this candidate, that component's weight (see HR_WEIGHTS/TB_WEIGHTS/
       HITS_WEIGHTS - the same numbers the "Model methodology" section
       already shows in the abstract), and their real product - the
       actual points that component contributed to this exact player's
       score. Sorted by that contribution, biggest first, so the real
       driver of a ranking is never buried in a fixed weight order.
    2. "Also scored tonight" - this same player's other real candidates
       across markets/games, when `player`/`all_props_by_player` are
       supplied - see `_other_props_html`.

    Real data only throughout: `{}` components (predates this field - see
    EdgeCandidate.components' docstring), an unrecognized market, and no
    other real candidates for this player all render nothing for their
    respective piece; the whole toggle renders nothing at all if BOTH
    pieces are empty, never a fabricated or zeroed-out breakdown.
    """
    weights = _WEIGHTS_BY_MARKET.get(market)
    detail_rows = ""
    if weights and components:
        rows = sorted(
            ((k, components[k], w) for k, w in weights.items() if k in components),
            key=lambda row: row[1] * row[2],
            reverse=True,
        )
        detail_rows = "".join(
            f'<div class="detail-row"><span class="label">{_esc(_component_label(k))}</span>'
            f'<span class="val">{v:.0f}/100 &times; {w * 100:.0f}% = <b>{v * w:.1f}</b></span></div>'
            for k, v, w in rows
        )
    other_props = _other_props_html(player, market, event, all_props_by_player) if player and event else ""
    if not detail_rows and not other_props:
        return ""
    return f'<span class="expand-toggle" data-role="expand">why? &#9662;</span><div class="detail-panel">{detail_rows}{other_props}</div>'


def _prop_row(
    e: EdgeCandidate,
    heat: Optional[HeatIndex],
    kind: str,
    all_props_by_player: Optional[Dict[str, List[EdgeCandidate]]] = None,
    l15_cuts: Optional[List[float]] = None,
    season_cuts: Optional[List[float]] = None,
) -> str:
    # bp_model_prob: Ballpark Pal's own independent model, when configured
    # (see edges.py's EdgeCandidate docstring) - "n/a" for 2+ TB and
    # whenever it isn't configured or has no data for this matchup.
    bp_cell = f'<td class="num secondary-col">{_fmt_opt_pct(e.bp_model_prob)}</td>'
    # clearance_cols (from report.py, shared with the console report so the
    # two never drift): (L15 literal count, season rate) - see its
    # docstring for why L5/L10 aren't shown here either. clearance_rates is
    # the same real numbers as raw 0-1 floats, for the sort columns and the
    # colored rating chips below - never guessed when clearance_cols itself
    # would show "n/a". l15_cuts/season_cuts are this table's own real
    # quintile cut points (see _quantile_cuts) - None/[] renders plain text.
    l15, szn = clearance_cols(heat, kind)
    l15_rate, season_rate = clearance_rates(heat, kind)
    l15_data = "" if l15_rate is None else f"{l15_rate:.4f}"
    season_data = "" if season_rate is None else f"{season_rate:.4f}"
    l15_chip = _chip(l15, _rate_bucket(l15_rate, l15_cuts or []))
    season_chip = _chip(szn, _rate_bucket(season_rate, season_cuts or []))
    clr_cells = f'<td class="num secondary-col" data-k="l15">{l15_chip}</td><td class="num secondary-col" data-k="season">{season_chip}</td>'
    verdict_rank = _VERDICT_RANK[_verdict(e.has_market_data, e.tier, e.ev_percent_model)[0]]
    if not e.has_market_data:
        return f"""
          <tr data-player="{_esc(e.player.lower())}" data-book="" data-tier="no_market" data-prob="{e.model_prob}" data-ev=""
              data-verdict="{verdict_rank}" data-fair="" data-edge="" data-evmarket="" data-books="" data-weather=""
              data-l15="{l15_data}" data-season="{season_data}">
            <td data-k="verdict">{_verdict_badge(e.has_market_data, e.tier, e.ev_percent_model)}</td>
            <td class="player" data-k="player">{_esc(e.player)}<div class="tier model">Model only &mdash; no market price</div>{_lineup_source_note(e.lineup_source)}{_component_detail_html(e.market, e.components, e.player, e.event, all_props_by_player)}</td>
            <td class="event secondary-col">{_esc(e.event)}</td><td class="num" data-k="prob">{_fmt_pct(e.model_prob)}</td>{bp_cell}
            <td colspan="3" class="wx-cell">no book currently quotes this prop</td>
            <td colspan="5" class="secondary-col"></td>
            {clr_cells}</tr>"""
    # e.tier (edges.py) is the shared source of truth for this
    # classification - results.py's pick recording uses the same property,
    # so a recorded pick's tier always matches what this page showed for it.
    if e.tier == "agree":
        tier = '<div class="tier agree">Model + market agree</div>'
    elif e.tier == "model_only_single_sided":
        # Real price, but a single-sided market (e.g. an "Over 0.5"-only home
        # run prop with no "Under" leg) - no no-vig consensus to compare
        # against, so this is model-vs-price EV only. See edges.py.
        tier = '<div class="tier model">Model only &mdash; single-sided market, no no-vig consensus</div>'
    else:
        tier = '<div class="tier model">Model only</div>'
    wind = "dome" if e.is_dome else f"{abs(e.wind_out_mph):.0f}mph {'out' if e.wind_out_mph > 0 else 'in' if e.wind_out_mph < 0 else 'calm'}"
    temp = f"{e.temp_f:.0f}°F" if e.temp_f is not None else "n/a"
    edge_cell = f"{e.edge_vs_market:+.1%}" if e.edge_vs_market is not None else "n/a"
    ev_market_cell = f"{e.ev_percent_market:+.1f}%" if e.ev_percent_market is not None else "n/a"
    fair_data = "" if e.market_fair_prob is None else f"{e.market_fair_prob:.4f}"
    edge_data = "" if e.edge_vs_market is None else f"{e.edge_vs_market:.4f}"
    evmarket_data = "" if e.ev_percent_market is None else f"{e.ev_percent_market:.2f}"
    edge_chip = _chip(edge_cell, _edge_bucket(e.edge_vs_market))
    ev_chip = _chip(f"{e.ev_percent_model:+.1f}%", _ev_bucket(e.ev_percent_model))
    return f"""
          <tr data-player="{_esc(e.player.lower())}" data-book="{_esc(e.best_line.sportsbook.lower())}" data-tier="{_esc(e.tier)}" data-prob="{e.model_prob}" data-ev="{e.ev_percent_model}"
              data-verdict="{verdict_rank}" data-fair="{fair_data}" data-edge="{edge_data}" data-evmarket="{evmarket_data}"
              data-books="{e.books_quoting}" data-weather="{e.weather_boost_pct}" data-l15="{l15_data}" data-season="{season_data}">
            <td data-k="verdict">{_verdict_badge(e.has_market_data, e.tier, e.ev_percent_model)}</td>
            <td class="player" data-k="player">{_esc(e.player)}{tier}{_lineup_source_note(e.lineup_source)}{_component_detail_html(e.market, e.components, e.player, e.event, all_props_by_player)}</td>
            <td class="event secondary-col">{_esc(e.event)}</td>
            <td class="num" data-k="prob">{_fmt_pct(e.model_prob)}</td>
            {bp_cell}
            <td class="num pos" data-k="price">{e.best_line.odds:+d}</td>
            <td class="book" data-k="book">{_esc(e.best_line.sportsbook)}</td>
            <td class="num secondary-col" data-k="fair">{_fmt_opt_pct(e.market_fair_prob)}</td>
            <td class="num secondary-col" data-k="edge">{edge_chip}</td>
            <td class="num" data-k="ev">{ev_chip}</td>
            <td class="num secondary-col {'pos' if e.ev_percent_market is not None and e.ev_percent_market >= 0 else 'neg' if e.ev_percent_market is not None else ''}" data-k="evmarket">{ev_market_cell}</td>
            <td class="num secondary-col" data-k="books">{e.books_quoting}</td>
            <td class="wx-cell secondary-col" data-k="weather">{wind}, {temp} <b>{e.weather_boost_pct:+.1f}%</b></td>
            {clr_cells}
          </tr>"""


def _breakeven_cell(breakeven: Optional[int]) -> str:
    if breakeven is None:
        return ""
    return f'<div class="breakeven">beat {breakeven:+d}</div>'


def _take_bet_button(r: RecommendedBet, game_date_iso: str) -> str:
    """A client-side-only "I took this bet" toggle - see the My Bets
    section's disclosure for why this stays in the browser (localStorage)
    rather than writing anywhere server-side: a self-reported "I took
    this" is real, but it isn't verified ground truth of what actually
    happened, so it's never mixed into this project's own real,
    server-recorded performance tracking (results.py/data/results/*.jsonl) -
    that stays strictly resolved-from-real-outcomes only.

    `key` is stable per (date, market, player, event) so a page reload -
    even a full regeneration on the next run - re-marks the same button
    "taken" if its localStorage record still exists, and so the same real
    pick recorded twice in one day (this project's own snapshot
    convention - see PickRecord.recorded_at's docstring) still maps to
    one taken-bet entry, not two.
    """
    key = f"{game_date_iso}|{r.market}|{r.player.strip().lower()}|{r.event.strip().lower()}"
    return (
        f'<button type="button" class="take-btn" data-key="{_esc(key)}" '
        f'data-player="{_esc(r.player)}" data-market-label="{_esc(r.market_label)}" '
        f'data-event="{_esc(r.event)}" data-game-date="{_esc(game_date_iso)}" '
        f'data-price="{r.best_price}" data-book="{_esc(r.best_book)}" '
        f'data-units="{r.units:g}">Log this bet</button>'
    )


def _reco_row(r: RecommendedBet, game_date_iso: str, all_props_by_player: Optional[Dict[str, List[EdgeCandidate]]] = None) -> str:
    edge_cell = f"{r.edge_vs_market:+.1%}" if r.edge_vs_market is not None else "n/a"
    label, css_class = _verdict(True, r.tier, r.ev_percent_model)
    edge_chip = _chip(edge_cell, _edge_bucket(r.edge_vs_market))
    return f"""
      <div class="reco-row">
        <div><span class="verdict {css_class}" style="margin-right:8px;">{_esc(label)}</span><div class="who">{_esc(r.player)}</div><div class="bet">{_esc(r.market_label)}</div>{_lineup_source_note(r.lineup_source)}{_component_detail_html(r.market, r.components, r.player, r.event, all_props_by_player)}</div>
        <div class="event">{_esc(r.event)}</div>
        <div class="price num"><b class="pos">{r.best_price:+d}</b> {_esc(r.best_book)}{_breakeven_cell(r.breakeven)}</div>
        <div class="prob num">{_fmt_pct(r.model_prob)} model<div class="mkt-fair">{_fmt_opt_pct(r.market_fair_prob)} market fair</div></div>
        <div class="edge num">{edge_chip}</div>
        <div class="reco-units"><div class="n">{r.units:g}u</div><div class="lbl">size</div>{_take_bet_button(r, game_date_iso)}</div>
      </div>"""


# A real slate can clear the bar with far more plays than fit on one
# screen (every real +EV pick, no cap on the underlying selection) - past
# this many, the rest sit behind a "show N more" toggle rather than
# forcing everyone to scroll past all of them just to reach the next
# section. Nothing is hidden, just deferred one click.
_RECO_VISIBLE_CAP = 8


def _reco_group(
    title: str, hint: str, recs: List[RecommendedBet], game_date_iso: str, all_props_by_player: Optional[Dict[str, List[EdgeCandidate]]] = None
) -> str:
    if not recs:
        list_wrap = '<div class="reco-empty">No real plays cleared the bar here right now.</div>'
    else:
        visible, rest = recs[:_RECO_VISIBLE_CAP], recs[_RECO_VISIBLE_CAP:]
        body = "".join(_reco_row(r, game_date_iso, all_props_by_player) for r in visible)
        more_html = ""
        if rest:
            rest_rows = "".join(_reco_row(r, game_date_iso, all_props_by_player) for r in rest)
            more_html = f'<details class="reco-more"><summary>Show {len(rest)} more</summary>{rest_rows}</details>'
        list_wrap = f'<div class="reco-list">{body}{more_html}</div>'
    return f"""
    <div class="reco-group">
      <div class="reco-group-head"><h3>{_esc(title)}</h3><span class="hint">{_esc(hint)}</span></div>
      {list_wrap}
    </div>"""


def _recommended_bets_section(
    strong: List[RecommendedBet],
    speculative: List[RecommendedBet],
    game_date_iso: str,
    all_props_by_player: Optional[Dict[str, List[EdgeCandidate]]] = None,
) -> str:
    strong_html = _reco_group(
        f"Strong plays ({len(strong)})",
        "Model + market both see real value - our fundamentals and the market's own cross-book pricing agree",
        strong,
        game_date_iso,
        all_props_by_player,
    )
    speculative_html = _reco_group(
        f"Speculative ({len(speculative)})",
        "Model only, no market confirmation - real edge by our own numbers, but nothing else backs it up. Sized smaller, treat with more scrutiny",
        speculative,
        game_date_iso,
        all_props_by_player,
    )
    return f"""
  <section class="section" id="reco" style="margin-top:0;">
    <div class="section-head">
      <h2>Tonight's Recommended Bets</h2>
      <span class="hint">Every real +EV play that clears the bar, sized to a conservative fraction of Kelly</span>
    </div>
    {strong_html}
    {speculative_html}
    <div class="reco-disclosure">
      <b>How sizing works:</b> "size" is fractional Kelly - quarter-Kelly (0.25x) for Strong plays, an extra-conservative
      1/8-Kelly (0.125x) for Speculative ones - expressed in units where <b>1 unit = 1% of your bankroll</b> (this project
      doesn't know your actual bankroll). Every size is floored at 0.5u and capped at 3u regardless of what the raw math
      says, so one overconfident number can't recommend an outsized bet.
      <b>This model's probability is a hand-weighted heuristic (see mlb_props/scoring.py), not a calibrated prediction</b> -
      real Kelly sizing assumes that number is genuinely accurate, which is exactly what hasn't been proven yet (see the
      <a href="performance.html">Performance</a> page for the real, growing track record). Treat every size here as a
      conservative starting point to adjust with your own judgment, never as a guarantee. Never bet more than you can
      afford to lose. <b>"beat +N"/"beat -N"</b> under the price is the exact number to check against your actual
      sportsbook right now - every price on this page is a snapshot from whenever this report last ran, so if your
      book's real price is at least that good, the bet is still +EV even though the number shown here has moved;
      worse than that, it no longer is. <b>"market fair"</b> is the real cross-book no-vig consensus - genuinely "n/a"
      only for Speculative plays, where by definition no second book quotes the other side to de-vig against (see
      the Speculative hint above); every Strong play always has a real one, since that agreement is what makes it Strong.
    </div>
  </section>"""


def _my_bets_section() -> str:
    """Entirely client-side: no data from `report` feeds this - it's
    populated on load and updated on click by the JS at the bottom of the
    page, reading/writing `localStorage`. See `_take_bet_button`'s
    docstring for why this stays browser-local rather than being recorded
    anywhere server-side.
    """
    return """
  <section class="section" id="my-bets">
    <div class="section-head">
      <h2>My Bets</h2>
      <span class="hint" id="my-bets-count">0 bets logged</span>
    </div>
    <p class="reco-disclosure" style="margin-top:0;">
      Click <b>&ldquo;Log this bet&rdquo;</b> on a Recommended Bet above to record that you actually took it, at the
      recommended size. <b>Stored only in this browser</b> (localStorage) - it never leaves your device, doesn't sync
      across devices/browsers, isn't backed up anywhere, and clearing your browser data will lose it. This is your
      personal log, kept deliberately separate from this project's own real, server-recorded performance tracking
      (see the <a href="performance.html">Performance</a> page) - a self-reported "I took this" is real, but it isn't
      verified ground truth of what actually happened, so it's never mixed into the official track record.
    </p>
    <div class="table-scroll">
      <table class="props" style="min-width:0;">
        <thead><tr><th>Date</th><th>Player</th><th>Bet</th><th>Event</th><th>Price</th><th>Size</th><th></th></tr></thead>
        <tbody id="my-bets-tbody"></tbody>
      </table>
    </div>
    <div class="empty" id="my-bets-empty">No bets logged yet.</div>
  </section>"""


def _weight_rows(weights: dict) -> str:
    rows = []
    for name, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
        label = _component_label(name)
        rows.append(
            f'<div class="weight-row"><div class="wname">{_esc(label)}</div>'
            f'<div class="weight-bar"><i style="width:{w * 100:.0f}%"></i></div>'
            f'<div class="wval num">{w * 100:.0f}%</div></div>'
        )
    return "\n".join(rows)


def _prop_table(
    title: str,
    hint: str,
    edges: List[EdgeCandidate],
    prob_header: str,
    top: int,
    heat_by_player: Dict[str, HeatIndex],
    kind: str,
    table_id: str,
    all_props_by_player: Optional[Dict[str, List[EdgeCandidate]]] = None,
) -> str:
    if not edges:
        return f"""
    <div class="section-head"><h2>{_esc(title)}</h2><span class="hint">{_esc(hint)}</span></div>
    <div class="empty">No candidates scored for this slate.</div>"""
    shown = edges[:top]
    # This table's own real quintile cuts for the two clearance-rate
    # columns (see _quantile_cuts) - computed once here from every shown
    # row's real rate, rather than re-deriving it per row.
    l15_vals, season_vals = [], []
    for e in shown:
        l15_rate, season_rate = clearance_rates(heat_by_player.get(e.player), kind)
        if l15_rate is not None:
            l15_vals.append(l15_rate)
        if season_rate is not None:
            season_vals.append(season_rate)
    l15_cuts = _quantile_cuts(l15_vals)
    season_cuts = _quantile_cuts(season_vals)
    rows = "".join(
        _prop_row(e, heat_by_player.get(e.player), kind, all_props_by_player, l15_cuts, season_cuts) for e in shown
    )
    # Real books actually seen in this table, for the filter dropdown - never
    # a fixed list (a book with zero real prices tonight shouldn't appear as
    # a selectable, always-empty filter option).
    books = sorted({e.best_line.sportsbook for e in shown if e.has_market_data})
    book_options = "".join(f'<option value="{_esc(b.lower())}">{_esc(b)}</option>' for b in books)
    return f"""
    <div class="section-head"><h2>{_esc(title)}</h2><span class="hint">{_esc(hint)}</span></div>
    <div class="filter-bar">
      <input type="search" id="{table_id}-search" placeholder="Search player…">
      <select id="{table_id}-book"><option value="">All books</option>{book_options}</select>
      <select id="{table_id}-tier">
        <option value="">All tiers</option>
        <option value="agree">Model + market agree</option>
        <option value="model_only">Model only</option>
        <option value="model_only_single_sided">Model only (single-sided)</option>
        <option value="no_market">No market price</option>
      </select>
      <span class="filter-count" id="{table_id}-count"></span>
    </div>
    <input type="checkbox" class="view-toggle" id="{table_id}-full">
    <label class="view-toggle-wrap" for="{table_id}-full">Show full detail (matchup, BP model, market fair, edge, EV vs market, books, weather, clearance history)</label>
    <div class="table-scroll">
      <table class="props sortable" id="{table_id}-table">
        <thead><tr>
          <th data-k="verdict">Verdict<span class="arrow">▾</span></th>
          <th data-k="player">Player<span class="arrow">▾</span></th><th class="secondary-col">Matchup</th>
          <th data-k="prob">{_esc(prob_header)}<span class="arrow">▾</span></th><th class="secondary-col">BP Model</th>
          <th data-k="price">Best price<span class="arrow">▾</span></th><th data-k="book">Book<span class="arrow">▾</span></th>
          <th class="secondary-col" data-k="fair">Market fair<span class="arrow">▾</span></th><th class="secondary-col" data-k="edge">Edge<span class="arrow">▾</span></th>
          <th data-k="ev">EV (model)<span class="arrow">▾</span></th><th class="secondary-col" data-k="evmarket">EV (market)<span class="arrow">▾</span></th>
          <th class="secondary-col" data-k="books">Books<span class="arrow">▾</span></th><th class="secondary-col" data-k="weather">Weather<span class="arrow">▾</span></th>
          <th class="secondary-col" data-k="l15" title="Real games this player actually cleared this line, out of the last 15 games played">L15 clear<span class="arrow">▾</span></th>
          <th class="secondary-col" data-k="season" title="Same real clearance rate over the full season - the baseline to judge L15 clear against">Season rate<span class="arrow">▾</span></th>
        </tr></thead>
        <tbody id="{table_id}-tbody">{rows}
        </tbody>
      </table>
    </div>"""


def _quick_nav() -> str:
    """Jump straight to a section instead of scrolling past everything
    above it - the actual fix for a page that's grown long, without
    deleting any of the real content people asked to keep. Sticky, so
    it's still one tap away no matter how far down the page you've
    scrolled.
    """
    links = [
        ("#reco", "Recommended"),
        ("#my-bets", "My Bets"),
        ("#props-hr", "HR Props"),
        ("#props-tb", "2+ TB"),
        ("#props-hits", "1+ Hits"),
        ("#envs", "Matchups"),
        ("#hot", "Who's Hot"),
        ("#method", "Methodology"),
    ]
    return '<nav class="quick-nav">' + "".join(f'<a href="{href}">{_esc(label)}</a>' for href, label in links) + "</nav>"


def render_html_report(
    report: SlateReport,
    top: int = 15,
    is_mock: bool = False,
    generated_at: Optional[datetime] = None,
) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    envs = report.matchup_environments
    hot = report.hot_batters
    hr = report.hr_edges
    tb = report.tb_edges
    hits = report.hits_edges
    heat_by_player = heat_lookup(hot)

    # Two real, already-scored-candidate lookups, shared by every "click to
    # see more" feature below (game cards, prop-table/reco-row "why?"
    # panels) - built once here rather than per-row, from data this
    # project already computed this run.
    all_candidates = hr + tb + hits
    candidates_by_event: Dict[str, List[EdgeCandidate]] = {}
    all_props_by_player: Dict[str, List[EdgeCandidate]] = {}
    for c in all_candidates:
        candidates_by_event.setdefault(c.event, []).append(c)
        all_props_by_player.setdefault(c.player.strip().lower(), []).append(c)

    top_hr = next((e for e in hr if e.has_market_data), hr[0] if hr else None)
    top_tb = next((e for e in tb if e.has_market_data), tb[0] if tb else None)
    top_hits = next((e for e in hits if e.has_market_data), hits[0] if hits else None)
    best_env = envs[0] if envs else None
    hottest = hot[0] if hot else None
    strong_recs, speculative_recs = build_recommended_bets(report)

    def tile(label: str, value: str, sub: str) -> str:
        return f'<div class="tile"><div class="label">{_esc(label)}</div><div class="value">{_esc(value)}</div><div class="sub">{sub}</div></div>'

    def _top_tile_sub(top) -> str:
        if not top.has_market_data:
            return f"model {_fmt_pct(top.model_prob)} &middot; no market price"
        vs_market = f" vs market <b>{_fmt_pct(top.market_fair_prob)}</b>" if top.market_fair_prob is not None else " (single-sided market, no no-vig consensus)"
        return f"model <b>{_fmt_pct(top.model_prob)}</b>{vs_market} &middot; <span class=\"num pos\">{top.best_line.odds:+d}</span> {_esc(top.best_line.sportsbook)}"

    tiles = []
    if top_hr:
        tiles.append(tile("Top HR Prop", top_hr.player, _top_tile_sub(top_hr)))
    if top_tb:
        tiles.append(tile("Top 2+ TB Prop", top_tb.player, _top_tile_sub(top_tb)))
    if top_hits:
        tiles.append(tile("Top 1+ Hits Prop", top_hits.player, _top_tile_sub(top_hits)))
    if best_env:
        m = best_env.matchup
        tiles.append(tile("Best HR Environment", m.venue, f"{_esc(m.away_team)} @ {_esc(m.home_team)} &middot; env score <b>{best_env.environment_score:.1f}</b>/100"))
    if hottest:
        tiles.append(tile("Hottest Hitter", hottest.player, f"L15 wOBA <b>{hottest.last15_woba:.3f}</b> vs season <b>{hottest.season_woba:.3f}</b> &middot; z <b>{hottest.z_score:+.2f}</b>"))

    status_class = "sample" if is_mock else "live"
    status_text = (
        "SAMPLE OUTPUT &mdash; synthetic demo data (mock mode)."
        if is_mock
        else f"LIVE &mdash; fetched {generated_at.strftime('%Y-%m-%d %H:%M UTC')}."
    )
    status_detail = (
        "Statcast, schedule, weather and odds are all synthetic. Run without --mock (and with network access) for real picks."
        if is_mock
        else "Statcast (pybaseball), MLB Stats API, Open-Meteo, and Betstamp odds fetched fresh this run - nothing cached."
    )

    env_cards = "\n".join(
        _env_card(env, i + 1, candidates_by_event.get(f"{env.matchup.away_team} @ {env.matchup.home_team}", []))
        for i, env in enumerate(envs[:10])
    )
    hot_rows = "\n".join(_hot_row(i + 1, h) for i, h in enumerate(hot[:10]))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Longball Board</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;700;800&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>{_STYLE}</style>
</head>
<body>
<div class="wrap">
  {nav_html("board")}
  <header class="top">
    <div class="brand-block">
      <div class="brand"><span class="mark">MLB PROPS</span></div>
      <h1 class="title">Longball Board</h1>
      <p class="subtitle">Home run &amp; 2+ total bases prop rankings &mdash; Statcast quality of contact, matchup/platoon edges, recent form, ballpark &amp; weather, cross-book odds.</p>
    </div>
    <div class="meta">
      <div class="date num">{_esc(report.game_date.isoformat())}</div>
      <div>{len(report.slate)}-game slate &middot; regenerated every run</div>
    </div>
  </header>

  <div class="status-bug {status_class}">
    <span class="dot"></span>
    {status_text}
    <span class="detail">{status_detail}</span>
  </div>

  {_quick_nav()}

  {_recommended_bets_section(strong_recs, speculative_recs, report.game_date.isoformat(), all_props_by_player)}

  {_my_bets_section()}

  <section class="section">
    <span class="eyebrow">At a glance</span>
    <div class="tiles">
      {''.join(tiles) if tiles else '<div class="empty">Not enough data scored yet.</div>'}
    </div>
  </section>

  <section class="section" id="props-hr">
    {_prop_table("Best home run props", 'Ranked by our model’s EV% against the best live price - "agree" = model & market both see value', hr, "Model P(HR)", top, heat_by_player, "hr", "hr", all_props_by_player)}
  </section>

  <section class="section" id="props-tb">
    {_prop_table("Best 2+ total bases props", "Ranked by our model's EV% against the best live price", tb, "Model P(2+ TB)", top, heat_by_player, "tb2", "tb", all_props_by_player)}
  </section>

  <section class="section" id="props-hits">
    {_prop_table("Best 1+ hits props", "Ranked by our model's EV% against the best live price", hits, "Model P(1+ Hits)", top, heat_by_player, "hit", "hits", all_props_by_player)}
  </section>

  <details class="section" id="envs">
    <summary class="collapse-head">
      <h2>Best HR matchups on the slate</h2>
      <span class="hint">Park factor &middot; wind/temp &middot; opposing starter vulnerability</span>
      <span class="details-arrow">&#9662;</span>
    </summary>
    <div class="env-grid">{env_cards if env_cards else '<div class="empty">No games on this slate.</div>'}</div>
  </details>

  <details class="section" id="hot">
    <summary class="collapse-head">
      <h2>Who's hot</h2>
      <span class="hint">Last-15-day wOBA vs. season baseline, as a z-score</span>
      <span class="details-arrow">&#9662;</span>
    </summary>
    <div class="hot-list">{hot_rows if hot_rows else '<div class="empty">No batters scored.</div>'}</div>
  </details>

  <details class="section" id="method">
    <summary class="collapse-head">
      <h2>How the score is built</h2>
      <span class="hint">mlb_props/scoring.py &mdash; every weight below, verbatim</span>
      <span class="details-arrow">&#9662;</span>
    </summary>
    <div class="method-grid">
      <div class="method-card"><h3>Home run score (weights)</h3>{_weight_rows(HR_WEIGHTS)}</div>
      <div class="method-card"><h3>2+ total bases score (weights)</h3>{_weight_rows(TB_WEIGHTS)}</div>
      <div class="method-card"><h3>1+ hits score (weights)</h3>{_weight_rows(HITS_WEIGHTS)}<p class="hint" style="margin-top:8px;">"Batter K%" and "Pitcher K% Allowed" are real per-player strikeout rates (see scoring.py) - a season-long rate stat, not a whiff rate specific to tonight's exact pitch-mix matchup.</p></div>
    </div>
    <div class="sources">
      <span class="source-chip">Statcast batted-ball quality &rarr; pybaseball / Baseball Savant</span>
      <span class="source-chip">Platoon, BvP, pitch-mix &rarr; Statcast pitch logs</span>
      <span class="source-chip">Recent form (7/15/30d) &rarr; FanGraphs range stats</span>
      <span class="source-chip">Slate + probable pitchers &rarr; MLB Stats API</span>
      <span class="source-chip">Wind + temperature &rarr; Open-Meteo</span>
      <span class="source-chip">Cross-book odds + no-vig fair price &rarr; Betstamp</span>
    </div>
  </details>

  <footer>
    <p><strong>Weather is a first-class input:</strong> each pick's wind (mph, in/out) and temperature feed a heuristic HR-odds shift, weighted directly into both scores (8% of the HR score, 5% of the 2+ TB score) and shown per-pick in the Weather column.</p>
    <p><strong>Model score &ne; prediction.</strong> It's a transparent, hand-weighted heuristic calibrated to realistic MLB base rates, meant to be cross-checked against the market's own no-vig consensus (Market fair column) - not treated as ground truth.</p>
    <p>"Model + market agree" rows are where our fundamentals and the market's own cross-book pricing both say a price is generous; "model only" rows lean on our score alone and warrant more scrutiny.</p>
    <p><strong>Data quality notes (permanent, applies every run):</strong></p>
    <p>&middot; <strong>"EV%" means model vs. market, not "market is wrong."</strong> Every EV% figure is our model's probability compared against the book's own no-vig fair price. A positive EV% is our model disagreeing with the market in the bettor's favor, not proof the market is mispriced - the market could just as easily be right and our model wrong. Weigh it as one informed opinion against another, not a guarantee.</p>
    <p>&middot; <strong>Pull-air% is permanently unavailable for the HR score (6% of its weight).</strong> Neither Baseball Savant leaderboard this project pulls carries a pull-rate column, and FanGraphs (which does) returns 403 to every request from this environment's hosting provider. That component defaults to 0 for every player, every run - a real, disclosed gap, not a hidden zero.</p>
    <p>&middot; <strong>"BP Model" (HR/Hits tables, when configured) is Ballpark Pal's own independent model</strong> - a genuine second opinion, not this project's model shown twice. Their real numbers are per-plate-appearance; converted here to a per-game figure via P(at least 1 in ~4.3 PA) so it's comparable to "Model". Agreement is more reassuring than either model alone; disagreement means two different models read the matchup differently, not that one is wrong. "n/a" means not configured or no data for that matchup.</p>
    <p>Every prop table is sortable (click a column header) and filterable (search/book/tier, above each table). See the <a href="performance.html">Performance</a> page for this model's real track record - past picks resolved against what actually happened, plus closing-line value.</p>
  </footer>
</div>
<script>
(function(){{
  // Shared by every "why?" toggle on the page (prop tables + Recommended
  // Bets) - one delegated listener rather than wiring each row, so this
  // still works after applySort() below re-appends rows elsewhere in the
  // DOM.
  document.addEventListener('click', function(e) {{
    var toggle = e.target.closest('.expand-toggle');
    if (!toggle) return;
    var panel = toggle.nextElementSibling;
    if (!panel || !panel.classList.contains('detail-panel')) return;
    var open = panel.classList.toggle('open');
    toggle.innerHTML = open ? 'why? &#9652;' : 'why? &#9662;';
    // Confirmed live: a prop table's player column is sticky (stays
    // pinned while the wide table scrolls horizontally - see
    // site_style.py), so an opened panel wider than that pinned column
    // could render past the edge of the screen with no way to scroll to
    // it (scrolling a container never moves a sticky element). CSS
    // (td.player:has(.detail-panel.open){{position:static}}) now makes
    // the cell scroll normally again once its own panel is open, which
    // is what actually fixes it - this just brings it into view
    // automatically so nobody has to go hunting for the right scroll
    // position by hand.
    if (open) {{
      panel.scrollIntoView({{behavior: 'smooth', block: 'nearest', inline: 'nearest'}});
    }}
  }});

  function initPropTable(id) {{
    var table = document.getElementById(id + '-table');
    if (!table) return;
    var tbody = document.getElementById(id + '-tbody');
    var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
    var search = document.getElementById(id + '-search');
    var bookSel = document.getElementById(id + '-book');
    var tierSel = document.getElementById(id + '-tier');
    var countEl = document.getElementById(id + '-count');
    var thead = table.querySelector('thead');
    var sortKey = null, sortDir = 1;

    function applyFilters() {{
      var q = search.value.trim().toLowerCase();
      var book = bookSel.value;
      var tier = tierSel.value;
      var visible = 0;
      rows.forEach(function(row) {{
        var show = true;
        if (q && row.dataset.player.indexOf(q) === -1) show = false;
        if (book && row.dataset.book !== book) show = false;
        if (tier && row.dataset.tier !== tier) show = false;
        row.classList.toggle('hidden-row', !show);
        if (show) visible++;
      }});
      countEl.textContent = visible + ' of ' + rows.length + ' shown';
    }}

    // Every column with a real "better" direction sorts best-first on the
    // very first click (biggest number on top) - a second click on the
    // same header reverses it. Text columns (player/book) start A-Z, the
    // normal expectation there - no real "best" name/book to lead with.
    var NUMERIC_KEYS = ['prob', 'price', 'ev', 'verdict', 'fair', 'edge', 'evmarket', 'books', 'weather', 'l15', 'season'];

    function applySort(key) {{
      var numeric = NUMERIC_KEYS.indexOf(key) !== -1;
      if (sortKey === key) {{ sortDir = -sortDir; }} else {{ sortKey = key; sortDir = numeric ? -1 : 1; }}
      rows.sort(function(a, b) {{
        var av, bv;
        if (numeric) {{
          av = parseFloat(a.dataset[key]) || 0;
          bv = parseFloat(b.dataset[key]) || 0;
        }} else {{
          av = (a.dataset[key] || '').toLowerCase();
          bv = (b.dataset[key] || '').toLowerCase();
        }}
        if (av < bv) return -1 * sortDir;
        if (av > bv) return 1 * sortDir;
        return 0;
      }});
      rows.forEach(function(row) {{ tbody.appendChild(row); }});
      thead.querySelectorAll('th').forEach(function(th) {{
        th.classList.toggle('sorted', th.getAttribute('data-k') === key);
      }});
    }}

    thead.querySelectorAll('th[data-k]').forEach(function(th) {{
      th.addEventListener('click', function() {{ applySort(th.getAttribute('data-k')); }});
    }});
    [search, bookSel, tierSel].forEach(function(el) {{
      el.addEventListener('input', applyFilters);
      el.addEventListener('change', applyFilters);
    }});
    applyFilters();
  }}
  ['hr', 'tb', 'hits'].forEach(initPropTable);

  // My Bets: entirely client-side, see _take_bet_button's/_my_bets_section's
  // docstrings in html_report.py for why this stays in localStorage rather
  // than being recorded anywhere server-side. One JSON array under a single
  // key - small enough (a real season of taken bets is at most a few
  // hundred rows) that there's no need for anything fancier.
  var TAKEN_KEY = 'mlbPropsTakenBets';

  function loadTakenBets() {{
    try {{
      var raw = localStorage.getItem(TAKEN_KEY);
      return raw ? JSON.parse(raw) : [];
    }} catch (e) {{ return []; }}
  }}
  function saveTakenBets(list) {{
    try {{ localStorage.setItem(TAKEN_KEY, JSON.stringify(list)); }} catch (e) {{ /* storage unavailable (private mode, quota) - the click still updates the page, just won't survive reload */ }}
  }}

  function renderMyBets() {{
    var tbody = document.getElementById('my-bets-tbody');
    if (!tbody) return;
    var list = loadTakenBets().slice().sort(function(a, b) {{
      return (b.takenAt || '').localeCompare(a.takenAt || '');
    }});
    tbody.innerHTML = '';
    var totalUnits = 0;
    list.forEach(function(b) {{
      totalUnits += (parseFloat(b.units) || 0);
      var priceText = b.price != null && b.price !== '' ? (parseFloat(b.price) > 0 ? '+' : '') + b.price + ' ' + (b.book || '') : 'n/a';
      var tr = document.createElement('tr');
      var cells = [b.gameDate || '', b.player || '', b.marketLabel || '', b.event || '', priceText, (b.units != null ? b.units + 'u' : '')];
      cells.forEach(function(text, i) {{
        var td = document.createElement('td');
        if (i === 1) td.className = 'player';
        if (i === 3) td.className = 'event';
        if (i === 4 || i === 5) td.className = 'num';
        td.textContent = text;
        tr.appendChild(td);
      }});
      var removeTd = document.createElement('td');
      var removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'untake-btn';
      removeBtn.setAttribute('data-key', b.key);
      removeBtn.textContent = 'Remove';
      removeTd.appendChild(removeBtn);
      tr.appendChild(removeTd);
      tbody.appendChild(tr);
    }});
    var countEl = document.getElementById('my-bets-count');
    if (countEl) countEl.textContent = list.length + ' bet' + (list.length === 1 ? '' : 's') + ' logged' + (list.length ? ' · ' + (Math.round(totalUnits * 10) / 10) + 'u total' : '');
    var emptyEl = document.getElementById('my-bets-empty');
    if (emptyEl) emptyEl.style.display = list.length ? 'none' : '';
    // Re-sync every "Log this bet" button on the page (today's board only
    // ever shows today's picks, but a taken bet from a prior day the
    // button no longer exists for still lives in the log above).
    document.querySelectorAll('.take-btn').forEach(function(btn) {{
      var key = btn.getAttribute('data-key');
      var taken = list.some(function(b) {{ return b.key === key; }});
      btn.textContent = taken ? '✓ Taken — click to undo' : 'Log this bet';
      btn.classList.toggle('taken', taken);
    }});
  }}

  document.addEventListener('click', function(e) {{
    var takeBtn = e.target.closest('.take-btn');
    if (takeBtn) {{
      var key = takeBtn.getAttribute('data-key');
      var list = loadTakenBets();
      var idx = -1;
      for (var i = 0; i < list.length; i++) {{ if (list[i].key === key) {{ idx = i; break; }} }}
      if (idx !== -1) {{
        list.splice(idx, 1);
      }} else {{
        list.push({{
          key: key,
          player: takeBtn.getAttribute('data-player'),
          marketLabel: takeBtn.getAttribute('data-market-label'),
          event: takeBtn.getAttribute('data-event'),
          gameDate: takeBtn.getAttribute('data-game-date'),
          price: takeBtn.getAttribute('data-price'),
          book: takeBtn.getAttribute('data-book'),
          units: takeBtn.getAttribute('data-units'),
          takenAt: new Date().toISOString()
        }});
      }}
      saveTakenBets(list);
      renderMyBets();
      return;
    }}
    var removeBtn = e.target.closest('.untake-btn');
    if (removeBtn) {{
      var removeKey = removeBtn.getAttribute('data-key');
      saveTakenBets(loadTakenBets().filter(function(b) {{ return b.key !== removeKey; }}));
      renderMyBets();
    }}
  }});

  renderMyBets();
}})();
</script>
</body>
</html>
"""
