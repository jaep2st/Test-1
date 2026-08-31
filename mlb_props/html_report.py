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

from .betting import LiveValueBet, RecommendedBet, build_recommended_bets
from .edges import EdgeCandidate
from .hot_streak import HeatIndex
from .pipeline import MatchupEnvironment, SlateReport
from .report import clearance_cols, heat_lookup
from .scoring import HITS_WEIGHTS, HR_WEIGHTS, TB_WEIGHTS
from .site_style import STYLE as _STYLE
from .site_style import nav_html


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


def _env_card(env: MatchupEnvironment, rank: int) -> str:
    m = env.matchup
    wind_class = "wind-out" if env.weather_boost_pct > 0 else ("wind-in" if env.weather_boost_pct < 0 else "")
    pitchers = " vs ".join(p for p in (m.away_pitcher, m.home_pitcher) if p) or "Probable pitchers TBA"
    return f"""
      <div class="env-card">
        <div class="rank">#{rank} ENVIRONMENT</div>
        <div class="matchup">{_esc(m.away_team)} @ {_esc(m.home_team)}</div>
        <div class="pitchers">{_esc(pitchers)}</div>
        <div class="park">{_esc(m.venue)} &middot; park HR factor {env.park_hr_factor:.0f} (neutral = 100)</div>
        <div class="env-bar-row"><div class="env-bar"><i style="width:{max(0, min(100, env.environment_score)):.1f}%"></i></div><div class="env-score num">{env.environment_score:.1f}</div></div>
        <div class="env-chips">
          <span class="chip {wind_class}">{env.weather_boost_pct:+.1f}% weather</span>
        </div>
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


def _prop_row(e: EdgeCandidate, heat: Optional[HeatIndex], kind: str) -> str:
    # bp_model_prob: Ballpark Pal's own independent model, when configured
    # (see edges.py's EdgeCandidate docstring) - "n/a" for 2+ TB and
    # whenever it isn't configured or has no data for this matchup.
    bp_cell = f'<td class="num">{_fmt_opt_pct(e.bp_model_prob)}</td>'
    # clearance_cols (from report.py, shared with the console report so the
    # two never drift): (L15 literal count, season rate) - see its
    # docstring for why L5/L10 aren't shown here either.
    l15, szn = clearance_cols(heat, kind)
    clr_cells = f'<td class="num">{_esc(l15)}</td><td class="num">{_esc(szn)}</td>'
    if not e.has_market_data:
        return f"""
          <tr data-player="{_esc(e.player.lower())}" data-book="" data-tier="no_market" data-prob="{e.model_prob}" data-ev="">
            <td class="player" data-k="player">{_esc(e.player)}<div class="tier model">Model only &mdash; no market price</div></td>
            <td class="event">{_esc(e.event)}</td><td class="num" data-k="prob">{_fmt_pct(e.model_prob)}</td>{bp_cell}
            <td colspan="6" class="wx-cell">no book currently quotes this prop</td>{clr_cells}</tr>"""
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
    return f"""
          <tr data-player="{_esc(e.player.lower())}" data-book="{_esc(e.best_line.sportsbook.lower())}" data-tier="{_esc(e.tier)}" data-prob="{e.model_prob}" data-ev="{e.ev_percent_model}">
            <td class="player" data-k="player">{_esc(e.player)}{tier}</td>
            <td class="event">{_esc(e.event)}</td>
            <td class="num" data-k="prob">{_fmt_pct(e.model_prob)}</td>
            {bp_cell}
            <td class="num pos" data-k="price">{e.best_line.odds:+d}</td>
            <td class="book" data-k="book">{_esc(e.best_line.sportsbook)}</td>
            <td class="num">{_fmt_opt_pct(e.market_fair_prob)}</td>
            <td class="num {'pos' if e.edge_vs_market is not None and e.edge_vs_market >= 0 else 'neg' if e.edge_vs_market is not None else ''}">{edge_cell}</td>
            <td class="num {'pos' if e.ev_percent_model >= 0 else 'neg'}" data-k="ev">{e.ev_percent_model:+.1f}%</td>
            <td class="num {'pos' if e.ev_percent_market is not None and e.ev_percent_market >= 0 else 'neg' if e.ev_percent_market is not None else ''}">{ev_market_cell}</td>
            <td class="num">{e.books_quoting}</td>
            <td class="wx-cell">{wind}, {temp} <b>{e.weather_boost_pct:+.1f}%</b></td>
            {clr_cells}
          </tr>"""


def _reco_row(r: RecommendedBet) -> str:
    edge_cell = f"{r.edge_vs_market:+.1%}" if r.edge_vs_market is not None else "n/a"
    return f"""
      <div class="reco-row">
        <div><div class="who">{_esc(r.player)}</div><div class="bet">{_esc(r.market_label)}</div></div>
        <div class="event">{_esc(r.event)}</div>
        <div class="price num"><b class="pos">{r.best_price:+d}</b> {_esc(r.best_book)}</div>
        <div class="prob num">{_fmt_pct(r.model_prob)} model</div>
        <div class="edge num {'pos' if r.edge_vs_market is not None and r.edge_vs_market >= 0 else 'neg' if r.edge_vs_market is not None else ''}">{edge_cell}</div>
        <div class="reco-units"><div class="n">{r.units:g}u</div><div class="lbl">size</div></div>
      </div>"""


def _reco_group(title: str, hint: str, recs: List[RecommendedBet]) -> str:
    body = (
        "".join(_reco_row(r) for r in recs)
        if recs
        else '<div class="reco-empty">No real plays cleared the bar here right now.</div>'
    )
    list_wrap = f'<div class="reco-list">{body}</div>' if recs else body
    return f"""
    <div class="reco-group">
      <div class="reco-group-head"><h3>{_esc(title)}</h3><span class="hint">{_esc(hint)}</span></div>
      {list_wrap}
    </div>"""


def _recommended_bets_section(strong: List[RecommendedBet], speculative: List[RecommendedBet]) -> str:
    strong_html = _reco_group(
        f"Strong plays ({len(strong)})",
        "Model + market both see real value - our fundamentals and the market's own cross-book pricing agree",
        strong,
    )
    speculative_html = _reco_group(
        f"Speculative ({len(speculative)})",
        "Model only, no market confirmation - real edge by our own numbers, but nothing else backs it up. Sized smaller, treat with more scrutiny",
        speculative,
    )
    return f"""
  <section class="section" style="margin-top:0;">
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
      afford to lose.
    </div>
  </section>"""


def _live_row(b: LiveValueBet) -> str:
    return f"""
      <div class="reco-row">
        <div><div class="who">{_esc(b.player)}</div><div class="bet">{_esc(b.market_label)}</div></div>
        <div class="event">{_esc(b.event)}</div>
        <div class="price num"><b class="pos">{b.best_price:+d}</b> {_esc(b.best_book)}</div>
        <div class="prob num">{_fmt_pct(b.fair_prob)} mkt</div>
        <div class="edge num pos">{b.ev_percent:+.1f}%</div>
        <div class="reco-units"><div class="n">{b.units:g}u</div><div class="lbl">size</div></div>
      </div>"""


def _live_bets_section(live_bets: List[LiveValueBet]) -> str:
    body = (
        "".join(_live_row(b) for b in live_bets)
        if live_bets
        else '<div class="reco-empty">No real live cross-book value right now - check back once more games are underway.</div>'
    )
    list_wrap = f'<div class="reco-list">{body}</div>' if live_bets else body
    return f"""
  <section class="section">
    <div class="section-head">
      <h2>Live Right Now ({len(live_bets)})</h2>
      <span class="hint">Already-started games only - real cross-book value, not this model's own score</span>
    </div>
    {list_wrap}
    <div class="reco-disclosure">
      <b>This is a different kind of signal than the recommendations above.</b> This project's own model has no
      live-game-state awareness at all (no tracking of a batter's plate appearances remaining today), so a live bet is
      never sized off this model's probability - there isn't one to use. "mkt" here is the real, de-vigged consensus
      probability across the books quoting both sides of this exact live line right now, and "size" is an extra-
      conservative 1/8-Kelly against that consensus (same units convention as above). The real risk with a live price
      is less about that consensus being wrong and more about <b>timing</b>: a live line can reprice or disappear
      within seconds, so a real edge here might already be gone by the time you'd act on it. Verify the price is still
      live on your book before betting it.
    </div>
  </section>"""


def _weight_rows(weights: dict) -> str:
    rows = []
    for name, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
        label = name.replace("_", " ").replace("pct", "%").title().replace("Hr", "HR").replace("Fb", "FB").replace("Iso", "ISO").replace("Xslg", "xSLG")
        rows.append(
            f'<div class="weight-row"><div class="wname">{_esc(label)}</div>'
            f'<div class="weight-bar"><i style="width:{w * 100:.0f}%"></i></div>'
            f'<div class="wval num">{w * 100:.0f}%</div></div>'
        )
    return "\n".join(rows)


def _prop_table(
    title: str, hint: str, edges: List[EdgeCandidate], prob_header: str, top: int, heat_by_player: Dict[str, HeatIndex], kind: str, table_id: str
) -> str:
    if not edges:
        return f"""
    <div class="section-head"><h2>{_esc(title)}</h2><span class="hint">{_esc(hint)}</span></div>
    <div class="empty">No candidates scored for this slate.</div>"""
    shown = edges[:top]
    rows = "".join(_prop_row(e, heat_by_player.get(e.player), kind) for e in shown)
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
    <div class="table-scroll">
      <table class="props sortable" id="{table_id}-table">
        <thead><tr>
          <th data-k="player">Player<span class="arrow">▾</span></th><th>Matchup</th>
          <th data-k="prob">{_esc(prob_header)}<span class="arrow">▾</span></th><th>BP Model</th>
          <th data-k="price">Best price<span class="arrow">▾</span></th><th data-k="book">Book<span class="arrow">▾</span></th>
          <th>Market fair</th><th>Edge</th><th data-k="ev">EV (model)<span class="arrow">▾</span></th><th>EV (market)</th><th>Books</th><th>Weather</th>
          <th title="Real games this player actually cleared this line, out of the last 15 games played">L15 clear</th>
          <th title="Same real clearance rate over the full season - the baseline to judge L15 clear against">Season rate</th>
        </tr></thead>
        <tbody id="{table_id}-tbody">{rows}
        </tbody>
      </table>
    </div>"""


def render_html_report(
    report: SlateReport,
    top: int = 15,
    is_mock: bool = False,
    generated_at: Optional[datetime] = None,
    live_bets: Optional[List[LiveValueBet]] = None,
) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    envs = report.matchup_environments
    hot = report.hot_batters
    hr = report.hr_edges
    tb = report.tb_edges
    hits = report.hits_edges
    heat_by_player = heat_lookup(hot)

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

    env_cards = "\n".join(_env_card(env, i + 1) for i, env in enumerate(envs[:10]))
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

  {_recommended_bets_section(strong_recs, speculative_recs)}

  {_live_bets_section(live_bets or [])}

  <section class="section">
    <span class="eyebrow">At a glance</span>
    <div class="tiles">
      {''.join(tiles) if tiles else '<div class="empty">Not enough data scored yet.</div>'}
    </div>
  </section>

  <section class="section">
    <div class="section-head">
      <h2>Best HR matchups on the slate</h2>
      <span class="hint">Park factor &middot; wind/temp &middot; opposing starter vulnerability</span>
    </div>
    <div class="env-grid">{env_cards if env_cards else '<div class="empty">No games on this slate.</div>'}</div>
  </section>

  <section class="section">
    <div class="section-head">
      <h2>Who's hot</h2>
      <span class="hint">Last-15-day wOBA vs. season baseline, as a z-score</span>
    </div>
    <div class="hot-list">{hot_rows if hot_rows else '<div class="empty">No batters scored.</div>'}</div>
  </section>

  <section class="section">
    {_prop_table("Best home run props", 'Ranked by our model’s EV% against the best live price - "agree" = model & market both see value', hr, "Model P(HR)", top, heat_by_player, "hr", "hr")}
  </section>

  <section class="section">
    {_prop_table("Best 2+ total bases props", "Ranked by our model's EV% against the best live price", tb, "Model P(2+ TB)", top, heat_by_player, "tb2", "tb")}
  </section>

  <section class="section">
    {_prop_table("Best 1+ hits props", "Ranked by our model's EV% against the best live price", hits, "Model P(1+ Hits)", top, heat_by_player, "hit", "hits")}
  </section>

  <section class="section">
    <div class="section-head">
      <h2>How the score is built</h2>
      <span class="hint">mlb_props/scoring.py &mdash; every weight below, verbatim</span>
    </div>
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
  </section>

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

    function applySort(key) {{
      if (sortKey === key) {{ sortDir = -sortDir; }} else {{ sortKey = key; sortDir = 1; }}
      var numeric = key === 'prob' || key === 'price' || key === 'ev';
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
}})();
</script>
</body>
</html>
"""
