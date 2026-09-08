"""Renders a `WeekReport` as a self-contained, styled HTML page - reuses
`mlb_props/site_style.py`'s shared look (see `ncaaf/site_style.py`). Used
by `ncaaf_main.py --html-out` and the `mlb-props-report` GitHub Actions
workflow to publish a live page to `public/ncaaf/index.html`.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import List, Optional

from .betting import MIN_EV_PERCENT_TO_RECOMMEND, RecommendedBet, build_recommended_bets
from .edges import EdgeCandidate
from .market import book_display_name
from .pipeline import WeekReport
from .ratings import OFFENSE_POINTS_STD, POWER_RATING_POINTS_STD, W_ELO, W_SP_OVERALL
from .site_style import STYLE as _STYLE
from .site_style import nav_html

_TIER_LABEL = {"agree": "STRONG BET", "model_only": "SPECULATIVE", "model_only_single_sided": "SPECULATIVE", "no_market": "NO PRICE"}
_TIER_CLASS = {"agree": "verdict-strong", "model_only": "verdict-speculative", "model_only_single_sided": "verdict-speculative", "no_market": "verdict-none"}


def _esc(s: object) -> str:
    return html.escape(str(s))


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def _picks_table(candidates: List[EdgeCandidate]) -> str:
    if not candidates:
        return '<div class="empty">No priced candidates this run.</div>'
    rows = []
    for c in candidates:
        label = _TIER_LABEL.get(c.tier, c.tier)
        css = _TIER_CLASS.get(c.tier, "verdict-none")
        price = f"{c.best_line.odds:+d}" if c.best_line else "n/a"
        book = book_display_name(c.best_line.sportsbook) if c.best_line else "n/a"
        ev_model = f"{c.ev_percent_model:+.1f}%" if c.ev_percent_model is not None else "n/a"
        ev_market = f"{c.ev_percent_market:+.1f}%" if c.ev_percent_market is not None else "n/a"
        rows.append(
            f"<tr><td><span class='verdict {css}'>{_esc(label)}</span></td>"
            f"<td class='player'>{_esc(c.event)}</td>"
            f"<td>{_esc(c.market)}</td>"
            f"<td>{_esc(c.selection_label)}</td>"
            f"<td class='num'>{_fmt_pct(c.model_prob)}</td>"
            f"<td class='num secondary-col'>{_fmt_pct(c.market_fair_prob)}</td>"
            f"<td class='num'>{_esc(price)}</td>"
            f"<td class='book'>{_esc(book)}</td>"
            f"<td class='num'>{_esc(ev_model)}</td>"
            f"<td class='num secondary-col'>{_esc(ev_market)}</td>"
            f"<td class='num secondary-col'>{c.books_quoting}</td></tr>"
        )
    return f"""
    <label class="view-toggle-wrap"><input class="view-toggle" type="checkbox"> Show full detail</label>
    <div class="table-scroll"><table class="props">
      <thead><tr>
        <th>Verdict</th><th data-k="player">Game</th><th>Market</th><th>Selection</th><th>Model %</th>
        <th class="secondary-col">Market Fair %</th><th>Price</th><th>Book</th><th>EV (model)</th>
        <th class="secondary-col">EV (market)</th><th class="secondary-col">Books</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div>"""


def _predicted_scores_table(report: WeekReport) -> str:
    if not report.scores:
        return '<div class="empty">No games could be scored this week.</div>'
    rows = []
    for event, score in sorted(report.scores.items()):
        fav = score.matchup.home_team if score.predicted_margin >= 0 else score.matchup.away_team
        rows.append(
            f"<tr><td class='player'>{_esc(event)}</td>"
            f"<td class='num'>{score.predicted_away_points:.1f} - {score.predicted_home_points:.1f}</td>"
            f"<td class='num'>{score.predicted_margin:+.1f} ({_esc(fav)})</td>"
            f"<td class='num'>{score.predicted_total:.1f}</td>"
            f"<td class='num'>{score.home_win_prob:.0%}</td>"
            f"<td class='secondary-col'>{_esc(score.context.venue)}{' (neutral)' if score.context.neutral_site else ''}</td>"
            f"<td class='num secondary-col'>{score.context.total_margin_adjustment_pts:+.1f}</td></tr>"
        )
    return f"""
    <div class="table-scroll"><table class="props">
      <thead><tr><th data-k="player">Game</th><th>Pred. Score (Away-Home)</th><th>Margin</th><th>Total</th>
        <th>Home Win %</th><th class="secondary-col">Venue</th><th class="secondary-col">Context Adj.</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div>"""


def _reco_row(r: RecommendedBet) -> str:
    return f"""
      <div class="reco-row">
        <div class="who">{_esc(r.event)}</div>
        <div class="event">{_esc(r.market_label)}: {_esc(r.selection)}</div>
        <div class="price num"><b>{r.best_price:+d}</b> {_esc(book_display_name(r.best_book))}</div>
        <div class="prob num">{_fmt_pct(r.model_prob)}</div>
        <div class="edge num">{f'{r.edge_vs_market:+.1%}' if r.edge_vs_market is not None else 'n/a'}</div>
        <div class="reco-units"><span class="n">{r.units:g}u</span><span class="lbl">EV {r.ev_percent_model:+.1f}%</span></div>
      </div>"""


def _reco_section(strong: List[RecommendedBet], speculative: List[RecommendedBet]) -> str:
    if not strong and not speculative:
        return '<div class="reco-empty">Nothing currently clears the recommendation bar (min EV% ' + f"{MIN_EV_PERCENT_TO_RECOMMEND:g}%).</div>"
    parts = []
    if strong:
        parts.append(
            f'<div class="reco-group"><div class="reco-group-head"><h3>Strong ({len(strong)})</h3>'
            '<span class="hint">Model + market cross-book consensus agree</span></div>'
            f'<div class="reco-list">{"".join(_reco_row(r) for r in strong)}</div></div>'
        )
    if speculative:
        parts.append(
            f'<div class="reco-group"><div class="reco-group-head"><h3>Speculative ({len(speculative)})</h3>'
            '<span class="hint">Model only - no independent market confirmation</span></div>'
            f'<div class="reco-list">{"".join(_reco_row(r) for r in speculative)}</div></div>'
        )
    return "".join(parts)


def render_html_report(report: WeekReport, top: int = 60, is_mock: bool = False) -> str:
    strong, speculative = build_recommended_bets(report.candidates)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status_class = "sample" if is_mock else "live"
    status_text = "SAMPLE DATA" if is_mock else "LIVE DATA"
    status_detail = "Synthetic --mock output for demonstration." if is_mock else "Real CFBD ratings + The Odds API cross-book prices."

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NCAAF Model - Week {report.week}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;700;800&family=Big+Shoulders+Display:wght@700;800&family=IBM+Plex+Mono:wght@500;600;700&display=swap" rel="stylesheet">
<style>{_STYLE}</style></head>
<body><div class="wrap">
{nav_html("board")}
<header class="top">
  <div class="brand"><span class="mark">NCAAF</span>
    <div><h1 class="title">The Sharpest Model</h1>
      <div class="subtitle">Blended SP+ + Elo power ratings, situational context, and real cross-book odds - ranked +EV spread/moneyline/total picks for Week {report.week}.</div>
    </div>
  </div>
  <div class="meta"><div class="date">{_esc(report.season)} &middot; Week {report.week} ({_esc(report.season_type)})</div><div>Generated {generated_at}</div></div>
</header>
<div class="status-bug {status_class}"><span class="dot"></span>{status_text}<span class="detail">{status_detail}</span></div>

<section class="section">
  <div class="tiles">
    <div class="tile"><div class="label">Games on slate</div><div class="value">{len(report.slate)}</div></div>
    <div class="tile"><div class="label">Games scored</div><div class="value">{len(report.scores)}</div></div>
    <div class="tile"><div class="label">Strong bets</div><div class="value">{len(strong)}</div></div>
    <div class="tile"><div class="label">Speculative bets</div><div class="value">{len(speculative)}</div></div>
  </div>
</section>

<section class="section">
  <div class="section-head"><h2>Recommended Bets</h2><span class="hint">Fractional-Kelly sized, 1 unit = 1% bankroll</span></div>
  {_reco_section(strong, speculative)}
  <div class="reco-disclosure"><b>Not financial advice.</b> Sizing uses conservative fractional Kelly against a
  transparent, hand-built statistical model - not a trained/calibrated one. Always confirm the live price at your
  own sportsbook before betting; every price here is a snapshot from generation time.</div>
</section>

<details class="section" open>
  <summary class="collapse-head"><h2>Predicted Scores</h2><span class="hint">All {len(report.scores)} scored games</span><span class="details-arrow">&#9662;</span></summary>
  {_predicted_scores_table(report)}
</details>

<details class="section" open>
  <summary class="collapse-head"><h2>Ranked +EV Picks</h2><span class="hint">Top {min(top, len(report.candidates))} of {len(report.candidates)}</span><span class="details-arrow">&#9662;</span></summary>
  {_picks_table(report.candidates[:top])}
</details>

<details class="section">
  <summary class="collapse-head"><h2>Methodology</h2><span class="hint">How this model works</span><span class="details-arrow">&#9662;</span></summary>
  <div class="method-grid">
    <div class="method-card"><h3>Power rating blend</h3>
      <div class="weight-row"><span class="wname">SP+ (opponent-adjusted)</span><div class="weight-bar"><i style="width:{W_SP_OVERALL*100:.0f}%"></i></div><span class="wval">{W_SP_OVERALL:.0%}</span></div>
      <div class="weight-row"><span class="wname">Elo (results + MOV)</span><div class="weight-bar"><i style="width:{W_ELO*100:.0f}%"></i></div><span class="wval">{W_ELO:.0%}</span></div>
      <p style="font-size:12.5px;color:var(--ink-muted);margin-top:10px;">Each z-scored across the league then rescaled to points-above-average (std &asymp; {POWER_RATING_POINTS_STD:g} pts overall, {OFFENSE_POINTS_STD:g} pts offense/defense). Margin/win-probability come from the blended overall rating; totals come from the separate offense/defense split (SP+ + PPA) - two independent estimates kept as a real, visible cross-check rather than silently reconciled.</p>
    </div>
    <div class="method-card"><h3>Situational context</h3>
      <p style="font-size:13px;">Home-field advantage (~2.3 pts, 0 at a neutral site), a real altitude bonus at high-elevation venues (Wyoming, Air Force, Colorado, etc.), a rest-days edge, the away team's real travel distance, and live wind/precipitation at kickoff (totals only, via Open-Meteo).</p>
    </div>
  </div>
  <div class="sources">
    <span class="source-chip">College Football Data (SP+, PPA, schedule, venues)</span>
    <span class="source-chip">This project's own Elo (elo.py)</span>
    <span class="source-chip">The Odds API (live cross-book odds)</span>
    <span class="source-chip">Open-Meteo (kickoff weather)</span>
  </div>
</details>

<footer>
  <p><b>EV% means model vs. market, not "the market is wrong."</b> A positive EV% means our model disagrees with the
  market's own no-vig fair price in the bettor's favor - it is not proof of a real mispricing. The market could just
  as easily be right and the model wrong. This model uses a continuous-normal approximation for cover/win/total
  probabilities, which does not separately model real "key number" push risk (especially around 3 and 7 points).</p>
  <p>Model probabilities are a transparent, hand-built statistical estimate, not a trained/calibrated one - see
  ncaaf/scoring.py and ncaaf/ratings.py in the repository for exact weights and every documented caveat.</p>
</footer>
</div></body></html>"""
