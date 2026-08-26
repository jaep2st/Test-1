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
from typing import List, Optional

from .edges import EdgeCandidate
from .pipeline import MatchupEnvironment, SlateReport
from .scoring import HR_WEIGHTS, TB_WEIGHTS

_STYLE = """
:root{
  --bg:#F6F8F2; --surface:#FFFFFF; --surface-2:#ECF1E5; --border:#D6DECB;
  --ink:#172319; --ink-muted:#57614F; --accent:#B36D0C; --accent-strong:#8A5308;
  --accent-tint:#F3E3C6; --positive:#1C7A4C; --positive-tint:#DFF1E7;
  --negative:#B23A2E; --negative-tint:#F6E1DE; --info:#3E6E96;
  --shadow: 0 1px 2px rgba(23,35,25,0.06), 0 8px 24px -12px rgba(23,35,25,0.18);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0A130E; --surface:#101C15; --surface-2:#16241B; --border:#223328;
    --ink:#E9F1E6; --ink-muted:#9DB0A2; --accent:#FFC155; --accent-strong:#FFD98A;
    --accent-tint:#3A2C0F; --positive:#6FE0A0; --positive-tint:#123322;
    --negative:#FF8A7A; --negative-tint:#3A1913; --info:#8FC3EA;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 30px -14px rgba(0,0,0,0.6);
  }
}
:root[data-theme="dark"]{
  --bg:#0A130E; --surface:#101C15; --surface-2:#16241B; --border:#223328;
  --ink:#E9F1E6; --ink-muted:#9DB0A2; --accent:#FFC155; --accent-strong:#FFD98A;
  --accent-tint:#3A2C0F; --positive:#6FE0A0; --positive-tint:#123322;
  --negative:#FF8A7A; --negative-tint:#3A1913; --info:#8FC3EA;
  --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 30px -14px rgba(0,0,0,0.6);
}
*{ box-sizing:border-box; }
body{ margin:0; background:var(--bg); color:var(--ink); font-family:"Public Sans",-apple-system,"Segoe UI",sans-serif; line-height:1.45; -webkit-font-smoothing:antialiased; }
h1,h2,h3{ font-family:"Big Shoulders Display","Arial Narrow",sans-serif; text-wrap:balance; margin:0; }
.num{ font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums; }
a{ color:var(--info); }
.wrap{ max-width:1180px; margin:0 auto; padding:28px 20px 80px; }
header.top{ display:flex; align-items:flex-end; justify-content:space-between; gap:20px; flex-wrap:wrap; padding-bottom:18px; margin-bottom:16px; border-bottom:3px solid var(--ink); }
.brand{ display:flex; align-items:baseline; gap:12px; }
.brand .mark{ font-family:"Big Shoulders Display",sans-serif; font-weight:800; font-size:15px; letter-spacing:.14em; color:var(--accent); border:2px solid var(--accent); border-radius:4px; padding:3px 7px; line-height:1; }
h1.title{ font-size:clamp(30px,4.6vw,44px); font-weight:800; letter-spacing:.01em; }
.subtitle{ color:var(--ink-muted); font-size:14.5px; margin-top:4px; max-width:56ch; }
.meta{ text-align:right; font-size:13px; color:var(--ink-muted); }
.meta .date{ font-family:"IBM Plex Mono",monospace; font-size:15px; color:var(--ink); font-weight:600; }
.status-bug{ display:flex; align-items:center; gap:10px; border-radius:8px; padding:10px 14px; font-size:13.5px; font-weight:600; margin-bottom:26px; border:1.5px solid; }
.status-bug.live{ background:var(--positive-tint); border-color:var(--positive); color:var(--positive); }
.status-bug.sample{ background:var(--negative-tint); border-color:var(--negative); color:var(--negative); }
.status-bug .dot{ width:8px; height:8px; border-radius:50%; background:currentColor; flex:none; }
.status-bug span.detail{ color:var(--ink-muted); font-weight:400; }
.section{ margin-top:44px; }
.section-head{ display:flex; align-items:baseline; justify-content:space-between; gap:16px; margin-bottom:14px; flex-wrap:wrap; }
.section-head h2{ font-size:24px; font-weight:700; letter-spacing:.01em; }
.section-head .hint{ font-size:12.5px; color:var(--ink-muted); text-transform:uppercase; letter-spacing:.08em; }
.eyebrow{ font-family:"Big Shoulders Display",sans-serif; font-weight:700; font-size:12.5px; letter-spacing:.16em; text-transform:uppercase; color:var(--accent); margin-bottom:6px; display:block; }
.tiles{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }
@media (max-width:820px){ .tiles{ grid-template-columns:repeat(2,1fr); } }
.tile{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 16px 14px; box-shadow:var(--shadow); }
.tile .label{ font-size:11.5px; text-transform:uppercase; letter-spacing:.09em; color:var(--ink-muted); font-weight:600; }
.tile .value{ font-size:24px; font-weight:600; margin-top:6px; }
.tile .sub{ font-size:13px; color:var(--ink-muted); margin-top:4px; }
.tile .sub b{ color:var(--ink); font-weight:600; }
.env-grid{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }
@media (max-width:760px){ .env-grid{ grid-template-columns:1fr; } }
.env-card{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 18px; box-shadow:var(--shadow); }
.env-card .rank{ font-family:"Big Shoulders Display",sans-serif; font-weight:800; font-size:13px; color:var(--accent); }
.env-card .matchup{ font-size:18px; font-weight:600; margin-top:2px; }
.env-card .pitchers{ font-size:13px; color:var(--ink-muted); margin-top:2px; }
.env-card .park{ font-size:12.5px; color:var(--ink-muted); margin-top:8px; }
.env-bar-row{ display:flex; align-items:center; gap:10px; margin-top:10px; }
.env-bar{ flex:1; height:8px; border-radius:5px; background:var(--surface-2); overflow:hidden; }
.env-bar > i{ display:block; height:100%; background:linear-gradient(90deg,var(--accent-strong),var(--accent)); border-radius:5px; }
.env-score{ font-size:14px; font-weight:600; width:38px; text-align:right; }
.env-chips{ display:flex; gap:8px; margin-top:10px; flex-wrap:wrap; }
.chip{ display:inline-flex; align-items:center; gap:5px; font-size:12px; font-weight:600; border-radius:20px; padding:4px 10px; border:1px solid var(--border); background:var(--surface-2); color:var(--ink); }
.chip.wind-out{ color:var(--positive); border-color:var(--positive); background:var(--positive-tint); }
.chip.wind-in{ color:var(--negative); border-color:var(--negative); background:var(--negative-tint); }
.hot-list{ background:var(--surface); border:1px solid var(--border); border-radius:10px; box-shadow:var(--shadow); overflow:hidden; }
.hot-row{ display:grid; grid-template-columns:26px 1.4fr 90px 1fr 60px; align-items:center; gap:14px; padding:11px 18px; border-bottom:1px solid var(--border); }
.hot-row:last-child{ border-bottom:none; }
.hot-row .rank{ font-family:"IBM Plex Mono",monospace; color:var(--ink-muted); font-size:13px; }
.hot-row .name{ font-weight:600; font-size:14.5px; }
.label-pill{ font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; border-radius:20px; padding:3px 9px; text-align:center; }
.label-scorching, .label-ice-cold{ background:var(--negative-tint); color:var(--negative); }
.label-hot{ background:var(--accent-tint); color:var(--accent-strong); }
.label-steady, .label-cold{ background:var(--surface-2); color:var(--ink-muted); }
.heat-bar{ height:7px; border-radius:5px; background:var(--surface-2); overflow:hidden; position:relative; }
.heat-bar > i{ position:absolute; top:0; bottom:0; background:var(--accent); border-radius:5px; }
.hot-row .z{ text-align:right; font-size:13.5px; font-weight:600; }
.table-scroll{ overflow-x:auto; border:1px solid var(--border); border-radius:10px; box-shadow:var(--shadow); background:var(--surface); }
table.props{ border-collapse:collapse; width:100%; min-width:1080px; font-size:13.5px; }
table.props thead th{ position:sticky; top:0; background:var(--surface-2); text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-muted); padding:10px 12px; border-bottom:1px solid var(--border); font-weight:700; white-space:nowrap; }
table.props tbody td{ padding:10px 12px; border-bottom:1px solid var(--border); white-space:nowrap; }
table.props tbody tr:last-child td{ border-bottom:none; }
table.props tbody tr:hover{ background:var(--surface-2); }
table.props td.player{ font-weight:600; white-space:normal; }
table.props td.event{ color:var(--ink-muted); font-size:12.5px; white-space:normal; }
.tier{ display:inline-block; font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; border-radius:5px; padding:2px 6px; margin-top:3px; }
.tier.agree{ background:var(--positive-tint); color:var(--positive); }
.tier.model{ background:var(--surface-2); color:var(--ink-muted); }
.pos{ color:var(--positive); }
.neg{ color:var(--negative); }
.book{ text-transform:capitalize; color:var(--ink-muted); font-size:12.5px; }
.wx-cell{ font-size:12.5px; color:var(--ink-muted); }
.wx-cell b{ color:var(--ink); font-weight:600; }
.method-grid{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }
@media (max-width:760px){ .method-grid{ grid-template-columns:1fr; } }
.method-card{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:18px 20px; box-shadow:var(--shadow); }
.method-card h3{ font-size:16px; font-weight:700; margin-bottom:10px; }
.weight-row{ display:flex; align-items:center; gap:10px; font-size:13px; padding:4px 0; }
.weight-row .wname{ width:150px; color:var(--ink-muted); flex:none; }
.weight-bar{ flex:1; height:6px; border-radius:4px; background:var(--surface-2); overflow:hidden; }
.weight-bar > i{ display:block; height:100%; background:var(--accent); }
.weight-row .wval{ width:34px; text-align:right; font-weight:600; }
.sources{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
.source-chip{ font-size:12px; border:1px solid var(--border); background:var(--surface-2); color:var(--ink-muted); border-radius:20px; padding:4px 10px; }
footer{ margin-top:50px; padding-top:20px; border-top:1px solid var(--border); font-size:12.5px; color:var(--ink-muted); }
footer p{ max-width:80ch; margin:0 0 8px; }
.empty{ color:var(--ink-muted); font-size:14px; padding:18px; }
"""


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


def _prop_row(e: EdgeCandidate) -> str:
    if not e.has_market_data:
        return f"""
          <tr><td class="player">{_esc(e.player)}<div class="tier model">Model only &mdash; no market price</div></td>
            <td class="event">{_esc(e.event)}</td><td class="num">{_fmt_pct(e.model_prob)}</td>
            <td colspan="7" class="wx-cell">no book currently quotes this prop</td></tr>"""
    both_agree = e.ev_percent_model is not None and e.ev_percent_model > 0 and e.edge_vs_market is not None and e.edge_vs_market > 0
    if both_agree:
        tier = '<div class="tier agree">Model + market agree</div>'
    elif e.market_fair_prob is None:
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
          <tr>
            <td class="player">{_esc(e.player)}{tier}</td>
            <td class="event">{_esc(e.event)}</td>
            <td class="num">{_fmt_pct(e.model_prob)}</td>
            <td class="num pos">{e.best_line.odds:+d}</td>
            <td class="book">{_esc(e.best_line.sportsbook)}</td>
            <td class="num">{_fmt_opt_pct(e.market_fair_prob)}</td>
            <td class="num {'pos' if e.edge_vs_market is not None and e.edge_vs_market >= 0 else 'neg' if e.edge_vs_market is not None else ''}">{edge_cell}</td>
            <td class="num {'pos' if e.ev_percent_model >= 0 else 'neg'}">{e.ev_percent_model:+.1f}%</td>
            <td class="num {'pos' if e.ev_percent_market is not None and e.ev_percent_market >= 0 else 'neg' if e.ev_percent_market is not None else ''}">{ev_market_cell}</td>
            <td class="num">{e.books_quoting}</td>
            <td class="wx-cell">{wind}, {temp} <b>{e.weather_boost_pct:+.1f}%</b></td>
          </tr>"""


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


def _prop_table(title: str, hint: str, edges: List[EdgeCandidate], prob_header: str, top: int) -> str:
    if not edges:
        return f"""
    <div class="section-head"><h2>{_esc(title)}</h2><span class="hint">{_esc(hint)}</span></div>
    <div class="empty">No candidates scored for this slate.</div>"""
    rows = "".join(_prop_row(e) for e in edges[:top])
    return f"""
    <div class="section-head"><h2>{_esc(title)}</h2><span class="hint">{_esc(hint)}</span></div>
    <div class="table-scroll">
      <table class="props">
        <thead><tr>
          <th>Player</th><th>Matchup</th><th>{_esc(prob_header)}</th><th>Best price</th><th>Book</th>
          <th>Market fair</th><th>Edge</th><th>EV (model)</th><th>EV (market)</th><th>Books</th><th>Weather</th>
        </tr></thead>
        <tbody>{rows}
        </tbody>
      </table>
    </div>"""


def render_html_report(report: SlateReport, top: int = 15, is_mock: bool = False, generated_at: Optional[datetime] = None) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    envs = report.matchup_environments
    hot = report.hot_batters
    hr = report.hr_edges
    tb = report.tb_edges

    top_hr = next((e for e in hr if e.has_market_data), hr[0] if hr else None)
    top_tb = next((e for e in tb if e.has_market_data), tb[0] if tb else None)
    best_env = envs[0] if envs else None
    hottest = hot[0] if hot else None

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

  <section class="section" style="margin-top:0;">
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
    {_prop_table("Best home run props", 'Ranked by our model’s EV% against the best live price - "agree" = model & market both see value', hr, "Model P(HR)", top)}
  </section>

  <section class="section">
    {_prop_table("Best 2+ total bases props", "Ranked by our model's EV% against the best live price", tb, "Model P(2+ TB)", top)}
  </section>

  <section class="section">
    <div class="section-head">
      <h2>How the score is built</h2>
      <span class="hint">mlb_props/scoring.py &mdash; every weight below, verbatim</span>
    </div>
    <div class="method-grid">
      <div class="method-card"><h3>Home run score (weights)</h3>{_weight_rows(HR_WEIGHTS)}</div>
      <div class="method-card"><h3>2+ total bases score (weights)</h3>{_weight_rows(TB_WEIGHTS)}</div>
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
  </footer>
</div>
</body>
</html>
"""
