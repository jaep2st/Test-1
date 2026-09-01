"""Shared look for every page this project publishes - the Longball Board
report (`html_report.py`) and the Performance dashboard
(`performance_report.py`). Kept in one place so the two pages read as one
product (same colors, same type, same tiles/table conventions) instead of
visually drifting apart, and so a design tweak only needs to happen once.
"""

from __future__ import annotations

STYLE = """
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
table.props{ border-collapse:collapse; width:100%; min-width:1220px; font-size:13.5px; }
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

/* --- Shared nav (both pages) --- */
.site-nav{ display:flex; gap:6px; margin-bottom:22px; }
.site-nav a{ text-decoration:none; font-size:13.5px; font-weight:600; color:var(--ink-muted); padding:7px 14px; border-radius:20px; border:1px solid var(--border); background:var(--surface); }
.site-nav a.active{ color:var(--bg); background:var(--ink); border-color:var(--ink); }

/* --- Performance dashboard: filters + sortable tables --- */
.filter-bar{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:14px; }
.filter-bar input[type="search"], .filter-bar select{
  font:inherit; font-size:13px; padding:7px 11px; border-radius:8px; border:1px solid var(--border);
  background:var(--surface); color:var(--ink);
}
.filter-bar input[type="search"]{ min-width:200px; }
.filter-count{ font-size:12.5px; color:var(--ink-muted); margin-left:auto; }
table.sortable thead th{ cursor:pointer; user-select:none; }
table.sortable thead th:hover{ color:var(--ink); }
table.sortable thead th .arrow{ margin-left:3px; opacity:.5; }
table.sortable thead th.sorted .arrow{ opacity:1; color:var(--accent); }
tr.hidden-row{ display:none; }

/* --- Calibration chart --- */
.calib-wrap{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:18px 20px 10px; box-shadow:var(--shadow); }
.calib-legend{ display:flex; gap:18px; font-size:12.5px; color:var(--ink-muted); margin-bottom:8px; flex-wrap:wrap; }
.calib-legend .sw{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:5px; vertical-align:-1px; }

.sample-note{ font-size:13px; color:var(--ink-muted); background:var(--surface-2); border:1px solid var(--border); border-radius:8px; padding:10px 14px; margin-bottom:18px; }
.sample-note b{ color:var(--ink); }

/* --- Recommended Bets --- */
.reco-group{ margin-bottom:22px; }
.reco-group:last-child{ margin-bottom:0; }
.reco-group-head{ display:flex; align-items:baseline; gap:10px; margin-bottom:10px; }
.reco-group-head h3{ font-size:16px; font-weight:700; }
.reco-group-head .hint{ font-size:12px; color:var(--ink-muted); }
.reco-list{ background:var(--surface); border:1px solid var(--border); border-radius:10px; box-shadow:var(--shadow); overflow:hidden; }
.reco-row{ display:grid; grid-template-columns:1.6fr 1fr .8fr .9fr .7fr 70px; align-items:center; gap:14px; padding:13px 18px; border-bottom:1px solid var(--border); }
.reco-row:last-child{ border-bottom:none; }
.reco-row .who{ font-weight:700; font-size:14.5px; }
.reco-row .bet{ font-size:12.5px; color:var(--ink-muted); margin-top:2px; }
.reco-row .event{ font-size:12.5px; color:var(--ink-muted); }
.reco-row .price{ font-size:13.5px; }
.reco-row .price b{ font-weight:700; }
.reco-row .breakeven{ font-size:10.5px; color:var(--ink-muted); margin-top:1px; }
.reco-row .prob{ font-size:13px; }
.reco-row .edge{ font-size:13px; }
.reco-units{ display:flex; flex-direction:column; align-items:flex-end; }
.reco-units .n{ font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace; font-weight:700; font-size:17px; color:var(--accent-strong); }
.reco-units .lbl{ font-size:10px; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-muted); }
.reco-empty{ padding:16px 18px; color:var(--ink-muted); font-size:13.5px; background:var(--surface); border:1px solid var(--border); border-radius:10px; }
.reco-disclosure{ font-size:12.5px; color:var(--ink-muted); background:var(--surface-2); border:1px solid var(--border); border-radius:8px; padding:12px 16px; margin-top:16px; }
.reco-disclosure b{ color:var(--ink); }
@media (max-width:760px){
  .reco-row{ grid-template-columns:1fr 1fr; grid-template-areas:"who units" "bet bet" "event event" "price prob"; row-gap:6px; }
  .reco-row .who{ grid-area:who; }
  .reco-units{ grid-area:units; align-items:flex-end; }
  .reco-row .bet{ grid-area:bet; }
  .reco-row .event{ grid-area:event; }
  .reco-row .price{ grid-area:price; }
  .reco-row .prob{ grid-area:prob; }
  .reco-row .edge{ display:none; }
}
"""


def nav_html(active: str) -> str:
    """`active`: "board" or "performance" - marks the current page's tab.
    Both published pages (index.html, performance.html) live in the same
    `public/` directory, so plain relative filenames link between them.
    """
    board_class = "active" if active == "board" else ""
    perf_class = "active" if active == "performance" else ""
    return (
        '<nav class="site-nav">'
        f'<a class="{board_class}" href="index.html">Today’s Board</a>'
        f'<a class="{perf_class}" href="performance.html">Performance</a>'
        "</nav>"
    )
