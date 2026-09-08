"""Renders the real Performance dashboard - calibration, closing-line
value, hit rate by market/tier, and a real units ledger - from everything
recorded under `<data_dir>` (see `results.py`/`backtest.py`). The `ncaaf`
counterpart to `mlb_props/performance_report.py`.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

from .backtest import (
    calibration_buckets,
    clv_summary,
    hit_rate_by_market,
    hit_rate_by_tier,
    load_all_clv,
    load_all_picks,
    load_all_results,
    resolve_picks,
    units_by_week,
    units_ledger,
    units_summary,
)
from .site_style import STYLE as _STYLE
from .site_style import nav_html

REFIT_READY_DAYS = 40  # picks - see refit.py's MIN_PICKS_TO_FIT, same bar


def _esc(s: object) -> str:
    return html.escape(str(s))


def _calibration_svg(buckets) -> str:
    w, h, pad = 640, 260, 36
    points_pred = []
    points_actual = []
    for b in buckets:
        x = pad + (b.lo + (b.hi - b.lo) / 2) * (w - 2 * pad)
        if b.predicted_mean is not None:
            points_pred.append((x, h - pad - b.predicted_mean * (h - 2 * pad)))
        if b.actual_rate is not None:
            points_actual.append((x, h - pad - b.actual_rate * (h - 2 * pad)))
    diag = f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{pad}" stroke="var(--border)" stroke-dasharray="4,4"/>'

    def polyline(points, color):
        if len(points) < 2:
            return "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>' for x, y in points)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        circles = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>' for x, y in points)
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>{circles}'

    return f"""<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" aria-label="Calibration chart">
      <line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="var(--ink-muted)"/>
      <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="var(--ink-muted)"/>
      {diag}
      {polyline(points_actual, 'var(--accent)')}
      {polyline(points_pred, 'var(--info)')}
    </svg>"""


def _sortable_table(headers, rows, id_prefix: str) -> str:
    head = "".join(f'<th data-k="{i}">{_esc(h)}<span class="arrow">&#9650;</span></th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="table-scroll"><table class="props sortable" id="{id_prefix}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_performance_report(data_dir: str) -> str:
    picks = load_all_picks(data_dir)
    results = load_all_results(data_dir)
    clv_records = load_all_clv(data_dir)
    resolved = resolve_picks(picks, results)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_resolved = len(resolved)
    n_picks = len(picks)

    buckets = calibration_buckets(resolved)
    clv = clv_summary(clv_records)
    by_market = hit_rate_by_market(resolved)
    by_tier = hit_rate_by_tier(resolved)
    ledger = units_ledger(resolved)
    summary = units_summary(ledger)
    weekly = units_by_week(ledger)

    sample_note = ""
    if n_resolved < REFIT_READY_DAYS:
        sample_note = (
            f'<div class="sample-note">Only <b>{n_resolved}</b> real resolved pick(s) so far - every stat below '
            f"is real, but small samples are noisy. Weight refitting (see ncaaf/refit.py) needs at least "
            f"<b>{REFIT_READY_DAYS}</b> to produce a reliable result.</div>"
        )

    market_rows = [[g.key, g.n, f"{g.hit_rate:.1%}"] for g in by_market]
    tier_rows = [[g.key, g.n, f"{g.hit_rate:.1%}"] for g in by_tier]
    weekly_rows = [[f"{w.season} wk{w.week}", f"{w.net_units:+.2f}", f"{w.cumulative_units:+.2f}"] for w in weekly]

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NCAAF Model - Performance</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;700;800&family=Big+Shoulders+Display:wght@700;800&family=IBM+Plex+Mono:wght@500;600;700&display=swap" rel="stylesheet">
<style>{_STYLE}</style></head>
<body><div class="wrap">
{nav_html("performance")}
<header class="top">
  <div class="brand"><span class="mark">NCAAF</span><div><h1 class="title">Performance</h1>
  <div class="subtitle">Real recorded picks vs. real resolved outcomes - calibration, closing-line value, hit rate, and a real units ledger. Nothing here is a backtest against synthetic data.</div></div></div>
  <div class="meta"><div class="date">{n_picks} picks recorded</div><div>Generated {generated_at}</div></div>
</header>
{sample_note}

<section class="section">
  <div class="tiles">
    <div class="tile"><div class="label">Resolved picks</div><div class="value">{n_resolved}</div></div>
    <div class="tile"><div class="label">Mean CLV</div><div class="value">{f'{clv.mean_clv_percent:+.2f}%' if clv.mean_clv_percent is not None else 'n/a'}</div><div class="sub">n={clv.n}</div></div>
    <div class="tile"><div class="label">Beat closing line</div><div class="value">{f'{clv.beat_close_percent:.0f}%' if clv.beat_close_percent is not None else 'n/a'}</div></div>
    <div class="tile"><div class="label">Net units</div><div class="value">{summary.net_units:+.2f}</div><div class="sub">{summary.n_bets} bets, ROI {f'{summary.roi_percent:+.1f}%' if summary.roi_percent is not None else 'n/a'}</div></div>
  </div>
</section>

<section class="section">
  <div class="section-head"><h2>Calibration</h2><span class="hint">Predicted probability vs. real outcome rate, by decile</span></div>
  <div class="calib-wrap">
    <div class="calib-legend"><span><i class="sw" style="background:var(--info)"></i>Predicted</span><span><i class="sw" style="background:var(--accent)"></i>Actual</span></div>
    {_calibration_svg(buckets)}
  </div>
</section>

<section class="section">
  <div class="section-head"><h2>Hit Rate</h2></div>
  <div class="method-grid">
    <div class="method-card"><h3>By market</h3>{_sortable_table(["Market", "N", "Hit rate"], market_rows, "byMarket")}</div>
    <div class="method-card"><h3>By tier</h3>{_sortable_table(["Tier", "N", "Hit rate"], tier_rows, "byTier")}</div>
  </div>
</section>

<section class="section">
  <div class="section-head"><h2>Units by Week</h2><span class="hint">Real, would-have-bet outcomes at recommended sizing</span></div>
  {_sortable_table(["Week", "Net units", "Cumulative"], weekly_rows, "byWeek") if weekly_rows else '<div class="empty">No resolved recommended bets yet.</div>'}
</section>

<footer>
  <p>Every number on this page is computed from real recorded picks (`data/ncaaf/picks/`) joined against real
  resolved outcomes (`data/ncaaf/results/`) and real closing-line snapshots (`data/ncaaf/clv/`) - never a
  synthetic backtest. See ncaaf/results.py and ncaaf/backtest.py for exactly how each stat is computed, and
  ncaaf/refit.py for the (proposal-only) weight-refitting mechanism this data feeds.</p>
</footer>
</div>
<script>
document.querySelectorAll('table.sortable').forEach(function(table){{
  table.querySelectorAll('th').forEach(function(th, idx){{
    th.addEventListener('click', function(){{
      var tbody = table.querySelector('tbody');
      var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
      var asc = !th.classList.contains('sorted') || th.dataset.dir === 'desc';
      table.querySelectorAll('th').forEach(function(h){{ h.classList.remove('sorted'); }});
      th.classList.add('sorted'); th.dataset.dir = asc ? 'asc' : 'desc';
      rows.sort(function(a, b){{
        var av = a.children[idx].textContent, bv = b.children[idx].textContent;
        var an = parseFloat(av.replace(/[^0-9.+-]/g, '')), bn = parseFloat(bv.replace(/[^0-9.+-]/g, ''));
        var cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : av.localeCompare(bv);
        return asc ? cmp : -cmp;
      }});
      rows.forEach(function(r){{ tbody.appendChild(r); }});
    }});
  }});
}});
</script>
</body></html>"""
