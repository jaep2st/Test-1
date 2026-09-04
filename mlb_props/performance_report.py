"""Renders `public/performance.html` - the "is this model actually
working" dashboard, computed entirely from `mlb_props/backtest.py`'s real
recorded picks/results/CLV (see `mlb_props/results.py`). Same visual system
as `html_report.py` (shared `site_style.py`) so the two pages read as one
product; linked together by the top nav both pages render.

Every number here carries its own sample size next to it - this project's
`data/` history starts empty the day this ships, so small-N is the honest
normal state for a while, not hidden.
"""

from __future__ import annotations

import html
import os
from datetime import datetime, timezone
from typing import List, Optional

from .backtest import (
    CalibrationBucket,
    HitRateGroup,
    calibration_buckets,
    clv_summary,
    hit_rate_by_lineup_source,
    hit_rate_by_market,
    hit_rate_by_run_hour,
    hit_rate_by_tier,
    load_all_clv,
    load_all_picks,
    load_all_results,
    recorded_at_et,
    resolve_picks,
)
from .historical_backtest import HistoricalBacktestRun, load_historical_backtest_runs
from .market import book_display_name
from .refit import MIN_PICKS_TO_FIT, BlendResult, RefitResult, fit_all_market_blends, refit_all_markets
from .site_style import STYLE, nav_html

# Real days of results needed before a weight refit (see scoring.py) would
# be fitting against something more than noise - roughly a full slate of
# games every day for a few weeks. Purely a UI signpost, not a hard gate
# enforced anywhere in code.
REFIT_READY_DAYS = 21


def _esc(s: object) -> str:
    return html.escape(str(s))


def _fmt_pct(x: Optional[float], digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%" if x is not None else "n/a"


def _fmt_signed_pct(x: Optional[float], digits: int = 1) -> str:
    return f"{x:+.{digits}f}%" if x is not None else "n/a"


_MARKET_LABELS = {
    "batter_home_runs": "1+ HR",
    "batter_total_bases": "2+ Total Bases",
    "batter_hits": "1+ Hits",
}
_TIER_LABELS = {
    "agree": "Model + market agree",
    "model_only": "Model only",
    "model_only_single_sided": "Model only (single-sided market)",
    "no_market": "No market price",
}
_LINEUP_SOURCE_LABELS = {
    "confirmed": "Lineup confirmed",
    "active_roster": "Active roster (lineup not posted yet)",
}


def _tile(label: str, value: str, sub: str) -> str:
    return f'<div class="tile"><div class="label">{_esc(label)}</div><div class="value">{_esc(value)}</div><div class="sub">{sub}</div></div>'


def _calibration_svg(buckets: List[CalibrationBucket]) -> str:
    """A simple inline-SVG reliability chart: bars for real observed hit
    rate per decile, a diagonal reference line for "perfectly calibrated",
    and a dot for what the model predicted in that decile. No chart
    library - consistent with the rest of this zero-dependency site.
    """
    w, h = 640, 260
    pad_l, pad_b, pad_t, pad_r = 46, 34, 14, 14
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
    n = len(buckets) or 1
    bar_w = plot_w / n

    def x_of(i: float) -> float:
        return pad_l + i * plot_w

    def y_of(rate: float) -> float:
        return pad_t + (1.0 - rate) * plot_h

    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Calibration chart: predicted probability vs. real observed outcome rate">'
    ]
    # Axes
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h - pad_b}" stroke="var(--border)" />'
        f'<line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}" stroke="var(--border)" />'
    )
    # Perfect-calibration reference diagonal
    parts.append(
        f'<line x1="{x_of(0):.1f}" y1="{y_of(0):.1f}" x2="{x_of(1):.1f}" y2="{y_of(1):.1f}" '
        'stroke="var(--ink-muted)" stroke-dasharray="4 4" stroke-width="1.5" />'
    )
    for i, b in enumerate(buckets):
        cx = x_of((i + 0.5) / n)
        if b.n > 0:
            bar_h = max(2.0, b.actual_rate * plot_h)
            bar_x = x_of(i / n) + bar_w * 0.2
            parts.append(
                f'<rect x="{bar_x:.1f}" y="{(h - pad_b - bar_h):.1f}" width="{bar_w * 0.6:.1f}" height="{bar_h:.1f}" '
                f'rx="2" fill="var(--accent)" fill-opacity="0.75"><title>{b.lo:.0%}-{b.hi:.0%} predicted: '
                f'{b.actual_rate:.1%} actual over {b.n} picks</title></rect>'
            )
            py = y_of(b.predicted_mean)
            parts.append(f'<circle cx="{cx:.1f}" cy="{py:.1f}" r="4" fill="var(--info)"><title>predicted mean {b.predicted_mean:.1%}</title></circle>')
        parts.append(
            f'<text x="{cx:.1f}" y="{h - pad_b + 16}" font-size="10.5" text-anchor="middle" fill="var(--ink-muted)">{b.lo:.0%}</text>'
        )
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<text x="{pad_l - 6}" y="{y_of(frac) + 3:.1f}" font-size="10.5" text-anchor="end" fill="var(--ink-muted)">{frac:.0%}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _hit_rate_table(title: str, groups: List[HitRateGroup], label_map: dict) -> str:
    if not groups:
        return f'<div class="method-card"><h3>{_esc(title)}</h3><div class="empty">No resolved picks yet.</div></div>'
    rows = "".join(
        f"<tr><td>{_esc(label_map.get(g.key, g.key))}</td><td class=\"num\">{g.hit_rate:.1%}</td><td class=\"num\">{g.n}</td></tr>"
        for g in groups
    )
    return f"""
    <div class="method-card">
      <h3>{_esc(title)}</h3>
      <table class="props" style="min-width:0;">
        <thead><tr><th>Group</th><th>Real hit rate</th><th>N</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


def _refit_card(r: RefitResult, distinct_days: int) -> str:
    label = _MARKET_LABELS.get(r.market, r.market)
    notes = []
    if not r.reliable:
        notes.append(
            f'<div class="empty">Only {r.n_train} real training row{"s" if r.n_train != 1 else ""} so far '
            f"(need &ge;{MIN_PICKS_TO_FIT}) - shown for transparency, not yet actionable.</div>"
        )
    if distinct_days < REFIT_READY_DAYS:
        notes.append(
            f'<div class="empty">Also under {REFIT_READY_DAYS} real distinct days ({distinct_days} so far) - '
            "even a numerically-reliable fit this early can still be overfit to a handful of days' variance.</div>"
        )
    if r.fitted_test_log_loss is None or r.current_test_log_loss is None:
        verdict, verdict_cls = "Not enough held-out picks yet for a real comparison.", ""
    elif r.improves_on_current:
        verdict = (
            f"Fitted weights measurably beat the current hand-set ones on real held-out data "
            f"({r.fitted_test_log_loss:.3f} vs {r.current_test_log_loss:.3f} log-loss, lower is better)."
        )
        verdict_cls = "pos"
    else:
        verdict = (
            f"Not yet distinguishable from the current hand-set model on real held-out data "
            f"({r.fitted_test_log_loss:.3f} vs {r.current_test_log_loss:.3f} log-loss, lower is better)."
        )
        verdict_cls = ""
    comp_rows = "".join(
        f"<tr><td>{_esc(k.replace('_', ' ').title())}</td>"
        f'<td class="num">{r.current_weights.get(k, 0.0):.2f}</td>'
        f'<td class="num">{r.fitted_importance.get(k, 0.0):.2f}</td></tr>'
        for k in r.current_weights
    )
    return f"""
    <div class="method-card">
      <h3>{_esc(label)} <span class="hint">n={r.n_train} train / {r.n_test} test</span></h3>
      {''.join(notes)}
      <div class="{verdict_cls}" style="font-size:13px;margin:6px 0 10px;">{_esc(verdict)}</div>
      <table class="props" style="min-width:0;">
        <thead><tr><th>Component</th><th>Current weight</th><th>Fitted importance</th></tr></thead>
        <tbody>{comp_rows}</tbody>
      </table>
    </div>"""


def _refit_section(refit_results: List[RefitResult], distinct_days: int) -> str:
    if not refit_results:
        body = (
            '<div class="empty">No resolved pick yet carries real component features to fit from - this needs '
            "picks recorded after mlb_props/refit.py shipped (see PickRecord.components's docstring).</div>"
        )
    else:
        body = '<div class="method-grid">' + "".join(_refit_card(r, distinct_days) for r in refit_results) + "</div>"
    return f"""
  <section class="section">
    <div class="section-head">
      <h2>Weight refit check</h2>
      <span class="hint">Real logistic-regression fit vs. the current hand-set weights, on real held-out data</span>
    </div>
    {body}
    <div class="reco-disclosure">
      <b>This is a transparent comparison, never a live behavior change.</b> mlb_props/scoring.py's weights stay
      exactly what they are regardless of what's shown here - a fitted result is real evidence to weigh, not
      something this project applies automatically. "Fitted importance" is each component's normalized share of the
      logistic regression's own coefficients - a genuinely different function from scoring.py's normalize-then-
      calibrate pipeline, so treat it as a relative-importance signal to inform a manual weight change, never as a
      literal drop-in replacement for the current-weight column next to it.
    </div>
  </section>"""


def _fmt_loss(x: Optional[float]) -> str:
    return f"{x:.3f}" if x is not None else "n/a"


def _blend_card(r: BlendResult) -> str:
    label = _MARKET_LABELS.get(r.market, r.market)
    notes = []
    if not r.reliable:
        notes.append(
            f'<div class="empty">Only {r.n_train} real training row{"s" if r.n_train != 1 else ""} so far '
            f"(need &ge;{MIN_PICKS_TO_FIT}) - shown for transparency, not yet actionable.</div>"
        )
    if r.blended_test_log_loss is None or r.model_only_test_log_loss is None:
        verdict, verdict_cls = "Not enough held-out picks yet for a real comparison.", ""
    elif r.improves_on_model_only:
        verdict = (
            "Blending in the market's own fair probability measurably beats pure model_prob (today's live "
            f"behavior) on real held-out data ({r.blended_test_log_loss:.3f} vs {r.model_only_test_log_loss:.3f} "
            "log-loss, lower is better)."
        )
        verdict_cls = "pos"
    else:
        verdict = (
            "Not yet distinguishable from pure model_prob (today's live behavior) on real held-out data "
            f"({r.blended_test_log_loss:.3f} vs {r.model_only_test_log_loss:.3f} log-loss, lower is better)."
        )
        verdict_cls = ""
    return f"""
    <div class="method-card">
      <h3>{_esc(label)} <span class="hint">n={r.n_train} train / {r.n_test} test</span></h3>
      {''.join(notes)}
      <div class="{verdict_cls}" style="font-size:13px;margin:6px 0 10px;">{_esc(verdict)}</div>
      <table class="props" style="min-width:0;">
        <thead><tr><th>Weight on model_prob (&alpha;)</th><th>Blended log-loss</th><th>Pure model log-loss</th><th>Pure market log-loss</th></tr></thead>
        <tbody><tr>
          <td class="num">{r.best_alpha:.1f}</td>
          <td class="num">{_fmt_loss(r.blended_test_log_loss)}</td>
          <td class="num">{_fmt_loss(r.model_only_test_log_loss)}</td>
          <td class="num">{_fmt_loss(r.market_only_test_log_loss)}</td>
        </tr></tbody>
      </table>
    </div>"""


def _blend_section(blend_results: List[BlendResult]) -> str:
    if not blend_results:
        body = (
            '<div class="empty">No resolved pick yet carries a real market_fair_prob to blend against - this '
            "needs a two-sided market price at pick time (see EdgeCandidate's module docstring).</div>"
        )
    else:
        body = '<div class="method-grid">' + "".join(_blend_card(r) for r in blend_results) + "</div>"
    return f"""
  <section class="section">
    <div class="section-head">
      <h2>Market blend check</h2>
      <span class="hint">Real log-odds blend of model_prob and the market's own fair probability, on real held-out data</span>
    </div>
    {body}
    <div class="reco-disclosure">
      <b>Bill Benter's real, documented technique:</b> combine a fundamentals-based model with the market's own
      odds, since the market already encodes real public information the model doesn't see, rather than trusting
      either one alone. This is a transparent comparison, never a live behavior change: bet sizing (betting.py)
      keeps using model_prob exactly as it does today regardless of what's shown here. &alpha; is the fitted
      weight on model_prob (1 &minus; &alpha; on the market's own fair probability) that minimized real held-out
      log-loss; &alpha; near 1 means the market adds little beyond what the model already knows, &alpha; near 0
      means the market is carrying most of the real signal here.
    </div>
  </section>"""


_BUCKET_ORDER = ["Hot (z >= +1.0)", "Neutral (-1.0 < z < +1.0)", "Cold (z <= -1.0)"]
_BUCKET_SHORT_LABELS = {"Hot (z >= +1.0)": "Hot", "Neutral (-1.0 < z < +1.0)": "Neutral", "Cold (z <= -1.0)": "Cold"}


def _historical_backtest_section(runs: List[HistoricalBacktestRun]) -> str:
    """Real hot/cold z-score signal backtest results (see
    historical_backtest.py) - unlike everything else on this page, this
    isn't live-forward pick tracking, it's a direct check against already-
    played real MLB history, run on demand via `--historical-backtest-
    hot-streak` (not part of the regular schedule - see that workflow
    input's own description). Persisted (`record_historical_backtest_run`)
    so a run's real evidence stays visible here instead of only ever
    existing as one run's console text.
    """
    if not runs:
        body = (
            '<div class="empty">No historical backtest run yet - trigger the "historical_backtest_hot_streak" '
            "workflow_dispatch input with a real date range to check whether the hot/cold z-score signal actually "
            "predicted real past MLB games (see mlb_props/historical_backtest.py).</div>"
        )
        return f"""
  <section class="section">
    <div class="section-head">
      <h2>Historical hot-streak signal backtest</h2>
      <span class="hint">Does the real hot/cold z-score signal actually predict already-played MLB history?</span>
    </div>
    {body}
  </section>"""

    latest = max(runs, key=lambda r: r.run_at)
    rows = []
    overall = latest.overall_rates
    rows.append(
        "<tr><td>Overall baseline</td><td class=\"num\">-</td>"
        f'<td class="num">{_fmt_pct(overall.get("hr"))}</td><td class="num">{_fmt_pct(overall.get("tb2"))}</td>'
        f'<td class="num">{_fmt_pct(overall.get("hit"))}</td></tr>'
    )
    for label in _BUCKET_ORDER:
        b = latest.buckets.get(label, {})
        n = int(b.get("n", 0))
        rows.append(
            f"<tr><td>{_esc(_BUCKET_SHORT_LABELS[label])}</td><td class=\"num\">{n}</td>"
            f'<td class="num">{_fmt_pct(b.get("hr")) if n else "n/a"}</td>'
            f'<td class="num">{_fmt_pct(b.get("tb2")) if n else "n/a"}</td>'
            f'<td class="num">{_fmt_pct(b.get("hit")) if n else "n/a"}</td></tr>'
        )
    table = f"""
      <table class="props" style="min-width:0;">
        <thead><tr><th>Bucket</th><th>n</th><th>1+ HR</th><th>2+ TB</th><th>1+ Hit</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>"""

    other_runs_note = ""
    if len(runs) > 1:
        other_runs_note = f'<div class="hint" style="margin-top:8px;">{len(runs)} real backtest runs recorded total - showing the most recent.</div>'

    return f"""
  <section class="section">
    <div class="section-head">
      <h2>Historical hot-streak signal backtest</h2>
      <span class="hint">Does the real hot/cold z-score signal actually predict already-played MLB history?</span>
    </div>
    <div class="method-card">
      <h3>{_esc(latest.start_date)} to {_esc(latest.end_date)} <span class="hint">n={latest.n_observations} real (player, game) observations</span></h3>
      {table}
      {other_runs_note}
    </div>
    <div class="reco-disclosure">
      <b>Read this like the calibration chart above:</b> if the signal is real, "Hot" rates should sit above the
      overall baseline and "Cold" below it, in every market, not just by chance in one. Computed as of the day
      BEFORE each real game (see hot_streak.py's as_of parameter) - no lookahead. Small buckets (roughly n&lt;30)
      are noisy; read a small split as suggestive; never conclusive. This is a real, direct check against already-
      played history, not live-forward tracking - it doesn't accumulate on its own the way the rest of this page
      does, only when this diagnostic is run again.
    </div>
  </section>"""


def _methodology_section() -> str:
    """A short, factual statement of what's real vs. heuristic vs. proposal-
    only on this site - answers "is this quality, backed-up data" with
    something checkable against the actual code, not a claim."""
    return """
  <section class="section">
    <div class="section-head">
      <h2>Methodology &amp; data sources</h2>
      <span class="hint">What's real, what's heuristic, and what's a proposal - stated plainly</span>
    </div>
    <div class="method-grid">
      <div class="method-card">
        <h3>Real data</h3>
        <ul style="margin:0;padding-left:18px;font-size:13px;line-height:1.7;">
          <li>Batted-ball quality &amp; hot-streak form: Baseball Savant / Statcast</li>
          <li>Rosters, probable pitchers, real boxscores: MLB Stats API</li>
          <li>Wind &amp; temperature: Open-Meteo</li>
          <li>Sportsbook prices &amp; cross-book de-vigged consensus: The Odds API</li>
        </ul>
      </div>
      <div class="method-card">
        <h3>Heuristic - not calibrated</h3>
        <ul style="margin:0;padding-left:18px;font-size:13px;line-height:1.7;">
          <li>model_prob: a hand-weighted composite score (mlb_props/scoring.py), never a trained model</li>
          <li>Ballpark Pal's own model_prob: shown as a second opinion only, never blended in automatically</li>
        </ul>
      </div>
      <div class="method-card">
        <h3>Validated against real outcomes</h3>
        <ul style="margin:0;padding-left:18px;font-size:13px;line-height:1.7;">
          <li>Closing-line value (CLV) - the standard proxy for real betting skill, tracked on every priced pick</li>
          <li>Calibration - predicted probability vs. real observed hit rate, by decile</li>
          <li>Weight refit &amp; market blend checks - walk-forward held-out log-loss, never a random-shuffle split</li>
        </ul>
      </div>
      <div class="method-card">
        <h3>Proposal only - never auto-applied</h3>
        <ul style="margin:0;padding-left:18px;font-size:13px;line-height:1.7;">
          <li>Weight refit and market blend results above: real, validated evidence a human reviews before
          manually changing scoring.py or betting.py - this site never silently changes its own live behavior.</li>
        </ul>
      </div>
    </div>
  </section>"""


def render_performance_report(data_dir: str, generated_at: Optional[datetime] = None) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)

    picks = load_all_picks(data_dir)
    results = load_all_results(data_dir)
    clv_records = load_all_clv(data_dir)
    resolved = resolve_picks(picks, results)

    buckets = calibration_buckets(resolved)
    clv = clv_summary(clv_records)
    by_market = hit_rate_by_market(resolved)
    by_tier = hit_rate_by_tier(resolved)
    by_lineup_source = hit_rate_by_lineup_source(resolved)

    distinct_days = len({r.pick.game_date for r in resolved})
    overall_rate = (sum(1 for r in resolved if r.won) / len(resolved)) if resolved else None

    refit_ready = distinct_days >= REFIT_READY_DAYS
    refit_note = (
        f"<b>{distinct_days} real day{'s' if distinct_days != 1 else ''} of resolved picks collected.</b> "
        + (
            "That's enough real data for a meaningful weight refit - see scoring.py's component weights."
            if refit_ready
            else f"Weight refitting (see scoring.py) becomes meaningful around {REFIT_READY_DAYS} days of "
            "real results - until then the model's weights stay the hand-set values shown on Today's Board, "
            "and every stat below is exactly what it says: real, but still a small sample."
        )
    )

    tiles = [
        # CLV leads: unlike win rate, closing-line value is the standard way
        # sharp bettors (and sportsbooks themselves) measure real skill,
        # since the closing line is the most efficient price a market ever
        # prints - see the "Methodology & data sources" section below.
        _tile(
            "Mean CLV",
            _fmt_signed_pct(clv.mean_clv_percent) if clv.n else "n/a",
            f"vs. the closing price, across <b>{clv.n}</b> price{'s' if clv.n != 1 else ''} tracked &mdash; "
            "the standard proxy for real skill",
        ),
        _tile(
            "Beat the close",
            _fmt_pct(clv.beat_close_percent / 100.0 if clv.beat_close_percent is not None else None)
            if clv.n
            else "n/a",
            "share of tracked picks priced better than the eventual closing line",
        ),
        _tile(
            "Resolved picks",
            str(len(resolved)),
            f"across <b>{distinct_days}</b> real day{'s' if distinct_days != 1 else ''} of games",
        ),
        _tile(
            "Real hit rate",
            _fmt_pct(overall_rate) if overall_rate is not None else "n/a",
            "share of resolved picks that actually cleared their line",
        ),
    ]

    calib_svg = _calibration_svg(buckets)
    hit_rate_market_html = _hit_rate_table("Real hit rate by market", by_market, _MARKET_LABELS)
    hit_rate_tier_html = _hit_rate_table("Real hit rate by tier", by_tier, _TIER_LABELS)
    hit_rate_lineup_source_html = _hit_rate_table("Real hit rate by lineup source", by_lineup_source, _LINEUP_SOURCE_LABELS)
    by_hour = hit_rate_by_run_hour(resolved)
    hit_rate_hour_html = _hit_rate_table("Real hit rate by hour recorded (ET)", by_hour, {})
    refit_results = refit_all_markets(resolved)
    refit_section_html = _refit_section(refit_results, distinct_days)
    blend_results = fit_all_market_blends(resolved)
    blend_section_html = _blend_section(blend_results)
    historical_backtest_runs = load_historical_backtest_runs(os.path.join(data_dir, "historical_backtest", "runs.jsonl"))
    historical_backtest_section_html = _historical_backtest_section(historical_backtest_runs)
    methodology_section_html = _methodology_section()

    log_rows = []
    for r in sorted(resolved, key=lambda r: (r.pick.game_date, r.pick.player), reverse=True):
        p = r.pick
        outcome_cell = '<span class="pos">Won</span>' if r.won else '<span class="neg">Lost</span>'
        price_cell = f"{p.best_price:+d}" if p.best_price is not None else "n/a"
        recorded_local = recorded_at_et(p)
        recorded_cell = recorded_local.strftime("%Y-%m-%d %H:%M ET")
        log_rows.append(
            "<tr>"
            f'<td class="num" data-k="date">{_esc(p.game_date)}</td>'
            f'<td class="num" data-k="time">{_esc(recorded_cell)}</td>'
            f'<td class="player" data-k="player">{_esc(p.player)}</td>'
            f'<td data-k="market">{_esc(_MARKET_LABELS.get(p.market, p.market))}</td>'
            f'<td class="event">{_esc(p.event)}</td>'
            f'<td data-k="tier">{_esc(_TIER_LABELS.get(p.tier, p.tier))}</td>'
            f'<td class="num" data-k="prob">{_fmt_pct(p.model_prob)}</td>'
            f'<td class="num" data-k="price">{price_cell}</td>'
            f'<td class="book" data-k="book">{_esc(book_display_name(p.best_book) or "n/a")}</td>'
            f'<td data-k="outcome">{outcome_cell}</td>'
            "</tr>"
        )
    log_body = "".join(log_rows)

    markets_options = "".join(
        f'<option value="{_esc(k)}">{_esc(v)}</option>' for k, v in _MARKET_LABELS.items()
    )
    tiers_options = "".join(f'<option value="{_esc(k)}">{_esc(v)}</option>' for k, v in _TIER_LABELS.items())

    generated_str = generated_at.strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Longball Board — Performance</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;700;800&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>{STYLE}</style>
</head>
<body>
<div class="wrap">
  {nav_html("performance")}
  <header class="top">
    <div class="brand-block">
      <div class="brand"><span class="mark">MLB PROPS</span></div>
      <h1 class="title">Performance</h1>
      <p class="subtitle">Every pick this model has ever made, resolved against what actually happened, plus closing-line value - the real track record behind Today's Board.</p>
    </div>
    <div class="meta">
      <div class="date num">{_esc(generated_str)}</div>
      <div>recomputed every run, from real recorded data</div>
    </div>
  </header>

  <div class="sample-note">{refit_note}</div>

  <section class="section" style="margin-top:0;">
    <span class="eyebrow">At a glance</span>
    <div class="tiles">{''.join(tiles)}</div>
  </section>

  <section class="section">
    <div class="section-head">
      <h2>Calibration</h2>
      <span class="hint">Predicted probability decile vs. real observed hit rate</span>
    </div>
    <div class="calib-wrap">
      <div class="calib-legend">
        <span><i class="sw" style="background:var(--accent);opacity:.75;"></i>real observed hit rate</span>
        <span><i class="sw" style="background:var(--info);"></i>model's predicted mean</span>
        <span>dashed line = perfectly calibrated</span>
      </div>
      {calib_svg if resolved else '<div class="empty">No resolved picks yet - calibration fills in as results.py records real outcomes.</div>'}
    </div>
  </section>

  <section class="section">
    <div class="section-head"><h2>Hit rate breakdowns</h2><span class="hint">Real outcomes only - never model-predicted</span></div>
    <div class="method-grid">
      {hit_rate_market_html}
      {hit_rate_tier_html}
      {hit_rate_lineup_source_html}
      {hit_rate_hour_html}
    </div>
  </section>

  {refit_section_html}

  {blend_section_html}

  {historical_backtest_section_html}

  {methodology_section_html}

  <section class="section">
    <div class="section-head">
      <h2>Pick history</h2>
      <span class="hint">Every resolved pick, sortable and filterable</span>
    </div>
    <div class="filter-bar">
      <input type="search" id="pf-search" placeholder="Search player…">
      <select id="pf-market"><option value="">All markets</option>{markets_options}</select>
      <select id="pf-tier"><option value="">All tiers</option>{tiers_options}</select>
      <select id="pf-outcome"><option value="">Won or lost</option><option value="won">Won</option><option value="lost">Lost</option></select>
      <span class="filter-count" id="pf-count"></span>
    </div>
    <div class="table-scroll">
      <table class="props sortable" id="pf-table">
        <thead><tr>
          <th data-k="date">Date<span class="arrow">▾</span></th>
          <th data-k="time">Recorded (ET)<span class="arrow">▾</span></th>
          <th data-k="player">Player<span class="arrow">▾</span></th>
          <th data-k="market">Market<span class="arrow">▾</span></th>
          <th>Matchup</th>
          <th data-k="tier">Tier<span class="arrow">▾</span></th>
          <th data-k="prob">Model P()<span class="arrow">▾</span></th>
          <th data-k="price">Price<span class="arrow">▾</span></th>
          <th data-k="book">Book<span class="arrow">▾</span></th>
          <th data-k="outcome">Result<span class="arrow">▾</span></th>
        </tr></thead>
        <tbody id="pf-tbody">{log_body if log_body else ''}</tbody>
      </table>
      {'' if log_body else '<div class="empty">No resolved picks recorded yet.</div>'}
    </div>
  </section>

  <footer>
    <p><strong>Where this data comes from:</strong> every run appends that day's scored candidates to a permanent record (mlb_props/results.py); once a game is final, the picked player's real Statcast log resolves whether the pick actually cleared its line. Nothing here is recomputed model output - it's what really happened.</p>
    <p><strong>CLV (closing-line value)</strong> compares the price recorded at pick time to the best matching price right before lock. Beating the closing line consistently is the standard way sharp bettors measure real edge, independent of whether any single pick hit - a losing pick that beat the close was still a good bet; a winning pick that didn't wasn't necessarily a skillful one.</p>
    <p>Small sample sizes are the honest normal state early on - every number above shows its own N. Nothing here should be read as a guarantee.</p>
  </footer>
</div>
<script>
(function(){{
  var tbody = document.getElementById('pf-tbody');
  var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
  var search = document.getElementById('pf-search');
  var marketSel = document.getElementById('pf-market');
  var tierSel = document.getElementById('pf-tier');
  var outcomeSel = document.getElementById('pf-outcome');
  var countEl = document.getElementById('pf-count');
  var thead = document.querySelector('#pf-table thead');
  var sortKey = null, sortDir = 1;

  function cellText(row, k) {{
    var cell = row.querySelector('[data-k="' + k + '"]');
    return cell ? cell.textContent.trim() : '';
  }}

  function applyFilters() {{
    var q = search.value.trim().toLowerCase();
    var market = marketSel.value;
    var tier = tierSel.value;
    var outcome = outcomeSel.value;
    var visible = 0;
    rows.forEach(function(row) {{
      var player = cellText(row, 'player').toLowerCase();
      var rowMarketLabel = cellText(row, 'market');
      var rowTierLabel = cellText(row, 'tier');
      var rowOutcome = cellText(row, 'outcome').toLowerCase();
      var show = true;
      if (q && player.indexOf(q) === -1) show = false;
      if (market && rowMarketLabel !== marketSel.options[marketSel.selectedIndex].text) show = false;
      if (tier && rowTierLabel !== tierSel.options[tierSel.selectedIndex].text) show = false;
      if (outcome && rowOutcome.indexOf(outcome) === -1) show = false;
      row.classList.toggle('hidden-row', !show);
      if (show) visible++;
    }});
    countEl.textContent = visible + ' of ' + rows.length + ' picks';
  }}

  function applySort(key) {{
    if (sortKey === key) {{ sortDir = -sortDir; }} else {{ sortKey = key; sortDir = 1; }}
    var numeric = key === 'prob' || key === 'price' || key === 'date';
    rows.sort(function(a, b) {{
      var av = cellText(a, key), bv = cellText(b, key);
      if (numeric) {{
        av = parseFloat(av.replace('%', '')) || 0;
        bv = parseFloat(bv.replace('%', '')) || 0;
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
  [search, marketSel, tierSel, outcomeSel].forEach(function(el) {{
    el.addEventListener('input', applyFilters);
    el.addEventListener('change', applyFilters);
  }});
  applyFilters();
}})();
</script>
</body>
</html>
"""
