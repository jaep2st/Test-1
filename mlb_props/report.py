"""Renders a `SlateReport` as readable console text: best matchups, who's
hot, top home run props, and top 2+ total bases props.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .edges import EdgeCandidate
from .hot_streak import HeatIndex
from .market import book_display_name
from .pipeline import MatchupEnvironment, SlateReport


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_opt_pct(x) -> str:
    return _fmt_pct(x) if x is not None else "n/a"


def _fmt_opt_signed_pct(x, digits: int = 1) -> str:
    return f"{x:+.{digits}%}" if x is not None else "n/a"


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def render_matchup_environments(environments: List[MatchupEnvironment], top: int = 10) -> str:
    lines = ["## Best HR Matchups on the Slate", ""]
    if not environments:
        lines.append("(no games found)")
        return "\n".join(lines)
    lines.append(f"{'Matchup':<34} {'Park':<22} {'Env':>6} {'ParkHR':>7} {'Wx':>7}")
    for env in environments[:top]:
        m = env.matchup
        matchup_str = _truncate(f"{m.away_team} @ {m.home_team}", 34)
        lines.append(
            f"{matchup_str:<34} {_truncate(m.venue, 22):<22} {env.environment_score:>6.1f} "
            f"{env.park_hr_factor:>6.0f} {env.weather_boost_pct:>+6.1f}%"
        )
    return "\n".join(lines)


def render_hot_batters(hot_batters: List, top: int = 10) -> str:
    lines = ["", "## Who's Hot", ""]
    if not hot_batters:
        lines.append("(no batters scored)")
        return "\n".join(lines)
    lines.append(f"{'Player':<26} {'Label':<10} {'L15 wOBA':>9} {'Season wOBA':>12} {'Z-score':>8}")
    for h in hot_batters[:top]:
        lines.append(f"{h.player:<26} {h.label:<10} {h.last15_woba:>9.3f} {h.season_woba:>12.3f} {h.z_score:>+8.2f}")
    return "\n".join(lines)


def heat_lookup(hot_batters: List[HeatIndex]) -> Dict[str, HeatIndex]:
    # `hot_batters` (despite the name) carries every scored candidate's
    # HeatIndex, not just the hot ones - pipeline.py appends one per
    # candidate unconditionally; "Who's Hot" just slices the top few by
    # z-score for display. Safe to build a full player -> HeatIndex lookup
    # from it here.
    return {h.player: h for h in hot_batters}


# kind -> (ClearanceWindow attribute for the count, ClearanceWindow property
# for the rate), one pair per market this report ranks props for.
_CLEARANCE_ATTRS = {
    "hr": ("hr_games", "hr_rate"),
    "tb2": ("tb2_games", "tb2_rate"),
    "hit": ("hit_games", "hit_rate"),
}


def clearance_cols(heat: Optional[HeatIndex], kind: str) -> "tuple[str, str]":
    """(L15 literal count, season rate) for the given market. These two
    numbers are the ones this project treats as signal: L15 is a real
    15-game sample - large enough to mean something, and the same window
    this report already uses for the wOBA z-score - and season rate is the
    trustworthy long-run baseline to judge it against. L5/L10 are computed
    and available on the HeatIndex itself (`clear_l5`/`clear_l10`) but left
    out of this compact table on purpose: a 5-or-10-game window is exactly
    the size a single hot or cold week produces on its own, noise more
    often than a real signal. "n/a" means no per-game log was available for
    this player (only `StatcastHotStreakProvider` computes these).
    """
    if heat is None:
        return "n/a", "n/a"
    count_attr, rate_attr = _CLEARANCE_ATTRS[kind]
    window = heat.clear_l15
    l15 = f"{getattr(window, count_attr)}/{window.games}" if window is not None and window.games else "n/a"
    season = heat.clear_season
    rate = getattr(season, rate_attr) if season is not None and season.games else None
    szn = f"{rate * 100:.0f}%" if rate is not None else "n/a"
    return l15, szn


def clearance_rates(heat: Optional[HeatIndex], kind: str) -> "tuple[Optional[float], Optional[float]]":
    """The same real (L15 rate, season rate) `clearance_cols` formats for
    display, but as raw 0-1 floats instead of strings ("3/15", "45%") -
    for sorting a real numeric column by actual clearance rate rather than
    alphabetically by its formatted text. `None` (never a guessed 0) for
    whatever `clearance_cols` would show as "n/a".
    """
    if heat is None:
        return None, None
    count_attr, rate_attr = _CLEARANCE_ATTRS[kind]
    window = heat.clear_l15
    l15_rate = (getattr(window, count_attr) / window.games) if window is not None and window.games else None
    season = heat.clear_season
    season_rate = getattr(season, rate_attr) if season is not None and season.games else None
    return l15_rate, season_rate


def _model_only_line(e: EdgeCandidate, heat: Optional[HeatIndex], kind: str) -> str:
    bp = f" | BP model: {_fmt_pct(e.bp_model_prob)}" if e.bp_model_prob is not None else ""
    l15, szn = clearance_cols(heat, kind)
    return f"  - {e.player} ({e.event}): model score {e.model_score:.0f}/100, est. {_fmt_pct(e.model_prob)}{bp} | L15 clear: {l15} | season rate: {szn}"


def _render_edge_table(title: str, edges: List[EdgeCandidate], top: int, heat_by_player: Dict[str, HeatIndex], kind: str) -> str:
    lines = [f"## {title}", ""]
    if not edges:
        lines.append("(no candidates)")
        return "\n".join(lines)

    priced = [e for e in edges if e.has_market_data]
    if not priced:
        lines.append("(no market prices matched - showing model-only ranking below)")
        for e in edges[:top]:
            lines.append(_model_only_line(e, heat_by_player.get(e.player), kind))
        return "\n".join(lines)

    lines.append(
        f"{'Player':<20} {'Event':<16} {'Model':>7} {'BP Mdl':>7} {'Best':>7} {'Book':<10} "
        f"{'MktFair':>8} {'Edge':>7} {'EV(mdl)':>8} {'EV(mkt)':>8} {'Bks':>4} {'L15Clr':>7} {'SznRt':>6} {'Weather':<22}"
    )
    for e in priced[:top]:
        wind = "dome" if e.is_dome else f"{abs(e.wind_out_mph):.0f}mph {'out' if e.wind_out_mph > 0 else 'in' if e.wind_out_mph < 0 else 'calm'}"
        temp = f"{e.temp_f:.0f}F" if e.temp_f is not None else "n/a"
        weather = f"{wind}, {temp} ({e.weather_boost_pct:+.1f}%)"
        # market_fair_prob/edge_vs_market/ev_percent_market are None for a
        # single-sided market (real price, but no second side to de-vig a
        # fair consensus from - see edges.py's module docstring); shown as
        # "n/a" rather than crashing or hiding an otherwise-real price.
        ev_market = f"{e.ev_percent_market:+.1f}%" if e.ev_percent_market is not None else "n/a"
        # bp_model_prob: Ballpark Pal's own independent model, when
        # configured - see edges.py's EdgeCandidate docstring. "n/a" for
        # 2+ TB (no honest analog exists) and whenever it isn't configured
        # or has no data for this matchup, same as any other missing
        # signal in this report.
        bp_model = _fmt_opt_pct(e.bp_model_prob)
        l15, szn = clearance_cols(heat_by_player.get(e.player), kind)
        lines.append(
            f"{_truncate(e.player, 20):<20} {_truncate(e.event, 16):<16} {_fmt_pct(e.model_prob):>7} {bp_model:>7} "
            f"{e.best_line.odds:>+7d} {_truncate(book_display_name(e.best_line.sportsbook), 13):<13} "
            f"{_fmt_opt_pct(e.market_fair_prob):>8} {_fmt_opt_signed_pct(e.edge_vs_market):>7} "
            f"{e.ev_percent_model:>+7.1f}% {ev_market:>8} {e.books_quoting:>4} {l15:>7} {szn:>6} {weather:<22}"
        )

    # Real market prices only cover whichever candidates a book happens to
    # quote (confirmed live: some events, some players - never all of
    # them). Priced rows above are the highest-value info, but the rest of
    # the field shouldn't just vanish from the report because one other
    # player had a price - backfill remaining slots with model-only rows
    # so this table still surfaces the top candidates overall, priced or not.
    remaining = top - len(priced[:top])
    unpriced = [e for e in edges if not e.has_market_data][:remaining]
    if unpriced:
        lines.append("")
        lines.append(f"(model-only - no book currently quotes these {len(unpriced)} candidates)")
        for e in unpriced:
            lines.append(_model_only_line(e, heat_by_player.get(e.player), kind))
    return "\n".join(lines)


def render_hr_props(edges: List[EdgeCandidate], heat_by_player: Optional[Dict[str, HeatIndex]] = None, top: int = 15) -> str:
    return _render_edge_table("Best Home Run Props (+EV, ranked)", edges, top, heat_by_player or {}, "hr")


def render_total_bases_props(edges: List[EdgeCandidate], heat_by_player: Optional[Dict[str, HeatIndex]] = None, top: int = 15) -> str:
    return _render_edge_table("Best 2+ Total Bases Props (+EV, ranked)", edges, top, heat_by_player or {}, "tb2")


def render_hits_props(edges: List[EdgeCandidate], heat_by_player: Optional[Dict[str, HeatIndex]] = None, top: int = 15) -> str:
    return _render_edge_table("Best 1+ Hits Props (+EV, ranked)", edges, top, heat_by_player or {}, "hit")


def render_report(report: SlateReport, top: int = 15) -> str:
    header = f"# MLB Home Run, 2+ Total Bases & 1+ Hits Report - {report.game_date.isoformat()}\n"
    heat_by_player = heat_lookup(report.hot_batters)
    sections = [
        header,
        render_matchup_environments(report.matchup_environments),
        render_hot_batters(report.hot_batters),
        "",
        render_hr_props(report.hr_edges, heat_by_player, top),
        "",
        render_total_bases_props(report.tb_edges, heat_by_player, top),
        "",
        render_hits_props(report.hits_edges, heat_by_player, top),
        "",
        "---",
        "Model scores are a transparent heuristic (see mlb_props/scoring.py), not a",
        "calibrated prediction - cross-check against the market's own no-vig consensus",
        "(the 'Mkt Fair' column) and your own judgment before betting anything.",
        "",
        "DATA QUALITY NOTES (permanent, applies every run):",
        "- 'EV%' means model vs. market, not 'market is wrong.' Every EV% figure is",
        "  our model's probability compared against the book's own no-vig fair price.",
        "  A positive EV% is our model disagreeing with the market in the bettor's",
        "  favor, not proof the market is mispriced - it could just as easily be right",
        "  and our model wrong. Weigh it as one informed opinion against another.",
        "- Pull-air% is permanently unavailable for the HR score (6% of its weight).",
        "  Neither Baseball Savant leaderboard this project pulls carries a pull-rate",
        "  column, and FanGraphs (which does) returns 403 from this hosting provider.",
        "  That component defaults to 0 for every player, every run - a disclosed",
        "  gap, not a hidden zero.",
        "- 'BP Model' (HR/Hits tables only, when configured) is Ballpark Pal's own",
        "  independent model - a genuine second opinion, not this project's model shown",
        "  twice. Their real numbers are per-plate-appearance; converted here to a",
        "  per-game figure via P(at least 1 in ~4.3 PA) so it's comparable to 'Model'.",
        "  Agreement between the two is more reassuring than either alone; disagreement",
        "  means two different models read the matchup differently, not that one is",
        "  wrong. 'n/a' means not configured or Ballpark Pal has no data for that matchup.",
        "- 'L15Clr'/'SznRt' are real per-game clearance counts, not another model -",
        "  literal 'did this player actually clear this exact line in this real game',",
        "  counted from Baseball Savant's per-PA log. L15Clr is X/Y = cleared X of the",
        "  last Y games actually played (Y can be under 15 for a recent call-up/return",
        "  from IL); SznRt is the same rate over the full season, the baseline L15Clr",
        "  should be read against. A 5- or 10-game window is deliberately not shown",
        "  here - too small a sample to separate a real hot streak from noise - but is",
        "  available on request. 'n/a' means no per-game log was available for that",
        "  player, never a hidden zero.",
    ]
    return "\n".join(sections)
