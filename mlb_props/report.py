"""Renders a `SlateReport` as readable console text: best matchups, who's
hot, top home run props, and top 2+ total bases props.
"""

from __future__ import annotations

from typing import List

from .edges import EdgeCandidate
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


def _render_edge_table(title: str, edges: List[EdgeCandidate], top: int) -> str:
    lines = [f"## {title}", ""]
    if not edges:
        lines.append("(no candidates)")
        return "\n".join(lines)

    priced = [e for e in edges if e.has_market_data]
    if not priced:
        lines.append("(no market prices matched - showing model-only ranking below)")
        for e in edges[:top]:
            lines.append(f"  - {e.player} ({e.event}): model score {e.model_score:.0f}/100, est. {_fmt_pct(e.model_prob)}")
        return "\n".join(lines)

    lines.append(
        f"{'Player':<20} {'Event':<16} {'Model':>7} {'Best':>7} {'Book':<10} "
        f"{'MktFair':>8} {'Edge':>7} {'EV(mdl)':>8} {'EV(mkt)':>8} {'Bks':>4} {'Weather':<22}"
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
        lines.append(
            f"{_truncate(e.player, 20):<20} {_truncate(e.event, 16):<16} {_fmt_pct(e.model_prob):>7} "
            f"{e.best_line.odds:>+7d} {e.best_line.sportsbook:<10} "
            f"{_fmt_opt_pct(e.market_fair_prob):>8} {_fmt_opt_signed_pct(e.edge_vs_market):>7} "
            f"{e.ev_percent_model:>+7.1f}% {ev_market:>8} {e.books_quoting:>4} {weather:<22}"
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
            lines.append(f"  - {e.player} ({e.event}): model score {e.model_score:.0f}/100, est. {_fmt_pct(e.model_prob)}")
    return "\n".join(lines)


def render_hr_props(edges: List[EdgeCandidate], top: int = 15) -> str:
    return _render_edge_table("Best Home Run Props (+EV, ranked)", edges, top)


def render_total_bases_props(edges: List[EdgeCandidate], top: int = 15) -> str:
    return _render_edge_table("Best 2+ Total Bases Props (+EV, ranked)", edges, top)


def render_hits_props(edges: List[EdgeCandidate], top: int = 15) -> str:
    return _render_edge_table("Best 1+ Hits Props (+EV, ranked)", edges, top)


def render_report(report: SlateReport, top: int = 15) -> str:
    header = f"# MLB Home Run, 2+ Total Bases & 1+ Hits Report - {report.game_date.isoformat()}\n"
    sections = [
        header,
        render_matchup_environments(report.matchup_environments),
        render_hot_batters(report.hot_batters),
        "",
        render_hr_props(report.hr_edges, top),
        "",
        render_total_bases_props(report.tb_edges, top),
        "",
        render_hits_props(report.hits_edges, top),
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
    ]
    return "\n".join(sections)
