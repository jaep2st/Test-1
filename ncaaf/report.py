"""Plain-text console report: this week's predicted scores, then ranked
+EV picks across spreads/moneylines/totals.
"""

from __future__ import annotations

from typing import List

from .betting import build_recommended_bets
from .edges import EdgeCandidate
from .market import book_display_name
from .pipeline import WeekReport

_TIER_LABELS = {
    "agree": "STRONG (model + market agree)",
    "model_only": "model only",
    "model_only_single_sided": "model only (single-sided market)",
    "no_market": "no market price",
}


def render_report(report: WeekReport, top: int = 25) -> str:
    lines: List[str] = []
    lines.append(f"NCAAF WEEK {report.week} ({report.season}, {report.season_type}) - {len(report.slate)} game(s) on the slate")
    lines.append("=" * 78)
    lines.append("")

    lines.append("PREDICTED SCORES")
    lines.append("-" * 78)
    if not report.scores:
        lines.append("No games could be scored this week (no power ratings resolved, or an empty slate).")
    for event, score in sorted(report.scores.items()):
        fav = score.matchup.home_team if score.predicted_margin >= 0 else score.matchup.away_team
        lines.append(
            f"{event:45s} {score.predicted_away_points:5.1f} - {score.predicted_home_points:5.1f}  "
            f"(pred margin {score.predicted_margin:+.1f}, {fav} | total {score.predicted_total:.1f} | "
            f"home win {score.home_win_prob:.0%})"
        )
    lines.append("")

    lines.append(f"TOP {top} +EV PICKS (all markets)")
    lines.append("-" * 78)
    if not report.candidates:
        lines.append("No priced candidates this run (no odds fetched, or nothing cleared the min-EV bar).")
    for c in report.candidates[:top]:
        lines.append(_describe_candidate(c))
    lines.append("")

    strong, speculative = build_recommended_bets(report.candidates)
    lines.append("RECOMMENDED BETS")
    lines.append("-" * 78)
    if not strong and not speculative:
        lines.append("Nothing currently clears the recommendation bar.")
    else:
        if strong:
            lines.append(f"Strong ({len(strong)}):")
            for r in strong:
                lines.append(
                    f"  {r.event} - {r.market_label} {r.selection}: {r.units:g}u @ {book_display_name(r.best_book)} "
                    f"{r.best_price:+d} (model {r.model_prob:.1%}, EV {r.ev_percent_model:+.1f}%)"
                )
        if speculative:
            lines.append(f"Speculative ({len(speculative)}):")
            for r in speculative:
                lines.append(
                    f"  {r.event} - {r.market_label} {r.selection}: {r.units:g}u @ {book_display_name(r.best_book)} "
                    f"{r.best_price:+d} (model {r.model_prob:.1%}, EV {r.ev_percent_model:+.1f}%)"
                )
    lines.append("")
    lines.append(
        "Model probabilities are a transparent, hand-built statistical estimate (blended SP+ + Elo power ratings "
        "plus situational context) - not a trained/calibrated model. EV% is our model vs. the market's own no-vig "
        "price, not proof the market is wrong. See README.md and ncaaf/scoring.py for the full methodology and "
        "known caveats."
    )
    return "\n".join(lines)


def _describe_candidate(c: EdgeCandidate) -> str:
    tier = _TIER_LABELS.get(c.tier, c.tier)
    if not c.has_market_data:
        return f"[{tier}] {c.event} - {c.market} {c.selection_label}: model {c.model_prob:.1%}, no market price"
    return (
        f"[{tier}] {c.event} - {c.market} {c.selection_label}: model {c.model_prob:.1%} vs market "
        f"{c.market_fair_prob:.1%} [edge {c.edge_vs_market:+.1%}] | best {book_display_name(c.best_line.sportsbook)} "
        f"{c.best_line.odds:+d} (EV {c.ev_percent_model:+.1f}% model, {c.ev_percent_market:+.1f}% market) | "
        f"{c.books_quoting} books"
    )
