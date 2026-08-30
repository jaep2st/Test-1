"""Renders a `SlateReport` as a print-ready PDF - the same data as
`report.py`'s console text and `html_report.py`'s web page, laid out for a
daily hand-off document: matchup environments, who's hot, and the three
ranked prop tables, each including the real per-game clearance columns
(see `report.py`'s `clearance_cols`) alongside every other signal.

Reuses `report.py`'s `heat_lookup`/`clearance_cols` rather than
re-deriving them, so the PDF, the HTML page, and the console text can
never quietly drift apart on what a given row's numbers mean.

Requires `reportlab` (`pip install reportlab`) - not a hard dependency of
the rest of this project, so the import is deferred into
`render_pdf_report` itself; callers that never ask for a PDF never pay for
it, and a clear `RuntimeError` (not an ImportError stack trace) tells you
what to install if you do.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .edges import EdgeCandidate
from .hot_streak import HeatIndex
from .pipeline import MatchupEnvironment, SlateReport
from .report import clearance_cols, heat_lookup

# kind -> the column header to show for that market's model probability -
# same convention as html_report.py's prob_header.
_PROB_HEADER = {"hr": "Model P(HR)", "tb2": "Model P(2+ TB)", "hit": "Model P(1+ Hits)"}
_TITLE = {"hr": "Best Home Run Props", "tb2": "Best 2+ Total Bases Props", "hit": "Best 1+ Hits Props"}


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def _fmt_signed_pct(x: Optional[float], digits: int = 1) -> str:
    return f"{x:+.{digits}%}" if x is not None else "n/a"


def _weather_str(e: EdgeCandidate) -> str:
    wind = "dome" if e.is_dome else f"{abs(e.wind_out_mph):.0f}mph {'out' if e.wind_out_mph > 0 else 'in' if e.wind_out_mph < 0 else 'calm'}"
    temp = f"{e.temp_f:.0f}F" if e.temp_f is not None else "n/a"
    return f"{wind}, {temp} ({e.weather_boost_pct:+.1f}%)"


def _prop_table_rows(edges: List[EdgeCandidate], heat_by_player: Dict[str, HeatIndex], kind: str, top: int, cell) -> "tuple[list, list]":
    """Returns (priced_rows, model_only_rows) as lists of cells ready to
    hand to a reportlab Table - same priced-first, model-only-backfill
    logic as report.py's `_render_edge_table`, kept in sync by eye since
    reportlab's Table needs plain cell lists rather than the text-line
    format that function returns. `cell` wraps any column whose text can
    run long (player name, event, weather) in a word-wrapping Paragraph -
    a plain string that's wider than its column silently overflows into
    the neighboring cell in reportlab's Table instead of wrapping, so
    every free-text column needs it; short numeric/code columns don't.
    """
    priced = [e for e in edges if e.has_market_data][:top]
    remaining = top - len(priced)
    unpriced = [e for e in edges if not e.has_market_data][:max(remaining, 0)] if priced else edges[:top]

    priced_rows = []
    for e in priced:
        l15, szn = clearance_cols(heat_by_player.get(e.player), kind)
        priced_rows.append([
            cell(e.player), cell(e.event), _fmt_pct(e.model_prob), _fmt_pct(e.bp_model_prob),
            f"{e.best_line.odds:+d}", e.best_line.sportsbook, _fmt_pct(e.market_fair_prob),
            _fmt_signed_pct(e.edge_vs_market), f"{e.ev_percent_model:+.1f}%",
            f"{e.ev_percent_market:+.1f}%" if e.ev_percent_market is not None else "n/a",
            str(e.books_quoting), l15, szn, cell(_weather_str(e)),
        ])

    model_only_rows = []
    for e in unpriced:
        l15, szn = clearance_cols(heat_by_player.get(e.player), kind)
        bp = _fmt_pct(e.bp_model_prob)
        model_only_rows.append([cell(e.player), cell(e.event), f"{e.model_score:.0f}/100", _fmt_pct(e.model_prob), bp, l15, szn])

    return priced_rows, model_only_rows


def render_pdf_report(
    report: SlateReport,
    out_path: str,
    top: int = 15,
    is_mock: bool = False,
    generated_at: Optional[datetime] = None,
) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("reportlab is required for --pdf-out. Install with `pip install reportlab`.") from exc

    generated_at = generated_at or datetime.now(timezone.utc)
    heat_by_player = heat_lookup(report.hot_batters)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=22, spaceAfter=4)
    sub_style = ParagraphStyle("SubX", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#555555"), spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#8A5308"))
    note = ParagraphStyle("Note", parent=styles["Normal"], fontSize=8, leading=11.5, textColor=colors.HexColor("#444444"), spaceBefore=2)
    cell_style = ParagraphStyle("CellX", parent=styles["Normal"], fontSize=7.2, leading=8.6)

    def cell(text: str) -> "Paragraph":
        # Wraps free-text (player names, "Away @ Home" events, weather
        # strings) so a value wider than its column word-wraps to a second
        # line instead of silently overflowing into the neighboring cell -
        # the failure mode a plain string hits in a reportlab Table the
        # moment real team/player names run long (confirmed live: "Los
        # Angeles Dodgers @ San Francisco Giants" at this table's width).
        # Headers stay plain strings - short enough (single words) to never
        # need it, and TableStyle's own header FONTNAME/TEXTCOLOR commands
        # only apply to plain-string cells, not a Paragraph's own styling.
        return Paragraph(html.escape(text), cell_style)

    page = landscape(letter)
    doc = SimpleDocTemplate(out_path, pagesize=page, topMargin=0.45 * inch, bottomMargin=0.45 * inch, leftMargin=0.4 * inch, rightMargin=0.4 * inch)
    story = []

    status = "SAMPLE OUTPUT - synthetic mock data" if is_mock else "LIVE data"
    story.append(Paragraph("Longball Board", title_style))
    story.append(Paragraph(
        f"MLB Home Run, 2+ Total Bases &amp; 1+ Hits Report &mdash; {report.game_date.isoformat()} "
        f"&middot; {status} &middot; generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        sub_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#172319"), spaceAfter=8))

    base_table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16241B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3E3C6")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D6DECB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]

    # --- Best HR matchups on the slate ---
    story.append(Paragraph("Best HR Matchups on the Slate", h2))
    env_header = ["Matchup", "Park", "Env", "ParkHR", "Wx"]
    env_rows = [env_header]
    for env in report.matchup_environments[:10]:
        m = env.matchup
        env_rows.append([
            cell(f"{m.away_team} @ {m.home_team}"), cell(m.venue), f"{env.environment_score:.1f}",
            f"{env.park_hr_factor:.0f}", f"{env.weather_boost_pct:+.1f}%",
        ])
    t = Table(env_rows, repeatRows=1, colWidths=[2.6 * inch, 1.8 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch])
    t.setStyle(TableStyle(base_table_style))
    story.append(t)

    # --- Who's hot ---
    story.append(Paragraph("Who's Hot", h2))
    hot_header = ["Player", "Label", "L15 wOBA", "Season wOBA", "Z-score"]
    hot_rows = [hot_header]
    for h in report.hot_batters[:10]:
        hot_rows.append([cell(h.player), h.label, f"{h.last15_woba:.3f}", f"{h.season_woba:.3f}", f"{h.z_score:+.2f}"])
    t = Table(hot_rows, repeatRows=1, colWidths=[2.2 * inch, 1.1 * inch, 1.1 * inch, 1.3 * inch, 1.0 * inch])
    t.setStyle(TableStyle(base_table_style))
    story.append(t)

    # --- The three ranked prop tables ---
    prop_header = [
        "Player", "Event", "Model", "BP Mdl", "Price", "Book", "MktFair",
        "Edge", "EV(mdl)", "EV(mkt)", "Bks", "L15Clr", "SznRt", "Weather",
    ]
    prop_col_widths = [
        1.3 * inch, 1.7 * inch, 0.55 * inch, 0.55 * inch, 0.5 * inch, 0.65 * inch, 0.5 * inch,
        0.45 * inch, 0.6 * inch, 0.55 * inch, 0.35 * inch, 0.55 * inch, 0.5 * inch, 1.2 * inch,
    ]
    model_only_header = ["Player", "Event", "Score", "Model", "BP Mdl", "L15Clr", "SznRt"]
    model_only_col_widths = [1.4 * inch, 1.9 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch]

    for edges, kind in ((report.hr_edges, "hr"), (report.tb_edges, "tb2"), (report.hits_edges, "hit")):
        story.append(Paragraph(f"{_TITLE[kind]} (+EV, ranked)", h2))
        priced_rows, model_only_rows = _prop_table_rows(edges, heat_by_player, kind, top, cell)
        if priced_rows:
            t = Table([prop_header] + priced_rows, repeatRows=1, colWidths=prop_col_widths)
            t.setStyle(TableStyle(base_table_style))
            story.append(t)
        if model_only_rows:
            story.append(Paragraph(
                "Model-only &mdash; no book currently quotes these candidates:" if priced_rows
                else "No market prices matched this slate - model-only ranking:",
                note,
            ))
            t = Table([model_only_header] + model_only_rows, repeatRows=1, colWidths=model_only_col_widths)
            t.setStyle(TableStyle(base_table_style))
            story.append(t)
        if not priced_rows and not model_only_rows:
            story.append(Paragraph("(no candidates)", note))

    # --- Footer data-quality notes - verbatim from report.py's footer, so
    # the PDF never promises something the underlying model doesn't. ---
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#D6DECB"), spaceAfter=6))
    footer_lines = [
        "Model scores are a transparent heuristic (see mlb_props/scoring.py), not a calibrated prediction - "
        "cross-check against the market's own no-vig consensus (MktFair) and your own judgment before betting anything.",
        "<b>EV%</b>: model vs. market, not \"market is wrong.\" A positive EV% is our model disagreeing with the "
        "market in the bettor's favor, not proof the market is mispriced.",
        "<b>Pull-air%</b> is permanently unavailable for the HR score (6% of its weight) - a disclosed gap, not a hidden zero.",
        "<b>BP Model</b> (HR/Hits only): Ballpark Pal's own independent model, converted to a per-game figure - a "
        "genuine second opinion, never blended into our own model_prob/EV math. \"n/a\" = not configured or no data for that matchup.",
        "<b>L15Clr / SznRt</b>: real per-game clearance counts, not another model - literal \"did this player actually "
        "clear this exact line in this real game,\" counted from Baseball Savant's per-PA log. L15Clr is X/Y = cleared "
        "X of the last Y games actually played (Y can be under 15 for a recent call-up/return from IL); SznRt is the "
        "same rate over the full season, the baseline L15Clr should be read against. A 5- or 10-game window is "
        "deliberately not shown here - too small a sample to separate a real hot streak from noise. \"n/a\" means no "
        "per-game log was available for that player, never a hidden zero.",
    ]
    for line in footer_lines:
        story.append(Paragraph(line, note))

    doc.build(story)
