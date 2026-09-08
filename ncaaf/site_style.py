"""Reuses `mlb_props/site_style.py`'s shared visual system (same colors,
type, tile/table conventions) so this repo's two products - MLB props and
this NCAAF game-line model - read as one site, not two visually unrelated
pages. Only the navigation differs, since the NCAAF pages are published
under a `ncaaf/` subdirectory of the same GitHub Pages site (see
`.github/workflows/mlb-props-report.yml`) - relative links need an extra
`../` to reach the MLB pages from here.
"""

from __future__ import annotations

from mlb_props.site_style import STYLE  # noqa: F401 - re-exported for ncaaf/html_report.py and ncaaf/performance_report.py


def nav_html(active: str) -> str:
    """`active`: "board" or "performance" - marks the current NCAAF page's
    tab. Also links back to the MLB Props site's own two pages one
    directory up.
    """
    board_class = "active" if active == "board" else ""
    perf_class = "active" if active == "performance" else ""
    return (
        '<nav class="site-nav">'
        f'<a class="{board_class}" href="index.html">NCAAF Board</a>'
        f'<a class="{perf_class}" href="performance.html">Performance</a>'
        '<a href="../index.html">MLB Props</a>'
        "</nav>"
    )
