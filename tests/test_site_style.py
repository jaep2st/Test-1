"""Covers mlb_props/site_style.py: real, reported UI bugs and their fixes.

Confirmed live (user report): clicking a prop table's "why?" toggle opened
a detail-panel that got visually cut off at the screen edge with no way to
scroll to the rest of it. Root cause: `table.props td.player` is
`position:sticky` (kept pinned in place while a wide prop table scrolls
horizontally - see the "Mobile: sticky verdict + player columns" comment
below), and the "why?" detail-panel renders *inside* that same cell (see
html_report.py's `_component_detail_html`). A sticky-positioned element
never moves when its scroll container is scrolled, so once the panel grew
wider than the pinned column, its overflow was stuck at a fixed screen
position - scrolling the table did nothing to reveal it.
"""

from mlb_props.site_style import STYLE


def test_sticky_player_column_unsticks_itself_while_its_own_panel_is_open():
    # The actual fix: un-stick just the one cell whose own detail-panel is
    # open, so it (and the panel inside it) scroll normally again like
    # every other column - a fixed-position cell was never compatible with
    # content that can grow wider than it.
    assert 'td.player:has(.detail-panel.open){ position:static' in STYLE


def test_detail_panel_never_wider_than_the_viewport():
    # Belt-and-suspenders: on a narrow (mobile) screen, the panel shrinks
    # to fit rather than relying on horizontal scroll to reach it at all.
    assert "max-width:min(380px, calc(100vw - 48px))" in STYLE
