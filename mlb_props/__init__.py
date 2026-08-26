"""MLB home run and 2+ total bases prop finder.

Built on top of `odds_monitor` (odds fetching + no-vig EV math) and adds
MLB-specific signal: Statcast batted-ball quality, platoon/matchup/pitch-mix
edges, recent form ("who's hot"), and ballpark/weather context, combined
into a composite score and cross-checked against the market. See
`mlb_props/README` section of the repo README for the full pipeline and
data sources.
"""
