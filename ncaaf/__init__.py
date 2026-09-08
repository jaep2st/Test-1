"""NCAA football (FBS) game-line betting model: schedule -> power ratings
(SP+ + an independently-computed Elo, blended) -> situational context (home
field, rest, travel, altitude, weather) -> a predicted score/margin per game
-> cross-book odds (spreads/moneylines/totals) -> ranked +EV picks.

Mirrors `mlb_props/`'s pipeline -> edges -> report -> results/backtest
architecture (see that package's docstrings for the shared design
philosophy: transparent, hand-inspectable components; "unknown stays
unknown" rather than a guessed number; a real, persisted pick/result/CLV
history instead of a synthetic backtest). The main structural difference is
the bet type: `mlb_props` scores individual player props, this package
scores whole games (point spread, moneyline, total) - see
`ncaaf/scoring.py`'s module docstring for the model itself.
"""
