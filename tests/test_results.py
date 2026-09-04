"""Covers mlb_props/results.py: recording picks, resolving real outcomes
via Statcast (reusing hot_streak.game_outcomes_from_events), and closing-
line-value (CLV) recording. See that module's docstring.
"""

from datetime import date, datetime, timezone

import pandas as pd

from mlb_props.edges import EdgeCandidate
from mlb_props.pipeline import SlateReport
from mlb_props.results import (
    load_clv,
    load_picks,
    load_results,
    record_closing_odds,
    record_picks,
    resolve_player_game_outcome,
    resolve_results_for_date,
)
from odds_monitor.models import PropLine


def _edge(player, market, event="Team A @ Team B", price=650, book="draftkings", tier_inputs=None, **overrides):
    tier_inputs = tier_inputs or {}
    defaults = dict(
        player=player,
        market=market,
        event=event,
        model_score=70.0,
        model_prob=0.15,
        market_fair_prob=0.10,
        best_line=PropLine(player=player, team=None, league="mlb", market=market, side="yes", line=0.5, odds=price, sportsbook=book, event=event),
        ev_percent_model=25.0,
        ev_percent_market=15.0,
        edge_vs_market=0.05,
        price_spread_percent=None,
        books_quoting=4,
        park="Test Park",
        wind_out_mph=0.0,
        temp_f=70.0,
        is_dome=False,
        weather_boost_pct=0.0,
        bp_model_prob=None,
    )
    defaults.update(tier_inputs)
    defaults.update(overrides)
    return EdgeCandidate(**defaults)


def _report():
    return SlateReport(
        game_date=date(2026, 8, 20),
        slate=[],
        matchup_environments=[],
        hot_batters=[],
        hr_edges=[_edge("Player One", "batter_home_runs")],
        tb_edges=[_edge("Player Two", "batter_total_bases", price=-140, book="fanduel")],
        hits_edges=[_edge("Player Three", "batter_hits", price=-120, book="betmgm")],
    )


def test_record_picks_writes_one_line_per_edge_across_all_markets(tmp_path):
    out = tmp_path / "picks" / "2026-08-20.jsonl"
    n = record_picks(_report(), str(out), recorded_at=datetime(2026, 8, 20, 18, 0, tzinfo=timezone.utc))
    assert n == 3
    picks = load_picks(str(out))
    assert {p.player for p in picks} == {"Player One", "Player Two", "Player Three"}
    assert all(p.game_date == "2026-08-20" for p in picks)
    hr_pick = next(p for p in picks if p.player == "Player One")
    assert hr_pick.tier == "agree"
    assert hr_pick.best_price == 650
    assert hr_pick.best_book == "draftkings"


def test_record_picks_appends_rather_than_overwrites(tmp_path):
    out = tmp_path / "picks" / "2026-08-20.jsonl"
    record_picks(_report(), str(out), recorded_at=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc))
    record_picks(_report(), str(out), recorded_at=datetime(2026, 8, 20, 18, 0, tzinfo=timezone.utc))
    assert len(load_picks(str(out))) == 6  # two snapshots of the same slate


def test_load_picks_on_missing_file_returns_empty_list(tmp_path):
    assert load_picks(str(tmp_path / "nope.jsonl")) == []


class _FakePyb:
    def __init__(self, id_lookup_df, batter_log_df):
        self._id_lookup_df = id_lookup_df
        self._batter_log_df = batter_log_df
        self.statcast_batter_calls = []

    def playerid_lookup(self, last, first):
        return self._id_lookup_df

    def statcast_batter(self, start, end, player_id):
        self.statcast_batter_calls.append((start, end, player_id))
        return self._batter_log_df


def test_resolve_player_game_outcome_reads_a_real_hr_game():
    id_df = pd.DataFrame({"key_mlbam": [660271]})
    log_df = pd.DataFrame({"game_date": ["2026-08-20", "2026-08-20"], "events": ["home_run", "strikeout"]})
    pyb = _FakePyb(id_df, log_df)
    outcome = resolve_player_game_outcome(pyb, {}, "Test Player", date(2026, 8, 20))
    assert outcome is not None
    assert outcome.got_hr is True
    assert outcome.got_2plus_tb is True
    assert outcome.got_hit is True
    assert outcome.hit_for("batter_home_runs") is True
    assert outcome.hit_for("batter_total_bases") is True
    assert outcome.hit_for("batter_hits") is True
    assert outcome.hit_for("not_a_real_market") is None


def test_resolve_player_game_outcome_returns_none_when_no_mlbam_id():
    pyb = _FakePyb(pd.DataFrame({"key_mlbam": []}), pd.DataFrame())
    assert resolve_player_game_outcome(pyb, {}, "Nobody Real", date(2026, 8, 20)) is None


def test_resolve_player_game_outcome_returns_none_when_player_did_not_play():
    id_df = pd.DataFrame({"key_mlbam": [660271]})
    empty_log = pd.DataFrame({"game_date": [], "events": []})
    pyb = _FakePyb(id_df, empty_log)
    assert resolve_player_game_outcome(pyb, {}, "Test Player", date(2026, 8, 20)) is None


def test_resolve_results_for_date_fetches_each_distinct_player_only_once(tmp_path):
    picks_path = tmp_path / "picks" / "2026-08-20.jsonl"
    report = SlateReport(
        game_date=date(2026, 8, 20),
        slate=[],
        matchup_environments=[],
        hot_batters=[],
        hr_edges=[_edge("Same Player", "batter_home_runs")],
        tb_edges=[_edge("Same Player", "batter_total_bases")],  # appears twice this slate
        hits_edges=[],
    )
    record_picks(report, str(picks_path), recorded_at=datetime(2026, 8, 20, 18, 0, tzinfo=timezone.utc))

    id_df = pd.DataFrame({"key_mlbam": [660271]})
    log_df = pd.DataFrame({"game_date": ["2026-08-20"], "events": ["single"]})
    pyb = _FakePyb(id_df, log_df)

    out = tmp_path / "results" / "2026-08-20.jsonl"
    n = resolve_results_for_date(pyb, str(picks_path), str(out), date(2026, 8, 20))
    assert n == 1
    assert len(pyb.statcast_batter_calls) == 1  # one real fetch, not one per pick
    results = load_results(str(out))
    assert results[0].got_hit is True
    assert results[0].got_hr is False


def test_resolve_results_for_date_with_no_recorded_picks_is_a_noop(tmp_path):
    pyb = _FakePyb(pd.DataFrame({"key_mlbam": []}), pd.DataFrame())
    n = resolve_results_for_date(pyb, str(tmp_path / "nope.jsonl"), str(tmp_path / "out.jsonl"), date(2026, 8, 20))
    assert n == 0
    assert not (tmp_path / "out.jsonl").exists()


def test_record_closing_odds_computes_positive_clv_when_pick_price_beat_the_close(tmp_path):
    picks_path = tmp_path / "picks" / "2026-08-20.jsonl"
    record_picks(_report(), str(picks_path), recorded_at=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc))

    # Closing price for Player One's HR prop shortened from +650 to +550 -
    # the recorded pick beat the close, so CLV should be positive.
    closing_lines = [
        PropLine(player="Player One", team=None, league="mlb", market="batter_home_runs", side="yes", line=0.5, odds=550, sportsbook="draftkings", event="Team A @ Team B"),
    ]
    out = tmp_path / "clv" / "2026-08-20.jsonl"
    n = record_closing_odds(str(picks_path), closing_lines, str(out), recorded_at=datetime(2026, 8, 20, 22, 0, tzinfo=timezone.utc))
    assert n == 1
    records = load_clv(str(out))
    assert records[0].player == "Player One"
    assert records[0].clv_percent > 0
    assert records[0].closing_price == 550


def test_record_closing_odds_skips_picks_with_no_matching_closing_line(tmp_path):
    picks_path = tmp_path / "picks" / "2026-08-20.jsonl"
    record_picks(_report(), str(picks_path), recorded_at=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc))
    out = tmp_path / "clv" / "2026-08-20.jsonl"
    n = record_closing_odds(str(picks_path), [], str(out))
    assert n == 0


def test_record_closing_odds_ignores_a_longer_shot_tier_at_the_same_side(tmp_path):
    # Real recorded CLV data (2026-09-04) showed batter_home_runs CLV
    # averaging -53.85% with "closing" prices like +8000 - a real book
    # quoting several point tiers under the same market/side ("1+ HR",
    # "2+ HR", "3+ HR", all outcome name "yes") got its longest-shot tier
    # picked as the "closing" price for what was actually always the
    # standard "1+ HR" (0.5) pick, since nothing here filtered by line.
    # Exactly the bug edges.py's _single_sided_lookup already guards
    # against - this is the same fix for record_closing_odds.
    picks_path = tmp_path / "picks" / "2026-08-20.jsonl"
    record_picks(_report(), str(picks_path), recorded_at=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc))

    closing_lines = [
        # The real standard line this pick was actually recorded at.
        PropLine(player="Player One", team=None, league="mlb", market="batter_home_runs", side="yes", line=0.5, odds=550, sportsbook="draftkings", event="Team A @ Team B"),
        # A much longer-shot tier, same side, same book - must NOT be picked
        # even though its price is far better (higher decimal odds).
        PropLine(player="Player One", team=None, league="mlb", market="batter_home_runs", side="yes", line=2.5, odds=8000, sportsbook="draftkings", event="Team A @ Team B"),
    ]
    out = tmp_path / "clv" / "2026-08-20.jsonl"
    n = record_closing_odds(str(picks_path), closing_lines, str(out), recorded_at=datetime(2026, 8, 20, 22, 0, tzinfo=timezone.utc))
    assert n == 1
    records = load_clv(str(out))
    assert records[0].player == "Player One"
    assert records[0].closing_price == 550  # the real 1+ HR line, not the 2.5 longshot tier


def test_record_closing_odds_keeps_only_the_latest_same_day_snapshot(tmp_path):
    picks_path = tmp_path / "picks" / "2026-08-20.jsonl"
    early = _edge("Player One", "batter_home_runs", price=700)
    late = _edge("Player One", "batter_home_runs", price=600)
    report_early = SlateReport(date(2026, 8, 20), [], [], [], [early], [], [])
    report_late = SlateReport(date(2026, 8, 20), [], [], [], [late], [], [])
    record_picks(report_early, str(picks_path), recorded_at=datetime(2026, 8, 20, 11, 0, tzinfo=timezone.utc))
    record_picks(report_late, str(picks_path), recorded_at=datetime(2026, 8, 20, 18, 0, tzinfo=timezone.utc))

    closing_lines = [
        PropLine(player="Player One", team=None, league="mlb", market="batter_home_runs", side="yes", line=0.5, odds=600, sportsbook="draftkings", event="Team A @ Team B"),
    ]
    out = tmp_path / "clv" / "2026-08-20.jsonl"
    record_closing_odds(str(picks_path), closing_lines, str(out), recorded_at=datetime(2026, 8, 20, 22, 0, tzinfo=timezone.utc))
    records = load_clv(str(out))
    assert len(records) == 1
    assert records[0].pick_price == 600  # the later snapshot's price, not the earlier +700
