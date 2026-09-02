from datetime import date

from mlb_props.context import MockParkWeatherProvider
from mlb_props.hot_streak import MockHotStreakProvider
from mlb_props.html_report import render_html_report
from mlb_props.market import MockMlbPropsOddsProvider
from mlb_props.matchup import MockMatchupProvider
from mlb_props.pipeline import run_pipeline
from mlb_props.schedule import MockScheduleProvider
from mlb_props.statcast import MockStatcastProvider


def _report(seed=5):
    schedule = MockScheduleProvider()
    slate = schedule.get_slate(date(2026, 8, 26))
    events_by_batter, all_batters = {}, []
    for game in slate:
        event = f"{game.away_team} @ {game.home_team}"
        for b in game.away_batters + game.home_batters:
            events_by_batter[b] = event
            all_batters.append(b)
    return run_pipeline(
        game_date=date(2026, 8, 26),
        schedule=schedule,
        statcast=MockStatcastProvider(seed=seed),
        matchup_provider=MockMatchupProvider(seed=seed),
        hot_streak=MockHotStreakProvider(seed=seed),
        park_weather=MockParkWeatherProvider(seed=seed),
        odds=MockMlbPropsOddsProvider(batters=all_batters, events_by_batter=events_by_batter, seed=seed),
    )


def test_html_report_is_well_formed_and_self_contained():
    html_text = render_html_report(_report(), is_mock=True)
    assert html_text.strip().startswith("<!doctype html>")
    assert html_text.count("<html") == 1
    assert html_text.count("</html>") == 1
    # self-contained: no external script/stylesheet hosts other than Google Fonts
    assert "fonts.googleapis.com" in html_text
    assert "src=\"http" not in html_text


def test_html_report_has_a_quick_nav_linking_every_section():
    html_text = render_html_report(_report())
    for anchor in ("#reco", "#props-hr", "#props-tb", "#props-hits", "#envs", "#hot", "#method"):
        assert f'href="{anchor}"' in html_text, f"quick-nav is missing a link to {anchor}"
    for section_id in ("reco", "props-hr", "props-tb", "props-hits", "envs", "hot", "method"):
        assert f'id="{section_id}"' in html_text, f"no element on the page actually has id={section_id!r}"


def test_html_report_reference_sections_are_collapsed_by_default():
    # Matchups/Who's-hot/Methodology are real reference material, not a
    # bet to place - they default closed so the page opens on what
    # matters (Recommended Bets + the prop tables), full detail always
    # one click away via native <details>, never deleted.
    html_text = render_html_report(_report())
    for section_id in ("envs", "hot", "method"):
        opening_tag_idx = html_text.index(f'<details class="section" id="{section_id}">')
        # the opening tag must not carry the "open" attribute
        tag_end = html_text.index(">", opening_tag_idx)
        assert "open" not in html_text[opening_tag_idx:tag_end]


def test_html_report_shows_sample_banner_only_in_mock_mode():
    mock_html = render_html_report(_report(), is_mock=True)
    assert "SAMPLE OUTPUT" in mock_html
    live_html = render_html_report(_report(), is_mock=False)
    assert "SAMPLE OUTPUT" not in live_html
    assert "LIVE" in live_html


def test_html_report_includes_every_hr_and_tb_pick():
    report = _report()
    html_text = render_html_report(report, top=50)
    for edge in report.hr_edges + report.tb_edges:
        assert edge.player in html_text


def test_html_report_escapes_player_names_safely():
    # A player name with markup-like characters must render escaped, not as
    # literal injected HTML. Checked directly against a crafted candidate
    # (rather than asserting no "<script>" appears anywhere on the page) -
    # the page now ships its own legitimate inline <script> for the
    # sort/filter/search controls, so that blanket assertion no longer
    # distinguishes "safely escaped" from "a real feature".
    from mlb_props.edges import EdgeCandidate
    from mlb_props.html_report import _prop_row

    malicious = EdgeCandidate(
        player="<script>alert(1)</script>",
        market="batter_home_runs",
        event="Team A @ Team B",
        model_score=50.0,
        model_prob=0.12,
        market_fair_prob=None,
        best_line=None,
        ev_percent_model=None,
        ev_percent_market=None,
        edge_vs_market=None,
        price_spread_percent=None,
        books_quoting=0,
        park="Test Park",
        wind_out_mph=0.0,
        temp_f=70.0,
        is_dome=False,
        weather_boost_pct=0.0,
    )
    row_html = _prop_row(malicious, None, "hr")
    assert "<script>alert(1)</script>" not in row_html
    assert "&lt;script&gt;" in row_html


def test_verdict_matches_the_real_recommended_bets_classification():
    from mlb_props.betting import MIN_EV_PERCENT_TO_RECOMMEND
    from mlb_props.html_report import _verdict

    # Exactly the same real bar/tier logic that decides Recommended Bets
    # membership - the verdict must never disagree with it.
    assert _verdict(False, "no_market", None) == ("NO PRICE YET", "verdict-none")
    assert _verdict(True, "model_only", None) == ("PASS", "verdict-pass")
    assert _verdict(True, "model_only", MIN_EV_PERCENT_TO_RECOMMEND - 0.1) == ("PASS", "verdict-pass")
    assert _verdict(True, "model_only", MIN_EV_PERCENT_TO_RECOMMEND) == ("SPECULATIVE", "verdict-speculative")
    assert _verdict(True, "agree", MIN_EV_PERCENT_TO_RECOMMEND) == ("STRONG BET", "verdict-strong")


def test_ev_bucket_uses_the_real_recommend_bar_as_its_boundary():
    from mlb_props.betting import MIN_EV_PERCENT_TO_RECOMMEND
    from mlb_props.html_report import _ev_bucket

    assert _ev_bucket(None) is None
    assert _ev_bucket(-15.0) == 0
    assert _ev_bucket(-5.0) == 1
    assert _ev_bucket(0.0) == 2
    assert _ev_bucket(MIN_EV_PERCENT_TO_RECOMMEND) == 3  # the exact real bar Recommended Bets uses
    assert _ev_bucket(15.0) == 4


def test_edge_bucket_buckets_by_real_sign_and_magnitude():
    from mlb_props.html_report import _edge_bucket

    assert _edge_bucket(None) is None
    assert _edge_bucket(-0.10) == 0
    assert _edge_bucket(-0.05) == 1
    assert _edge_bucket(-0.01) == 2
    assert _edge_bucket(0.0) == 3
    assert _edge_bucket(0.06) == 4


def test_quantile_cuts_splits_a_real_value_spread_into_quintiles():
    from mlb_props.html_report import _quantile_cuts, _rate_bucket

    values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    cuts = _quantile_cuts(values)
    assert len(cuts) == 4
    # the lowest real value sorts into the bottom bucket, the highest into the top
    assert _rate_bucket(values[0], cuts) == 0
    assert _rate_bucket(values[-1], cuts) == 4


def test_quantile_cuts_and_rate_bucket_are_none_safe():
    from mlb_props.html_report import _quantile_cuts, _rate_bucket

    assert _quantile_cuts([]) == []
    assert _rate_bucket(0.5, []) is None
    assert _rate_bucket(None, [0.1, 0.2, 0.3, 0.4]) is None


def test_chip_wraps_a_real_bucket_and_passes_through_otherwise():
    from mlb_props.html_report import _chip

    assert _chip("+5.0%", 4) == '<span class="rate-chip rate-chip-4">+5.0%</span>'
    assert _chip("n/a", None) == "n/a"


def test_html_report_prop_rows_show_colored_rating_chips():
    from mlb_props.betting import build_recommended_bets

    report = _report()
    strong, _ = build_recommended_bets(report)
    assert strong  # a real +EV pick, so the EV/edge chips below have a real value to render
    html_text = render_html_report(report)
    assert "rate-chip" in html_text


def test_html_report_prop_tables_show_a_verdict_badge():
    html_text = render_html_report(_report())
    assert 'data-k="verdict"' in html_text
    assert "verdict-strong" in html_text or "verdict-speculative" in html_text


def test_html_report_prop_table_headers_are_all_sortable():
    html_text = render_html_report(_report())
    for key in ("verdict", "player", "prob", "price", "book", "fair", "edge", "ev", "evmarket", "books", "weather", "l15", "season"):
        assert f'data-k="{key}"' in html_text, f"{key} column header is missing its sort arrow"


def test_html_report_prop_rows_carry_a_real_sortable_value_per_metric():
    from mlb_props.betting import build_recommended_bets

    report = _report()
    strong, _ = build_recommended_bets(report)
    assert strong  # a real +EV pick with market data, so every new attribute below has a real value to check
    html_text = render_html_report(report)
    for attr in ("data-verdict=", "data-fair=", "data-edge=", "data-evmarket=", "data-books=", "data-weather=", "data-l15=", "data-season="):
        assert attr in html_text, f"{attr} never appears on any prop row"


def test_html_report_recommended_bets_show_the_real_market_fair_value():
    from mlb_props.betting import RecommendedBet
    from mlb_props.html_report import _reco_row

    r = RecommendedBet(
        player="Player A", market="batter_home_runs", market_label="1+ HR", event="Team A @ Team B",
        tier="agree", model_prob=0.20, market_fair_prob=0.15, edge_vs_market=0.05, ev_percent_model=10.0,
        best_price=200, best_book="draftkings", books_quoting=2, units=1.0, full_kelly_percent=4.0, breakeven=400,
    )
    row_html = _reco_row(r)
    assert "15.0% market fair" in row_html
    assert "verdict-strong" in row_html


def _candidate(player, market, event, ev_percent_model=6.0, has_market=True, best_price=150, book="draftkings"):
    from mlb_props.edges import EdgeCandidate
    from odds_monitor.models import PropLine

    best_line = PropLine(player=player, team=None, league="mlb", market=market, side="yes", line=0.5, odds=best_price, sportsbook=book, event=event) if has_market else None
    return EdgeCandidate(
        player=player, market=market, event=event, model_score=70.0, model_prob=0.15, market_fair_prob=0.12,
        best_line=best_line, ev_percent_model=ev_percent_model if has_market else None, ev_percent_market=None,
        edge_vs_market=None, price_spread_percent=None, books_quoting=2 if has_market else 0, park="Test Park",
        wind_out_mph=0.0, temp_f=70.0, is_dome=False, weather_boost_pct=0.0,
    )


def test_other_props_html_lists_a_players_other_real_candidates():
    from mlb_props.html_report import _other_props_html

    hr = _candidate("Player A", "batter_home_runs", "Team A @ Team B")
    tb = _candidate("Player A", "batter_total_bases", "Team A @ Team B")
    lookup = {"player a": [hr, tb]}
    html_text = _other_props_html("Player A", "batter_home_runs", "Team A @ Team B", lookup)
    assert "Also scored tonight" in html_text
    assert "2+ TB" in html_text
    # Must not list itself back.
    assert html_text.count("Team A @ Team B") == 1


def test_other_props_html_is_empty_with_no_other_real_candidates():
    from mlb_props.html_report import _other_props_html

    hr = _candidate("Player A", "batter_home_runs", "Team A @ Team B")
    lookup = {"player a": [hr]}
    assert _other_props_html("Player A", "batter_home_runs", "Team A @ Team B", lookup) == ""
    assert _other_props_html("Player A", "batter_home_runs", "Team A @ Team B", None) == ""


def test_component_detail_html_includes_other_props_section_when_present():
    from mlb_props.html_report import _component_detail_html

    hr = _candidate("Player A", "batter_home_runs", "Team A @ Team B")
    hits = _candidate("Player A", "batter_hits", "Team A @ Team B")
    lookup = {"player a": [hr, hits]}
    html_text = _component_detail_html("batter_home_runs", {}, "Player A", "Team A @ Team B", lookup)
    # No real components recorded, but a real other-market candidate exists -
    # the toggle must still render for that reason alone.
    assert "expand-toggle" in html_text
    assert "Also scored tonight" in html_text


def test_html_report_shows_cross_market_props_for_a_real_player():
    # The mock fixture scores every batter across all three markets, so a
    # real cross-market "Also scored tonight" section should genuinely
    # appear somewhere on the page.
    html_text = render_html_report(_report())
    assert "Also scored tonight" in html_text


def test_fmt_start_time_et_converts_real_utc_to_us_eastern():
    from mlb_props.html_report import _fmt_start_time_et

    # 23:10 UTC on a summer date is EDT (UTC-4) - 7:10 PM ET.
    assert _fmt_start_time_et("2026-08-20T23:10:00Z") == "7:10 PM ET"


def test_fmt_start_time_et_is_tbd_for_missing_or_malformed():
    from mlb_props.html_report import _fmt_start_time_et

    assert _fmt_start_time_et(None) == "TBD"
    assert _fmt_start_time_et("") == "TBD"
    assert _fmt_start_time_et("not a real timestamp") == "TBD"


def test_html_report_env_cards_show_a_real_game_roster_and_start_time():
    from mlb_props.edges import EdgeCandidate
    from mlb_props.html_report import _env_card
    from mlb_props.pipeline import MatchupEnvironment
    from mlb_props.schedule import ProbableMatchup

    matchup = ProbableMatchup(
        away_team="Team A", home_team="Team B", venue="Test Park",
        away_pitcher="Pitcher A", home_pitcher="Pitcher B",
        game_time_utc="2026-08-20T23:10:00Z", status="Pre-Game",
    )
    env = MatchupEnvironment(
        matchup=matchup, park_hr_factor=100.0, weather_boost_pct=2.0,
        away_pitcher_vulnerability=None, home_pitcher_vulnerability=None, environment_score=60.0,
    )
    candidate = EdgeCandidate(
        player="Slugger One", market="batter_home_runs", event="Team A @ Team B", model_score=70.0,
        model_prob=0.15, market_fair_prob=0.12, best_line=None, ev_percent_model=6.0, ev_percent_market=None,
        edge_vs_market=None, price_spread_percent=None, books_quoting=0, park="Test Park", wind_out_mph=0.0,
        temp_f=70.0, is_dome=False, weather_boost_pct=2.0,
    )
    html_text = _env_card(env, 1, [candidate])
    assert "7:10 PM ET" in html_text
    assert "Pre-Game" in html_text
    assert "Slugger One" in html_text
    assert "1+ HR" in html_text
    assert "expand-toggle" in html_text


def test_html_report_recommended_bets_market_fair_is_honestly_na_when_absent():
    from mlb_props.betting import RecommendedBet
    from mlb_props.html_report import _reco_row

    r = RecommendedBet(
        player="Player B", market="batter_home_runs", market_label="1+ HR", event="Team A @ Team B",
        tier="model_only_single_sided", model_prob=0.20, market_fair_prob=None, edge_vs_market=None,
        ev_percent_model=10.0, best_price=200, best_book="draftkings", books_quoting=1, units=1.0,
        full_kelly_percent=4.0, breakeven=400,
    )
    row_html = _reco_row(r)
    assert "n/a market fair" in row_html
    assert "verdict-speculative" in row_html


def test_reco_group_caps_visible_rows_and_defers_the_rest():
    from mlb_props.betting import RecommendedBet
    from mlb_props.html_report import _RECO_VISIBLE_CAP, _reco_group

    def _bet(i):
        return RecommendedBet(
            player=f"Player {i}", market="batter_home_runs", market_label="1+ HR", event="Team A @ Team B",
            tier="agree", model_prob=0.20, market_fair_prob=0.15, edge_vs_market=0.05, ev_percent_model=10.0,
            best_price=200, best_book="draftkings", books_quoting=2, units=1.0, full_kelly_percent=4.0, breakeven=400,
        )

    recs = [_bet(i) for i in range(_RECO_VISIBLE_CAP + 3)]
    html_text = _reco_group("Strong plays", "hint", recs)
    for r in recs[:_RECO_VISIBLE_CAP]:
        assert r.player in html_text
    assert "Show 3 more" in html_text
    assert "reco-more" in html_text


def test_reco_group_shows_no_expander_at_or_under_the_cap():
    from mlb_props.betting import RecommendedBet
    from mlb_props.html_report import _RECO_VISIBLE_CAP, _reco_group

    def _bet(i):
        return RecommendedBet(
            player=f"Player {i}", market="batter_home_runs", market_label="1+ HR", event="Team A @ Team B",
            tier="agree", model_prob=0.20, market_fair_prob=0.15, edge_vs_market=0.05, ev_percent_model=10.0,
            best_price=200, best_book="draftkings", books_quoting=2, units=1.0, full_kelly_percent=4.0, breakeven=400,
        )

    recs = [_bet(i) for i in range(_RECO_VISIBLE_CAP)]
    html_text = _reco_group("Strong plays", "hint", recs)
    assert "reco-more" not in html_text
    for r in recs:
        assert r.player in html_text


def test_html_report_shows_recommended_bets_with_real_units():
    from mlb_props.betting import build_recommended_bets

    report = _report()
    strong, speculative = build_recommended_bets(report)
    assert strong  # the mock fixture reliably produces at least one real +EV pick
    html_text = render_html_report(report)
    assert "Tonight's Recommended Bets" in html_text
    assert "Strong plays" in html_text
    assert f"{strong[0].units:g}u" in html_text
    assert strong[0].player in html_text


def test_html_report_shows_the_breakeven_price_next_to_each_recommended_bet():
    from mlb_props.betting import build_recommended_bets

    report = _report()
    strong, _ = build_recommended_bets(report)
    assert strong
    html_text = render_html_report(report)
    assert f"beat {strong[0].breakeven:+d}" in html_text


def test_component_detail_html_sorts_by_real_weighted_contribution():
    from mlb_props.html_report import _component_detail_html

    # barrel_pct (weight .18) at 80/100 = 14.4 contribution; hard_hit_pct
    # (weight .13) at 90/100 = 11.7 - barrel_pct should still lead despite
    # the lower raw value, because contribution (value*weight) is what's
    # sorted, not the raw value or the weight alone.
    html_text = _component_detail_html("batter_home_runs", {"barrel_pct": 80.0, "hard_hit_pct": 90.0})
    assert "expand-toggle" in html_text
    assert "detail-panel" in html_text
    assert "Barrel %" in html_text
    assert html_text.index("Barrel %") < html_text.index("Hard Hit %")


def test_component_detail_html_is_empty_with_no_recorded_components():
    from mlb_props.html_report import _component_detail_html

    assert _component_detail_html("batter_home_runs", {}) == ""


def test_component_detail_html_is_empty_for_an_unrecognized_market():
    from mlb_props.html_report import _component_detail_html

    assert _component_detail_html("not_a_real_market", {"barrel_pct": 80.0}) == ""


def test_html_report_prop_tables_carry_a_real_why_drilldown():
    html_text = render_html_report(_report())
    assert "expand-toggle" in html_text
    assert "detail-panel" in html_text


def test_html_report_recommended_bets_carry_a_real_why_drilldown():
    from mlb_props.betting import RecommendedBet
    from mlb_props.html_report import _reco_row

    r = RecommendedBet(
        player="Player A", market="batter_home_runs", market_label="1+ HR", event="Team A @ Team B",
        tier="agree", model_prob=0.20, market_fair_prob=0.15, edge_vs_market=0.05, ev_percent_model=10.0,
        best_price=200, best_book="draftkings", books_quoting=2, units=1.0, full_kelly_percent=4.0,
        breakeven=400, components={"barrel_pct": 80.0, "hard_hit_pct": 50.0},
    )
    row_html = _reco_row(r)
    assert "expand-toggle" in row_html
    assert "Barrel %" in row_html


def test_html_report_recommended_bets_empty_state_is_honest_not_hidden():
    from datetime import date as _date

    from mlb_props.pipeline import SlateReport

    empty_report = SlateReport(
        game_date=_date(2026, 8, 26), slate=[], matchup_environments=[], hot_batters=[],
        hr_edges=[], tb_edges=[], hits_edges=[],
    )
    html_text = render_html_report(empty_report)
    assert "No real plays cleared the bar" in html_text


def test_html_report_escapes_malicious_player_name_in_recommended_bets():
    from mlb_props.betting import RecommendedBet
    from mlb_props.html_report import _reco_row

    malicious = RecommendedBet(
        player="<script>alert(1)</script>", market="batter_home_runs", market_label="1+ HR",
        event="Team A @ Team B", tier="agree", model_prob=0.20, market_fair_prob=0.15,
        edge_vs_market=0.05, ev_percent_model=10.0, best_price=200, best_book="draftkings",
        books_quoting=2, units=1.0, full_kelly_percent=4.0, breakeven=400,
    )
    row_html = _reco_row(malicious)
    assert "<script>alert(1)</script>" not in row_html
    assert "&lt;script&gt;" in row_html
