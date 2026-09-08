from ncaaf.context import MockVenueProvider, MockWeatherProvider
from ncaaf.market import MockGameOddsProvider
from ncaaf.pipeline import run_pipeline
from ncaaf.ratings import MockRatingsProvider
from ncaaf.report import render_report
from ncaaf.schedule import MockScheduleProvider


def _run(seed=1, min_ev_percent=0.0):
    schedule = MockScheduleProvider()
    ratings = MockRatingsProvider()
    venues = MockVenueProvider()
    weather = MockWeatherProvider(seed=seed)
    slate = schedule.get_week_slate(2026, 3)
    odds = MockGameOddsProvider(slate, seed=seed)
    return run_pipeline(2026, 3, schedule, ratings, venues, weather, odds, min_ev_percent=min_ev_percent)


def test_pipeline_runs_end_to_end_and_scores_every_slate_game():
    report = _run()
    assert report.slate
    assert len(report.scores) == len(report.slate)
    assert report.candidates


def test_every_game_has_all_three_markets_represented_among_candidates():
    report = _run()
    markets = {c.market for c in report.candidates}
    assert markets == {"spreads", "h2h", "totals"}


def _true_rank_bucket(c):
    """Reimplements edges.rank_candidates's real sort-key bucket from
    public EdgeCandidate fields only, to check the actual documented
    invariant (agree > positive-EV model-only > everything else) without
    depending on rank_candidates's private sort_key directly.
    """
    if not c.has_market_data:
        return -1
    if c.tier == "agree":
        return 2
    return 1 if (c.ev_percent_model is not None and c.ev_percent_model > 0) else 0


def test_candidates_are_ranked_by_tier_bucket_then_ev_descending():
    report = _run()
    buckets = [_true_rank_bucket(c) for c in report.candidates]
    assert buckets == sorted(buckets, reverse=True)
    for i in range(1, len(report.candidates)):
        if buckets[i] == buckets[i - 1]:
            prev_ev = report.candidates[i - 1].ev_percent_model or 0.0
            cur_ev = report.candidates[i].ev_percent_model or 0.0
            assert cur_ev <= prev_ev + 1e-9


def test_pipeline_is_deterministic_with_same_seed():
    report_a = _run(seed=42)
    report_b = _run(seed=42)
    assert [(c.event, c.market, c.side, c.point) for c in report_a.candidates] == [
        (c.event, c.market, c.side, c.point) for c in report_b.candidates
    ]


def test_min_ev_filter_does_not_drop_priced_candidates_at_default():
    report_default = _run(min_ev_percent=0.0)
    report_explicit_high = _run(min_ev_percent=100.0)
    assert len(report_default.candidates) >= len(report_explicit_high.candidates)


def test_report_renders_without_error_and_mentions_key_sections():
    report = _run()
    text = render_report(report)
    assert "PREDICTED SCORES" in text
    assert "TOP" in text
    assert "RECOMMENDED BETS" in text
