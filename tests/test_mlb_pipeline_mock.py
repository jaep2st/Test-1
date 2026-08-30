from datetime import date

from mlb_props.ballparkpal import BallparkPalProvider, HitterParkFactor
from mlb_props.context import MockParkWeatherProvider
from mlb_props.hot_streak import MockHotStreakProvider
from mlb_props.market import MockMlbPropsOddsProvider
from mlb_props.matchup import MockMatchupProvider
from mlb_props.pipeline import run_pipeline
from mlb_props.report import render_report
from mlb_props.schedule import MockScheduleProvider
from mlb_props.statcast import MockStatcastProvider


def _run(seed=1, min_ev_percent=0.0, ballparkpal=None):
    schedule = MockScheduleProvider()
    slate = schedule.get_slate(date(2026, 8, 26))
    events_by_batter = {}
    all_batters = []
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
        min_ev_percent=min_ev_percent,
        ballparkpal=ballparkpal,
    )


def test_pipeline_runs_end_to_end_and_scores_every_slate_batter():
    report = _run()
    assert report.slate
    assert report.matchup_environments
    assert report.hot_batters
    assert report.hr_edges
    assert report.tb_edges
    assert report.hits_edges


def test_matchup_environments_are_sorted_best_first():
    report = _run()
    scores = [e.environment_score for e in report.matchup_environments]
    assert scores == sorted(scores, reverse=True)


def test_hot_batters_are_sorted_hottest_first():
    report = _run()
    z_scores = [h.z_score for h in report.hot_batters]
    assert z_scores == sorted(z_scores, reverse=True)


def test_priced_edges_are_not_dropped_at_the_default_min_ev():
    # Confirmed live (2026-08-29): at the documented "0 = show all" default,
    # a real priced candidate with negative model EV used to vanish from
    # the report entirely (not shown priced, not shown model-only) instead
    # of just being ranked lower - see rank_candidates' docstring. The mock
    # odds provider's outlier-book mechanic (see market.py) guarantees a
    # realistic spread of EVs across many candidates/seeds, so summing
    # priced-edge counts across a few seeds reliably includes at least one
    # negative-EV real price if the bug ever regresses.
    total_priced = 0
    negative_ev_present = False
    for seed in range(1, 8):
        report = _run(seed=seed, min_ev_percent=0.0)
        for edge in report.hr_edges + report.tb_edges + report.hits_edges:
            if edge.has_market_data:
                total_priced += 1
                if edge.ev_percent_model < 0.0:
                    negative_ev_present = True
    assert total_priced > 0
    assert negative_ev_present, "expected at least one negative-EV priced candidate across seeds - test may need a wider seed range"


def test_explicit_positive_min_ev_still_filters_priced_candidates():
    # The filtering feature itself still works when a caller opts in with a
    # real threshold above 0 - only the buggy default (0, "show all") was
    # supposed to never drop anything.
    report = _run(seed=1, min_ev_percent=0.0)
    unfiltered_priced = [e for e in report.hr_edges + report.tb_edges + report.hits_edges if e.has_market_data]
    assert any(e.ev_percent_model < 50.0 for e in unfiltered_priced)  # sanity: not every candidate is a huge outlier

    report_filtered = _run(seed=1, min_ev_percent=50.0)
    filtered_priced = [e for e in report_filtered.hr_edges + report_filtered.tb_edges + report_filtered.hits_edges if e.has_market_data]
    assert all(e.ev_percent_model >= 50.0 for e in filtered_priced)
    assert len(filtered_priced) < len(unfiltered_priced)


def test_report_renders_without_error_and_mentions_key_sections():
    report = _run()
    text = render_report(report)
    assert "Best HR Matchups" in text
    assert "Who's Hot" in text
    assert "Best Home Run Props" in text
    assert "Best 2+ Total Bases Props" in text
    assert "Best 1+ Hits Props" in text


def test_pipeline_is_deterministic_with_same_seed():
    report_a = _run(seed=42)
    report_b = _run(seed=42)
    assert [e.player for e in report_a.hr_edges] == [e.player for e in report_b.hr_edges]


class _FakeBallparkPalProvider(BallparkPalProvider):
    """Returns a real, known factor for exactly one player - lets a test
    prove the override actually reaches the final report instead of just
    trusting the wiring.
    """

    def __init__(self, player: str, factor: HitterParkFactor):
        self._key = player.strip().lower()
        self._factor = factor

    def get_hitter_park_factor(self, player, game_date):
        return self._factor if player.strip().lower() == self._key else None


def test_ballparkpal_factor_overrides_weather_boost_for_that_batter_only():
    # Aaron Judge is a real MockScheduleProvider batter (New York Yankees @
    # Baltimore Orioles). A distinctive weather value (10.0%) that couldn't
    # plausibly come from MockParkWeatherProvider's own randomized range by
    # coincidence proves the override actually took effect on his result.
    override = HitterParkFactor(player_name="Aaron Judge", home_runs=1.20, home_runs_stadium=1.10, home_runs_weather=0.10)
    report = _run(seed=1, ballparkpal=_FakeBallparkPalProvider("Aaron Judge", override))

    judge_edges = [e for e in report.hr_edges + report.tb_edges if e.player == "Aaron Judge"]
    assert judge_edges
    for edge in judge_edges:
        assert edge.weather_boost_pct == 10.0

    # Nobody else on the slate should be affected - the fake provider only
    # has data for Aaron Judge, and everyone else must fall back to
    # MockParkWeatherProvider's own (seeded, non-10.0) value unchanged.
    others = [e for e in report.hr_edges + report.tb_edges if e.player != "Aaron Judge"]
    assert others
    assert any(e.weather_boost_pct != 10.0 for e in others)


def test_no_ballparkpal_provider_leaves_scoring_unchanged():
    # Default (ballparkpal=None) must behave identically to the pipeline
    # before this integration existed - same seed, same result.
    with_default = _run(seed=7)
    without_arg = run_pipeline(
        game_date=date(2026, 8, 26),
        schedule=MockScheduleProvider(),
        statcast=MockStatcastProvider(seed=7),
        matchup_provider=MockMatchupProvider(seed=7),
        hot_streak=MockHotStreakProvider(seed=7),
        park_weather=MockParkWeatherProvider(seed=7),
        odds=MockMlbPropsOddsProvider(
            batters=[b for g in MockScheduleProvider().get_slate(date(2026, 8, 26)) for b in g.away_batters + g.home_batters],
            events_by_batter={
                b: f"{g.away_team} @ {g.home_team}"
                for g in MockScheduleProvider().get_slate(date(2026, 8, 26))
                for b in g.away_batters + g.home_batters
            },
            seed=7,
        ),
    )
    assert [e.weather_boost_pct for e in with_default.hr_edges] == [e.weather_boost_pct for e in without_arg.hr_edges]


