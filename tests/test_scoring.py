from mlb_props.context import MockParkWeatherProvider
from mlb_props.hot_streak import MockHotStreakProvider
from mlb_props.matchup import MockMatchupProvider
from mlb_props.scoring import compute_hits_score, compute_hr_score, compute_total_bases_score
from mlb_props.statcast import BatterProfile, PitcherProfile


def _elite_batter():
    return BatterProfile(
        player="Elite Slugger",
        team="NYY",
        bats="R",
        pa=550,
        ab=480,
        hr=45,
        barrel_pct=20.0,
        hard_hit_pct=55.0,
        avg_exit_velo=94.5,
        avg_launch_angle=18.0,
        sweet_spot_pct=42.0,
        pull_air_pct=44.0,
        hr_fb_pct=30.0,
        iso=0.300,
        xwoba=0.420,
        xslg=0.630,
        k_pct=14.0,
    )


def _weak_batter():
    return BatterProfile(
        player="Weak Contact Hitter",
        team="MIA",
        bats="L",
        pa=400,
        ab=360,
        hr=4,
        barrel_pct=4.0,
        hard_hit_pct=30.0,
        avg_exit_velo=86.0,
        avg_launch_angle=8.0,
        sweet_spot_pct=24.0,
        pull_air_pct=14.0,
        hr_fb_pct=7.0,
        iso=0.110,
        xwoba=0.300,
        xslg=0.370,
        k_pct=29.0,
    )


def _bad_pitcher():
    return PitcherProfile(
        player="Gopher Ball Guy",
        team="COL",
        throws="R",
        ip=90.0,
        barrel_pct_allowed=11.0,
        hard_hit_pct_allowed=45.0,
        avg_exit_velo_allowed=91.0,
        hr_per_9=2.0,
        hr_fb_pct_allowed=17.0,
        xwoba_allowed=0.355,
        xslg_allowed=0.460,
        pitch_mix={"FF": 0.5, "SL": 0.3, "CH": 0.2},
        k_pct_allowed=16.0,
    )


def _good_pitcher():
    return PitcherProfile(
        player="Ace Righty",
        team="SD",
        throws="R",
        ip=140.0,
        barrel_pct_allowed=4.0,
        hard_hit_pct_allowed=29.0,
        avg_exit_velo_allowed=86.5,
        hr_per_9=0.7,
        hr_fb_pct_allowed=6.5,
        xwoba_allowed=0.285,
        xslg_allowed=0.365,
        pitch_mix={"FF": 0.4, "SL": 0.35, "CU": 0.25},
        k_pct_allowed=27.0,
    )


def test_elite_batter_vs_bad_pitcher_scores_higher_than_weak_batter_vs_good_pitcher():
    park = MockParkWeatherProvider(seed=1).get_context("Coors Field")
    heat = MockHotStreakProvider(seed=1).get_heat_index("Elite Slugger")
    matchup = MockMatchupProvider(seed=1).get_matchup("Elite Slugger", "R", "Gopher Ball Guy", "R", {})

    good_result = compute_hr_score(_elite_batter(), _bad_pitcher(), matchup, park, heat)

    park2 = MockParkWeatherProvider(seed=1).get_context("Oracle Park")
    heat2 = MockHotStreakProvider(seed=1).get_heat_index("Weak Contact Hitter")
    matchup2 = MockMatchupProvider(seed=1).get_matchup("Weak Contact Hitter", "L", "Ace Righty", "R", {})
    bad_result = compute_hr_score(_weak_batter(), _good_pitcher(), matchup2, park2, heat2)

    assert good_result.score > bad_result.score
    assert good_result.model_prob > bad_result.model_prob


def test_hr_model_prob_stays_within_calibrated_bounds():
    park = MockParkWeatherProvider(seed=2).get_context("Coors Field")
    heat = MockHotStreakProvider(seed=2).get_heat_index("Elite Slugger")
    matchup = MockMatchupProvider(seed=2).get_matchup("Elite Slugger", "R", "Gopher Ball Guy", "R", {})
    result = compute_hr_score(_elite_batter(), _bad_pitcher(), matchup, park, heat)
    assert 0.0 < result.model_prob <= 0.30


def test_total_bases_score_favors_higher_iso_and_slg():
    park = MockParkWeatherProvider(seed=3).get_context("Yankee Stadium")
    heat = MockHotStreakProvider(seed=3).get_heat_index("Elite Slugger")
    matchup = MockMatchupProvider(seed=3).get_matchup("Elite Slugger", "R", "Gopher Ball Guy", "R", {})
    good = compute_total_bases_score(_elite_batter(), _bad_pitcher(), matchup, park, heat)

    park2 = MockParkWeatherProvider(seed=3).get_context("Yankee Stadium")
    heat2 = MockHotStreakProvider(seed=3).get_heat_index("Weak Contact Hitter")
    matchup2 = MockMatchupProvider(seed=3).get_matchup("Weak Contact Hitter", "L", "Gopher Ball Guy", "R", {})
    bad = compute_total_bases_score(_weak_batter(), _bad_pitcher(), matchup2, park2, heat2)

    assert good.score > bad.score
    assert good.model_prob > bad.model_prob


def test_score_components_sum_to_the_overall_score():
    from mlb_props.scoring import HR_WEIGHTS

    park = MockParkWeatherProvider(seed=4).get_context("Fenway Park")
    heat = MockHotStreakProvider(seed=4).get_heat_index("Elite Slugger")
    matchup = MockMatchupProvider(seed=4).get_matchup("Elite Slugger", "R", "Gopher Ball Guy", "R", {})
    result = compute_hr_score(_elite_batter(), _bad_pitcher(), matchup, park, heat)
    recomputed = sum(result.components[k] * w for k, w in HR_WEIGHTS.items())
    assert round(recomputed, 1) == result.score


def test_hits_score_favors_higher_contact_quality_and_better_matchup():
    park = MockParkWeatherProvider(seed=5).get_context("Yankee Stadium")
    heat = MockHotStreakProvider(seed=5).get_heat_index("Elite Slugger")
    matchup = MockMatchupProvider(seed=5).get_matchup("Elite Slugger", "R", "Gopher Ball Guy", "R", {})
    good = compute_hits_score(_elite_batter(), _bad_pitcher(), matchup, park, heat)

    park2 = MockParkWeatherProvider(seed=5).get_context("Yankee Stadium")
    heat2 = MockHotStreakProvider(seed=5).get_heat_index("Weak Contact Hitter")
    matchup2 = MockMatchupProvider(seed=5).get_matchup("Weak Contact Hitter", "L", "Good Pitcher", "R", {})
    bad = compute_hits_score(_weak_batter(), _good_pitcher(), matchup2, park2, heat2)

    assert good.score > bad.score
    assert good.model_prob > bad.model_prob


def test_hits_model_prob_stays_within_calibrated_bounds():
    park = MockParkWeatherProvider(seed=6).get_context("Coors Field")
    heat = MockHotStreakProvider(seed=6).get_heat_index("Elite Slugger")
    matchup = MockMatchupProvider(seed=6).get_matchup("Elite Slugger", "R", "Gopher Ball Guy", "R", {})
    result = compute_hits_score(_elite_batter(), _bad_pitcher(), matchup, park, heat)
    # 1+ hits is a much higher-base-rate event than a HR or 2+ TB (a real
    # everyday hitter clears 50% most games) - the calibration anchors in
    # scoring.py reflect that, so the bounds here are intentionally much
    # higher than test_hr_model_prob_stays_within_calibrated_bounds' 0.30.
    assert 0.0 < result.model_prob <= 0.90


def test_hits_score_components_sum_to_the_overall_score():
    from mlb_props.scoring import HITS_WEIGHTS

    park = MockParkWeatherProvider(seed=7).get_context("Fenway Park")
    heat = MockHotStreakProvider(seed=7).get_heat_index("Elite Slugger")
    matchup = MockMatchupProvider(seed=7).get_matchup("Elite Slugger", "R", "Gopher Ball Guy", "R", {})
    result = compute_hits_score(_elite_batter(), _bad_pitcher(), matchup, park, heat)
    recomputed = sum(result.components[k] * w for k, w in HITS_WEIGHTS.items())
    assert round(recomputed, 1) == result.score


def test_hits_score_does_not_use_park_or_weather_as_scoring_inputs():
    # See scoring.py's note: park/weather are HR-oriented and have no real
    # relationship to a batter making contact for a hit, so they're carried
    # through on the result only for display, never weighted into the score.
    from mlb_props.scoring import HITS_WEIGHTS

    assert "park_factor" not in HITS_WEIGHTS
    assert "weather_boost" not in HITS_WEIGHTS

    heat = MockHotStreakProvider(seed=8).get_heat_index("Elite Slugger")
    matchup = MockMatchupProvider(seed=8).get_matchup("Elite Slugger", "R", "Gopher Ball Guy", "R", {})
    # Two genuinely different park/weather contexts (different seeds, so
    # different randomized wind/temp/park-factor) - if park/weather were
    # scoring inputs here, these would differ; they must not.
    park_a = MockParkWeatherProvider(seed=1).get_context("Coors Field")
    park_b = MockParkWeatherProvider(seed=99).get_context("Oracle Park")
    assert park_a.park_hr_factor != park_b.park_hr_factor  # sanity: contexts really do differ

    result_a = compute_hits_score(_elite_batter(), _bad_pitcher(), matchup, park_a, heat)
    result_b = compute_hits_score(_elite_batter(), _bad_pitcher(), matchup, park_b, heat)
    assert result_a.score == result_b.score


def test_hits_score_rewards_a_lower_batter_strikeout_rate_holding_everything_else_equal():
    # Isolates batter_k_pct: two otherwise-identical batters, only k_pct
    # differs. The low-strikeout one must score higher - this is the real
    # signal that used to be a documented blind spot (see scoring.py's
    # HITS_WEIGHTS note).
    from dataclasses import replace

    park = MockParkWeatherProvider(seed=9).get_context("Fenway Park")
    heat = MockHotStreakProvider(seed=9).get_heat_index("Elite Slugger")
    matchup = MockMatchupProvider(seed=9).get_matchup("Elite Slugger", "R", "Gopher Ball Guy", "R", {})

    contact_hitter = _elite_batter()
    whiff_prone = replace(contact_hitter, k_pct=32.0)
    assert contact_hitter.k_pct < whiff_prone.k_pct

    good = compute_hits_score(contact_hitter, _bad_pitcher(), matchup, park, heat)
    bad = compute_hits_score(whiff_prone, _bad_pitcher(), matchup, park, heat)

    assert good.score > bad.score
    assert good.components["batter_k_pct"] > bad.components["batter_k_pct"]


def test_hits_score_rewards_facing_a_lower_strikeout_pitcher_holding_everything_else_equal():
    # Isolates pitcher_k_pct_allowed the same way.
    from dataclasses import replace

    park = MockParkWeatherProvider(seed=10).get_context("Fenway Park")
    heat = MockHotStreakProvider(seed=10).get_heat_index("Elite Slugger")
    matchup = MockMatchupProvider(seed=10).get_matchup("Elite Slugger", "R", "Gopher Ball Guy", "R", {})

    weak_k_pitcher = _bad_pitcher()
    strikeout_pitcher = replace(weak_k_pitcher, k_pct_allowed=30.0)
    assert weak_k_pitcher.k_pct_allowed < strikeout_pitcher.k_pct_allowed

    good = compute_hits_score(_elite_batter(), weak_k_pitcher, matchup, park, heat)
    bad = compute_hits_score(_elite_batter(), strikeout_pitcher, matchup, park, heat)

    assert good.score > bad.score
    assert good.components["pitcher_k_pct_allowed"] > bad.components["pitcher_k_pct_allowed"]


def test_unenriched_k_pct_scores_as_neutral_not_as_a_maximum():
    # Regression test for a real bug caught against live data 2026-08-29:
    # BatterProfile.k_pct/PitcherProfile.k_pct_allowed default to None when
    # enrichment hasn't run or failed to resolve a real value (confirmed
    # live: happens for real players, not just in theory). A plain float
    # 0.0 default used to feed straight into the inverted normalize below
    # and come out as 100.0 - the *maximum* possible contact score - for
    # data we simply don't have. It must come out as a neutral 50.0 instead.
    from dataclasses import replace

    park = MockParkWeatherProvider(seed=11).get_context("Fenway Park")
    heat = MockHotStreakProvider(seed=11).get_heat_index("Elite Slugger")
    matchup = MockMatchupProvider(seed=11).get_matchup("Elite Slugger", "R", "Gopher Ball Guy", "R", {})

    unenriched_batter = replace(_elite_batter(), k_pct=None)
    unenriched_pitcher = replace(_bad_pitcher(), k_pct_allowed=None)
    assert unenriched_batter.k_pct is None
    assert unenriched_pitcher.k_pct_allowed is None

    result = compute_hits_score(unenriched_batter, unenriched_pitcher, matchup, park, heat)

    assert result.components["batter_k_pct"] == 50.0
    assert result.components["pitcher_k_pct_allowed"] == 50.0


def test_batter_profile_and_pitcher_profile_default_k_pct_to_none_not_zero():
    # The dataclass defaults themselves must stay None, not 0.0 - see the
    # regression test above for why 0.0 is actively dangerous here, not
    # just imprecise.
    from dataclasses import replace

    from mlb_props.statcast import BatterProfile, PitcherProfile

    batter = _elite_batter()
    default_batter = replace(batter, k_pct=BatterProfile.__dataclass_fields__["k_pct"].default)
    assert default_batter.k_pct is None

    pitcher = _bad_pitcher()
    default_pitcher = replace(pitcher, k_pct_allowed=PitcherProfile.__dataclass_fields__["k_pct_allowed"].default)
    assert default_pitcher.k_pct_allowed is None
