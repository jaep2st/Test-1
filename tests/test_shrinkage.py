"""Covers hot_streak._shrunk_woba_stdev - the sample-size-aware correction
that stops a hot streak built on a handful of real plate appearances from
swinging z_score (and therefore the hot_streak component of every score in
scoring.py) exactly as hard as one built on a full 15-day sample.
"""

from mlb_props.hot_streak import REFERENCE_PA_FOR_15D_STDEV, WOBA_15D_STDEV, _shrunk_woba_stdev


def test_full_reference_sample_is_unaffected():
    # At or above the reference PA count, the effective stdev matches the
    # original unshrunk constant exactly - no behavior change for an
    # everyday player with a full real sample.
    assert _shrunk_woba_stdev(REFERENCE_PA_FOR_15D_STDEV) == WOBA_15D_STDEV
    assert _shrunk_woba_stdev(REFERENCE_PA_FOR_15D_STDEV * 3) == WOBA_15D_STDEV


def test_small_sample_is_shrunk_toward_zero_via_a_wider_stdev():
    small = _shrunk_woba_stdev(6)
    large = _shrunk_woba_stdev(REFERENCE_PA_FOR_15D_STDEV)
    assert small > large  # a wider stdev means the same raw wOBA gap produces a smaller |z|


def test_shrinkage_scales_with_inverse_sqrt_of_pa():
    # Standard-error scaling: half the reference PA should widen the
    # stdev by sqrt(2), not some arbitrary amount.
    half_reference = REFERENCE_PA_FOR_15D_STDEV // 2
    stdev = _shrunk_woba_stdev(half_reference)
    assert abs(stdev - WOBA_15D_STDEV * (REFERENCE_PA_FOR_15D_STDEV / half_reference) ** 0.5) < 1e-9


def test_zero_pa_produces_infinite_stdev_ie_zero_effective_z():
    assert _shrunk_woba_stdev(0) == float("inf")


def test_a_real_streak_actually_moves_the_score_less_on_a_tiny_sample():
    # The concrete real-world bug this fixes: a 2026-08-30-style 3-PA burst
    # (e.g. 2-for-3 with a HR) shouldn't read as "scorching" the same way a
    # real 20-for-58 stretch would.
    tiny_sample_z = 0.060 / _shrunk_woba_stdev(3)  # e.g. a +.060 wOBA gap on 3 PA
    full_sample_z = 0.060 / _shrunk_woba_stdev(REFERENCE_PA_FOR_15D_STDEV)  # same gap, full sample
    assert abs(tiny_sample_z) < abs(full_sample_z)
