from ncaaf.elo import BASE_RATING, GameResult, compute_elo_ratings, elo_diff_to_points


def test_winner_rating_increases_and_loser_decreases():
    games = [GameResult(season=2026, week=1, home_team="A", away_team="B", home_points=28, away_points=14)]
    ratings = compute_elo_ratings(games)
    assert ratings["A"] > BASE_RATING
    assert ratings["B"] < BASE_RATING


def test_bigger_margin_moves_rating_more():
    close = compute_elo_ratings([GameResult(season=2026, week=1, home_team="A", away_team="B", home_points=21, away_points=17)])
    blowout = compute_elo_ratings([GameResult(season=2026, week=1, home_team="A", away_team="B", home_points=49, away_points=3)])
    assert (blowout["A"] - BASE_RATING) > (close["A"] - BASE_RATING)


def test_upset_loss_lowers_a_big_favorites_rating():
    # Team A is already a big favorite (many prior blowout wins); losing to
    # a real underdog it's never faced before should cost real rating.
    warmup = [
        GameResult(season=2026, week=w, home_team="A", away_team=f"Cupcake{w}", home_points=45, away_points=7)
        for w in range(1, 6)
    ]
    ratings_before = compute_elo_ratings(warmup)
    assert ratings_before["A"] > BASE_RATING + 50

    upset_loss = warmup + [GameResult(season=2026, week=6, home_team="A", away_team="Underdog", home_points=10, away_points=13)]
    ratings_after_upset = compute_elo_ratings(upset_loss)
    assert ratings_after_upset["A"] < ratings_before["A"]


def test_neutral_site_removes_home_field_from_expected_score():
    home_game = compute_elo_ratings([GameResult(season=2026, week=1, home_team="A", away_team="B", home_points=24, away_points=24 - 1, neutral_site=False)])
    neutral_game = compute_elo_ratings([GameResult(season=2026, week=1, home_team="A", away_team="B", home_points=24, away_points=24 - 1, neutral_site=True)])
    # A one-point home win is a smaller real over-performance at a neutral
    # site (no home-field boost baked into the expectation), so it should
    # move A's rating up MORE at a neutral site than at home.
    assert (neutral_game["A"] - BASE_RATING) > (home_game["A"] - BASE_RATING)


def test_season_transition_regresses_ratings_toward_base():
    dominant = [
        GameResult(season=2025, week=w, home_team="A", away_team=f"Cupcake{w}", home_points=50, away_points=3)
        for w in range(1, 8)
    ]
    end_of_2025 = compute_elo_ratings(dominant)
    assert end_of_2025["A"] > BASE_RATING + 80

    # No 2026 games for A yet - compute_elo_ratings only regresses at the
    # start of a season that actually has a processed game in it, so add
    # one throwaway neutral game to trigger the transition.
    with_2026_opener = compute_elo_ratings(
        dominant + [GameResult(season=2026, week=1, home_team="Filler1", away_team="Filler2", home_points=20, away_points=20 - 1)]
    )
    assert BASE_RATING < with_2026_opener["A"] < end_of_2025["A"]


def test_elo_diff_to_points_is_symmetric_and_zero_at_equal_rating():
    assert elo_diff_to_points(0.0) == 0.0
    assert elo_diff_to_points(50.0) == -elo_diff_to_points(-50.0)
    assert elo_diff_to_points(25.0) > 0
