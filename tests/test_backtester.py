from datetime import date, timedelta

from src.backtesting.backtester import Backtester, compare_weight_configurations
from src.backtesting.metrics import calculate_metrics
from src.models.match import Match
from src.models.player import Player


def test_backtester_excludes_cutoff_and_future_matches() -> None:
    player = Player(
        "Giocatore sintetico",
        [
            Match(date(2026, 1, 1), "2026", minutes_played=90, rating=6),
            Match(date(2026, 1, 8), "2026", minutes_played=90, rating=8, goals=1),
            Match(date(2026, 1, 15), "2026", minutes_played=90, rating=10, goals=2),
        ],
    )

    results = Backtester().run(
        player,
        current_season="2026",
        cutoffs=[date(2026, 1, 8)],
    )

    assert len(results) == 1
    result = results[0]
    assert result.target_match.date == date(2026, 1, 8)
    assert result.prediction.expected_rating_if_playing == 6
    assert result.prediction.expected_goals_if_playing == 0
    assert result.actual_rating == 8
    assert result.actual_goals == 1
    assert result.actual_minutes_played == 90
    assert result.actual_availability == 1


def test_backtester_repeats_for_multiple_cutoffs_and_skips_missing_targets() -> None:
    player = Player(
        "Giocatore sintetico",
        [
            Match(date(2026, 1, 1), "2026", minutes_played=90, rating=6),
            Match(date(2026, 1, 8), "2026", minutes_played=90, rating=7),
        ],
    )

    results = Backtester().run(
        player,
        current_season="2026",
        cutoffs=[date(2026, 1, 1), date(2026, 1, 8), date(2026, 1, 20)],
    )

    assert [result.target_match.date for result in results] == [
        date(2026, 1, 1),
        date(2026, 1, 8),
    ]


def test_weight_configuration_comparison_returns_metrics_for_each_configuration() -> None:
    player = Player(
        "Giocatore sintetico",
        [
            Match(date(2026, 1, 1) + timedelta(days=7 * index), "2026", minutes_played=90, rating=6 + index)
            for index in range(4)
        ],
    )
    configurations = [
        {"season": 0.50, "last_10": 0.30, "last_5": 0.20},
        {"season": 0.60, "last_10": 0.25, "last_5": 0.15},
        {"season": 0.40, "last_10": 0.35, "last_5": 0.25},
    ]

    comparison = compare_weight_configurations(
        player,
        current_season="2026",
        cutoffs=[date(2026, 1, 8), date(2026, 1, 15)],
        weight_configurations=configurations,
    )

    assert len(comparison) == 3
    assert all(metrics.observations == 2 for metrics in comparison.values())


def test_backtest_result_exposes_team_match_fantasy_prediction() -> None:
    player = Player(
        "Giocatore sintetico",
        [
            Match(date(2026, 1, 1), "2026", minutes_played=90, rating=6),
            Match(date(2026, 1, 8), "2026", minutes_played=90, rating=7, assists=1),
        ],
    )
    result = Backtester().run(
        player, current_season="2026", cutoffs=[date(2026, 1, 8)]
    )[0]

    assert result.expected_fantasy_points_per_team_match == result.prediction.expected_fantasy_points_per_team_match


def test_metrics_can_be_calculated_from_backtest_results() -> None:
    player = Player(
        "Giocatore sintetico",
        [
            Match(date(2026, 1, 1), "2026", minutes_played=90, rating=6),
            Match(date(2026, 1, 8), "2026", minutes_played=90, rating=8, goals=1, assists=1),
        ],
    )
    results = Backtester().run(
        player, current_season="2026", cutoffs=[date(2026, 1, 8)]
    )

    metrics = calculate_metrics(results)
    assert metrics.observations == 1
    assert metrics.rating_observations == 1
    assert metrics.mae_rating == 2
    assert metrics.mae_goals == 1
    assert metrics.mae_assists == 1
