from datetime import date

import pytest

from src.backtesting.backtester_types import BacktestResult
from src.backtesting.metrics import calculate_metrics
from src.models.match import Match
from src.models.prediction import PlayerPrediction


def test_rating_mae_ignores_target_without_rating() -> None:
    prediction = PlayerPrediction(
        availability_probability=1,
        expected_minutes_if_playing=90,
        expected_rating_if_playing=6,
        expected_goals_if_playing=0,
        expected_assists_if_playing=0,
        expected_penalties_scored_if_playing=0,
        expected_penalties_missed_if_playing=0,
        expected_yellow_cards_if_playing=0,
        expected_red_cards_if_playing=0,
        expected_own_goals_if_playing=0,
    )
    result = BacktestResult(
        player="Sintetico",
        prediction_cutoff=date(2026, 1, 1),
        target_match=Match(date(2026, 1, 8), "2026"),
        prediction=prediction,
    )

    metrics = calculate_metrics([result])

    assert metrics.rating_observations == 0
    assert metrics.observations == 1


def _prediction(
    *,
    availability: float = 1,
    rating: float = 6,
    goals: float = 1,
    assists: float = 1,
) -> PlayerPrediction:
    return PlayerPrediction(
        availability_probability=availability,
        expected_minutes_if_playing=90,
        expected_rating_if_playing=rating,
        expected_goals_if_playing=goals,
        expected_assists_if_playing=assists,
        expected_penalties_scored_if_playing=0,
        expected_penalties_missed_if_playing=0,
        expected_yellow_cards_if_playing=0,
        expected_red_cards_if_playing=0,
        expected_own_goals_if_playing=0,
    )


def test_metrics_separate_conditional_and_team_match_errors() -> None:
    result = BacktestResult(
        player="Sintetico",
        prediction_cutoff=date(2026, 1, 1),
        target_match=Match(date(2026, 1, 8), "2026", minutes_played=90, rating=6, goals=0, assists=0),
        prediction=_prediction(availability=0.5),
    )

    metrics = calculate_metrics([result])

    assert metrics.mae_fantasy_points_if_playing == 4
    assert metrics.mae_fantasy_points_per_team_match == 1
    assert metrics.mae_goals_if_playing == 1
    assert metrics.mae_goals_per_team_match == pytest.approx(0.5)
    assert metrics.mae_assists_if_playing == 1
    assert metrics.mae_assists_per_team_match == pytest.approx(0.5)


def test_metrics_exclude_conditional_events_for_absent_player() -> None:
    result = BacktestResult(
        player="Sintetico",
        prediction_cutoff=date(2026, 1, 1),
        target_match=Match(date(2026, 1, 8), "2026"),
        prediction=_prediction(availability=0.5),
    )

    metrics = calculate_metrics([result])

    assert metrics.fantasy_points_if_playing_observations == 0
    assert metrics.goals_if_playing_observations == 0
    assert metrics.assists_if_playing_observations == 0
    assert metrics.mae_goals_per_team_match == pytest.approx(0.5)
    assert metrics.mae_assists_per_team_match == pytest.approx(0.5)


def test_missing_rating_makes_actual_fantasy_points_unavailable() -> None:
    result = BacktestResult(
        player="Sintetico",
        prediction_cutoff=date(2026, 1, 1),
        target_match=Match(date(2026, 1, 8), "2026", minutes_played=20, goals=1),
        prediction=_prediction(),
    )

    assert result.actual_fantasy_points is None
    metrics = calculate_metrics([result])
    assert metrics.fantasy_points_if_playing_observations == 0
    assert metrics.fantasy_points_per_team_match_observations == 0
