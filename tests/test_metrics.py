from datetime import date

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
