"""Error metrics for player prediction backtests."""

from dataclasses import dataclass
from math import nan
from collections.abc import Sequence

from .backtester_types import BacktestResult


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Mean absolute errors calculated over backtest observations."""

    mae_rating: float
    mae_fantasy_points: float
    mae_goals: float
    mae_assists: float
    rating_observations: int
    observations: int


def _mae(errors: list[float]) -> float:
    return sum(errors) / len(errors) if errors else nan


def calculate_metrics(results: Sequence[BacktestResult]) -> BacktestMetrics:
    """Calculate MAE without using any observations beyond each target match."""

    rating_errors = [
        abs(result.prediction.expected_rating_if_playing - result.target_match.rating)
        for result in results
        if result.target_match.rating is not None and result.target_match.minutes_played > 0
    ]
    return BacktestMetrics(
        mae_rating=_mae(rating_errors),
        mae_fantasy_points=_mae([
            abs(
                result.prediction.expected_fantasy_points_if_playing
                - result.actual_fantasy_points
            )
            for result in results
        ]),
        mae_goals=_mae([
            abs(result.prediction.expected_goals_if_playing - result.target_match.goals)
            for result in results
        ]),
        mae_assists=_mae([
            abs(result.prediction.expected_assists_if_playing - result.target_match.assists)
            for result in results
        ]),
        rating_observations=len(rating_errors),
        observations=len(results),
    )
