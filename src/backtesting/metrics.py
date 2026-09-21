"""Error metrics for player prediction backtests."""

from dataclasses import dataclass
from math import nan
from collections.abc import Sequence

from .backtester_types import BacktestResult


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Mean absolute errors calculated over backtest observations."""

    mae_rating: float
    mae_fantasy_points_if_playing: float
    mae_fantasy_points_per_team_match: float
    mae_goals_if_playing: float
    mae_goals_per_team_match: float
    mae_assists_if_playing: float
    mae_assists_per_team_match: float
    rating_observations: int
    fantasy_points_if_playing_observations: int
    fantasy_points_per_team_match_observations: int
    goals_if_playing_observations: int
    goals_per_team_match_observations: int
    assists_if_playing_observations: int
    assists_per_team_match_observations: int
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
    fantasy_points_if_playing_errors = [
        abs(result.expected_fantasy_points_if_playing - result.actual_fantasy_points)
        for result in results
        if result.target_match.minutes_played > 0 and result.actual_fantasy_points is not None
    ]
    fantasy_points_per_team_match_errors = [
        abs(result.expected_fantasy_points_per_team_match - result.actual_fantasy_points)
        for result in results
        if result.actual_fantasy_points is not None
    ]
    goals_if_playing_errors = [
        abs(result.expected_goals_if_playing - result.actual_goals)
        for result in results
        if result.target_match.minutes_played > 0
    ]
    goals_per_team_match_errors = [
        abs(result.prediction.availability_probability * result.expected_goals_if_playing - result.actual_goals)
        for result in results
    ]
    assists_if_playing_errors = [
        abs(result.expected_assists_if_playing - result.actual_assists)
        for result in results
        if result.target_match.minutes_played > 0
    ]
    assists_per_team_match_errors = [
        abs(result.prediction.availability_probability * result.expected_assists_if_playing - result.actual_assists)
        for result in results
    ]
    return BacktestMetrics(
        mae_rating=_mae(rating_errors),
        mae_fantasy_points_if_playing=_mae(fantasy_points_if_playing_errors),
        mae_fantasy_points_per_team_match=_mae(fantasy_points_per_team_match_errors),
        mae_goals_if_playing=_mae(goals_if_playing_errors),
        mae_goals_per_team_match=_mae(goals_per_team_match_errors),
        mae_assists_if_playing=_mae(assists_if_playing_errors),
        mae_assists_per_team_match=_mae(assists_per_team_match_errors),
        rating_observations=len(rating_errors),
        fantasy_points_if_playing_observations=len(fantasy_points_if_playing_errors),
        fantasy_points_per_team_match_observations=len(fantasy_points_per_team_match_errors),
        goals_if_playing_observations=len(goals_if_playing_errors),
        goals_per_team_match_observations=len(goals_per_team_match_errors),
        assists_if_playing_observations=len(assists_if_playing_errors),
        assists_per_team_match_observations=len(assists_per_team_match_errors),
        observations=len(results),
    )
