"""Analytical reporting for backtest comparisons."""

from dataclasses import dataclass
from math import isnan

from src.models.player import Player
from src.prediction.player_predictor import PlayerPredictor

from .backtester import Backtester
from .backtester_types import BacktestResult
from .metrics import BacktestMetrics, calculate_metrics


@dataclass(frozen=True, slots=True)
class AggregateBacktestMetrics:
    """Aggregate error indicators for a configuration."""

    availability_mae: float
    minutes_mae: float
    rating_mae: float
    fantasy_points_mae: float
    availability_observations: int
    minutes_observations: int
    rating_observations: int
    fantasy_points_observations: int


@dataclass(frozen=True, slots=True)
class BacktestReportRow:
    """Analytical report for a single weight configuration."""

    configuration: str
    results: tuple[BacktestResult, ...]
    metrics: BacktestMetrics
    aggregate_metrics: AggregateBacktestMetrics

    @property
    def prediction_count(self) -> int:
        return len(self.results)

    def as_row(self) -> dict[str, float | int | str]:
        return {
            "configuration": self.configuration,
            "predictions": self.prediction_count,
            "mae_rating": self.metrics.mae_rating,
            "mae_fantasy_points": self.metrics.mae_fantasy_points,
            "mae_goals": self.metrics.mae_goals,
            "mae_assists": self.metrics.mae_assists,
        }


def _mean_absolute_error(errors: list[float]) -> float:
    return sum(errors) / len(errors) if errors else float("nan")


def calculate_aggregate_metrics(results: tuple[BacktestResult, ...]) -> AggregateBacktestMetrics:
    availability_errors = [
        abs(result.availability_probability - result.actual_availability)
        for result in results
    ]
    minutes_errors = [
        abs(result.expected_minutes_if_playing - result.actual_minutes_played)
        for result in results
        if result.actual_minutes_played > 0
    ]
    rating_errors = [
        abs(result.expected_rating_if_playing - result.actual_rating)
        for result in results
        if result.actual_rating is not None and result.actual_minutes_played > 0
    ]
    fantasy_errors = [
        abs(result.expected_fantasy_points_per_team_match - result.actual_fantasy_points)
        for result in results
    ]

    return AggregateBacktestMetrics(
        availability_mae=_mean_absolute_error(availability_errors),
        minutes_mae=_mean_absolute_error(minutes_errors),
        rating_mae=_mean_absolute_error(rating_errors),
        fantasy_points_mae=_mean_absolute_error(fantasy_errors),
        availability_observations=len(availability_errors),
        minutes_observations=len(minutes_errors),
        rating_observations=len(rating_errors),
        fantasy_points_observations=len(fantasy_errors),
    )


def compare_weight_configurations_report(
    player: Player,
    *,
    current_season: str,
    cutoffs,
    weight_configurations: list[dict[str, float]],
) -> list[BacktestReportRow]:
    """Compare multiple weight configurations and return analytical rows."""

    rows: list[BacktestReportRow] = []
    for weights in weight_configurations:
        predictor = Backtester(PlayerPredictor(weights=weights))
        results = tuple(predictor.run(player, current_season=current_season, cutoffs=cutoffs))
        config_label = ", ".join(
            f"{name}={weights.get(name, 0.0):.2f}"
            for name in ("season", "last_10", "last_5")
        )
        rows.append(
            BacktestReportRow(
                configuration=config_label,
                results=results,
                metrics=calculate_metrics(results),
                aggregate_metrics=calculate_aggregate_metrics(results),
            )
        )
    return rows


def render_backtest_report(rows: list[BacktestReportRow]) -> str:
    """Render a compact table for the analytical report."""

    headers = [
        "configurazione",
        "predizioni",
        "MAE voto",
        "MAE fantapunti",
        "MAE gol",
        "MAE assist",
    ]
    lines = [
        " | ".join(headers),
        " | ".join(["-" * len(header) for header in headers]),
    ]

    for row in rows:
        values = [
            row.configuration,
            str(row.prediction_count),
            _format_value(row.metrics.mae_rating),
            _format_value(row.metrics.mae_fantasy_points),
            _format_value(row.metrics.mae_goals),
            _format_value(row.metrics.mae_assists),
        ]
        lines.append(" | ".join(values))
    return "\n".join(lines)


def _format_value(value: float) -> str:
    return "nan" if isnan(value) else f"{value:.3f}"
