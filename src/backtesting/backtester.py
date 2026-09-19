"""Walk-forward backtesting for PlayerPredictor."""

from collections.abc import Iterable, Sequence
from datetime import date

from src.models.match import Match
from src.models.player import Player
from src.models.prediction import PlayerPrediction
from src.prediction.player_predictor import PlayerPredictor
from .backtester_types import BacktestResult
from .metrics import BacktestMetrics, calculate_metrics


class Backtester:
    """Evaluate predictions using only the history available at each cutoff."""

    def __init__(self, predictor: PlayerPredictor | None = None) -> None:
        self.predictor = predictor or PlayerPredictor()

    def run(
        self,
        player: Player,
        *,
        current_season: str,
        cutoffs: Iterable[date],
    ) -> list[BacktestResult]:
        matches = sorted(
            (match for match in player.matches if match.season == current_season),
            key=lambda match: match.date,
        )
        results: list[BacktestResult] = []
        for cutoff in sorted(set(cutoffs)):
            target = next((match for match in matches if match.date >= cutoff), None)
            if target is None:
                continue
            history = [match for match in matches if match.date < cutoff]
            historical_player = Player(player.name, history)
            prediction = self.predictor.predict(
                historical_player,
                current_season=current_season,
            )
            results.append(BacktestResult(player.name, cutoff, target, prediction))
        return results


def compare_weight_configurations(
    player: Player,
    *,
    current_season: str,
    cutoffs: Iterable[date],
    weight_configurations: Sequence[dict[str, float]],
) -> dict[str, BacktestMetrics]:
    """Return metrics for every weight configuration without ranking them."""

    comparison: dict[str, BacktestMetrics] = {}
    for weights in weight_configurations:
        predictor = Backtester(PlayerPredictor(weights=weights))
        results = predictor.run(
            player,
            current_season=current_season,
            cutoffs=cutoffs,
        )
        label = ", ".join(
            f"{name}={weights[name]:.2f}"
            for name in ("season", "last_10", "last_5")
        )
        comparison[label] = calculate_metrics(results)
    return comparison
