"""Player prediction orchestration."""

from datetime import date

from src.models.player import Player
from src.models.prediction import PlayerPrediction
from .statistics import calculate_weighted_statistics


class PlayerPredictor:
    """Build deterministic projections from a player's historical matches."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights

    def predict(
        self,
        player: Player,
        *,
        current_season: str,
        as_of: date | None = None,
    ) -> PlayerPrediction:
        statistics = calculate_weighted_statistics(
            player.matches,
            current_season=current_season,
            as_of=as_of,
            weights=self.weights,
        )
        return PlayerPrediction(
            availability_probability=statistics.availability,
            expected_minutes_if_playing=statistics.minutes_if_playing,
            expected_rating_if_playing=statistics.rating_if_playing,
            expected_goals_if_playing=statistics.event_for_minutes(statistics.goals_per_90),
            expected_assists_if_playing=statistics.event_for_minutes(statistics.assists_per_90),
            expected_penalties_scored_if_playing=statistics.event_for_minutes(
                statistics.penalties_scored_per_90
            ),
            expected_penalties_missed_if_playing=statistics.event_for_minutes(
                statistics.penalties_missed_per_90
            ),
            expected_yellow_cards_if_playing=statistics.event_for_minutes(
                statistics.yellow_cards_per_90
            ),
            expected_red_cards_if_playing=statistics.event_for_minutes(statistics.red_cards_per_90),
            expected_own_goals_if_playing=statistics.event_for_minutes(
                statistics.own_goals_per_90
            ),
        )
