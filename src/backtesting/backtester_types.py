"""Shared backtesting types kept separate to avoid import cycles."""

from dataclasses import dataclass
from datetime import date

from src.models.match import Match
from src.models.prediction import PlayerPrediction


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """One prediction made before a target match and its observed outcome."""

    player: str
    prediction_cutoff: date
    target_match: Match
    prediction: PlayerPrediction

    @property
    def availability_probability(self) -> float:
        return self.prediction.availability_probability

    @property
    def expected_minutes_if_playing(self) -> float:
        return self.prediction.expected_minutes_if_playing

    @property
    def expected_rating_if_playing(self) -> float:
        return self.prediction.expected_rating_if_playing

    @property
    def expected_goals_if_playing(self) -> float:
        return self.prediction.expected_goals_if_playing

    @property
    def expected_assists_if_playing(self) -> float:
        return self.prediction.expected_assists_if_playing

    @property
    def expected_fantasy_points_if_playing(self) -> float:
        return self.prediction.expected_fantasy_points_if_playing

    @property
    def expected_fantasy_points_per_team_match(self) -> float:
        return self.prediction.expected_fantasy_points_per_team_match

    @property
    def actual_rating(self) -> float | None:
        return self.target_match.rating

    @property
    def actual_minutes_played(self) -> int:
        return self.target_match.minutes_played

    @property
    def actual_availability(self) -> int:
        return int(self.target_match.minutes_played > 0)

    @property
    def actual_goals(self) -> int:
        return self.target_match.goals

    @property
    def actual_assists(self) -> int:
        return self.target_match.assists

    @property
    def actual_fantasy_points(self) -> float:
        if self.target_match.minutes_played == 0:
            return 0.0
        return (
            (self.target_match.rating or 0.0)
            + 3 * self.target_match.goals
            + self.target_match.assists
            + 3 * self.target_match.penalties_scored
            - 3 * self.target_match.penalties_missed
            - 0.5 * self.target_match.yellow_cards
            - self.target_match.red_cards
            - 3 * self.target_match.own_goals
        )
