"""Prediction result models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlayerPrediction:
    """Conditional player projections and their team-match contribution."""

    availability_probability: float
    expected_minutes_if_playing: float
    expected_rating_if_playing: float
    expected_goals_if_playing: float
    expected_assists_if_playing: float
    expected_penalties_scored_if_playing: float
    expected_penalties_missed_if_playing: float
    expected_yellow_cards_if_playing: float
    expected_red_cards_if_playing: float
    expected_own_goals_if_playing: float

    @property
    def expected_fantasy_points_if_playing(self) -> float:
        """Fantasy points conditional on the player appearing."""

        return (
            self.expected_rating_if_playing
            + 3 * self.expected_goals_if_playing
            + self.expected_assists_if_playing
            + 3 * self.expected_penalties_scored_if_playing
            - 3 * self.expected_penalties_missed_if_playing
            - 0.5 * self.expected_yellow_cards_if_playing
            - self.expected_red_cards_if_playing
            - 3 * self.expected_own_goals_if_playing
        )

    @property
    def expected_fantasy_points_per_team_match(self) -> float:
        """Expected contribution averaged over every team match."""

        return self.availability_probability * self.expected_fantasy_points_if_playing
