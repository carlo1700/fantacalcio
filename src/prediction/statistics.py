"""Deterministic, minute-normalized statistics."""

from dataclasses import dataclass
from datetime import date
from collections.abc import Sequence

from src.models.match import Match

DEFAULT_WEIGHTS = {"season": 0.5, "last_10": 0.3, "last_5": 0.2}


@dataclass(frozen=True, slots=True)
class WeightedStatistics:
    availability: float
    minutes_if_playing: float
    rating_if_playing: float
    goals_per_90: float
    assists_per_90: float
    penalties_scored_per_90: float
    penalties_missed_per_90: float
    yellow_cards_per_90: float
    red_cards_per_90: float
    own_goals_per_90: float

    def event_for_minutes(self, rate_per_90: float) -> float:
        return rate_per_90 * self.minutes_if_playing / 90


def _window_statistics(matches: Sequence[Match]) -> WeightedStatistics:
    if not matches:
        return WeightedStatistics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    appearances = [match for match in matches if match.minutes_played > 0]
    minutes = sum(match.minutes_played for match in matches)

    def per_90(attribute: str) -> float:
        return sum(getattr(match, attribute) for match in matches) / minutes * 90 if minutes else 0.0

    rated_appearances = [match for match in appearances if match.rating is not None]
    rating = (
        sum(match.rating for match in rated_appearances) / len(rated_appearances)
        if rated_appearances
        else 0.0
    )
    return WeightedStatistics(
        availability=len(appearances) / len(matches),
        minutes_if_playing=minutes / len(appearances) if appearances else 0.0,
        rating_if_playing=rating,
        goals_per_90=per_90("goals"),
        assists_per_90=per_90("assists"),
        penalties_scored_per_90=per_90("penalties_scored"),
        penalties_missed_per_90=per_90("penalties_missed"),
        yellow_cards_per_90=per_90("yellow_cards"),
        red_cards_per_90=per_90("red_cards"),
        own_goals_per_90=per_90("own_goals"),
    )


def calculate_weighted_statistics(
    matches: Sequence[Match],
    *,
    current_season: str,
    as_of: date | None = None,
    weights: dict[str, float] | None = None,
) -> WeightedStatistics:
    """Combine season, last-10 and last-5 statistics using explicit weights."""

    eligible = sorted(
        (match for match in matches if match.season == current_season and (as_of is None or match.date < as_of)),
        key=lambda match: match.date,
    )
    selected_weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if set(selected_weights) != set(DEFAULT_WEIGHTS) or any(value < 0 for value in selected_weights.values()):
        raise ValueError("weights must contain season, last_10 and last_5 with non-negative values")
    total_weight = sum(selected_weights.values())
    if total_weight <= 0:
        raise ValueError("weights must have a positive total")

    windows = {
        "season": eligible,
        "last_10": eligible[-10:],
        "last_5": eligible[-5:],
    }
    window_stats = {name: _window_statistics(window) for name, window in windows.items()}

    def weighted(attribute: str) -> float:
        return sum(
            selected_weights[name] * getattr(window_stats[name], attribute)
            for name in windows
        ) / total_weight

    return WeightedStatistics(
        *(weighted(attribute) for attribute in (
            "availability",
            "minutes_if_playing",
            "rating_if_playing",
            "goals_per_90",
            "assists_per_90",
            "penalties_scored_per_90",
            "penalties_missed_per_90",
            "yellow_cards_per_90",
            "red_cards_per_90",
            "own_goals_per_90",
        ))
    )
