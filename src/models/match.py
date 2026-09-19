"""Match-level data models."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Match:
    """Statistics recorded for one player in one match."""

    date: date
    season: str
    minutes_played: int = 0
    rating: float | None = None
    goals: int = 0
    assists: int = 0
    penalties_scored: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    own_goals: int = 0

    def __post_init__(self) -> None:
        if self.minutes_played < 0:
            raise ValueError("minutes_played cannot be negative")
        for field_name in (
            "goals",
            "assists",
            "penalties_scored",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "own_goals",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.rating is not None and self.rating < 0:
            raise ValueError("rating cannot be negative")
