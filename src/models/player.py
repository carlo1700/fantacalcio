"""Player domain model."""

from dataclasses import dataclass, field

from .match import Match


@dataclass(slots=True)
class Player:
    name: str
    matches: list[Match] = field(default_factory=list)
