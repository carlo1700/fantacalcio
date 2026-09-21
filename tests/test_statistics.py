from datetime import date, timedelta

from src.models.match import Match
from src.prediction.statistics import calculate_weighted_statistics


def test_statistics_are_normalized_by_minutes_and_weighted() -> None:
    matches = [
        Match(date(2026, 1, 1) + timedelta(days=index), "2026", minutes_played=90, rating=6)
        for index in range(10)
    ]
    matches[0] = Match(date(2026, 1, 1), "2026", minutes_played=90, rating=6, goals=1)

    stats = calculate_weighted_statistics(
        matches,
        current_season="2026",
        weights={"season": 1, "last_10": 0, "last_5": 0},
    )

    assert stats.availability == 1
    assert stats.minutes_if_playing == 90
    assert stats.goals_per_90 == 0.1
    assert stats.assists_per_90 == 0


def test_unplayed_matches_reduce_availability_but_not_minutes_if_playing() -> None:
    matches = [
        Match(date(2026, 1, 1), "2026", minutes_played=90, rating=6),
        Match(date(2026, 1, 2), "2026"),
    ]

    stats = calculate_weighted_statistics(
        matches,
        current_season="2026",
        weights={"season": 1, "last_10": 0, "last_5": 0},
    )

    assert stats.availability == 0.5
    assert stats.minutes_if_playing == 90


def test_statistics_cutoff_excludes_match_on_cutoff_date() -> None:
    matches = [
        Match(date(2026, 1, 1), "2026", minutes_played=90, rating=6),
        Match(date(2026, 1, 8), "2026", minutes_played=90, rating=10),
    ]

    stats = calculate_weighted_statistics(
        matches,
        current_season="2026",
        as_of=date(2026, 1, 8),
        weights={"season": 1, "last_10": 0, "last_5": 0},
    )

    assert stats.rating_if_playing == 6
