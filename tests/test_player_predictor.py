from datetime import date

import pytest

from src.models.match import Match
from src.models.player import Player
from src.models.prediction import PlayerPrediction
from src.prediction.player_predictor import PlayerPredictor


def test_prediction_keeps_if_playing_and_team_match_metrics_separate() -> None:
    player = Player(
        "Giocatore sintetico",
        [
            Match(
                date(2026, 1, 1),
                "2026",
                minutes_played=90,
                rating=6,
                goals=1,
                assists=1,
                penalties_scored=1,
                penalties_missed=1,
                yellow_cards=1,
                red_cards=1,
                own_goals=1,
            ),
            Match(date(2026, 1, 8), "2026"),
        ],
    )

    prediction = PlayerPredictor(weights={"season": 1, "last_10": 0, "last_5": 0}).predict(
        player, current_season="2026"
    )

    assert prediction.availability_probability == 0.5
    assert prediction.expected_minutes_if_playing == 90
    assert prediction.expected_rating_if_playing == 6
    assert prediction.expected_goals_if_playing == 1
    assert prediction.expected_assists_if_playing == 1
    assert prediction.expected_fantasy_points_if_playing == pytest.approx(5.5)
    assert prediction.expected_fantasy_points_per_team_match == pytest.approx(2.75)
    assert prediction.expected_minutes_per_team_match == pytest.approx(45)


def test_fantasy_points_include_all_primary_bonus_and_malus_values() -> None:
    prediction = PlayerPrediction(
        availability_probability=1,
        expected_minutes_if_playing=90,
        expected_rating_if_playing=6,
        expected_goals_if_playing=1,
        expected_assists_if_playing=1,
        expected_penalties_scored_if_playing=1,
        expected_penalties_missed_if_playing=1,
        expected_yellow_cards_if_playing=1,
        expected_red_cards_if_playing=1,
        expected_own_goals_if_playing=1,
    )

    assert prediction.expected_fantasy_points_if_playing == pytest.approx(5.5)
