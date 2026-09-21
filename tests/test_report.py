from datetime import date
import math

import pytest

from src.backtesting.report import (
    compare_weight_configurations_report,
    render_backtest_report,
)
from src.models.match import Match
from src.models.player import Player


def test_report_calculates_aggregate_metrics_for_unplayed_matches() -> None:
    player = Player(
        "Giocatore sintetico",
        [
            Match(date(2026, 1, 1), "2026", minutes_played=90, rating=6, goals=1, assists=1),
            Match(date(2026, 1, 8), "2026", minutes_played=0),
            Match(date(2026, 1, 15), "2026", minutes_played=90, rating=7, goals=2, assists=1),
        ],
    )

    report = compare_weight_configurations_report(
        player,
        current_season="2026",
        cutoffs=[date(2026, 1, 8)],
        weight_configurations=[{"season": 1.0, "last_10": 0.0, "last_5": 0.0}],
    )

    row = report[0]
    assert row.prediction_count == 1
    assert row.metrics.observations == 1
    assert row.metrics.rating_observations == 0
    assert math.isnan(row.metrics.mae_rating)
    assert row.aggregate_metrics.availability_observations == 1
    assert row.aggregate_metrics.fantasy_points_observations == 1
    assert row.aggregate_metrics.availability_mae == pytest.approx(1.0)
    assert row.aggregate_metrics.fantasy_points_per_team_match_mae >= 0.0

    rendered = render_backtest_report(report)
    assert "configurazione" in rendered
    assert "MAE fantapunti" in rendered


def test_report_compares_multiple_weight_configurations() -> None:
    player = Player(
        "Giocatore sintetico",
        [
            Match(date(2026, 1, 1), "2026", minutes_played=90, rating=6, goals=1),
            Match(date(2026, 1, 8), "2026", minutes_played=90, rating=7, goals=1),
            Match(date(2026, 1, 15), "2026", minutes_played=90, rating=8, goals=2),
            Match(date(2026, 1, 22), "2026", minutes_played=90, rating=7, goals=1),
        ],
    )

    report = compare_weight_configurations_report(
        player,
        current_season="2026",
        cutoffs=[date(2026, 1, 8), date(2026, 1, 15)],
        weight_configurations=[
            {"season": 0.50, "last_10": 0.30, "last_5": 0.20},
            {"season": 0.60, "last_10": 0.25, "last_5": 0.15},
        ],
    )

    assert len(report) == 2
    assert all(row.prediction_count == 2 for row in report)
    assert [row.configuration for row in report] == [
        "season=0.50, last_10=0.30, last_5=0.20",
        "season=0.60, last_10=0.25, last_5=0.15",
    ]


def test_report_handles_missing_rating_without_crashing() -> None:
    player = Player(
        "Giocatore sintetico",
        [
            Match(date(2026, 1, 1), "2026", minutes_played=90, rating=5, goals=1),
            Match(date(2026, 1, 8), "2026", minutes_played=0),
        ],
    )

    report = compare_weight_configurations_report(
        player,
        current_season="2026",
        cutoffs=[date(2026, 1, 8)],
        weight_configurations=[{"season": 1.0, "last_10": 0.0, "last_5": 0.0}],
    )

    row = report[0]
    assert row.metrics.rating_observations == 0
    assert math.isnan(row.metrics.mae_rating)
    assert row.aggregate_metrics.rating_mae == pytest.approx(float("nan"), nan_ok=True)
