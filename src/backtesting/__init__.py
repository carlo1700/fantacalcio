"""Backtesting utilities for deterministic player predictions."""

from .backtester import Backtester, compare_weight_configurations
from .backtester_types import BacktestResult
from .metrics import BacktestMetrics, calculate_metrics
from .report import (
    AggregateBacktestMetrics,
    BacktestReportRow,
    calculate_aggregate_metrics,
    compare_weight_configurations_report,
    render_backtest_report,
)

__all__ = [
    "AggregateBacktestMetrics",
    "BacktestMetrics",
    "BacktestReportRow",
    "BacktestResult",
    "Backtester",
    "calculate_aggregate_metrics",
    "calculate_metrics",
    "compare_weight_configurations",
    "compare_weight_configurations_report",
    "render_backtest_report",
]
