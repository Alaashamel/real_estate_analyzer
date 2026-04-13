"""
Evaluation Module for Real Estate Price Prediction
Metrics and visualization for model performance
"""

from .metrics import (
    ModelEvaluator,
    calculate_metrics,
    compare_modalities,
    plot_predictions,
    plot_error_distribution
)

__all__ = [
    'ModelEvaluator',
    'calculate_metrics',
    'compare_modalities',
    'plot_predictions',
    'plot_error_distribution'
]