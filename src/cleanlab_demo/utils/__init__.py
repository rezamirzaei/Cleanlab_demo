"""
Utility modules for Cleanlab Demo.

This package provides reusable utilities:
- download: File download with validation
- fs: Filesystem operations
- ml: Machine learning utilities
"""

from cleanlab_demo.utils.ml import (
    build_classifier_pipeline,
    build_multilabel_pipeline,
    build_regressor_pipeline,
    compute_classification_metrics,
    compute_multilabel_metrics,
    compute_regression_metrics,
    ensure_numpy_array,
    get_cv_predicted_probs,
    get_cv_predictions,
    labels_to_list_format,
    train_test_indices,
)


__all__ = [
    "build_classifier_pipeline",
    "build_multilabel_pipeline",
    "build_regressor_pipeline",
    "compute_classification_metrics",
    "compute_multilabel_metrics",
    "compute_regression_metrics",
    "ensure_numpy_array",
    "get_cv_predicted_probs",
    "get_cv_predictions",
    "labels_to_list_format",
    "train_test_indices",
]
