"""
Machine Learning utilities for Cleanlab Demo.

This module provides reusable utilities for common ML operations
including model building, evaluation, and data manipulation.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cleanlab_demo.core.constants import DEFAULT_CV_FOLDS, DEFAULT_MAX_ITER, DEFAULT_SEED


# =============================================================================
# Model Building
# =============================================================================


def build_classifier_pipeline(
    *,
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = DEFAULT_SEED,
    scale: bool = True,
) -> Pipeline:
    """
    Build a standard classification pipeline.

    Creates a pipeline with optional scaling and logistic regression.

    Args:
        max_iter: Maximum iterations for solver.
        seed: Random seed.
        scale: Whether to include StandardScaler.

    Returns:
        Scikit-learn Pipeline.

    Example:
        >>> pipeline = build_classifier_pipeline(max_iter=1000, seed=42)
        >>> pipeline.fit(X_train, y_train)
    """
    steps: list[tuple[str, Any]] = []

    if scale:
        steps.append(("scale", StandardScaler()))

    steps.append(
        (
            "model",
            LogisticRegression(
                max_iter=max_iter,
                solver="lbfgs",
                random_state=seed,
                n_jobs=1,
            ),
        )
    )

    return Pipeline(steps=steps)


def build_multilabel_pipeline(
    *,
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = DEFAULT_SEED,
    scale: bool = True,
) -> Pipeline:
    """
    Build a multilabel classification pipeline.

    Uses OneVsRestClassifier with logistic regression.

    Args:
        max_iter: Maximum iterations for solver.
        seed: Random seed.
        scale: Whether to include StandardScaler.

    Returns:
        Scikit-learn Pipeline for multilabel classification.
    """
    steps: list[tuple[str, Any]] = []

    if scale:
        steps.append(("scale", StandardScaler()))

    steps.append(
        (
            "model",
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=seed,
                )
            ),
        )
    )

    return Pipeline(steps=steps)


def build_regressor_pipeline(
    *,
    alpha: float = 1.0,
    seed: int = DEFAULT_SEED,
    scale: bool = True,
) -> Pipeline:
    """
    Build a standard regression pipeline.

    Creates a pipeline with optional scaling and ridge regression.

    Args:
        alpha: Regularization strength.
        seed: Random seed.
        scale: Whether to include StandardScaler.

    Returns:
        Scikit-learn Pipeline.
    """
    steps: list[tuple[str, Any]] = []

    if scale:
        steps.append(("scale", StandardScaler()))

    steps.append(
        (
            "model",
            Ridge(alpha=alpha, random_state=seed),
        )
    )

    return Pipeline(steps=steps)


# =============================================================================
# Cross-Validation Utilities
# =============================================================================


def get_cv_predicted_probs(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    model: Pipeline | BaseEstimator,
    *,
    cv_folds: int = DEFAULT_CV_FOLDS,
    seed: int = DEFAULT_SEED,
    stratified: bool = True,
) -> np.ndarray:
    """
    Get cross-validated predicted probabilities.

    Args:
        X: Feature matrix.
        y: Labels.
        model: Classifier with predict_proba method.
        cv_folds: Number of CV folds.
        seed: Random seed.
        stratified: Use stratified folds for classification.

    Returns:
        Array of predicted probabilities shape (n_samples, n_classes).
    """
    if stratified:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    pred_probs = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )

    return np.asarray(pred_probs, dtype=float)


def get_cv_predictions(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    model: Pipeline | BaseEstimator,
    *,
    cv_folds: int = DEFAULT_CV_FOLDS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """
    Get cross-validated predictions.

    Args:
        X: Feature matrix.
        y: Labels/targets.
        model: Model with predict method.
        cv_folds: Number of CV folds.
        seed: Random seed.

    Returns:
        Array of predictions.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict",
        n_jobs=1,
    )

    return np.asarray(predictions)


# =============================================================================
# Evaluation Metrics
# =============================================================================


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    *,
    average: str = "macro",
) -> dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (optional).
        average: Averaging method for F1 ('macro', 'micro', 'weighted').

    Returns:
        Dictionary of metric names to values.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        f"{average}_f1": float(f1_score(y_true, y_pred, average=average)),
    }

    if y_proba is not None:
        with suppress(ValueError):
            metrics["log_loss"] = float(log_loss(y_true, y_proba))

    return metrics


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute multilabel classification metrics.

    Args:
        y_true: True label matrix (n_samples, n_labels).
        y_pred: Predicted label matrix.

    Returns:
        Dictionary of metric names to values.
    """
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute standard regression metrics.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metric names to values.
    """
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# =============================================================================
# Data Utilities
# =============================================================================


def labels_to_list_format(y: np.ndarray) -> list[list[int]]:
    """
    Convert binary label matrix to list-of-lists format.

    Required by Cleanlab's multilabel functions.

    Args:
        y: Binary label matrix of shape (n_samples, n_labels).

    Returns:
        List where each element is a list of active label indices.

    Example:
        >>> y = np.array([[1, 0, 1], [0, 1, 0]])
        >>> labels_to_list_format(y)
        [[0, 2], [1]]
    """
    return [list(np.flatnonzero(row).astype(int)) for row in y]


def ensure_numpy_array(arr: Any) -> np.ndarray:
    """
    Ensure input is a numpy array.

    Args:
        arr: Input array-like.

    Returns:
        Numpy array.
    """
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, pd.Series):
        return np.asarray(arr.to_numpy())
    if isinstance(arr, pd.DataFrame):
        return np.asarray(arr.to_numpy())
    return np.asarray(arr)


def train_test_indices(
    n: int,
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate train/test split indices.

    Args:
        n: Total number of samples.
        test_size: Fraction for test set.
        seed: Random seed.

    Returns:
        Tuple of (train_indices, test_indices).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split_point = int(n * (1 - test_size))
    return indices[:split_point], indices[split_point:]
