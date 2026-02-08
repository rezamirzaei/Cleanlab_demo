"""
Type definitions for Cleanlab Demo.

This module provides type aliases and protocols for better type safety
and code documentation throughout the application.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt


# =============================================================================
# Type Aliases
# =============================================================================

# Array types
ArrayLike = npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]]
"""Numpy array of floats or integers."""

Labels = npt.NDArray[np.integer[Any]]
"""1D array of integer labels."""

Predictions = npt.NDArray[np.integer[Any]] | npt.NDArray[np.floating[Any]]
"""1D array of predictions (integer for classification, float for regression)."""

ProbabilityMatrix = npt.NDArray[np.floating[Any]]
"""2D array of class probabilities with shape (n_samples, n_classes)."""

DataFrameLike = "pd.DataFrame"
"""Pandas DataFrame type alias."""

# Generic type variables
TConfig = TypeVar("TConfig", bound="BaseConfig")
"""Type variable for configuration classes."""

TResult = TypeVar("TResult", bound="BaseResult")
"""Type variable for result classes."""

TData = TypeVar("TData")
"""Type variable for generic data types."""


# =============================================================================
# Protocols (Structural Subtyping)
# =============================================================================


@runtime_checkable
class BaseConfig(Protocol):
    """Protocol for configuration objects."""

    seed: int

    def model_dump(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        ...


@runtime_checkable
class BaseResult(Protocol):
    """Protocol for result objects."""

    timestamp: str

    def model_dump(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        ...

    def model_dump_json(self, **kwargs: Any) -> str:
        """Serialize to JSON string."""
        ...


@runtime_checkable
class DataProvider(Protocol[TConfig]):
    """
    Protocol for data providers.

    Data providers are responsible for loading and preparing datasets
    for specific tasks. They encapsulate data access logic and provide
    a consistent interface for tasks to consume data.
    """

    @property
    def name(self) -> str:
        """Human-readable name of the dataset."""
        ...

    def load(self, seed: int, **kwargs: Any) -> tuple[Any, Any]:
        """
        Load the dataset.

        Args:
            seed: Random seed for reproducibility.
            **kwargs: Additional loading options.

        Returns:
            Tuple of (features, labels).
        """
        ...


@runtime_checkable
class Task(Protocol[TConfig, TResult]):
    """
    Protocol for task implementations.

    Tasks encapsulate the logic for running specific ML experiments
    with Cleanlab. They receive data from providers and return
    structured results.
    """

    def run(self, config: TConfig) -> TResult:
        """
        Execute the task.

        Args:
            config: Task configuration.

        Returns:
            Task result with metrics and analysis.
        """
        ...


@runtime_checkable
class Evaluator(Protocol):
    """
    Protocol for model evaluators.

    Evaluators compute metrics for model predictions.
    """

    def evaluate(
        self, y_true: ArrayLike, y_pred: ArrayLike, **kwargs: Any
    ) -> dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Model predictions.
            **kwargs: Additional arguments (e.g., y_proba for classification).

        Returns:
            Dictionary of metric names to values.
        """
        ...


@runtime_checkable
class NoiseInjector(Protocol):
    """
    Protocol for noise injection strategies.

    Noise injectors add synthetic label noise for evaluation purposes.
    """

    def inject(
        self, labels: ArrayLike, *, frac: float, seed: int
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Inject noise into labels.

        Args:
            labels: Original labels.
            frac: Fraction of labels to corrupt.
            seed: Random seed.

        Returns:
            Tuple of (noisy_labels, corrupted_indices).
        """
        ...


# =============================================================================
# Result Type Helpers
# =============================================================================


class MetricsDict(dict[str, float]):
    """
    Dictionary subclass for storing metrics.

    Provides additional convenience methods for metric aggregation.
    """

    def mean(self) -> float:
        """Compute mean of all metrics."""
        if not self:
            return 0.0
        return sum(self.values()) / len(self)

    def to_percentage(self) -> MetricsDict:
        """Convert all values to percentages (multiply by 100)."""
        return MetricsDict({k: v * 100 for k, v in self.items()})

