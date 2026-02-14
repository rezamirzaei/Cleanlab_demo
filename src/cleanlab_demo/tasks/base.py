"""
Base classes for Cleanlab Demo tasks.

This module provides abstract base classes and common functionality
for all task implementations. Tasks follow a consistent pattern:

1. A DataProvider loads and prepares data
2. A Task orchestrates the experiment workflow
3. A Result captures metrics and analysis

Example:
    >>> class MyTask(BaseTask[MyConfig, MyResult]):
    ...     def run(self, config: MyConfig) -> MyResult:
    ...         data = self.data_provider.load(config.seed)
    ...         # ... task logic ...
    ...         return MyResult(...)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from cleanlab_demo.core.constants import (
    DEFAULT_NOISE_FRAC,
    DEFAULT_SEED,
)


# Type variables for generic task definitions
TConfig = TypeVar("TConfig", bound="DemoConfig")
TResult = TypeVar("TResult", bound="DemoResult")
TData = TypeVar("TData")


class DemoConfig(BaseModel):
    """
    Base configuration shared by all tasks.

    This class provides common configuration options that apply to
    all Cleanlab demo tasks. Subclasses can extend this with
    task-specific options.

    Attributes:
        seed: Random seed for reproducibility. Must be between 0 and 1,000,000.

    Example:
        >>> config = DemoConfig(seed=123)
        >>> config.seed
        123
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    seed: int = Field(
        default=DEFAULT_SEED,
        ge=0,
        le=1_000_000,
        description="Random seed for reproducibility",
    )


class DemoResult(BaseModel):
    """
    Base result shared by all tasks.

    This class provides common result fields that apply to all
    Cleanlab demo tasks. Subclasses extend this with task-specific
    metrics and analysis.

    Attributes:
        timestamp: ISO format timestamp of when the result was created.

    Example:
        >>> result = DemoResult()
        >>> result.timestamp  # e.g., "2026-02-08T12:00:00+00:00"
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO format timestamp of result creation",
    )


class BaseDataProvider(ABC, Generic[TData]):
    """
    Abstract base class for data providers.

    Data providers encapsulate the logic for loading and preparing
    datasets for specific tasks. They provide a consistent interface
    that tasks can rely on.

    Type Parameters:
        TData: The type of data returned by load() (e.g., tuple of arrays).

    Example:
        >>> class MyProvider(BaseDataProvider[tuple[pd.DataFrame, np.ndarray]]):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_dataset"
        ...
        ...     def load(self, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
        ...         return X, y
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name of the dataset.

        Returns:
            Dataset name for logging and reporting.
        """
        ...

    @abstractmethod
    def load(self, seed: int, **kwargs: Any) -> TData:
        """
        Load and return the dataset.

        Args:
            seed: Random seed for reproducible sampling/shuffling.
            **kwargs: Additional provider-specific options.

        Returns:
            Dataset in provider-specific format.

        Raises:
            DataLoadError: If loading fails.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class BaseTask(ABC, Generic[TConfig, TResult]):
    """
    Abstract base class for all Cleanlab demo tasks.

    Tasks orchestrate the complete workflow of an ML experiment:
    1. Load data from provider
    2. Prepare/preprocess data
    3. Train models (baseline and variants)
    4. Apply Cleanlab analysis
    5. Compute and return metrics

    Type Parameters:
        TConfig: Configuration type for this task.
        TResult: Result type returned by this task.

    Attributes:
        data_provider: Provider that supplies data for this task.

    Example:
        >>> class MyTask(BaseTask[MyConfig, MyResult]):
        ...     def __init__(self, provider: MyProvider):
        ...         self.data_provider = provider
        ...
        ...     def run(self, config: MyConfig) -> MyResult:
        ...         X, y = self.data_provider.load(config.seed)
        ...         # ... training and analysis ...
        ...         return MyResult(metrics=metrics)
    """

    data_provider: BaseDataProvider[Any]

    @abstractmethod
    def run(self, config: TConfig) -> TResult:
        """
        Execute the task and return results.

        This method contains the main logic of the task:
        - Data loading and preprocessing
        - Model training and evaluation
        - Cleanlab analysis
        - Metric computation

        Args:
            config: Task-specific configuration.

        Returns:
            Result object with metrics and analysis.

        Raises:
            TaskExecutionError: If task execution fails.
        """
        ...

    def __repr__(self) -> str:
        provider_name = getattr(self.data_provider, "name", "unknown")
        return f"{self.__class__.__name__}(provider={provider_name!r})"


# =============================================================================
# Common Utility Mixins
# =============================================================================


class NoiseInjectionMixin:
    """
    Mixin providing label noise injection utilities.

    This mixin provides methods for injecting synthetic label noise
    into datasets for evaluation purposes.
    """

    @staticmethod
    def inject_classification_noise(
        y: np.ndarray,
        *,
        frac: float = DEFAULT_NOISE_FRAC,
        seed: int = DEFAULT_SEED,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Inject random label noise into classification labels.

        Randomly flips a fraction of labels to other classes.

        Args:
            y: Original labels (1D integer array).
            frac: Fraction of labels to corrupt (0.0 to 1.0).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (noisy_labels, corrupted_indices).

        Example:
            >>> y = np.array([0, 1, 2, 0, 1])
            >>> y_noisy, flipped = NoiseInjectionMixin.inject_classification_noise(
            ...     y, frac=0.2, seed=42
            ... )
        """
        if frac <= 0:
            return y.copy(), np.array([], dtype=int)

        rng = np.random.default_rng(seed=seed)
        y_noisy = y.copy()
        n_flip = round(frac * len(y_noisy))

        if n_flip <= 0:
            return y_noisy, np.array([], dtype=int)

        classes = np.unique(y_noisy)
        flip_indices = rng.choice(len(y_noisy), size=n_flip, replace=False)

        for idx in flip_indices:
            current = y_noisy[int(idx)]
            other_classes = classes[classes != current]
            if len(other_classes) > 0:
                y_noisy[int(idx)] = rng.choice(other_classes)

        return y_noisy, np.sort(flip_indices.astype(int))

    @staticmethod
    def inject_multilabel_noise(
        y: np.ndarray,
        *,
        frac: float = DEFAULT_NOISE_FRAC,
        seed: int = DEFAULT_SEED,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Inject noise into multilabel binary matrix.

        Randomly flips individual label bits for a fraction of examples.

        Args:
            y: Label matrix of shape (n_samples, n_labels).
            frac: Fraction of examples to corrupt.
            seed: Random seed.

        Returns:
            Tuple of (noisy_labels, corrupted_example_indices).
        """
        if frac <= 0:
            return y.copy(), np.array([], dtype=int)

        rng = np.random.default_rng(seed=seed)
        y_noisy = y.copy()
        n, n_labels = y_noisy.shape
        n_noisy = round(frac * n)

        if n_noisy <= 0:
            return y_noisy, np.array([], dtype=int)

        noisy_indices = rng.choice(n, size=n_noisy, replace=False).astype(int)

        for idx in noisy_indices:
            j = int(rng.integers(0, n_labels))
            y_noisy[int(idx), j] = 1 - y_noisy[int(idx), j]

        return y_noisy, np.sort(noisy_indices)

    @staticmethod
    def inject_regression_noise(
        y: np.ndarray,
        *,
        frac: float = DEFAULT_NOISE_FRAC,
        scale: float = 1.0,
        seed: int = DEFAULT_SEED,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Inject noise into regression targets.

        Adds Gaussian noise scaled by the target's standard deviation.

        Args:
            y: Target values (1D float array).
            frac: Fraction of samples to corrupt.
            scale: Noise scale relative to y's std deviation.
            seed: Random seed.

        Returns:
            Tuple of (noisy_targets, corrupted_indices).
        """
        if frac <= 0:
            return y.copy(), np.array([], dtype=int)

        rng = np.random.default_rng(seed=seed)
        y_noisy = y.copy().astype(float)
        n_corrupt = round(frac * len(y_noisy))

        if n_corrupt <= 0:
            return y_noisy, np.array([], dtype=int)

        corrupt_indices = rng.choice(len(y_noisy), size=n_corrupt, replace=False)
        noise_std = float(np.std(y_noisy)) * scale
        y_noisy[corrupt_indices] += rng.normal(0, noise_std, size=n_corrupt)

        return y_noisy, np.sort(corrupt_indices.astype(int))


class PruningMixin:
    """
    Mixin providing data pruning utilities.

    This mixin provides methods for pruning flagged examples
    from training data based on Cleanlab issue detection.
    """

    @staticmethod
    def prune_by_indices(
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        prune_indices: set[int] | list[int] | np.ndarray,
    ) -> tuple[pd.DataFrame | np.ndarray, np.ndarray]:
        """
        Remove examples at specified indices.

        Args:
            X: Feature matrix.
            y: Labels.
            prune_indices: Indices to remove.

        Returns:
            Tuple of (X_pruned, y_pruned).
        """
        prune_set = set(map(int, prune_indices))
        keep_mask = np.ones(len(y), dtype=bool)

        if prune_set:
            keep_mask[list(prune_set)] = False

        X_pruned = X.iloc[keep_mask] if isinstance(X, pd.DataFrame) else X[keep_mask]
        y_pruned = y[keep_mask]

        return X_pruned, y_pruned

    @staticmethod
    def compute_prune_metrics(
        prune_indices: set[int],
        ground_truth_indices: set[int],
    ) -> dict[str, float]:
        """
        Compute precision and recall for pruning.

        Args:
            prune_indices: Indices that were pruned.
            ground_truth_indices: Indices that should have been pruned.

        Returns:
            Dictionary with precision and recall.
        """
        if not prune_indices:
            return {"precision": 0.0, "recall": 0.0}

        tp = len(prune_indices & ground_truth_indices)
        precision = float(tp / len(prune_indices)) if prune_indices else 0.0
        recall = float(tp / len(ground_truth_indices)) if ground_truth_indices else 0.0

        return {"precision": precision, "recall": recall}


class EvaluationMixin:
    """
    Mixin providing model evaluation utilities.

    Common evaluation patterns used across different tasks.
    """

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safely divide, returning default if denominator is zero."""
        return float(numerator / denominator) if denominator > 0 else default
