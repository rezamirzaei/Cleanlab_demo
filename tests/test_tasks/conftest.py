"""
Shared test fixtures for task tests.

This module provides reusable fixtures and utilities for testing
Cleanlab demo tasks with synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleanlab_demo.tasks.base import BaseDataProvider


# =============================================================================
# Synthetic Data Generators
# =============================================================================


def generate_classification_data(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic classification data.

    Creates linearly separable classes with some noise.
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(n_samples, n_features))
    # Create class boundaries based on first feature
    boundaries = np.linspace(X[:, 0].min(), X[:, 0].max(), n_classes + 1)[1:-1]
    y = np.digitize(X[:, 0], boundaries)

    # Add some noise to features
    X += rng.normal(scale=0.1, size=X.shape)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)

    return df, y.astype(int)


def generate_multilabel_data(
    n_samples: int = 300,
    n_features: int = 10,
    n_labels: int = 5,
    label_density: float = 0.3,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic multilabel data.

    Creates features correlated with label patterns.
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(n_samples, n_features))

    # Generate labels with some structure
    y = np.zeros((n_samples, n_labels), dtype=int)
    for j in range(n_labels):
        # Each label correlates with a subset of features
        relevant_features = X[:, j % n_features]
        threshold = np.percentile(relevant_features, 100 * (1 - label_density))
        y[:, j] = (relevant_features > threshold).astype(int)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)

    return df, y


def generate_regression_data(
    n_samples: int = 500,
    n_features: int = 10,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic regression data.

    Creates a linear relationship with Gaussian noise.
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(n_samples, n_features))

    # True coefficients
    true_coef = rng.uniform(-2, 2, size=n_features)
    y = X @ true_coef + rng.normal(scale=noise_std, size=n_samples)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)

    return df, y


def generate_token_data(
    n_sentences: int = 100,
    min_length: int = 5,
    max_length: int = 20,
    n_tags: int = 10,
    seed: int = 42,
) -> tuple[list[list[str]], list[list[int]]]:
    """
    Generate synthetic token classification data.

    Creates dummy sentences with POS-like tags.
    """
    rng = np.random.default_rng(seed)

    vocab = [f"word_{i}" for i in range(200)]

    sentences = []
    labels = []

    for _ in range(n_sentences):
        length = rng.integers(min_length, max_length + 1)
        sentence = rng.choice(vocab, size=length).tolist()
        tags = rng.integers(0, n_tags, size=length).tolist()
        sentences.append(sentence)
        labels.append(tags)

    return sentences, labels


# =============================================================================
# Mock Data Providers
# =============================================================================


class MockClassificationProvider(BaseDataProvider[tuple[pd.DataFrame, np.ndarray]]):
    """Mock provider for classification tasks."""

    def __init__(
        self,
        n_samples: int = 500,
        n_features: int = 10,
        n_classes: int = 3,
        name: str = "mock_classification",
    ):
        self._name = name
        self._n_samples = n_samples
        self._n_features = n_features
        self._n_classes = n_classes

    @property
    def name(self) -> str:
        return self._name

    def load(self, seed: int, **kwargs: Any) -> tuple[pd.DataFrame, np.ndarray]:
        return generate_classification_data(
            n_samples=self._n_samples,
            n_features=self._n_features,
            n_classes=self._n_classes,
            seed=seed,
        )


class MockMultilabelProvider(BaseDataProvider[tuple[pd.DataFrame, np.ndarray]]):
    """Mock provider for multilabel tasks."""

    def __init__(
        self,
        n_samples: int = 300,
        n_features: int = 10,
        n_labels: int = 5,
        name: str = "mock_multilabel",
    ):
        self._name = name
        self._n_samples = n_samples
        self._n_features = n_features
        self._n_labels = n_labels

    @property
    def name(self) -> str:
        return self._name

    def load(self, seed: int, **kwargs: Any) -> tuple[pd.DataFrame, np.ndarray]:
        return generate_multilabel_data(
            n_samples=self._n_samples,
            n_features=self._n_features,
            n_labels=self._n_labels,
            seed=seed,
        )


class MockRegressionProvider(BaseDataProvider[tuple[pd.DataFrame, np.ndarray]]):
    """Mock provider for regression tasks."""

    def __init__(
        self,
        n_samples: int = 500,
        n_features: int = 10,
        name: str = "mock_regression",
    ):
        self._name = name
        self._n_samples = n_samples
        self._n_features = n_features

    @property
    def name(self) -> str:
        return self._name

    def load(self, seed: int, **kwargs: Any) -> tuple[pd.DataFrame, np.ndarray]:
        return generate_regression_data(
            n_samples=self._n_samples,
            n_features=self._n_features,
            seed=seed,
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Fixture providing synthetic classification data."""
    return generate_classification_data(n_samples=300, n_classes=3, seed=42)


@pytest.fixture
def multilabel_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Fixture providing synthetic multilabel data."""
    return generate_multilabel_data(n_samples=200, n_labels=4, seed=42)


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Fixture providing synthetic regression data."""
    return generate_regression_data(n_samples=300, seed=42)


@pytest.fixture
def mock_classification_provider() -> MockClassificationProvider:
    """Fixture providing a mock classification data provider."""
    return MockClassificationProvider(n_samples=300, n_classes=3)


@pytest.fixture
def mock_multilabel_provider() -> MockMultilabelProvider:
    """Fixture providing a mock multilabel data provider."""
    return MockMultilabelProvider(n_samples=200, n_labels=4)


@pytest.fixture
def mock_regression_provider() -> MockRegressionProvider:
    """Fixture providing a mock regression data provider."""
    return MockRegressionProvider(n_samples=300)

