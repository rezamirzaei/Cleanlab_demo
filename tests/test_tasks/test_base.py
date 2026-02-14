"""Tests for task base classes and mixins."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleanlab_demo.tasks.base import (
    DemoConfig,
    DemoResult,
    EvaluationMixin,
    NoiseInjectionMixin,
    PruningMixin,
)


class TestDemoConfig:
    """Tests for DemoConfig base class."""

    def test_default_seed(self) -> None:
        """Test default seed value."""
        config = DemoConfig()
        assert config.seed == 42

    def test_custom_seed(self) -> None:
        """Test custom seed value."""
        config = DemoConfig(seed=123)
        assert config.seed == 123

    def test_seed_validation_min(self) -> None:
        """Test seed minimum validation."""
        with pytest.raises(ValueError):
            DemoConfig(seed=-1)

    def test_seed_validation_max(self) -> None:
        """Test seed maximum validation."""
        with pytest.raises(ValueError):
            DemoConfig(seed=2_000_000)

    def test_model_dump(self) -> None:
        """Test serialization."""
        config = DemoConfig(seed=100)
        data = config.model_dump()
        assert data == {"seed": 100}


class TestDemoResult:
    """Tests for DemoResult base class."""

    def test_timestamp_auto_generated(self) -> None:
        """Test that timestamp is automatically generated."""
        result = DemoResult()
        assert result.timestamp is not None
        assert "T" in result.timestamp  # ISO format check

    def test_timestamp_format(self) -> None:
        """Test timestamp is valid ISO format."""
        from datetime import datetime

        result = DemoResult()
        # Should not raise
        datetime.fromisoformat(result.timestamp.replace("+00:00", ""))

    def test_model_dump_json(self) -> None:
        """Test JSON serialization."""
        result = DemoResult()
        json_str = result.model_dump_json()
        assert "timestamp" in json_str


class TestNoiseInjectionMixin:
    """Tests for NoiseInjectionMixin."""

    def test_classification_noise_no_noise(self) -> None:
        """Test with frac=0 returns unchanged labels."""
        y = np.array([0, 1, 2, 0, 1, 2])
        y_noisy, flipped = NoiseInjectionMixin.inject_classification_noise(y, frac=0.0, seed=42)
        np.testing.assert_array_equal(y, y_noisy)
        assert len(flipped) == 0

    def test_classification_noise_with_noise(self) -> None:
        """Test with noise fraction."""
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_noisy, flipped = NoiseInjectionMixin.inject_classification_noise(y, frac=0.3, seed=42)

        # Some labels should be flipped
        assert len(flipped) == 3  # 30% of 10
        assert not np.array_equal(y, y_noisy)

        # Flipped indices should have different values
        for idx in flipped:
            assert y[idx] != y_noisy[idx]

    def test_classification_noise_reproducible(self) -> None:
        """Test that noise injection is reproducible with same seed."""
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        y_noisy1, flipped1 = NoiseInjectionMixin.inject_classification_noise(y, frac=0.3, seed=42)
        y_noisy2, flipped2 = NoiseInjectionMixin.inject_classification_noise(y, frac=0.3, seed=42)

        np.testing.assert_array_equal(y_noisy1, y_noisy2)
        np.testing.assert_array_equal(flipped1, flipped2)

    def test_multilabel_noise_no_noise(self) -> None:
        """Test multilabel with frac=0."""
        y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_noisy, corrupted = NoiseInjectionMixin.inject_multilabel_noise(y, frac=0.0, seed=42)
        np.testing.assert_array_equal(y, y_noisy)
        assert len(corrupted) == 0

    def test_multilabel_noise_with_noise(self) -> None:
        """Test multilabel with noise fraction."""
        y = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
            ]
        )
        y_noisy, corrupted = NoiseInjectionMixin.inject_multilabel_noise(y, frac=0.4, seed=42)

        assert len(corrupted) == 2  # 40% of 5
        assert not np.array_equal(y, y_noisy)

    def test_regression_noise_no_noise(self) -> None:
        """Test regression with frac=0."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_noisy, corrupted = NoiseInjectionMixin.inject_regression_noise(y, frac=0.0, seed=42)
        np.testing.assert_array_equal(y, y_noisy)
        assert len(corrupted) == 0

    def test_regression_noise_with_noise(self) -> None:
        """Test regression with noise fraction."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y_noisy, corrupted = NoiseInjectionMixin.inject_regression_noise(y, frac=0.3, seed=42)

        assert len(corrupted) == 3
        # Corrupted values should be different
        for idx in corrupted:
            assert y[idx] != y_noisy[idx]


class TestPruningMixin:
    """Tests for PruningMixin."""

    def test_prune_by_indices_numpy(self) -> None:
        """Test pruning with numpy arrays."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 1, 2, 3, 4])

        X_pruned, y_pruned = PruningMixin.prune_by_indices(X, y, {1, 3})

        assert len(X_pruned) == 3
        assert len(y_pruned) == 3
        np.testing.assert_array_equal(y_pruned, [0, 2, 4])

    def test_prune_by_indices_dataframe(self) -> None:
        """Test pruning with pandas DataFrame."""
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
        y = np.array([0, 1, 2, 3, 4])

        X_pruned, y_pruned = PruningMixin.prune_by_indices(X, y, [0, 2, 4])

        assert len(X_pruned) == 2
        assert len(y_pruned) == 2
        np.testing.assert_array_equal(y_pruned, [1, 3])

    def test_prune_by_indices_empty(self) -> None:
        """Test pruning with empty indices."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 2])

        X_pruned, y_pruned = PruningMixin.prune_by_indices(X, y, set())

        assert len(X_pruned) == 3
        assert len(y_pruned) == 3

    def test_compute_prune_metrics_perfect(self) -> None:
        """Test prune metrics with perfect precision/recall."""
        prune_indices = {1, 2, 3}
        ground_truth = {1, 2, 3}

        metrics = PruningMixin.compute_prune_metrics(prune_indices, ground_truth)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_compute_prune_metrics_partial(self) -> None:
        """Test prune metrics with partial overlap."""
        prune_indices = {1, 2, 5, 6}  # 2 correct, 2 wrong
        ground_truth = {1, 2, 3, 4}  # 2 found, 2 missed

        metrics = PruningMixin.compute_prune_metrics(prune_indices, ground_truth)

        assert metrics["precision"] == 0.5  # 2/4
        assert metrics["recall"] == 0.5  # 2/4

    def test_compute_prune_metrics_no_prune(self) -> None:
        """Test prune metrics with no pruning."""
        metrics = PruningMixin.compute_prune_metrics(set(), {1, 2, 3})

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_compute_prune_metrics_no_ground_truth(self) -> None:
        """Test prune metrics with no ground truth."""
        metrics = PruningMixin.compute_prune_metrics({1, 2}, set())

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0


class TestEvaluationMixin:
    """Tests for EvaluationMixin."""

    def test_safe_divide_normal(self) -> None:
        """Test normal division."""
        result = EvaluationMixin.safe_divide(10.0, 2.0)
        assert result == 5.0

    def test_safe_divide_by_zero(self) -> None:
        """Test division by zero returns default."""
        result = EvaluationMixin.safe_divide(10.0, 0.0)
        assert result == 0.0

    def test_safe_divide_custom_default(self) -> None:
        """Test division by zero with custom default."""
        result = EvaluationMixin.safe_divide(10.0, 0.0, default=-1.0)
        assert result == -1.0
