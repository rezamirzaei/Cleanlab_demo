"""Tests for ML utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cleanlab_demo.utils.ml import (
    build_classifier_pipeline,
    build_multilabel_pipeline,
    build_regressor_pipeline,
    compute_classification_metrics,
    compute_multilabel_metrics,
    compute_regression_metrics,
    ensure_numpy_array,
    labels_to_list_format,
    train_test_indices,
)


class TestPipelineBuilders:
    """Tests for pipeline builder functions."""

    def test_build_classifier_pipeline(self) -> None:
        """Test classifier pipeline creation."""
        pipeline = build_classifier_pipeline(max_iter=100, seed=42)

        assert "scale" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_build_classifier_pipeline_no_scale(self) -> None:
        """Test classifier pipeline without scaling."""
        pipeline = build_classifier_pipeline(scale=False)

        assert "scale" not in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_build_multilabel_pipeline(self) -> None:
        """Test multilabel pipeline creation."""
        pipeline = build_multilabel_pipeline(max_iter=100, seed=42)

        assert "scale" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_build_regressor_pipeline(self) -> None:
        """Test regressor pipeline creation."""
        pipeline = build_regressor_pipeline(alpha=0.5, seed=42)

        assert "scale" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_pipelines_are_fittable(self) -> None:
        """Test that built pipelines can be fitted."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, size=100)

        clf = build_classifier_pipeline()
        clf.fit(X, y)

        predictions = clf.predict(X)
        assert len(predictions) == 100


class TestMetricComputation:
    """Tests for metric computation functions."""

    def test_classification_metrics_perfect(self) -> None:
        """Test classification metrics with perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_classification_metrics_with_proba(self) -> None:
        """Test classification metrics with probabilities."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8],
            ]
        )

        metrics = compute_classification_metrics(y_true, y_pred, y_proba)

        assert "log_loss" in metrics
        assert metrics["log_loss"] < 0.5  # Should be low for good predictions

    def test_multilabel_metrics(self) -> None:
        """Test multilabel metrics."""
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])

        metrics = compute_multilabel_metrics(y_true, y_pred)

        assert metrics["micro_f1"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["subset_accuracy"] == 1.0
        assert metrics["hamming_loss"] == 0.0

    def test_regression_metrics(self) -> None:
        """Test regression metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = compute_regression_metrics(y_true, y_pred)

        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0

    def test_regression_metrics_imperfect(self) -> None:
        """Test regression metrics with errors."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        metrics = compute_regression_metrics(y_true, y_pred)

        assert metrics["mse"] > 0
        assert metrics["r2"] < 1.0
        assert metrics["r2"] > 0.9  # Should still be good


class TestDataUtilities:
    """Tests for data utility functions."""

    def test_labels_to_list_format(self) -> None:
        """Test conversion to list format."""
        y = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 1, 0],
            ]
        )

        result = labels_to_list_format(y)

        assert result == [[0, 2], [1, 3], [0, 1, 2]]

    def test_labels_to_list_format_empty_row(self) -> None:
        """Test conversion with row having no labels."""
        y = np.array(
            [
                [1, 0],
                [0, 0],  # No labels
                [0, 1],
            ]
        )

        result = labels_to_list_format(y)

        assert result == [[0], [], [1]]

    def test_ensure_numpy_array_from_ndarray(self) -> None:
        """Test with numpy array input."""
        arr = np.array([1, 2, 3])
        result = ensure_numpy_array(arr)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(arr, result)

    def test_ensure_numpy_array_from_series(self) -> None:
        """Test with pandas Series input."""
        series = pd.Series([1, 2, 3])
        result = ensure_numpy_array(series)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_ensure_numpy_array_from_list(self) -> None:
        """Test with list input."""
        lst = [1, 2, 3]
        result = ensure_numpy_array(lst)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_train_test_indices(self) -> None:
        """Test train/test index generation."""
        train_idx, test_idx = train_test_indices(100, test_size=0.2, seed=42)

        assert len(train_idx) == 80
        assert len(test_idx) == 20
        assert len(set(train_idx) & set(test_idx)) == 0  # No overlap

    def test_train_test_indices_reproducible(self) -> None:
        """Test reproducibility with same seed."""
        train1, test1 = train_test_indices(100, seed=42)
        train2, test2 = train_test_indices(100, seed=42)

        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(test1, test2)

    def test_train_test_indices_different_seeds(self) -> None:
        """Test different results with different seeds."""
        train1, _ = train_test_indices(100, seed=42)
        train2, _ = train_test_indices(100, seed=123)

        # Should be different (with very high probability)
        assert not np.array_equal(train1, train2)
