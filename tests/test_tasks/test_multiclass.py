"""Tests for multiclass classification task."""

from __future__ import annotations

import pytest

from tests.test_tasks.conftest import MockClassificationProvider


class TestMulticlassClassificationTask:
    """Tests for MulticlassClassificationTask."""

    def test_task_runs_successfully(
        self, mock_classification_provider: MockClassificationProvider
    ) -> None:
        """Test that task runs without errors."""
        from cleanlab_demo.tasks.multiclass import (
            MulticlassClassificationConfig,
            MulticlassClassificationTask,
        )

        task = MulticlassClassificationTask(mock_classification_provider)
        config = MulticlassClassificationConfig(
            noise_frac=0.1,
            cv_folds=3,
            prune_frac=0.05,
            seed=42,
        )

        result = task.run(config)

        assert result.dataset == mock_classification_provider.name
        assert result.n_train > 0
        assert result.n_test > 0
        assert result.n_classes == 3

    def test_task_returns_valid_metrics(
        self, mock_classification_provider: MockClassificationProvider
    ) -> None:
        """Test that task returns valid metrics."""
        from cleanlab_demo.tasks.multiclass import (
            MulticlassClassificationConfig,
            MulticlassClassificationTask,
        )

        task = MulticlassClassificationTask(mock_classification_provider)
        config = MulticlassClassificationConfig(
            noise_frac=0.0,
            cv_folds=3,
            seed=42,
        )

        result = task.run(config)

        # Check baseline metrics
        assert 0.0 <= result.metrics.baseline.accuracy <= 1.0
        assert 0.0 <= result.metrics.baseline.macro_f1 <= 1.0
        assert result.metrics.baseline.log_loss >= 0.0

        # Check pruned metrics
        assert 0.0 <= result.metrics.pruned_retrain.accuracy <= 1.0

    def test_task_with_noise_finds_issues(
        self, mock_classification_provider: MockClassificationProvider
    ) -> None:
        """Test that task with noise finds label issues."""
        from cleanlab_demo.tasks.multiclass import (
            MulticlassClassificationConfig,
            MulticlassClassificationTask,
        )

        task = MulticlassClassificationTask(mock_classification_provider)
        config = MulticlassClassificationConfig(
            noise_frac=0.2,  # 20% noise
            cv_folds=3,
            prune_frac=0.1,
            seed=42,
        )

        result = task.run(config)

        # Should detect some issues
        assert result.cleanlab.n_issues_found > 0
        assert result.noise.n_flipped > 0

    def test_task_reproducibility(
        self, mock_classification_provider: MockClassificationProvider
    ) -> None:
        """Test that task produces reproducible results."""
        from cleanlab_demo.tasks.multiclass import (
            MulticlassClassificationConfig,
            MulticlassClassificationTask,
        )

        task = MulticlassClassificationTask(mock_classification_provider)
        config = MulticlassClassificationConfig(
            noise_frac=0.1,
            cv_folds=3,
            seed=42,
        )

        result1 = task.run(config)
        result2 = task.run(config)

        assert result1.metrics.baseline.accuracy == result2.metrics.baseline.accuracy
        assert result1.cleanlab.n_issues_found == result2.cleanlab.n_issues_found

    def test_task_config_validation(self) -> None:
        """Test config validation."""
        from cleanlab_demo.tasks.multiclass import MulticlassClassificationConfig

        # Valid config
        config = MulticlassClassificationConfig(
            cv_folds=5,
            prune_frac=0.1,
        )
        assert config.cv_folds == 5

        # Invalid cv_folds
        with pytest.raises(ValueError):
            MulticlassClassificationConfig(cv_folds=1)

        # Invalid prune_frac
        with pytest.raises(ValueError):
            MulticlassClassificationConfig(prune_frac=1.5)


class TestMulticlassConfig:
    """Tests for MulticlassClassificationConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from cleanlab_demo.tasks.multiclass import MulticlassClassificationConfig

        config = MulticlassClassificationConfig()

        assert config.test_size == 0.2
        assert config.noise_frac == 0.0
        assert config.cv_folds == 5
        assert config.prune_frac == 0.02
        assert config.seed == 42

    def test_model_dump(self) -> None:
        """Test serialization."""
        from cleanlab_demo.tasks.multiclass import MulticlassClassificationConfig

        config = MulticlassClassificationConfig(
            test_size=0.3,
            noise_frac=0.1,
            cv_folds=3,
        )

        data = config.model_dump()

        assert data["test_size"] == 0.3
        assert data["noise_frac"] == 0.1
        assert data["cv_folds"] == 3


class TestMulticlassResult:
    """Tests for MulticlassClassificationResult."""

    def test_result_has_timestamp(
        self, mock_classification_provider: MockClassificationProvider
    ) -> None:
        """Test that result has timestamp."""
        from cleanlab_demo.tasks.multiclass import (
            MulticlassClassificationConfig,
            MulticlassClassificationTask,
        )

        task = MulticlassClassificationTask(mock_classification_provider)
        config = MulticlassClassificationConfig(cv_folds=3, seed=42)

        result = task.run(config)

        assert result.timestamp is not None
        assert "T" in result.timestamp

    def test_result_json_serialization(
        self, mock_classification_provider: MockClassificationProvider
    ) -> None:
        """Test that result can be serialized to JSON."""
        import json

        from cleanlab_demo.tasks.multiclass import (
            MulticlassClassificationConfig,
            MulticlassClassificationTask,
        )

        task = MulticlassClassificationTask(mock_classification_provider)
        config = MulticlassClassificationConfig(cv_folds=3, seed=42)

        result = task.run(config)
        json_str = result.model_dump_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert "dataset" in data
        assert "metrics" in data
        assert "cleanlab" in data
