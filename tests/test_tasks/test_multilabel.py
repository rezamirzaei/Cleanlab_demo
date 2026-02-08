"""Tests for multilabel classification task."""

from __future__ import annotations

import pytest

from tests.test_tasks.conftest import MockMultilabelProvider


class TestMultilabelClassificationTask:
    """Tests for MultilabelClassificationTask."""

    def test_task_runs_successfully(self, mock_multilabel_provider: MockMultilabelProvider) -> None:
        """Test that task runs without errors."""
        from cleanlab_demo.tasks.multilabel import (
            MultilabelClassificationConfig,
            MultilabelClassificationTask,
        )

        task = MultilabelClassificationTask(mock_multilabel_provider)
        config = MultilabelClassificationConfig(
            noise_frac=0.1,
            cv_folds=3,
            prune_frac=0.05,
            seed=42,
        )

        result = task.run(config)

        assert result.dataset == mock_multilabel_provider.name
        assert result.n_train > 0
        assert result.n_test > 0
        assert result.n_labels == 4

    def test_task_returns_valid_metrics(self, mock_multilabel_provider: MockMultilabelProvider) -> None:
        """Test that task returns valid metrics."""
        from cleanlab_demo.tasks.multilabel import (
            MultilabelClassificationConfig,
            MultilabelClassificationTask,
        )

        task = MultilabelClassificationTask(mock_multilabel_provider)
        config = MultilabelClassificationConfig(
            noise_frac=0.0,
            cv_folds=3,
            seed=42,
        )

        result = task.run(config)

        # Check baseline metrics
        assert 0.0 <= result.metrics.baseline.micro_f1 <= 1.0
        assert 0.0 <= result.metrics.baseline.macro_f1 <= 1.0
        assert 0.0 <= result.metrics.baseline.subset_accuracy <= 1.0
        assert 0.0 <= result.metrics.baseline.hamming_loss <= 1.0

    def test_task_with_noise_finds_issues(self, mock_multilabel_provider: MockMultilabelProvider) -> None:
        """Test that task with noise finds label issues."""
        from cleanlab_demo.tasks.multilabel import (
            MultilabelClassificationConfig,
            MultilabelClassificationTask,
        )

        task = MultilabelClassificationTask(mock_multilabel_provider)
        config = MultilabelClassificationConfig(
            noise_frac=0.2,
            cv_folds=3,
            prune_frac=0.1,
            seed=42,
        )

        result = task.run(config)

        assert result.cleanlab.n_issues_found > 0
        assert result.noise.n_noisy_examples > 0

    def test_task_reproducibility(self, mock_multilabel_provider: MockMultilabelProvider) -> None:
        """Test that task produces reproducible results."""
        from cleanlab_demo.tasks.multilabel import (
            MultilabelClassificationConfig,
            MultilabelClassificationTask,
        )

        task = MultilabelClassificationTask(mock_multilabel_provider)
        config = MultilabelClassificationConfig(
            noise_frac=0.1,
            cv_folds=3,
            seed=42,
        )

        result1 = task.run(config)
        result2 = task.run(config)

        assert result1.metrics.baseline.micro_f1 == result2.metrics.baseline.micro_f1


class TestMultilabelConfig:
    """Tests for MultilabelClassificationConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from cleanlab_demo.tasks.multilabel import MultilabelClassificationConfig

        config = MultilabelClassificationConfig()

        assert config.test_size == 0.2
        assert config.noise_frac == 0.0
        assert config.cv_folds == 5
        assert config.prune_frac == 0.05

    def test_validation(self) -> None:
        """Test config validation."""
        from cleanlab_demo.tasks.multilabel import MultilabelClassificationConfig

        # Invalid cv_folds
        with pytest.raises(ValueError):
            MultilabelClassificationConfig(cv_folds=0)

        # Invalid noise_frac
        with pytest.raises(ValueError):
            MultilabelClassificationConfig(noise_frac=-0.1)
