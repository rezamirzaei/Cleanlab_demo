"""Tests for outlier detection task."""

from __future__ import annotations

import pytest

from tests.test_tasks.conftest import MockRegressionProvider


class TestOutlierDetectionTask:
    """Tests for OutlierDetectionTask."""

    def test_task_runs_successfully(self, mock_regression_provider: MockRegressionProvider) -> None:
        """Test that task runs without errors."""
        # OutlierDetectionTask uses a different provider interface
        # Create a compatible provider
        from cleanlab_demo.data.providers import CaliforniaHousingOutlierProvider
        from cleanlab_demo.tasks.outlier import (
            OutlierDetectionConfig,
            OutlierDetectionTask,
        )

        task = OutlierDetectionTask(CaliforniaHousingOutlierProvider(max_rows=500))
        config = OutlierDetectionConfig(
            outlier_frac=0.1,
            seed=42,
        )

        result = task.run(config)

        assert result.dataset is not None
        assert result.n_rows > 0

    def test_task_with_synthetic_outliers(self) -> None:
        """Test that task detects synthetic outliers."""
        from cleanlab_demo.data.providers import CaliforniaHousingOutlierProvider
        from cleanlab_demo.tasks.outlier import (
            OutlierDetectionConfig,
            OutlierDetectionTask,
        )

        task = OutlierDetectionTask(CaliforniaHousingOutlierProvider(max_rows=500))
        config = OutlierDetectionConfig(
            outlier_frac=0.1,  # 10% synthetic outliers
            seed=42,
        )

        result = task.run(config)

        # Should have issue summary
        assert len(result.cleanlab.issue_summary) > 0
        assert result.cleanlab.issue_types is not None

    def test_task_reproducibility(self) -> None:
        """Test that task produces reproducible results."""
        from cleanlab_demo.data.providers import CaliforniaHousingOutlierProvider
        from cleanlab_demo.tasks.outlier import (
            OutlierDetectionConfig,
            OutlierDetectionTask,
        )

        provider = CaliforniaHousingOutlierProvider(max_rows=300)
        task = OutlierDetectionTask(provider)
        config = OutlierDetectionConfig(
            outlier_frac=0.05,
            seed=42,
        )

        result1 = task.run(config)
        result2 = task.run(config)

        assert result1.cleanlab.precision_vs_injected == result2.cleanlab.precision_vs_injected


class TestOutlierDetectionConfig:
    """Tests for OutlierDetectionConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from cleanlab_demo.tasks.outlier import OutlierDetectionConfig

        config = OutlierDetectionConfig()

        assert config.outlier_frac == 0.0
        assert config.seed == 42

    def test_validation(self) -> None:
        """Test config validation."""
        from cleanlab_demo.tasks.outlier import OutlierDetectionConfig

        # Valid config
        config = OutlierDetectionConfig(outlier_frac=0.1)
        assert config.outlier_frac == 0.1

        # Invalid outlier_frac (negative)
        with pytest.raises(ValueError):
            OutlierDetectionConfig(outlier_frac=-0.1)

        # Invalid outlier_frac (too high)
        with pytest.raises(ValueError):
            OutlierDetectionConfig(outlier_frac=0.5)


class TestOutlierDetectionResult:
    """Tests for OutlierDetectionResult."""

    def test_result_structure(self) -> None:
        """Test that result has expected structure."""
        from cleanlab_demo.data.providers import CaliforniaHousingOutlierProvider
        from cleanlab_demo.tasks.outlier import (
            OutlierDetectionConfig,
            OutlierDetectionTask,
        )

        task = OutlierDetectionTask(CaliforniaHousingOutlierProvider(max_rows=300))
        config = OutlierDetectionConfig(outlier_frac=0.05, seed=42)

        result = task.run(config)

        # Check required fields
        assert hasattr(result, "timestamp")
        assert hasattr(result, "dataset")
        assert hasattr(result, "n_rows")
        assert hasattr(result, "cleanlab")

    def test_result_serialization(self) -> None:
        """Test that result can be serialized."""
        import json

        from cleanlab_demo.data.providers import CaliforniaHousingOutlierProvider
        from cleanlab_demo.tasks.outlier import (
            OutlierDetectionConfig,
            OutlierDetectionTask,
        )

        task = OutlierDetectionTask(CaliforniaHousingOutlierProvider(max_rows=300))
        config = OutlierDetectionConfig(seed=42)

        result = task.run(config)
        json_str = result.model_dump_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert "dataset" in data
        assert "cleanlab" in data
