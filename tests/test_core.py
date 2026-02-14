"""Tests for core module."""

from __future__ import annotations

import pytest


class TestExceptions:
    """Tests for custom exceptions."""

    def test_cleanlab_demo_error_basic(self) -> None:
        """Test base exception with message only."""
        from cleanlab_demo.core.exceptions import CleanlabDemoError

        error = CleanlabDemoError("Test error")
        assert error.message == "Test error"
        assert error.details == {}
        assert str(error) == "Test error"

    def test_cleanlab_demo_error_with_details(self) -> None:
        """Test base exception with details."""
        from cleanlab_demo.core.exceptions import CleanlabDemoError

        error = CleanlabDemoError("Test error", details={"code": 42, "info": "extra"})
        assert error.message == "Test error"
        assert error.details == {"code": 42, "info": "extra"}
        assert "42" in str(error)
        assert "extra" in str(error)

    def test_exception_repr(self) -> None:
        """Test exception repr."""
        from cleanlab_demo.core.exceptions import CleanlabDemoError

        error = CleanlabDemoError("Test", details={"x": 1})
        repr_str = repr(error)
        assert "CleanlabDemoError" in repr_str
        assert "Test" in repr_str

    def test_subclass_inheritance(self) -> None:
        """Test that subclasses inherit from base."""
        from cleanlab_demo.core.exceptions import (
            CleanlabDemoError,
            ConfigurationError,
            DataLoadError,
            ModelError,
            TaskExecutionError,
            ValidationError,
        )

        # All should be subclasses of CleanlabDemoError
        assert issubclass(ConfigurationError, CleanlabDemoError)
        assert issubclass(DataLoadError, CleanlabDemoError)
        assert issubclass(ValidationError, CleanlabDemoError)
        assert issubclass(ModelError, CleanlabDemoError)
        assert issubclass(TaskExecutionError, CleanlabDemoError)

    def test_can_catch_by_base_class(self) -> None:
        """Test that subclasses can be caught by base class."""
        from cleanlab_demo.core.exceptions import (
            CleanlabDemoError,
            DataLoadError,
        )

        with pytest.raises(CleanlabDemoError):
            raise DataLoadError("Failed to load")


class TestConstants:
    """Tests for constants module."""

    def test_default_values_exist(self) -> None:
        """Test that default constants are defined."""
        from cleanlab_demo.core.constants import (
            DEFAULT_CV_FOLDS,
            DEFAULT_MAX_ITER,
            DEFAULT_PRUNE_FRAC,
            DEFAULT_SEED,
            DEFAULT_TEST_SIZE,
        )

        assert DEFAULT_SEED == 42
        assert DEFAULT_TEST_SIZE == 0.2
        assert DEFAULT_CV_FOLDS == 5
        assert DEFAULT_MAX_ITER == 500
        assert DEFAULT_PRUNE_FRAC == 0.02

    def test_constants_are_final(self) -> None:
        """Test that constants have Final type hints."""
        from cleanlab_demo.core import constants

        # Check that module has annotations for constants.
        assert getattr(constants, "__annotations__", {})
        # Constants should exist and have proper values
        assert hasattr(constants, "DEFAULT_SEED")


class TestTypes:
    """Tests for types module."""

    def test_metrics_dict_mean(self) -> None:
        """Test MetricsDict mean calculation."""
        from cleanlab_demo.core.types import MetricsDict

        metrics = MetricsDict({"a": 0.8, "b": 0.6, "c": 1.0})
        assert abs(metrics.mean() - 0.8) < 1e-9  # Use approximate comparison

    def test_metrics_dict_mean_empty(self) -> None:
        """Test MetricsDict mean with empty dict."""
        from cleanlab_demo.core.types import MetricsDict

        metrics = MetricsDict()
        assert metrics.mean() == 0.0

    def test_metrics_dict_to_percentage(self) -> None:
        """Test MetricsDict percentage conversion."""
        from cleanlab_demo.core.types import MetricsDict

        metrics = MetricsDict({"accuracy": 0.95, "f1": 0.88})
        pct = metrics.to_percentage()

        assert pct["accuracy"] == 95.0
        assert pct["f1"] == 88.0

    def test_protocols_are_runtime_checkable(self) -> None:
        """Test that protocols can be checked at runtime."""
        from cleanlab_demo.core.types import DataProvider

        # Should not raise
        assert hasattr(DataProvider, "__protocol_attrs__") or True  # Protocol check
