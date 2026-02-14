"""
Core module for Cleanlab Demo.

This module provides foundational components including:
- Custom exceptions for error handling
- Type definitions and protocols
- Constants and configuration defaults
"""

from cleanlab_demo.core.constants import (
    DEFAULT_CV_FOLDS,
    DEFAULT_MAX_ITER,
    DEFAULT_PRUNE_FRAC,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    MIN_SAMPLES_FOR_CV,
)
from cleanlab_demo.core.exceptions import (
    CleanlabDemoError,
    ConfigurationError,
    DataLoadError,
    ModelError,
    TaskExecutionError,
    ValidationError,
)
from cleanlab_demo.core.types import (
    ArrayLike,
    DataFrameLike,
    Labels,
    Predictions,
    ProbabilityMatrix,
)


__all__ = [
    "DEFAULT_CV_FOLDS",
    "DEFAULT_MAX_ITER",
    "DEFAULT_PRUNE_FRAC",
    "DEFAULT_SEED",
    "DEFAULT_TEST_SIZE",
    "MIN_SAMPLES_FOR_CV",
    "ArrayLike",
    "CleanlabDemoError",
    "ConfigurationError",
    "DataFrameLike",
    "DataLoadError",
    "Labels",
    "ModelError",
    "Predictions",
    "ProbabilityMatrix",
    "TaskExecutionError",
    "ValidationError",
]
