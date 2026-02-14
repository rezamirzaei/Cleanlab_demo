"""
Custom exceptions for Cleanlab Demo.

This module defines a hierarchy of exceptions for better error handling
and debugging throughout the application.

Exception Hierarchy:
    CleanlabDemoError (base)
    ├── ConfigurationError - Invalid configuration
    ├── DataLoadError - Data loading/parsing failures
    ├── ValidationError - Data/input validation failures
    ├── ModelError - Model training/prediction failures
    └── TaskExecutionError - Task execution failures
"""

from __future__ import annotations

from typing import Any


class CleanlabDemoError(Exception):
    """
    Base exception for all Cleanlab Demo errors.

    All custom exceptions in this package inherit from this class,
    making it easy to catch any Cleanlab Demo specific error.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional error context.

    Example:
        >>> try:
        ...     raise CleanlabDemoError("Something went wrong", details={"code": 42})
        ... except CleanlabDemoError as e:
        ...     print(e.message, e.details)
        Something went wrong {'code': 42}
    """

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


class ConfigurationError(CleanlabDemoError):
    """
    Raised when configuration is invalid or missing.

    This exception is raised when:
    - Required configuration values are missing
    - Configuration values are out of valid range
    - Configuration combinations are incompatible

    Example:
        >>> raise ConfigurationError(
        ...     "Invalid CV folds",
        ...     details={"cv_folds": 1, "min_required": 2}
        ... )
    """


class DataLoadError(CleanlabDemoError):
    """
    Raised when data loading fails.

    This exception is raised when:
    - Dataset file is not found
    - Dataset format is invalid
    - Network request for dataset fails
    - Data parsing encounters errors

    Example:
        >>> raise DataLoadError(
        ...     "Failed to load dataset",
        ...     details={"dataset": "adult_income", "path": "/data/adult.csv"}
        ... )
    """


class ValidationError(CleanlabDemoError):
    """
    Raised when data validation fails.

    This exception is raised when:
    - Input data has unexpected shape
    - Required columns are missing
    - Data types are incompatible
    - Value constraints are violated

    Example:
        >>> raise ValidationError(
        ...     "Feature matrix has wrong shape",
        ...     details={"expected": (100, 10), "got": (100, 5)}
        ... )
    """


class ModelError(CleanlabDemoError):
    """
    Raised when model operations fail.

    This exception is raised when:
    - Model training fails to converge
    - Prediction fails due to incompatible input
    - Model serialization/deserialization fails

    Example:
        >>> raise ModelError(
        ...     "Model failed to converge",
        ...     details={"model": "LogisticRegression", "max_iter": 100}
        ... )
    """


class TaskExecutionError(CleanlabDemoError):
    """
    Raised when task execution fails.

    This exception is raised when:
    - A task encounters an unrecoverable error during execution
    - Required dependencies for a task are missing
    - Task preconditions are not met

    Example:
        >>> raise TaskExecutionError(
        ...     "Multiclass classification task failed",
        ...     details={"task": "MulticlassClassificationTask", "stage": "cv_predict"}
        ... )
    """
