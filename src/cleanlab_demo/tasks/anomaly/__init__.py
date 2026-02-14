"""
Anomaly Detection Module.

This module provides a clean, object-oriented implementation for anomaly detection
using Cleanlab's Datalab and traditional ML methods. It supports multiple detection
strategies including kNN-based outlier detection, Isolation Forest, and Local Outlier Factor.

Key Components:
    - AnomalyDetectionTask: Main orchestrator for anomaly detection workflows
    - AnomalyDetectionConfig: Pydantic-validated configuration
    - AnomalyDetectionResult: Structured results with metrics
    - AnomalyDetectionStrategy: Abstract base for pluggable algorithms
    - DataProviders: Abstract interface for loading datasets

Example:
    >>> from cleanlab_demo.tasks.anomaly import (
    ...     AnomalyDetectionTask,
    ...     AnomalyDetectionConfig,
    ... )
    >>> from cleanlab_demo.data.providers.anomaly import SyntheticAnomalyProvider
    >>>
    >>> task = AnomalyDetectionTask(SyntheticAnomalyProvider())
    >>> config = AnomalyDetectionConfig(strategy="datalab_knn", contamination=0.01)
    >>> result = task.run(config)
    >>> print(f"Detected {result.summary.n_anomalies_detected} anomalies")
"""

from cleanlab_demo.tasks.anomaly.provider import AnomalyDetectionDataProvider
from cleanlab_demo.tasks.anomaly.schemas import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
    AnomalyRow,
    AnomalySummary,
    StrategyType,
)
from cleanlab_demo.tasks.anomaly.strategies import (
    AnomalyDetectionStrategy,
    DataLabKNNStrategy,
    IsolationForestStrategy,
    LocalOutlierFactorStrategy,
    get_strategy,
)
from cleanlab_demo.tasks.anomaly.task import AnomalyDetectionTask, run_anomaly_detection


__all__ = [
    "AnomalyDetectionConfig",
    "AnomalyDetectionDataProvider",
    "AnomalyDetectionResult",
    "AnomalyDetectionStrategy",
    "AnomalyDetectionTask",
    "AnomalyRow",
    "AnomalySummary",
    "DataLabKNNStrategy",
    "IsolationForestStrategy",
    "LocalOutlierFactorStrategy",
    "StrategyType",
    "get_strategy",
    "run_anomaly_detection",
]
