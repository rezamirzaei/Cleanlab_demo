"""
Anomaly Detection Schemas.

Pydantic models for anomaly detection configuration and results.
These schemas provide:
- Type validation and coercion
- Serialization to/from JSON
- Clear documentation of all parameters
- Immutable results for safety

Mathematical Notation:
    - n: total number of samples
    - k: number of detected anomalies
    - ε: contamination rate (k/n)
    - TP: true positives (correctly identified anomalies)
    - FP: false positives (normal samples flagged as anomalies)
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN) where FN = actual anomalies not detected
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cleanlab_demo.tasks.base import DemoConfig, DemoResult


# Type alias for supported detection strategies
StrategyType = Literal[
    "datalab_knn",  # Cleanlab Datalab with kNN distance
    "isolation_forest",  # Scikit-learn Isolation Forest
    "local_outlier_factor",  # Scikit-learn LOF
    "ensemble",  # Combination of multiple methods
]


class AnomalyRow(BaseModel):
    """
    Single anomaly detection result for one sample.

    Attributes:
        index: Original DataFrame index of the sample
        anomaly_score: Anomaly score (lower = more anomalous for most methods)
        is_anomaly: Whether flagged as anomaly by threshold
        features_summary: Optional dict of key feature values for inspection
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(description="Original row index in dataset")
    anomaly_score: float = Field(description="Anomaly score (interpretation varies by strategy)")
    is_anomaly: bool = Field(description="Whether flagged as anomaly")
    ground_truth: int | None = Field(
        default=None, description="Ground truth label if available (0=normal, 1=anomaly)"
    )


class StrategyResult(BaseModel):
    """
    Results from a single detection strategy.

    Attributes:
        strategy_name: Name of the strategy used
        n_detected: Number of anomalies detected
        threshold: Threshold used for classification
        execution_time_seconds: Time taken to run detection
    """

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    n_detected: int = Field(ge=0)
    threshold: float | None = Field(default=None)
    execution_time_seconds: float = Field(ge=0.0)

    # Metrics (only available if ground truth exists)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1_score: float | None = Field(default=None, ge=0.0, le=1.0)


class AnomalySummary(BaseModel):
    """
    Summary statistics for anomaly detection results.

    Mathematical Context:
        The contamination rate ε determines the fraction of samples
        flagged as anomalies. For unsupervised detection:

            threshold τ = percentile(scores, ε * 100)
            is_anomaly = score < τ  (for distance-based methods)

        For supervised evaluation with ground truth:
            Precision = |{detected ∩ actual}| / |{detected}|
            Recall = |{detected ∩ actual}| / |{actual}|
            F1 = 2 * Precision * Recall / (Precision + Recall)
    """

    model_config = ConfigDict(frozen=True)

    n_total: int = Field(ge=0, description="Total number of samples")
    n_anomalies_detected: int = Field(ge=0, description="Number flagged as anomalies")
    contamination_rate: float = Field(
        ge=0.0, le=1.0, description="Fraction of samples flagged (n_anomalies/n_total)"
    )

    # Ground truth metrics (None if no ground truth available)
    n_actual_anomalies: int | None = Field(
        default=None, description="Actual anomaly count from ground truth"
    )
    true_positives: int | None = Field(default=None, ge=0)
    false_positives: int | None = Field(default=None, ge=0)
    false_negatives: int | None = Field(default=None, ge=0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1_score: float | None = Field(default=None, ge=0.0, le=1.0)


class AnomalyDetectionConfig(DemoConfig):
    """
    Configuration for anomaly detection task.

    This config supports multiple detection strategies with customizable
    parameters. The Strategy pattern allows easy extension with new methods.

    Attributes:
        strategy: Detection algorithm to use
        contamination: Expected fraction of anomalies (0.0-0.5)
        n_neighbors: Number of neighbors for kNN-based methods
        top_k: Number of top anomalies to return in detail
        use_ground_truth: Whether to evaluate against labels if available
        random_state: Alias for seed (for sklearn compatibility)

    Example:
        >>> config = AnomalyDetectionConfig(
        ...     strategy="datalab_knn",
        ...     contamination=0.02,
        ...     n_neighbors=20,
        ... )
    """

    strategy: StrategyType = Field(default="datalab_knn", description="Detection strategy to use")

    contamination: float = Field(
        default=0.01, ge=0.001, le=0.5, description="Expected fraction of anomalies in the dataset"
    )

    n_neighbors: int = Field(
        default=20,
        ge=2,
        le=100,
        description="Number of neighbors for kNN-based methods (Datalab, LOF)",
    )

    n_estimators: int = Field(
        default=100, ge=10, le=500, description="Number of trees for Isolation Forest"
    )

    top_k: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Number of top anomalies to return in detailed results",
    )

    use_ground_truth: bool = Field(
        default=True, description="Whether to compute metrics against ground truth if available"
    )

    scale_features: bool = Field(
        default=True, description="Whether to standardize features before detection"
    )


class AnomalyDetectionResult(DemoResult):
    """
    Complete results from anomaly detection task.

    This immutable result object captures:
    - Configuration used for the run
    - Summary statistics and metrics
    - Detailed per-sample results for top anomalies
    - Timing information

    Example:
        >>> result = task.run(config)
        >>> print(f"Detected {result.summary.n_anomalies_detected} anomalies")
        >>> if result.summary.precision:
        ...     print(f"Precision: {result.summary.precision:.2%}")
        >>> for row in result.top_anomalies[:5]:
        ...     print(f"  Row {row.index}: score={row.anomaly_score:.4f}")
    """

    model_config = ConfigDict(frozen=True)

    task: Literal["anomaly_detection"] = "anomaly_detection"
    dataset: str = Field(description="Name of the dataset used")
    strategy_used: StrategyType = Field(description="Detection strategy that was used")

    summary: AnomalySummary = Field(description="Summary statistics and metrics")
    strategy_details: StrategyResult = Field(description="Strategy-specific results")

    top_anomalies: list[AnomalyRow] = Field(
        default_factory=list, description="Top K most anomalous samples with details"
    )

    # All anomaly scores for further analysis
    all_scores: list[float] | None = Field(
        default=None, description="Anomaly scores for all samples (optional, can be large)"
    )
    all_predictions: list[bool] | None = Field(
        default=None, description="Anomaly predictions for all samples (optional)"
    )
