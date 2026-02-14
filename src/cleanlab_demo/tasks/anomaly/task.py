"""
Anomaly Detection Task.

This module provides the main task orchestrator for anomaly detection.
It coordinates data loading, feature preparation, strategy execution,
and result aggregation.

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────┐
    │                   AnomalyDetectionTask                      │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  ┌──────────────┐    ┌─────────────┐    ┌───────────────┐  │
    │  │ DataProvider │───▶│ Preprocessor│───▶│   Strategy    │  │
    │  │              │    │ (Scaler)    │    │ (kNN/IF/LOF)  │  │
    │  └──────────────┘    └─────────────┘    └───────────────┘  │
    │                                                │            │
    │                                                ▼            │
    │                                         ┌───────────┐       │
    │                                         │  Result   │       │
    │                                         │ Builder   │       │
    │                                         └───────────┘       │
    └─────────────────────────────────────────────────────────────┘

Design Principles:
    - Dependency Injection: DataProvider and Strategy are injected
    - Single Responsibility: Task only orchestrates, doesn't implement detection
    - Immutable Results: Results are frozen Pydantic models
    - Comprehensive Logging: Timing and metrics tracked throughout
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from cleanlab_demo.tasks.anomaly.provider import AnomalyDetectionDataProvider
from cleanlab_demo.tasks.anomaly.schemas import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
    AnomalyRow,
    AnomalySummary,
    StrategyResult,
)
from cleanlab_demo.tasks.anomaly.strategies import (
    AnomalyDetectionStrategy,
    get_strategy,
)


class AnomalyDetectionTask:
    """
    Main task orchestrator for anomaly detection.

    This class coordinates the entire anomaly detection workflow:
    1. Load data from provider
    2. Preprocess features (scaling)
    3. Execute detection strategy
    4. Compute metrics (if ground truth available)
    5. Build and return results

    The task follows the Template Method pattern where the high-level
    algorithm is fixed, but individual steps can be customized via
    injected dependencies (provider, strategy).

    Attributes:
        data_provider: Provider for loading the dataset
        strategy: Detection strategy (optional, can be set in config)

    Example:
        >>> from cleanlab_demo.tasks.anomaly import AnomalyDetectionTask
        >>> from cleanlab_demo.data.providers.anomaly import SyntheticAnomalyProvider
        >>>
        >>> provider = SyntheticAnomalyProvider(n_samples=1000, contamination=0.05)
        >>> task = AnomalyDetectionTask(provider)
        >>>
        >>> config = AnomalyDetectionConfig(strategy="isolation_forest")
        >>> result = task.run(config)
        >>>
        >>> print(f"Detected: {result.summary.n_anomalies_detected}")
        >>> print(f"Precision: {result.summary.precision:.2%}")
    """

    def __init__(
        self,
        data_provider: AnomalyDetectionDataProvider,
        strategy: AnomalyDetectionStrategy | None = None,
    ) -> None:
        """
        Initialize the anomaly detection task.

        Args:
            data_provider: Data provider instance for loading data
            strategy: Optional pre-configured strategy. If None,
                      strategy is determined by config at runtime.
        """
        self.data_provider = data_provider
        self._strategy = strategy

    def run(self, config: AnomalyDetectionConfig) -> AnomalyDetectionResult:
        """
        Execute the complete anomaly detection workflow.

        This method orchestrates the entire detection pipeline:

        1. **Data Loading**: Load dataset from provider
        2. **Feature Extraction**: Separate features from labels
        3. **Preprocessing**: Standardize features (optional)
        4. **Strategy Selection**: Get strategy from config or use injected
        5. **Detection**: Run fit_predict on the strategy
        6. **Metrics Computation**: Calculate precision/recall if ground truth
        7. **Result Building**: Aggregate into immutable result object

        Args:
            config: Configuration specifying strategy, contamination, etc.

        Returns:
            AnomalyDetectionResult with summary, top anomalies, and metrics

        Raises:
            ValueError: If strategy is unknown
            DataLoadError: If data loading fails
        """
        # Step 1: Load data
        df = self.data_provider.load(seed=config.seed)

        # Step 2: Extract features and labels
        X_df, y_true = self.data_provider.get_features_and_labels(df)

        # Step 3: Preprocess features
        if config.scale_features:
            scaler = StandardScaler()
            X = scaler.fit_transform(X_df.to_numpy(dtype=np.float64))
        else:
            X = X_df.to_numpy(dtype=np.float64)

        # Step 4: Get strategy
        strategy = self._strategy or get_strategy(config.strategy)

        # Step 5: Run detection
        detection_start = time.perf_counter()
        scores, predictions = strategy.fit_predict(X, config)
        detection_time = time.perf_counter() - detection_start

        # Step 6: Compute metrics
        summary = self._compute_summary(
            scores=scores,
            predictions=predictions,
            y_true=y_true,
            config=config,
        )

        # Step 7: Build top anomalies list
        top_anomalies = self._build_top_anomalies(
            scores=scores,
            predictions=predictions,
            y_true=y_true,
            top_k=config.top_k,
        )

        # Build strategy result
        strategy_result = StrategyResult(
            strategy_name=strategy.name,
            n_detected=int(predictions.sum()),
            threshold=float(strategy.get_threshold(scores, config.contamination)),
            execution_time_seconds=detection_time,
            precision=summary.precision,
            recall=summary.recall,
            f1_score=summary.f1_score,
        )

        return AnomalyDetectionResult(
            dataset=self.data_provider.name,
            strategy_used=config.strategy,
            summary=summary,
            strategy_details=strategy_result,
            top_anomalies=top_anomalies,
            all_scores=scores.tolist() if len(scores) <= 50000 else None,
            all_predictions=predictions.tolist() if len(predictions) <= 50000 else None,
        )

    def _compute_summary(
        self,
        scores: np.ndarray,
        predictions: np.ndarray,
        y_true: np.ndarray | None,
        config: AnomalyDetectionConfig,
    ) -> AnomalySummary:
        """
        Compute summary statistics and metrics.

        Args:
            scores: Anomaly scores for each sample
            predictions: Boolean predictions (True = anomaly)
            y_true: Ground truth labels if available (0 = normal, 1 = anomaly)
            config: Detection configuration

        Returns:
            AnomalySummary with counts and metrics
        """
        n_total = len(scores)
        n_detected = int(predictions.sum())
        contamination_rate = n_detected / n_total if n_total > 0 else 0.0

        # Initialize metrics as None (no ground truth)
        n_actual = None
        tp = fp = fn = None
        precision = recall = f1 = None

        # Compute metrics if ground truth is available
        if y_true is not None and config.use_ground_truth:
            y_true_bool = y_true.astype(bool)
            n_actual = int(y_true_bool.sum())

            # True Positives: predicted anomaly AND actually anomaly
            tp = int(np.sum(predictions & y_true_bool))
            # False Positives: predicted anomaly BUT actually normal
            fp = int(np.sum(predictions & ~y_true_bool))
            # False Negatives: predicted normal BUT actually anomaly
            fn = int(np.sum(~predictions & y_true_bool))

            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0

            # Recall = TP / (TP + FN)
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0

            # F1 = 2 * P * R / (P + R)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        return AnomalySummary(
            n_total=n_total,
            n_anomalies_detected=n_detected,
            contamination_rate=contamination_rate,
            n_actual_anomalies=n_actual,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
        )

    def _build_top_anomalies(
        self,
        scores: np.ndarray,
        predictions: np.ndarray,
        y_true: np.ndarray | None,
        top_k: int,
    ) -> list[AnomalyRow]:
        """
        Build list of top K most anomalous samples.

        Samples are sorted by anomaly score (ascending, lower = more anomalous)
        and the top K are returned with details.

        Args:
            scores: Anomaly scores for each sample
            predictions: Boolean predictions
            y_true: Ground truth if available
            top_k: Number of top anomalies to return

        Returns:
            List of AnomalyRow objects for top anomalies
        """
        # Sort indices by score (ascending = most anomalous first)
        sorted_indices = np.argsort(scores)

        top_anomalies = []
        for raw_idx in sorted_indices[:top_k]:
            idx = int(raw_idx)
            gt = int(y_true[idx]) if y_true is not None else None

            top_anomalies.append(
                AnomalyRow(
                    index=idx,
                    anomaly_score=float(scores[idx]),
                    is_anomaly=bool(predictions[idx]),
                    ground_truth=gt,
                )
            )

        return top_anomalies

    def __repr__(self) -> str:
        return (
            f"AnomalyDetectionTask(provider={self.data_provider.name!r}, strategy={self._strategy})"
        )


def run_anomaly_detection(
    data_provider: AnomalyDetectionDataProvider,
    config: AnomalyDetectionConfig | None = None,
    **kwargs: Any,
) -> AnomalyDetectionResult:
    """
    Convenience function to run anomaly detection in one call.

    This function provides a simple functional interface to the
    AnomalyDetectionTask for quick experimentation.

    Args:
        data_provider: Data provider instance
        config: Configuration object. If None, created from kwargs.
        **kwargs: Configuration parameters if config is None

    Returns:
        AnomalyDetectionResult with all metrics and predictions

    Example:
        >>> from cleanlab_demo.tasks.anomaly import run_anomaly_detection
        >>> from cleanlab_demo.data.providers.anomaly import SyntheticAnomalyProvider
        >>>
        >>> result = run_anomaly_detection(
        ...     SyntheticAnomalyProvider(),
        ...     strategy="isolation_forest",
        ...     contamination=0.05,
        ... )
        >>> print(result.summary)
    """
    cfg = config or AnomalyDetectionConfig(**kwargs)
    return AnomalyDetectionTask(data_provider).run(cfg)
