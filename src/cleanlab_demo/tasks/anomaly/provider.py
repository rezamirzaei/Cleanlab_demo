"""
Anomaly Detection Data Provider.

Abstract base class for anomaly detection data providers.
Concrete implementations should provide datasets suitable for
anomaly/fraud detection tasks.

Mathematical Context:
    Anomaly detection assumes data follows a distribution P(x) where
    anomalies are samples from a different distribution Q(x) with
    P(Q) << P(P), i.e., anomalies are rare events.

    The contamination rate ε = n_anomalies / n_total typically ranges
    from 0.001 to 0.1 depending on the domain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd


if TYPE_CHECKING:
    import numpy as np


class AnomalyDetectionDataProvider(ABC):
    """
    Abstract base class for anomaly detection data providers.

    This class defines the interface that all anomaly detection data
    providers must implement. It follows the Strategy pattern, allowing
    different datasets to be plugged into the same detection pipeline.

    Design Principles:
        - Single Responsibility: Only handles data loading, not detection
        - Open/Closed: New datasets extend without modifying existing code
        - Liskov Substitution: All providers are interchangeable

    Attributes:
        name: Human-readable dataset identifier
        feature_cols: List of feature column names (optional, auto-detected if None)
        label_col: Column containing ground truth labels (if available)
        has_ground_truth: Whether dataset has known anomaly labels
        expected_contamination: Expected fraction of anomalies (0.0-1.0)

    Example:
        >>> class MyDataProvider(AnomalyDetectionDataProvider):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_dataset"
        ...
        ...     @property
        ...     def has_ground_truth(self) -> bool:
        ...         return True
        ...
        ...     def load(self, seed: int) -> pd.DataFrame:
        ...         return pd.read_csv("my_data.csv")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name of the dataset.

        Returns:
            Dataset name for logging, reporting, and identification.
        """
        ...

    @property
    @abstractmethod
    def has_ground_truth(self) -> bool:
        """
        Whether the dataset has ground truth anomaly labels.

        If True, the dataset contains a column with known anomaly labels
        (0 = normal, 1 = anomaly), enabling precision/recall evaluation.

        Returns:
            True if ground truth labels are available, False otherwise.
        """
        ...

    @property
    def label_col(self) -> str | None:
        """
        Column name containing ground truth anomaly labels.

        Returns:
            Column name if has_ground_truth is True, None otherwise.
            Convention: 0 = normal, 1 = anomaly
        """
        return None

    @property
    def feature_cols(self) -> list[str] | None:
        """
        List of feature column names to use for detection.

        If None, all numeric columns except label_col are used.

        Returns:
            List of column names or None for auto-detection.
        """
        return None

    @property
    def expected_contamination(self) -> float:
        """
        Expected fraction of anomalies in the dataset.

        This is used as a prior for detection algorithms and
        for setting default thresholds.

        Returns:
            Float in range [0.0, 1.0], default is 0.01 (1%)
        """
        return 0.01

    @abstractmethod
    def load(self, seed: int, **kwargs: Any) -> pd.DataFrame:
        """
            Load and return the dataset as a DataFrame.

            The returned DataFrame should contain:
            - Feature columns (numeric, will be scaled internally)
            - Optionally, a label column with ground truth (0/1)

            Args:
                seed: Random seed for reproducible sampling/shuffling.
                **kwargs: Additional provider-specific options (e.g., max_rows).

            Returns:
                DataFrame with features and optionally ground truth labels.

            Raises:
                DataLoadError: If loading fails.

        Example:
            >>> provider = SyntheticAnomalyProvider()
            >>> df = provider.load(seed=42, max_rows=10000)
            >>> df.shape
            (10000, 11)
        """
        ...

    def get_features_and_labels(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
        """
        Extract features and labels from loaded DataFrame.

        This helper method separates features from labels and handles
        the case where ground truth is not available.

        Args:
            df: DataFrame returned by load()

        Returns:
            Tuple of (X features DataFrame, y labels array or None)
        """
        import numpy as np

        if self.feature_cols is not None:
            X = df[self.feature_cols].copy()
        else:
            # Auto-detect: use all numeric columns except label
            exclude = {self.label_col} if self.label_col else set()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            X = df[[c for c in numeric_cols if c not in exclude]].copy()

        y = None
        if self.has_ground_truth and self.label_col and self.label_col in df.columns:
            y = df[self.label_col].to_numpy(dtype=int)

        return X, y

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"has_ground_truth={self.has_ground_truth}, "
            f"expected_contamination={self.expected_contamination:.2%})"
        )
