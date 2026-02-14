"""
Anomaly Detection Data Providers.

This module provides concrete data providers for anomaly detection tasks.
Each provider encapsulates dataset-specific loading logic while
conforming to the AnomalyDetectionDataProvider interface.

Available Providers:
    - SyntheticAnomalyProvider: Generates synthetic data with known anomalies
    - CreditCardFraudProvider: Credit Card Fraud Detection dataset (Kaggle)
    - CaliforniaHousingAnomalyProvider: California Housing with synthetic anomalies

Design Pattern:
    All providers implement the Strategy pattern, allowing interchangeable
    datasets with the same AnomalyDetectionTask.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleanlab_demo.tasks.anomaly.provider import AnomalyDetectionDataProvider


class SyntheticAnomalyProvider(AnomalyDetectionDataProvider):
    """
    Synthetic anomaly dataset provider for testing and demonstration.

    Generates a dataset with known anomalies for algorithm evaluation.
    Normal points are sampled from a multivariate Gaussian, while
    anomalies are sampled from a uniform distribution (scattered).

    Mathematical Model:
        Normal points: x ~ N(μ, Σ) where Σ = I (identity covariance)
        Anomaly points: x ~ U(a, b) where [a, b] = [-6, 6] (uniform)

    This creates a clear separation since anomalies are unlikely
    under the Gaussian model but spread uniformly.

    Attributes:
        n_samples: Total number of samples to generate
        n_features: Number of features per sample
        contamination: Fraction of samples that are anomalies

    Example:
        >>> provider = SyntheticAnomalyProvider(n_samples=1000, contamination=0.05)
        >>> df = provider.load(seed=42)
        >>> print(f"Shape: {df.shape}, Anomalies: {df['is_anomaly'].sum()}")
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        contamination: float = 0.05,
    ) -> None:
        """
        Initialize synthetic data provider.

        Args:
            n_samples: Total number of samples (normal + anomalies)
            n_features: Dimensionality of feature space
            contamination: Fraction of samples to be anomalies
        """
        self._n_samples = n_samples
        self._n_features = n_features
        self._contamination = contamination

    @property
    def name(self) -> str:
        return f"synthetic_anomaly_{self._n_samples}x{self._n_features}"

    @property
    def has_ground_truth(self) -> bool:
        return True

    @property
    def label_col(self) -> str:
        return "is_anomaly"

    @property
    def expected_contamination(self) -> float:
        return self._contamination

    def load(self, seed: int, **kwargs: Any) -> pd.DataFrame:
        """
        Generate synthetic anomaly dataset.

        Algorithm:
            1. Determine n_anomalies = contamination * n_samples
            2. Sample n_normal points from N(0, I)
            3. Sample n_anomalies points from U(-6, 6)
            4. Combine and shuffle

        Args:
            seed: Random seed for reproducibility

        Returns:
            DataFrame with features and 'is_anomaly' label column
        """
        rng = np.random.default_rng(seed=seed)

        n_anomalies = int(self._n_samples * self._contamination)
        n_normal = self._n_samples - n_anomalies

        # Generate normal points from standard Gaussian
        X_normal = rng.standard_normal((n_normal, self._n_features))
        y_normal = np.zeros(n_normal, dtype=int)

        # Generate anomalies from uniform distribution
        # Range [-6, 6] is unlikely under N(0, 1)
        X_anomalies = rng.uniform(-6, 6, (n_anomalies, self._n_features))
        y_anomalies = np.ones(n_anomalies, dtype=int)

        # Combine and shuffle
        X = np.vstack([X_normal, X_anomalies])
        y = np.concatenate([y_normal, y_anomalies])

        indices = rng.permutation(len(X))
        X = X[indices]
        y = y[indices]

        # Create DataFrame
        feature_cols = [f"feature_{i}" for i in range(self._n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df[self.label_col] = y

        return df


class CaliforniaHousingAnomalyProvider(AnomalyDetectionDataProvider):
    """
    California Housing dataset adapted for anomaly detection.

    Uses the sklearn California Housing dataset and optionally injects
    synthetic anomalies for evaluation. Without injection, relies on
    natural data distribution variations.

    The California Housing dataset contains 20,640 samples with 8 features:
        - MedInc: Median income in block group
        - HouseAge: Median house age in block group
        - AveRooms: Average number of rooms per household
        - AveBedrms: Average number of bedrooms per household
        - Population: Block group population
        - AveOccup: Average house occupancy
        - Latitude: Block group latitude
        - Longitude: Block group longitude

    Synthetic Anomaly Injection:
        Anomalies are created by scaling features by a large factor,
        creating points far from the normal distribution.

    Example:
        >>> provider = CaliforniaHousingAnomalyProvider(inject_anomalies=True)
        >>> df = provider.load(seed=42)
    """

    def __init__(
        self,
        max_rows: int = 10000,
        inject_anomalies: bool = True,
        contamination: float = 0.02,
        anomaly_scale: float = 10.0,
    ) -> None:
        """
        Initialize California Housing provider.

        Args:
            max_rows: Maximum number of samples to load
            inject_anomalies: Whether to inject synthetic anomalies
            contamination: Fraction of samples to make anomalous (if injecting)
            anomaly_scale: Scale factor for creating anomalies
        """
        self._max_rows = max_rows
        self._inject_anomalies = inject_anomalies
        self._contamination = contamination
        self._anomaly_scale = anomaly_scale

    @property
    def name(self) -> str:
        suffix = "_synthetic" if self._inject_anomalies else "_natural"
        return f"california_housing{suffix}"

    @property
    def has_ground_truth(self) -> bool:
        return self._inject_anomalies

    @property
    def label_col(self) -> str | None:
        return "is_anomaly" if self._inject_anomalies else None

    @property
    def expected_contamination(self) -> float:
        return self._contamination

    def load(self, seed: int, **kwargs: Any) -> pd.DataFrame:
        """
        Load California Housing data with optional anomaly injection.

        Args:
            seed: Random seed for reproducibility

        Returns:
            DataFrame with housing features and optional anomaly labels
        """
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing(as_frame=True)
        df = data.frame.drop(columns=["MedHouseVal"])  # Drop target

        # Sample if too large
        if len(df) > self._max_rows:
            df = df.sample(n=self._max_rows, random_state=seed).reset_index(drop=True)

        if self._inject_anomalies:
            df = self._inject_synthetic_anomalies(df, seed)

        return df

    def _inject_synthetic_anomalies(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        """
        Inject synthetic anomalies by scaling features.

        Args:
            df: Original DataFrame
            seed: Random seed

        Returns:
            DataFrame with injected anomalies and 'is_anomaly' column
        """
        rng = np.random.default_rng(seed=seed)

        n_anomalies = int(len(df) * self._contamination)
        anomaly_indices = rng.choice(len(df), size=n_anomalies, replace=False)

        # Create anomaly labels
        df["is_anomaly"] = 0
        df.loc[anomaly_indices, "is_anomaly"] = 1

        # Scale features for anomalies
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "is_anomaly"]

        for idx in anomaly_indices:
            # Randomly select features to corrupt
            n_features_to_corrupt = rng.integers(1, len(numeric_cols) // 2 + 1)
            features_to_corrupt = rng.choice(
                numeric_cols, size=n_features_to_corrupt, replace=False
            )

            for col in features_to_corrupt:
                # Scale by large factor with random sign
                sign = rng.choice([-1, 1])
                df.loc[idx, col] *= sign * self._anomaly_scale

        return df


class ForestCoverAnomalyProvider(AnomalyDetectionDataProvider):
    """
    Forest Cover Type dataset adapted for anomaly detection.

    Uses the sklearn Covertype dataset, treating rare classes as anomalies.
    This provides a realistic multi-class to anomaly conversion scenario.

    Dataset contains 581,012 samples with 54 features describing
    cartographic variables for 30x30 meter forest cells.

    Anomaly Definition:
        Rare forest cover types (classes with < 5% of samples) are
        treated as anomalies. This mirrors real-world scenarios where
        rare events need detection.

    Example:
        >>> provider = ForestCoverAnomalyProvider(max_rows=50000)
        >>> df = provider.load(seed=42)
        >>> print(f"Anomaly rate: {df['is_anomaly'].mean():.2%}")
    """

    def __init__(
        self,
        max_rows: int = 50000,
        rare_threshold: float = 0.05,
    ) -> None:
        """
        Initialize Forest Cover provider.

        Args:
            max_rows: Maximum samples to load
            rare_threshold: Classes with frequency < threshold are anomalies
        """
        self._max_rows = max_rows
        self._rare_threshold = rare_threshold

    @property
    def name(self) -> str:
        return "forest_cover_anomaly"

    @property
    def has_ground_truth(self) -> bool:
        return True

    @property
    def label_col(self) -> str:
        return "is_anomaly"

    @property
    def expected_contamination(self) -> float:
        # Approximate based on dataset statistics
        return 0.03

    def load(self, seed: int, **kwargs: Any) -> pd.DataFrame:
        """
        Load Forest Cover data with rare classes as anomalies.

        Args:
            seed: Random seed for reproducibility

        Returns:
            DataFrame with features and 'is_anomaly' label
        """
        from sklearn.datasets import fetch_covtype

        data = fetch_covtype(as_frame=True)
        df = data.frame.copy()

        # Sample if too large
        if len(df) > self._max_rows:
            df = df.sample(n=self._max_rows, random_state=seed).reset_index(drop=True)

        # Determine rare classes
        target_col = "Cover_Type"
        class_freq = df[target_col].value_counts(normalize=True)
        rare_classes = class_freq[class_freq < self._rare_threshold].index.tolist()

        # Create anomaly label
        df["is_anomaly"] = df[target_col].isin(rare_classes).astype(int)

        # Drop original target
        return df.drop(columns=[target_col])


class KDDCup99Provider(AnomalyDetectionDataProvider):
    """
    KDD Cup 1999 Network Intrusion dataset for anomaly detection.

    Classic benchmark dataset for network intrusion detection.
    Contains 4,898,431 samples with 41 features describing
    TCP connection records.

    Anomaly Definition:
        Normal connections are labeled 'normal', all other
        connection types (attacks) are treated as anomalies.

    Note:
        Due to the large size, only a subset is loaded by default.
        The 10% subset (kddcup.data_10_percent) is commonly used.

    Example:
        >>> provider = KDDCup99Provider(max_rows=100000)
        >>> df = provider.load(seed=42)
    """

    def __init__(
        self,
        max_rows: int = 100000,
        percent10: bool = True,
    ) -> None:
        """
        Initialize KDD Cup 99 provider.

        Args:
            max_rows: Maximum samples to load
            percent10: Use 10% subset (faster loading)
        """
        self._max_rows = max_rows
        self._percent10 = percent10

    @property
    def name(self) -> str:
        return "kddcup99"

    @property
    def has_ground_truth(self) -> bool:
        return True

    @property
    def label_col(self) -> str:
        return "is_anomaly"

    @property
    def expected_contamination(self) -> float:
        # KDD Cup has high attack rate (~20% in 10% subset)
        return 0.20

    def load(self, seed: int, **kwargs: Any) -> pd.DataFrame:
        """
        Load KDD Cup 99 data with attacks as anomalies.

        Args:
            seed: Random seed for reproducibility

        Returns:
            DataFrame with numeric features and 'is_anomaly' label
        """
        from sklearn.datasets import fetch_kddcup99

        data = fetch_kddcup99(
            subset=None,
            percent10=self._percent10,
            as_frame=True,
        )
        df = data.frame.copy()

        # Sample if too large
        if len(df) > self._max_rows:
            df = df.sample(n=self._max_rows, random_state=seed).reset_index(drop=True)

        # Create anomaly label (normal vs attacks)
        label_col = "labels"
        df["is_anomaly"] = (~df[label_col].isin([b"normal.", "normal."])).astype(int)

        # Drop non-numeric and label columns
        df = df.drop(columns=[label_col])

        # Keep only numeric columns + anomaly label
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[[*numeric_cols, "is_anomaly"]]


__all__ = [
    "AnomalyDetectionDataProvider",
    "CaliforniaHousingAnomalyProvider",
    "ForestCoverAnomalyProvider",
    "KDDCup99Provider",
    "SyntheticAnomalyProvider",
]
