"""
Anomaly Detection Strategies.

This module implements the Strategy pattern for anomaly detection algorithms.
Each strategy encapsulates a specific detection method, allowing easy
comparison and extension.

Mathematical Foundations:

1. **kNN Distance (Datalab)**:
   For each sample xᵢ, compute average distance to k nearest neighbors:

       score(xᵢ) = (1/k) Σⱼ∈kNN(xᵢ) d(xᵢ, xⱼ)

   where d(·,·) is Euclidean distance in standardized feature space.
   Higher scores indicate more isolated (anomalous) points.
   Cleanlab normalizes scores to [0, 1] where lower = more anomalous.

2. **Isolation Forest**:
   Anomalies are "few and different" — easier to isolate.
   For each sample, measure average path length h(x) across trees:

       score(x) = 2^(-E[h(x)] / c(n))

   where c(n) is average path length in unsuccessful search in BST.
   Scores close to 1 indicate anomalies, close to 0.5 = normal.

3. **Local Outlier Factor (LOF)**:
   Compares local density of a point to its neighbors:

       LOF(x) = (1/k) Σⱼ∈kNN(x) [lrd(xⱼ) / lrd(x)]

   where lrd = local reachability density.
   LOF ≈ 1 means similar density to neighbors (normal).
   LOF >> 1 means lower density than neighbors (anomaly).

Design Principles:
    - Strategy Pattern: Interchangeable algorithms with common interface
    - Single Responsibility: Each strategy does one thing well
    - Open/Closed: Add new strategies without modifying existing code
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from cleanlab_demo.tasks.anomaly.schemas import (
    AnomalyDetectionConfig,
    StrategyType,
)


class AnomalyDetectionStrategy(ABC):
    """
    Abstract base class for anomaly detection strategies.

    All strategies implement the same interface, allowing the Task
    to use them interchangeably (Strategy pattern).

    Subclasses must implement:
        - name: Identifier for the strategy
        - fit_predict: Core detection logic

    Example:
        >>> strategy = IsolationForestStrategy()
        >>> scores, predictions = strategy.fit_predict(X, config)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the strategy."""
        ...

    @abstractmethod
    def fit_predict(
        self,
        X: np.ndarray,
        config: AnomalyDetectionConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the model and predict anomaly scores.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            config: Detection configuration

        Returns:
            Tuple of:
                - scores: Anomaly scores for each sample (shape: n_samples,)
                          Lower scores = more anomalous (normalized convention)
                - predictions: Boolean array where True = anomaly
        """
        ...

    def get_threshold(self, scores: np.ndarray, contamination: float) -> float:
        """
        Compute threshold for anomaly classification.

        By convention, scores are normalized so lower = more anomalous.
        The threshold is set at the contamination percentile.

        Args:
            scores: Anomaly scores (lower = more anomalous)
            contamination: Expected fraction of anomalies

        Returns:
            Threshold value: samples with score < threshold are anomalies
        """
        return float(np.percentile(scores, contamination * 100))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DataLabKNNStrategy(AnomalyDetectionStrategy):
    """
    Cleanlab Datalab kNN-based outlier detection strategy.

    This strategy uses Cleanlab's Datalab module which computes
    outlier scores based on k-nearest neighbor distances.

    Algorithm:
        1. Standardize features to zero mean, unit variance
        2. Compute kNN distances for each sample
        3. Normalize distances to outlier scores in [0, 1]
        4. Lower scores indicate more anomalous samples

    Mathematical Details:
        The outlier score is based on the average distance to k neighbors:

            raw_score(xᵢ) = mean(||xᵢ - xⱼ||₂ for j in kNN(xᵢ))

        Cleanlab normalizes these to [0, 1] using a cumulative distribution,
        where lower percentile = more outlying.

    Note:
        Datalab requires a DataFrame with a label column. For unsupervised
        anomaly detection, we create a dummy label column.
    """

    @property
    def name(self) -> str:
        return "datalab_knn"

    def fit_predict(
        self,
        X: np.ndarray,
        config: AnomalyDetectionConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Use Cleanlab Datalab for kNN-based outlier detection.

        Note: Datalab expects a DataFrame with labels. We create
        a dummy regression target for unsupervised detection.
        """
        try:
            import pandas as pd
            from cleanlab.datalab.datalab import Datalab

            # Create DataFrame with dummy label for Datalab
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            df["_dummy_label"] = np.zeros(len(df))  # Dummy for unsupervised

            # Initialize Datalab with regression task (works for continuous dummy)
            datalab = Datalab(
                data=df,
                label_name="_dummy_label",
                task="regression",
                verbosity=0,
            )

            # Find outlier issues using kNN
            issue_types = {"outlier": {"knn": config.n_neighbors}}
            datalab.find_issues(features=X, issue_types=issue_types)

            # Extract results
            issues = datalab.get_issues()

            if "outlier_score" in issues.columns:
                # Datalab scores: lower = more outlying
                scores = issues["outlier_score"].fillna(0.5).to_numpy()
            else:
                # Fallback if outlier detection failed
                scores = np.full(len(X), 0.5)
        except ImportError:
            # Cleanlab is optional at runtime; fallback to pure sklearn kNN distances.
            # This preserves the lower-score-means-more-anomalous convention.
            from sklearn.neighbors import NearestNeighbors

            n_neighbors = min(config.n_neighbors + 1, len(X))
            knn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
            knn.fit(X)
            distances, _ = knn.kneighbors(X)

            # Drop self-distance (first column is usually 0.0 for the sample itself).
            neighbor_distances = distances[:, 1:] if distances.shape[1] > 1 else distances
            raw_scores = neighbor_distances.mean(axis=1)

            min_score, max_score = raw_scores.min(), raw_scores.max()
            if max_score > min_score:
                # Invert normalized distance: high distance -> low anomaly score
                scores = 1.0 - ((raw_scores - min_score) / (max_score - min_score))
            else:
                scores = np.full(len(X), 0.5)

        # Determine threshold and predictions
        threshold = self.get_threshold(scores, config.contamination)
        predictions = scores < threshold

        return scores.astype(np.float64), predictions.astype(bool)


class IsolationForestStrategy(AnomalyDetectionStrategy):
    """
    Isolation Forest anomaly detection strategy.

    Isolation Forest isolates anomalies by randomly selecting features
    and split values. Anomalies require fewer splits to isolate.

    Algorithm:
        1. Build ensemble of isolation trees (random splits)
        2. For each sample, compute average path length across trees
        3. Shorter paths indicate easier-to-isolate (anomalous) points

    Mathematical Details:
        The anomaly score s(x, n) for a sample x in dataset of size n:

            s(x, n) = 2^(-E[h(x)] / c(n))

        where:
            - h(x) = path length to isolate x
            - c(n) = 2H(n-1) - 2(n-1)/n ≈ average path in BST
            - H(i) = ln(i) + 0.5772... (harmonic number)

        Score interpretation:
            - s → 1: definite anomaly
            - s → 0.5: normal point
            - s → 0: definite normal (rare)

    Reference:
        Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
        "Isolation forest." ICDM 2008.
    """

    @property
    def name(self) -> str:
        return "isolation_forest"

    def fit_predict(
        self,
        X: np.ndarray,
        config: AnomalyDetectionConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Use sklearn's Isolation Forest for anomaly detection.
        """
        from sklearn.ensemble import IsolationForest

        clf = IsolationForest(
            n_estimators=config.n_estimators,
            contamination=config.contamination,
            random_state=config.seed,
            n_jobs=-1,
        )

        # Fit and predict (-1 = anomaly, 1 = normal)
        y_pred = clf.fit_predict(X)

        # Get anomaly scores (more negative = more anomalous)
        # score_samples returns: negative of average path length
        raw_scores = clf.score_samples(X)

        # Normalize to [0, 1] where lower = more anomalous
        # (consistent with Datalab convention)
        min_score, max_score = raw_scores.min(), raw_scores.max()
        if max_score > min_score:
            scores = (raw_scores - min_score) / (max_score - min_score)
        else:
            scores = np.full_like(raw_scores, 0.5)

        predictions = y_pred == -1  # -1 indicates anomaly

        return scores.astype(np.float64), predictions.astype(bool)


class LocalOutlierFactorStrategy(AnomalyDetectionStrategy):
    """
    Local Outlier Factor (LOF) anomaly detection strategy.

    LOF measures local deviation of density compared to neighbors.
    Points with substantially lower density than neighbors are outliers.

    Algorithm:
        1. For each point, find k nearest neighbors
        2. Compute local reachability density (LRD) for each point
        3. Compare LRD to neighbors' LRD to get LOF score

    Mathematical Details:
        For point p with k neighbors N_k(p):

        Reachability distance:
            reach_dist_k(p, o) = max(k_dist(o), d(p, o))

        Local reachability density:
            lrd_k(p) = 1 / (Σ_{o∈N_k(p)} reach_dist_k(p, o) / k)

        Local Outlier Factor:
            LOF_k(p) = (Σ_{o∈N_k(p)} lrd_k(o)) / (k · lrd_k(p))

        Score interpretation:
            - LOF ≈ 1: similar density to neighbors (normal)
            - LOF >> 1: lower density than neighbors (outlier)
            - LOF << 1: higher density than neighbors (cluster core)

    Reference:
        Breunig, Markus M., et al.
        "LOF: identifying density-based local outliers." ACM SIGMOD 2000.
    """

    @property
    def name(self) -> str:
        return "local_outlier_factor"

    def fit_predict(
        self,
        X: np.ndarray,
        config: AnomalyDetectionConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Use sklearn's Local Outlier Factor for anomaly detection.
        """
        from sklearn.neighbors import LocalOutlierFactor

        clf = LocalOutlierFactor(
            n_neighbors=config.n_neighbors,
            contamination=config.contamination,
            novelty=False,  # Use for outlier detection (not novelty)
            n_jobs=-1,
        )

        # Fit and predict (-1 = anomaly, 1 = normal)
        y_pred = clf.fit_predict(X)

        # Get negative outlier factor (more negative = more anomalous)
        raw_scores = clf.negative_outlier_factor_

        # Normalize to [0, 1] where lower = more anomalous
        min_score, max_score = raw_scores.min(), raw_scores.max()
        if max_score > min_score:
            scores = (raw_scores - min_score) / (max_score - min_score)
        else:
            scores = np.full_like(raw_scores, 0.5)

        predictions = y_pred == -1

        return scores.astype(np.float64), predictions.astype(bool)


class EnsembleStrategy(AnomalyDetectionStrategy):
    """
    Ensemble anomaly detection combining multiple strategies.

    This strategy combines predictions from multiple base strategies
    using score averaging and voting. Ensemble methods typically
    improve robustness by reducing individual method biases.

    Algorithm:
        1. Run all base strategies (Datalab, IF, LOF)
        2. Normalize scores to [0, 1] for each method
        3. Average scores across methods
        4. Apply threshold on averaged scores

    Mathematical Details:
        For M strategies with scores s₁(x), ..., sₘ(x):

            ensemble_score(x) = (1/M) Σₘ sₘ(x)

        This reduces variance: Var(ensemble) = Var(single)/M
        assuming uncorrelated strategies.
    """

    @property
    def name(self) -> str:
        return "ensemble"

    def fit_predict(
        self,
        X: np.ndarray,
        config: AnomalyDetectionConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Combine multiple strategies via score averaging.
        """
        strategies: list[AnomalyDetectionStrategy] = [
            IsolationForestStrategy(),
            LocalOutlierFactorStrategy(),
        ]

        # Try to add Datalab, but don't fail if unavailable.
        from importlib.util import find_spec

        if find_spec("cleanlab.datalab.datalab") is not None:
            strategies.insert(0, DataLabKNNStrategy())

        all_scores = []
        for strategy in strategies:
            try:
                scores, _ = strategy.fit_predict(X, config)
                all_scores.append(scores)
            except Exception:
                continue  # Skip failed strategies

        if not all_scores:
            # Fallback if all strategies fail
            scores = np.full(len(X), 0.5)
            predictions = np.zeros(len(X), dtype=bool)
            return scores, predictions

        # Average scores across strategies
        stacked = np.stack(all_scores, axis=0)
        ensemble_scores = np.mean(stacked, axis=0)

        # Apply threshold
        threshold = self.get_threshold(ensemble_scores, config.contamination)
        predictions = ensemble_scores < threshold

        return ensemble_scores.astype(np.float64), predictions.astype(bool)


# Strategy registry for factory pattern
_STRATEGY_REGISTRY: dict[StrategyType, type[AnomalyDetectionStrategy]] = {
    "datalab_knn": DataLabKNNStrategy,
    "isolation_forest": IsolationForestStrategy,
    "local_outlier_factor": LocalOutlierFactorStrategy,
    "ensemble": EnsembleStrategy,
}


def get_strategy(strategy_type: StrategyType) -> AnomalyDetectionStrategy:
    """
    Factory function to get a strategy instance by type.

    This implements the Factory pattern for creating strategies.

    Args:
        strategy_type: One of the supported strategy types

    Returns:
        Instance of the requested strategy

    Raises:
        ValueError: If strategy_type is not recognized

    Example:
        >>> strategy = get_strategy("isolation_forest")
        >>> scores, predictions = strategy.fit_predict(X, config)
    """
    if strategy_type not in _STRATEGY_REGISTRY:
        available = ", ".join(_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {strategy_type}. Available: {available}")

    return _STRATEGY_REGISTRY[strategy_type]()


def list_strategies() -> list[StrategyType]:
    """
    List all available detection strategies.

    Returns:
        List of strategy type identifiers
    """
    return list(_STRATEGY_REGISTRY.keys())
