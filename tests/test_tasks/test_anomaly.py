"""Tests for anomaly detection task."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleanlab_demo.data.providers.anomaly import (
    CaliforniaHousingAnomalyProvider,
    SyntheticAnomalyProvider,
)
from cleanlab_demo.tasks.anomaly import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
    AnomalyDetectionTask,
    run_anomaly_detection,
)
from cleanlab_demo.tasks.anomaly.strategies import (
    DataLabKNNStrategy,
    EnsembleStrategy,
    IsolationForestStrategy,
    LocalOutlierFactorStrategy,
    get_strategy,
    list_strategies,
)


class TestSyntheticAnomalyProvider:
    """Tests for SyntheticAnomalyProvider."""

    def test_load_returns_dataframe(self) -> None:
        """Test that load returns a pandas DataFrame."""
        provider = SyntheticAnomalyProvider(n_samples=100, contamination=0.1)
        df = provider.load(seed=42)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "is_anomaly" in df.columns

    def test_contamination_rate(self) -> None:
        """Test that contamination rate is approximately correct."""
        provider = SyntheticAnomalyProvider(n_samples=1000, contamination=0.05)
        df = provider.load(seed=42)

        actual_rate = df["is_anomaly"].mean()
        assert 0.04 <= actual_rate <= 0.06  # Allow some variance

    def test_reproducibility(self) -> None:
        """Test that same seed produces same data."""
        provider = SyntheticAnomalyProvider(n_samples=100)
        df1 = provider.load(seed=42)
        df2 = provider.load(seed=42)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_data(self) -> None:
        """Test that different seeds produce different data."""
        provider = SyntheticAnomalyProvider(n_samples=100)
        df1 = provider.load(seed=42)
        df2 = provider.load(seed=123)

        assert not df1.equals(df2)

    def test_provider_properties(self) -> None:
        """Test provider properties are correct."""
        provider = SyntheticAnomalyProvider(n_samples=500, contamination=0.03)

        assert provider.has_ground_truth is True
        assert provider.label_col == "is_anomaly"
        assert provider.expected_contamination == 0.03
        assert "synthetic" in provider.name


class TestAnomalyDetectionConfig:
    """Tests for AnomalyDetectionConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AnomalyDetectionConfig()

        assert config.strategy == "datalab_knn"
        assert config.contamination == 0.01
        assert config.n_neighbors == 20
        assert config.n_estimators == 100
        assert config.scale_features is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = AnomalyDetectionConfig(
            strategy="isolation_forest",
            contamination=0.05,
            n_neighbors=30,
            seed=123,
        )

        assert config.strategy == "isolation_forest"
        assert config.contamination == 0.05
        assert config.n_neighbors == 30
        assert config.seed == 123

    def test_contamination_validation(self) -> None:
        """Test that contamination is validated."""
        with pytest.raises(ValueError):
            AnomalyDetectionConfig(contamination=0.6)  # > 0.5

        with pytest.raises(ValueError):
            AnomalyDetectionConfig(contamination=0.0001)  # < 0.001


class TestAnomalyDetectionStrategies:
    """Tests for anomaly detection strategies."""

    @pytest.fixture
    def sample_data(self) -> np.ndarray:
        """Create sample data for testing."""
        rng = np.random.default_rng(42)
        # Normal data
        normal = rng.standard_normal((90, 5))
        # Anomalies (far from origin)
        anomalies = rng.uniform(5, 10, (10, 5))
        return np.vstack([normal, anomalies])

    def test_isolation_forest_strategy(self, sample_data: np.ndarray) -> None:
        """Test Isolation Forest strategy."""
        strategy = IsolationForestStrategy()
        config = AnomalyDetectionConfig(contamination=0.1, seed=42)

        scores, predictions = strategy.fit_predict(sample_data, config)

        assert len(scores) == 100
        assert len(predictions) == 100
        assert scores.dtype == np.float64
        assert predictions.dtype == bool
        assert predictions.sum() == 10  # ~contamination * n_samples

    def test_datalab_knn_strategy_falls_back_without_cleanlab(
        self,
        sample_data: np.ndarray,
    ) -> None:
        """Test DataLab strategy works even when cleanlab isn't installed."""
        strategy = DataLabKNNStrategy()
        config = AnomalyDetectionConfig(
            strategy="datalab_knn",
            contamination=0.1,
            n_neighbors=10,
            seed=42,
        )

        scores, predictions = strategy.fit_predict(sample_data, config)

        assert len(scores) == 100
        assert len(predictions) == 100
        assert np.all(np.isfinite(scores))
        assert scores.dtype == np.float64
        assert predictions.dtype == bool
        assert predictions.sum() == 10  # ~contamination * n_samples

    def test_lof_strategy(self, sample_data: np.ndarray) -> None:
        """Test Local Outlier Factor strategy."""
        strategy = LocalOutlierFactorStrategy()
        config = AnomalyDetectionConfig(contamination=0.1, n_neighbors=10, seed=42)

        scores, predictions = strategy.fit_predict(sample_data, config)

        assert len(scores) == 100
        # Allow some variance in detection count (contamination * n ± 2)
        assert 8 <= predictions.sum() <= 12

    def test_ensemble_strategy(self, sample_data: np.ndarray) -> None:
        """Test Ensemble strategy."""
        strategy = EnsembleStrategy()
        config = AnomalyDetectionConfig(contamination=0.1, seed=42)

        scores, predictions = strategy.fit_predict(sample_data, config)

        assert len(scores) == 100
        assert 5 <= predictions.sum() <= 15  # Allow some variance

    def test_get_strategy_factory(self) -> None:
        """Test strategy factory function."""
        strategy = get_strategy("isolation_forest")
        assert isinstance(strategy, IsolationForestStrategy)

        strategy = get_strategy("local_outlier_factor")
        assert isinstance(strategy, LocalOutlierFactorStrategy)

    def test_get_strategy_invalid(self) -> None:
        """Test strategy factory with invalid name."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("invalid_strategy")  # type: ignore

    def test_list_strategies(self) -> None:
        """Test listing available strategies."""
        strategies = list_strategies()

        assert "datalab_knn" in strategies
        assert "isolation_forest" in strategies
        assert "local_outlier_factor" in strategies
        assert "ensemble" in strategies


class TestAnomalyDetectionTask:
    """Tests for AnomalyDetectionTask."""

    @pytest.fixture
    def task(self) -> AnomalyDetectionTask:
        """Create task with synthetic provider."""
        provider = SyntheticAnomalyProvider(n_samples=200, contamination=0.05)
        return AnomalyDetectionTask(provider)

    def test_run_isolation_forest(self, task: AnomalyDetectionTask) -> None:
        """Test running task with Isolation Forest."""
        config = AnomalyDetectionConfig(
            strategy="isolation_forest",
            contamination=0.05,
            seed=42,
        )

        result = task.run(config)

        assert isinstance(result, AnomalyDetectionResult)
        assert result.task == "anomaly_detection"
        assert result.strategy_used == "isolation_forest"
        assert result.summary.n_total == 200
        assert result.summary.precision is not None
        assert result.summary.recall is not None

    def test_run_lof(self, task: AnomalyDetectionTask) -> None:
        """Test running task with LOF."""
        config = AnomalyDetectionConfig(
            strategy="local_outlier_factor",
            contamination=0.05,
            seed=42,
        )

        result = task.run(config)

        assert result.strategy_used == "local_outlier_factor"
        assert 0 <= result.summary.precision <= 1
        assert 0 <= result.summary.recall <= 1

    def test_top_anomalies(self, task: AnomalyDetectionTask) -> None:
        """Test that top anomalies are returned."""
        config = AnomalyDetectionConfig(
            strategy="isolation_forest",
            top_k=10,
            seed=42,
        )

        result = task.run(config)

        assert len(result.top_anomalies) == 10
        # Top anomalies should be sorted by score (ascending = most anomalous first)
        scores = [a.anomaly_score for a in result.top_anomalies]
        assert scores == sorted(scores)

    def test_reproducibility(self, task: AnomalyDetectionTask) -> None:
        """Test that same seed produces same results."""
        config = AnomalyDetectionConfig(
            strategy="isolation_forest",
            seed=42,
        )

        result1 = task.run(config)
        result2 = task.run(config)

        assert result1.summary.n_anomalies_detected == result2.summary.n_anomalies_detected
        assert result1.summary.precision == result2.summary.precision


class TestRunAnomalyDetection:
    """Tests for run_anomaly_detection convenience function."""

    def test_functional_api(self) -> None:
        """Test functional API."""
        result = run_anomaly_detection(
            SyntheticAnomalyProvider(n_samples=100, contamination=0.1),
            strategy="isolation_forest",
            contamination=0.1,
            seed=42,
        )

        assert isinstance(result, AnomalyDetectionResult)
        assert result.summary.n_total == 100

    def test_with_config_object(self) -> None:
        """Test with explicit config object."""
        provider = SyntheticAnomalyProvider(n_samples=100)
        config = AnomalyDetectionConfig(strategy="local_outlier_factor")

        result = run_anomaly_detection(provider, config=config)

        assert result.strategy_used == "local_outlier_factor"


class TestAnomalyDetectionResult:
    """Tests for AnomalyDetectionResult."""

    def test_result_is_immutable(self) -> None:
        """Test that result is immutable (frozen)."""
        result = run_anomaly_detection(
            SyntheticAnomalyProvider(n_samples=50),
            strategy="isolation_forest",
            seed=42,
        )

        with pytest.raises(Exception):  # Pydantic ValidationError
            result.dataset = "new_name"  # type: ignore

    def test_result_serialization(self) -> None:
        """Test that result can be serialized to JSON."""
        result = run_anomaly_detection(
            SyntheticAnomalyProvider(n_samples=50),
            strategy="isolation_forest",
            seed=42,
        )

        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        assert "anomaly_detection" in json_str


class TestCaliforniaHousingAnomalyProvider:
    """Tests for CaliforniaHousingAnomalyProvider."""

    def test_load_with_injection(self) -> None:
        """Test loading with synthetic anomaly injection."""
        provider = CaliforniaHousingAnomalyProvider(
            max_rows=500,
            inject_anomalies=True,
            contamination=0.02,
        )

        df = provider.load(seed=42)

        assert len(df) == 500
        assert "is_anomaly" in df.columns
        assert provider.has_ground_truth is True

    def test_load_without_injection(self) -> None:
        """Test loading without synthetic anomalies."""
        provider = CaliforniaHousingAnomalyProvider(
            max_rows=500,
            inject_anomalies=False,
        )

        df = provider.load(seed=42)

        assert len(df) == 500
        assert "is_anomaly" not in df.columns
        assert provider.has_ground_truth is False
