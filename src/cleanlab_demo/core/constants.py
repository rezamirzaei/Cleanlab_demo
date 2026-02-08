"""
Constants and default values for Cleanlab Demo.

This module centralizes all magic numbers and default configuration values
to make them easy to find, modify, and document.
"""

from __future__ import annotations

from typing import Final


# =============================================================================
# Random State & Reproducibility
# =============================================================================

DEFAULT_SEED: Final[int] = 42
"""Default random seed for reproducibility."""

# =============================================================================
# Data Splitting
# =============================================================================

DEFAULT_TEST_SIZE: Final[float] = 0.2
"""Default test set size as fraction of total data."""

MIN_TEST_SIZE: Final[float] = 0.05
"""Minimum allowed test set size."""

MAX_TEST_SIZE: Final[float] = 0.5
"""Maximum allowed test set size."""

# =============================================================================
# Cross-Validation
# =============================================================================

DEFAULT_CV_FOLDS: Final[int] = 5
"""Default number of cross-validation folds."""

MIN_CV_FOLDS: Final[int] = 2
"""Minimum number of CV folds."""

MAX_CV_FOLDS: Final[int] = 20
"""Maximum number of CV folds."""

MIN_SAMPLES_FOR_CV: Final[int] = 10
"""Minimum samples required per fold for cross-validation."""

# =============================================================================
# Model Training
# =============================================================================

DEFAULT_MAX_ITER: Final[int] = 500
"""Default maximum iterations for iterative solvers."""

DEFAULT_SOLVER: Final[str] = "lbfgs"
"""Default solver for logistic regression."""

DEFAULT_N_JOBS: Final[int] = 1
"""Default number of parallel jobs (1 = no parallelism)."""

# =============================================================================
# Cleanlab & Pruning
# =============================================================================

DEFAULT_PRUNE_FRAC: Final[float] = 0.02
"""Default fraction of training data to prune."""

MIN_PRUNE_FRAC: Final[float] = 0.0
"""Minimum prune fraction."""

MAX_PRUNE_FRAC: Final[float] = 0.5
"""Maximum prune fraction."""

DEFAULT_NOISE_FRAC: Final[float] = 0.0
"""Default synthetic noise fraction (0 = no synthetic noise)."""

MAX_NOISE_FRAC: Final[float] = 0.5
"""Maximum allowed noise fraction."""

# =============================================================================
# Data Loading
# =============================================================================

DEFAULT_MAX_ROWS: Final[int | None] = None
"""Default maximum rows to load (None = all rows)."""

MIN_ROWS_FOR_TRAINING: Final[int] = 100
"""Minimum rows required for meaningful training."""

DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 60
"""Timeout for dataset downloads."""

DOWNLOAD_CHUNK_SIZE: Final[int] = 8192
"""Chunk size for streaming downloads."""

# =============================================================================
# Vision Tasks
# =============================================================================

DEFAULT_SCORE_THRESHOLD: Final[float] = 0.3
"""Default confidence threshold for object detection."""

DEFAULT_MAX_IMAGES: Final[int] = 8
"""Default maximum images for vision tasks."""

MAX_IMAGES_LIMIT: Final[int] = 200
"""Hard limit on images to prevent memory issues."""

# =============================================================================
# Token Classification
# =============================================================================

DEFAULT_MAX_TRAIN_SENTENCES: Final[int] = 1000
"""Default maximum training sentences for token classification."""

DEFAULT_MAX_DEV_SENTENCES: Final[int] = 300
"""Default maximum dev sentences for token classification."""

# =============================================================================
# Multi-annotator
# =============================================================================

DEFAULT_MIN_RATINGS_PER_ITEM: Final[int] = 10
"""Default minimum ratings per item for multi-annotator tasks."""

DEFAULT_MIN_RATINGS_PER_ANNOTATOR: Final[int] = 100
"""Default minimum ratings per annotator."""

# =============================================================================
# File Paths & Directories
# =============================================================================

DEFAULT_DATA_DIR: Final[str] = "data"
"""Default directory for cached datasets."""

DEFAULT_ARTIFACTS_DIR: Final[str] = "artifacts"
"""Default directory for experiment artifacts."""

# =============================================================================
# Metric Thresholds
# =============================================================================

PERFECT_SCORE: Final[float] = 1.0
"""Perfect metric score."""

BASELINE_IMPROVEMENT_THRESHOLD: Final[float] = 0.01
"""Minimum improvement over baseline to be considered significant."""

# =============================================================================
# Logging
# =============================================================================

DEFAULT_LOG_LEVEL: Final[str] = "INFO"
"""Default logging level."""

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
"""Standard log format string."""

