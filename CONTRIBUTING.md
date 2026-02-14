# Contributing to Cleanlab Forge

First off, thank you for considering contributing to Cleanlab Forge! 🎉

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Finding Issues

- Look for issues labeled `good first issue` for beginner-friendly tasks
- Check `help wanted` labels for tasks needing community input
- Feel free to ask questions on any issue before starting work

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/cleanlab-forge.git
   cd cleanlab-forge
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev,ui,notebooks]"
   ```

4. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   pre-commit install --hook-type commit-msg
   ```

5. **Verify your setup:**
   ```bash
   pytest -v
   ruff check src/ tests/
   mypy src/cleanlab_demo
   ```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feat/anomaly-detection` — New feature
- `fix/lof-threshold` — Bug fix
- `docs/readme-update` — Documentation
- `refactor/task-base-class` — Code refactoring
- `test/add-anomaly-tests` — Test additions

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, missing semicolons, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding or correcting tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(anomaly): add ensemble detection strategy

- Combine Isolation Forest, LOF, and Datalab kNN
- Average scores across methods
- Add comprehensive tests

Closes #42
```

```
fix(tasks): handle empty DataFrame in outlier detection

The task now returns an empty result instead of crashing
when given a DataFrame with no rows.
```

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feat/your-feature
   ```

2. **Make your changes** following the style guide

3. **Run quality checks:**
   ```bash
   # Format code
   ruff format src/ tests/
   
   # Check linting
   ruff check src/ tests/
   
   # Type check
   mypy src/cleanlab_demo
   
   # Run tests
   pytest -v --cov=cleanlab_demo --cov-fail-under=45
   ```

4. **Commit your changes** using conventional commits

5. **Push to your fork:**
   ```bash
   git push origin feat/your-feature
   ```

6. **Open a Pull Request** with:
   - Clear title following conventional commits
   - Description of changes
   - Link to related issues
   - Screenshots (if UI changes)

7. **Address review feedback** by pushing additional commits

8. **Squash and merge** once approved

## Style Guide

### Python Style

We use **Ruff** for linting and formatting:

```python
# ✅ Good
def calculate_anomaly_score(
    features: np.ndarray,
    n_neighbors: int = 20,
) -> np.ndarray:
    """Calculate anomaly scores using kNN distance.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features)
        n_neighbors: Number of neighbors to consider
    
    Returns:
        Anomaly scores for each sample
    """
    ...

# ❌ Bad
def calc_score(features,n_neighbors=20):
    # no docstring, poor formatting
    ...
```

### Type Hints

All public functions must have type hints:

```python
from typing import Any

def process_data(
    data: pd.DataFrame,
    config: dict[str, Any],
    *,
    verbose: bool = False,
) -> ProcessResult:
    ...
```

### Imports

Organize imports in this order:
1. Standard library
2. Third-party
3. Local application

```python
# Standard library
from __future__ import annotations

import time
from typing import TYPE_CHECKING

# Third-party
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Local
from cleanlab_demo.tasks.base import DemoConfig, DemoResult
```

### Docstrings

Use Google-style docstrings:

```python
def detect_anomalies(
    X: np.ndarray,
    contamination: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect anomalies in the feature matrix.
    
    This function uses Isolation Forest to identify anomalous
    samples in the dataset.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features).
        contamination: Expected proportion of anomalies.
    
    Returns:
        Tuple containing:
            - scores: Anomaly scores for each sample
            - predictions: Boolean array where True indicates anomaly
    
    Raises:
        ValueError: If contamination is not in (0, 0.5].
    
    Example:
        >>> X = np.random.randn(100, 5)
        >>> scores, preds = detect_anomalies(X, contamination=0.05)
        >>> print(f"Found {preds.sum()} anomalies")
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest -v --cov=cleanlab_demo --cov-report=html

# Run specific test file
pytest tests/test_tasks/test_anomaly.py -v

# Run tests matching a pattern
pytest -v -k "anomaly"

# Run with parallel execution
pytest -v -n auto
```

### Writing Tests

Place tests in `tests/` mirroring the source structure:

```
src/cleanlab_demo/tasks/anomaly/task.py
tests/test_tasks/test_anomaly.py
```

Use pytest fixtures for common setup:

```python
import pytest

@pytest.fixture
def sample_provider():
    """Create a sample data provider for testing."""
    return SyntheticAnomalyProvider(n_samples=100, contamination=0.1)

class TestAnomalyDetection:
    def test_detects_anomalies(self, sample_provider):
        task = AnomalyDetectionTask(sample_provider)
        result = task.run(AnomalyDetectionConfig(seed=42))
        
        assert result.summary.n_anomalies_detected > 0
        assert result.summary.precision >= 0.5
```

### Test Categories

Use markers for test categories:

```python
@pytest.mark.slow
def test_large_dataset():
    """Test with large dataset (skip with -m 'not slow')."""
    ...

@pytest.mark.integration
def test_external_api():
    """Test requiring external resources."""
    ...
```

## Documentation

### README Updates

When adding new features:
1. Add entry to Table of Contents
2. Add detailed section following existing pattern
3. Include mathematical foundations if applicable
4. Add architecture diagram
5. Include code examples

### Docstrings

All public APIs must have docstrings:
- Classes: Describe purpose, attributes, examples
- Functions: Describe args, returns, raises, examples
- Modules: Describe contents and usage

### Notebooks

When adding notebooks:
1. Number sequentially (e.g., `12_anomaly_detection.ipynb`)
2. Include markdown cells explaining concepts
3. Clear outputs before committing
4. Test notebook runs end-to-end

## Questions?

Feel free to:
- Open an issue for discussion
- Ask in pull request comments
- Reach out to maintainers

Thank you for contributing! 🚀

