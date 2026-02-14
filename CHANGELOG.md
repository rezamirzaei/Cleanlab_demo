# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Anomaly Detection Module** (`cleanlab_demo.tasks.anomaly`)
  - Multiple detection strategies: DataLab kNN, Isolation Forest, LOF, Ensemble
  - Clean object-oriented API following Strategy and Factory patterns
  - Data providers: Synthetic, California Housing, Forest Cover, KDD Cup 99
  - Comprehensive Pydantic schemas for config and results
  - Full test coverage with 24 tests
  - Jupyter notebook tutorial (`12_anomaly_detection_cleanlab.ipynb`)
- Project renamed to **Cleanlab Forge** for better branding
- `CONTRIBUTING.md` guide for contributors
- `CHANGELOG.md` for tracking changes
- Additional `.gitignore` patterns: Parquet, Arrow, MLflow, DVC, W&B

### Changed
- Updated README with:
  - New anomaly detection section (Section 9)
  - Badges for Python version, license, code style
  - Updated project structure showing anomaly module
  - Enhanced feature list with anomaly detection
- Updated LICENSE author to match `pyproject.toml`

## [0.1.0] - 2024-12-01

### Added
- Initial release of Cleanlab Demo project
- Core tasks:
  - Binary classification (Adult Income)
  - Multi-class classification (CoverType)
  - Multi-label classification (Emotions)
  - Token classification (UD POS)
  - Regression (Bike Sharing)
  - Outlier detection (California Housing)
  - Multi-annotator + Active Learning (MovieLens)
  - Object detection + Segmentation (PennFudanPed)
- CLI interface with `cleanlab-demo` command
- Streamlit UI for interactive experiments
- 11 Jupyter notebook tutorials
- Docker Compose setup for UI and notebooks
- Pre-commit hooks with Ruff, mypy, and conventional commits
- Comprehensive test suite with 70%+ coverage
- Pydantic v2 for configuration and result validation
- Model factory supporting multiple sklearn estimators

### Dependencies
- cleanlab[datalab] >= 2.6.0
- scikit-learn >= 1.4.0
- pandas >= 2.2.0
- pydantic >= 2.6.1
- streamlit >= 1.32.0
- torch >= 2.1.0
- torchvision >= 0.16.0

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2024-12-01 | Initial release with 8 task types |
| 0.2.0 | TBD | Anomaly detection module |

[Unreleased]: https://github.com/your-org/cleanlab-forge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/cleanlab-forge/releases/tag/v0.1.0

