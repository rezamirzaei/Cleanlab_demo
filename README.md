# Cleanlab Forge

Cleanlab Forge is a production-oriented demo project for data quality in machine learning.
It combines model training workflows with Cleanlab-based issue detection for:

- label issues
- outliers and anomalies
- near-duplicates and non-IID slices
- regression, multiclass, multilabel, token, vision, and multi-annotator tasks

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,ui,notebooks]"
```

Run tests:

```bash
pytest -v
```

Run the CLI:

```bash
cleanlab-demo --help
```

Run the UI:

```bash
streamlit run src/cleanlab_demo/ui/streamlit_app.py
```

## Documentation

- Detailed guide (architecture, workflows, and mathematics): `README_DETAILED.md`
- Contribution process: `CONTRIBUTING.md`
- Change log: `CHANGELOG.md`

## CI Quality Gates

CI enforces:

- `ruff check`
- `ruff format --check`
- `mypy`
- `pytest --cov` (minimum 45%)
- package build verification
