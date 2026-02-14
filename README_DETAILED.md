# Cleanlab Forge

> **Data Quality Toolkit** — A comprehensive ML project demonstrating [Cleanlab](https://github.com/cleanlab/cleanlab) for automatic label issue detection, anomaly detection, and data quality analysis. This project downloads real-world datasets, trains multiple ML models, applies Cleanlab to find mislabeled data, and provides both a CLI and web UI.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

## Table of Contents

- [Features](#features)
- [Quickstart](#quickstart-docker)
- [Installation](#local-installation-venv)
- [CLI Usage](#cli-usage)
- [Demo Use Cases](#demo-use-cases)
  - [1. Binary Classification (Adult Income)](#1-binary-classification-adult-income)
  - [2. Multi-class Classification (CoverType)](#2-multi-class-classification-covertype)
  - [3. Multi-label Classification (Emotions)](#3-multi-label-classification-emotions)
  - [4. Token Classification (UD POS)](#4-token-classification-ud-pos)
  - [5. Regression (Bike Sharing)](#5-regression-bike-sharing)
  - [6. Outlier Detection (California Housing)](#6-outlier-detection-california-housing)
  - [7. Multi-annotator + Active Learning (MovieLens)](#7-multi-annotator--active-learning-movielens-100k)
  - [8. Object Detection + Segmentation (PennFudanPed)](#8-object-detection--image-segmentation-pennfudanped)
  - [9. Anomaly Detection](#9-anomaly-detection-synthetic--real-world)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [AI Report](#ai-report-optional)
- [Repo Hygiene](#repo-hygiene)
  - [Quality Standards](#quality-standards)
  - [Pre-commit Hooks](#pre-commit-hooks)
  - [Commit Message Convention](#commit-message-convention)
  - [CI/CD Pipeline](#cicd-pipeline)
  - [Running Quality Checks](#running-quality-checks-locally)
  - [Dependency Management](#dependency-management)
  - [Release Checklist](#release-checklist)
- [Development](#development)
- [License](#license)

## Features

- 📊 **Multiple Datasets**: UCI Adult Income (classification), UCI Bike Sharing (regression), California Housing (regression; natural outliers)
- 🤖 **Model Support**: Logistic Regression, k-NN, Random Forest, ExtraTrees, Histogram Gradient Boosting, Ridge
- 🔍 **Cleanlab Integration**: Label issues + Datalab (outliers/near-duplicates/non-iid) + optional prune & retrain comparison
- 🧩 **More Task Types**: Multi-label, token classification, multi-annotator labeling + active learning, object detection, image segmentation (via tasks + notebooks)
- 🚨 **Anomaly Detection**: Multiple strategies (Datalab kNN, Isolation Forest, LOF, Ensemble) with clean OOP API
- 📈 **Model Sweeps**: Compare multiple models on the same dataset
- 🎛️ **Streamlit UI**: End-to-end tabular runner + dedicated task pages
- 📓 **Jupyter Notebooks**: Step-by-step tutorials
- 🤖 **AI Reports**: Optional LLM-powered analysis reports (via pydantic-ai)
- ✅ **Production Quality**: CI/CD, pre-commit hooks, comprehensive testing, type safety

## Quickstart (Docker)

```bash
docker compose up --build
```

- **Streamlit UI**: http://localhost:8501
- **JupyterLab**: http://localhost:8888

## Local Installation (venv)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with all optional dependencies
pip install -U pip
pip install -e ".[dev,ui,notebooks,ai]"

# Run tests
pytest -v

# Verify installation
cleanlab-demo --version
```

## Run the UI (Streamlit)

```bash
streamlit run src/cleanlab_demo/ui/streamlit_app.py
```

Use the sidebar **Mode** selector to switch between the tabular end-to-end runner and the dedicated task pages.

## CLI Usage

```bash
# Show help
cleanlab-demo --help

# Run a single experiment
cleanlab-demo run --dataset adult_income --model logistic_regression

# Run with custom parameters
cleanlab-demo run -d adult_income -m random_forest --max-rows 5000 --cleanlab --prune --prune-fraction 0.02

# Save results to file
cleanlab-demo run -d adult_income -o results/experiment.json

# Run a model sweep (compare multiple models)
cleanlab-demo sweep adult_income
cleanlab-demo sweep adult_income -m logistic_regression -m random_forest --save-csv results.csv

# Download datasets
cleanlab-demo download-data adult_income
cleanlab-demo download-data bike_sharing
cleanlab-demo download-data california_housing
```

---

## Demo Use Cases

This section provides detailed explanations of each demo, covering:
- **Problem Statement**: What the task aims to solve
- **Mathematical Foundation**: The underlying algorithms and formulas
- **Architecture & Data Flow**: How components interact
- **How to Run**: Commands and notebooks

---

### 1. Binary Classification (Adult Income)

**Dataset**: UCI Adult Income (~48K samples)  
**Task**: Predict whether income exceeds $50K/year  
**Notebook**: `notebooks/01_quickstart_cleanlab.ipynb`

#### Problem Statement

Binary classification predicts one of two classes. Real-world datasets often contain **label errors** (mislabeled examples) that degrade model performance. Cleanlab identifies these errors without ground truth.

#### Mathematical Foundation

**Confident Learning** is the core algorithm behind Cleanlab's label issue detection:

1. **Out-of-Sample Predicted Probabilities**: Train a classifier and obtain predicted probabilities `p̂(y=k|x)` using k-fold cross-validation to avoid overfitting.

2. **Self-Confidence Score**: For each example `i` with given label `ỹᵢ`:
   ```
   sᵢ = p̂(ỹᵢ | xᵢ)
   ```
   Lower scores indicate the model finds the given label less plausible.

3. **Confident Joint Matrix `Ĉ`**: Counts examples where:
   - Given label = `i`
   - Predicted class (with confidence above threshold `tⱼ`) = `j`
   
   The threshold `tⱼ` is typically the average predicted probability for class `j`:
   ```
   tⱼ = (1/n) Σᵢ p̂(y=j|xᵢ)
   ```

4. **Noise Rate Estimation**: From `Ĉ`, estimate:
   - `P(ỹ=i | y*=j)`: probability of observing label `i` when true label is `j`
   - `P(y*=j | ỹ=i)`: probability true label is `j` given observed `i`

5. **Label Issue Ranking**: Examples are ranked by `sᵢ` (ascending) to identify likely mislabeled data.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Binary Classification Pipeline               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────────┐   │
│  │ UCI Data │───▶│ Preprocessor│───▶│ Feature Engineering  │   │
│  │ (48K)    │    │ (scaling,   │    │ (numeric + one-hot   │   │
│  └──────────┘    │  encoding)  │    │  categorical)        │   │
│                  └─────────────┘    └──────────┬───────────┘   │
│                                                 │               │
│                                                 ▼               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Stratified K-Fold Cross-Validation          │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Fold 1: Train on 80% → Predict probs on held-out 20%│ │  │
│  │  │ Fold 2: Train on 80% → Predict probs on held-out 20%│ │  │
│  │  │ ...                                                 │ │  │
│  │  │ Fold 5: Train on 80% → Predict probs on held-out 20%│ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │  Output: Out-of-sample predicted probabilities for ALL   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                 │               │
│                                                 ▼               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Cleanlab: find_label_issues              │  │
│  │  • Compute self-confidence scores                        │  │
│  │  • Build confident joint Ĉ                               │  │
│  │  • Rank examples by label quality                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                 │               │
│                                                 ▼               │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Output: Ranked list of likely mislabeled examples      │    │
│  │ • Can prune & retrain to improve model performance     │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

#### How to Run

```bash
# CLI
cleanlab-demo run -d adult_income -m logistic_regression --cleanlab

# With pruning and retraining
cleanlab-demo run -d adult_income -m random_forest --cleanlab --prune --prune-fraction 0.02

# Notebook
jupyter notebook notebooks/01_quickstart_cleanlab.ipynb
```

#### Metrics Reported

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correctness |
| Macro F1 | Average F1 across classes |
| Log Loss | Probabilistic prediction quality |
| Precision @ Prune | Of pruned examples, how many were truly noisy (if synthetic noise) |
| Recall @ Prune | Of truly noisy examples, how many were pruned |

---

### 2. Multi-class Classification (CoverType)

**Dataset**: Covertype (581K samples, 7 forest cover classes)  
**Task**: Predict forest cover type from cartographic features  
**Notebook**: `notebooks/05_multiclass_classification_covtype.ipynb`

#### Problem Statement

Multi-class classification extends binary classification to K>2 classes. Label noise can confuse similar classes (e.g., different tree species).

#### Mathematical Foundation

The same confident learning framework applies, extended to K classes:

1. **Softmax Probabilities**: Model outputs `p̂(y=k|x)` for all K classes, where:
   ```
   Σₖ p̂(y=k|x) = 1
   ```

2. **Confident Joint `Ĉ`**: Now a K×K matrix where `Ĉ[i,j]` counts examples with:
   - Given label = class `i`
   - Confidently predicted class = `j`

3. **Off-Diagonal Analysis**: Large off-diagonal entries `Ĉ[i,j]` (i≠j) indicate systematic confusion between classes `i` and `j`.

4. **Per-Class Thresholds**: Each class `j` has threshold:
   ```
   tⱼ = E[p̂(y=j|x)] = (1/n) Σᵢ p̂(y=j|xᵢ)
   ```

5. **Label Quality Score**: Same self-confidence `sᵢ = p̂(ỹᵢ|xᵢ)`

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               Multi-class Classification Pipeline               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────┐     ┌──────────────┐     ┌───────────────────┐ │
│  │ Covertype  │────▶│ StandardScaler│───▶│ Logistic Regression│ │
│  │ (581K, 54  │     │              │     │ (softmax output)  │ │
│  │  features) │     └──────────────┘     └─────────┬─────────┘ │
│  └────────────┘                                    │           │
│                                                    ▼           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Stratified K-Fold (preserves class ratios)     │  │
│  │                                                          │  │
│  │  For each fold:                                          │  │
│  │    • Train model on (K-1) folds                          │  │
│  │    • Predict P(y=k|x) for held-out fold                  │  │
│  │                                                          │  │
│  │  Result: pred_probs shape = (n_samples, n_classes)       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                    │           │
│                                                    ▼           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                Cleanlab: find_label_issues               │  │
│  │                                                          │  │
│  │  Input:                                                  │  │
│  │    • labels: array of integers [0, K-1]                  │  │
│  │    • pred_probs: array shape (n, K)                      │  │
│  │                                                          │  │
│  │  Algorithm:                                              │  │
│  │    1. Compute class thresholds tⱼ                        │  │
│  │    2. Build confident joint Ĉ (K×K matrix)               │  │
│  │    3. Estimate noise rates                               │  │
│  │    4. Rank by self-confidence                            │  │
│  │                                                          │  │
│  │  Output: indices of likely mislabeled examples           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                    │           │
│                                                    ▼           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Prune & Retrain: Remove flagged examples, retrain      │    │
│  │ Compare: baseline_accuracy vs pruned_accuracy          │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

#### Noise Injection (for Evaluation)

To measure precision/recall of label issue detection:
```python
def inject_multiclass_label_noise(y, frac, seed):
    """Flip frac% of labels to a random different class."""
    n_flip = round(frac * len(y))
    flip_indices = random.choice(len(y), n_flip)
    for idx in flip_indices:
        y[idx] = random_different_class(y[idx])
    return y, flip_indices
```

#### How to Run

```bash
# Notebook (recommended for this task)
jupyter notebook notebooks/05_multiclass_classification_covtype.ipynb
```

---

### 3. Multi-label Classification (Emotions)

**Dataset**: OpenML Emotions (593 samples, 6 emotion labels)  
**Task**: Predict multiple emotions present in music  
**Notebook**: `notebooks/06_multilabel_classification_emotions.ipynb`

#### Problem Statement

Multi-label classification assigns multiple labels per example (e.g., a song can be both "happy" and "energetic"). Label errors include:
- **Missing labels**: A relevant label is absent
- **Extra labels**: An irrelevant label is present

#### Mathematical Foundation

1. **Binary Relevance Model**: Train K independent binary classifiers, one per label:
   ```
   p̂(yₖ=1|x) for k ∈ {1, ..., K}
   ```

2. **Multi-label Self-Confidence**: For example `i` with label set `Yᵢ`:
   ```
   sᵢ = Πₖ∈Yᵢ p̂(yₖ=1|xᵢ) × Πₖ∉Yᵢ p̂(yₖ=0|xᵢ)
   ```
   Or more commonly, the minimum (worst) confidence:
   ```
   sᵢ = min(min_{k∈Yᵢ} p̂(yₖ=1|xᵢ), min_{k∉Yᵢ} p̂(yₖ=0|xᵢ))
   ```

3. **Per-Label Analysis**: Cleanlab examines each label independently to find:
   - Examples where label k is present but shouldn't be
   - Examples where label k is absent but should be present

4. **Aggregated Score**: Combines per-label scores into example-level label quality.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Multi-label Classification Pipeline                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────┐     ┌──────────────────────────────────────┐   │
│  │ Emotions   │────▶│ Labels: Binary matrix (n × K)        │   │
│  │ Dataset    │     │ [1, 0, 1, 0, 0, 1] = happy, not sad, │   │
│  └────────────┘     │  surprised, not angry, not fear, calm│   │
│                     └──────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 OneVsRestClassifier                      │  │
│  │  ┌───────────────────────────────────────────────────┐   │  │
│  │  │ Label 1: Binary classifier → p̂(y₁=1|x)           │   │  │
│  │  │ Label 2: Binary classifier → p̂(y₂=1|x)           │   │  │
│  │  │ ...                                               │   │  │
│  │  │ Label K: Binary classifier → p̂(yₖ=1|x)           │   │  │
│  │  └───────────────────────────────────────────────────┘   │  │
│  │  Output: pred_probs shape = (n_samples, K)              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                    │           │
│                                                    ▼           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Cleanlab: multilabel_classification.filter       │  │
│  │                                                          │  │
│  │  Input:                                                  │  │
│  │    • labels: list of lists [[0,2,5], [1], [0,1,3], ...]  │  │
│  │    • pred_probs: array shape (n, K)                      │  │
│  │                                                          │  │
│  │  Algorithm:                                              │  │
│  │    1. Compute per-label confidence scores                │  │
│  │    2. Aggregate to example-level quality                 │  │
│  │    3. Rank by quality (ascending)                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                    │           │
│                                                    ▼           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Metrics: Micro/Macro F1, Subset Accuracy, Hamming Loss │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

#### Noise Injection

```python
def inject_multilabel_noise(y, frac_examples, seed):
    """Flip one random label bit per selected example."""
    n_noisy = round(frac_examples * n)
    for idx in selected_indices:
        j = random_label_index()
        y[idx, j] = 1 - y[idx, j]  # Flip 0→1 or 1→0
    return y, noisy_indices
```

#### How to Run

```bash
jupyter notebook notebooks/06_multilabel_classification_emotions.ipynb
```

#### Metrics Reported

| Metric | Description |
|--------|-------------|
| Micro F1 | F1 computed globally across all labels |
| Macro F1 | Average F1 per label |
| Subset Accuracy | Exact match of all labels |
| Hamming Loss | Fraction of incorrectly predicted labels |

---

### 4. Token Classification (UD POS)

**Dataset**: Universal Dependencies English Web Treebank  
**Task**: Part-of-Speech tagging (each token → POS tag)  
**Notebook**: `notebooks/07_token_classification_ud_pos.ipynb`

#### Problem Statement

Token classification assigns a label to each token in a sequence. In NER or POS tagging, annotators may mislabel individual words. Cleanlab identifies likely token-level errors.

#### Mathematical Foundation

1. **Per-Token Predictions**: For sentence with tokens `[w₁, w₂, ..., wₜ]`:
   ```
   p̂(tag=k | wᵢ, context) for each token i, class k
   ```

2. **Token-Level Self-Confidence**:
   ```
   sᵢ,ₜ = p̂(given_tag_t | wₜ, context)
   ```
   Low scores indicate the model disagrees with the annotation.

3. **Sentence-Level Aggregation**: Aggregate token scores to find problematic sentences:
   ```
   s_sentence = min(sᵢ,₁, sᵢ,₂, ..., sᵢ,ₜ)  # or mean
   ```

4. **Issue Identification**: Returns `(sentence_idx, token_idx)` pairs for likely errors.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Token Classification Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Input: CoNLL-U Format                                  │    │
│  │ "The    DET"                                           │    │
│  │ "quick  ADJ"                                           │    │
│  │ "brown  ADJ"                                           │    │
│  │ "fox    NOUN"                                          │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               Feature Extraction per Token               │  │
│  │  • Word shape (capitalized, all-caps, numeric, etc.)     │  │
│  │  • Prefix/suffix (first/last 2-3 chars)                  │  │
│  │  • Position in sentence                                  │  │
│  │  • Character n-grams                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Logistic Regression (per token)             │  │
│  │  Input: token features                                   │  │
│  │  Output: P(tag=k | token_features) for all K tags        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │      Cleanlab: token_classification.filter               │  │
│  │                                                          │  │
│  │  Input:                                                  │  │
│  │    • labels: list of lists [[tag₁, tag₂, ...], ...]      │  │
│  │    • pred_probs: list of arrays [arr₁, arr₂, ...]        │  │
│  │      where arrᵢ has shape (n_tokens_in_sent_i, K)        │  │
│  │                                                          │  │
│  │  Output: list of (sentence_idx, token_idx) tuples        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Prune & Retrain: Remove flagged (sent, token) pairs    │    │
│  │ Metrics: Token Accuracy, Macro F1                      │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

#### Noise Injection

```python
def inject_token_noise(labels, frac_tokens, seed, n_classes):
    """Corrupt frac_tokens% of all tokens across all sentences."""
    total_tokens = sum(len(sent) for sent in labels)
    n_corrupt = round(frac_tokens * total_tokens)
    # Randomly select (sent_idx, token_idx) pairs to corrupt
    for (s, t) in selected_pairs:
        labels[s][t] = random_different_tag(labels[s][t])
    return labels, corrupted_pairs
```

#### How to Run

```bash
jupyter notebook notebooks/07_token_classification_ud_pos.ipynb
```

---

### 5. Regression (Bike Sharing)

**Dataset**: UCI Bike Sharing (17K samples)  
**Task**: Predict hourly bike rental count  
**Notebook**: `notebooks/08_regression_cleanlearning_bike_sharing.ipynb`

#### Problem Statement

Regression predicts continuous values. Label errors manifest as **outlier labels** — values that are implausible given the features (e.g., recording 10 rentals when 1000 were expected).

#### Mathematical Foundation

Cleanlab's regression `CleanLearning` uses a different approach than classification:

1. **Model Predictions via CV**: Obtain out-of-sample predictions `ŷᵢ` using cross-validation.

2. **Residuals**: Compute residuals for each example:
   ```
   rᵢ = yᵢ - ŷᵢ
   ```

3. **Uncertainty Estimation**: Bootstrap the model to estimate prediction uncertainty:
   ```
   σᵢ = std(ŷᵢ across bootstrap samples)
   ```

4. **Normalized Residuals**: Standardize residuals by uncertainty:
   ```
   zᵢ = |rᵢ| / σᵢ
   ```

5. **Label Quality Score**: Lower is worse:
   ```
   label_quality_i = 1 / (1 + zᵢ)
   ```
   Examples with large residuals relative to uncertainty are flagged.

6. **Aleatoric vs Epistemic**: The method accounts for:
   - **Aleatoric uncertainty**: Inherent noise in the data
   - **Epistemic uncertainty**: Model uncertainty due to limited data

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Regression Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────┐     ┌──────────────────────────────────────┐   │
│  │ Bike Share │────▶│ Features: hour, temp, humidity,      │   │
│  │ Dataset    │     │ windspeed, season, holiday, etc.     │   │
│  └────────────┘     └──────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Cleanlab: CleanLearning (Regression)             │  │
│  │                                                          │  │
│  │  model = HistGradientBoostingRegressor()                 │  │
│  │  cleaner = CleanLearning(model=model, cv_n_folds=5)      │  │
│  │                                                          │  │
│  │  Algorithm:                                              │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ 1. K-Fold Cross-Validation                         │  │  │
│  │  │    • For each fold: train, predict held-out ŷ      │  │  │
│  │  │    • Collect all out-of-sample predictions         │  │  │
│  │  ├────────────────────────────────────────────────────┤  │  │
│  │  │ 2. Bootstrap Uncertainty Estimation                │  │  │
│  │  │    • Resample training data B times               │  │  │
│  │  │    • Train model on each bootstrap sample         │  │  │
│  │  │    • Compute σᵢ = std of predictions              │  │  │
│  │  ├────────────────────────────────────────────────────┤  │  │
│  │  │ 3. Compute Label Quality                           │  │  │
│  │  │    • rᵢ = yᵢ - ŷᵢ (residual)                       │  │  │
│  │  │    • zᵢ = |rᵢ| / σᵢ (normalized)                   │  │  │
│  │  │    • quality_i = 1/(1 + zᵢ)                        │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                          │  │
│  │  Output: DataFrame with 'label_quality' per example      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Rank by label_quality (ascending), prune worst         │    │
│  │ Retrain on clean data, compare metrics                 │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

#### Noise Injection

```python
def inject_regression_label_noise(y, frac, seed, scale=5.0):
    """Add large Gaussian noise to frac% of labels."""
    n_corrupt = round(frac * len(y))
    idx = random.choice(len(y), n_corrupt)
    sigma = std(y)
    y[idx] += normal(0, scale * sigma, size=n_corrupt)
    return y, idx
```

#### How to Run

```bash
jupyter notebook notebooks/08_regression_cleanlearning_bike_sharing.ipynb
```

#### Metrics Reported

| Metric | Description |
|--------|-------------|
| R² | Coefficient of determination |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |

---

### 6. Outlier Detection (California Housing)

**Dataset**: California Housing (20K samples)  
**Task**: Detect outliers, near-duplicates, and non-IID data  
**Notebook**: `notebooks/10_outlier_detection_datalab_california_housing.ipynb`

#### Problem Statement

Beyond label issues, datasets have **data quality issues**:
- **Outliers**: Examples far from the data distribution
- **Near-duplicates**: Almost identical examples
- **Non-IID**: Data not independently and identically distributed

Cleanlab's `Datalab` detects these automatically.

#### Mathematical Foundation

**Outlier Detection (kNN Distance)**:

1. **Feature Standardization**: Scale features to zero mean, unit variance.

2. **k-Nearest Neighbors**: For each example, find k nearest neighbors in feature space.

3. **Outlier Score**: Based on average distance to k neighbors:
   ```
   outlier_score_i = mean(dist(xᵢ, xⱼ) for j in kNN(i))
   ```
   Normalized to [0, 1] where lower = more outlying.

4. **Threshold**: Examples with scores below threshold are flagged.

**Near-Duplicate Detection**:

1. **Distance Computation**: Compute pairwise distances (or use kNN).

2. **Near-Duplicate Score**: Based on minimum distance:
   ```
   near_dup_score_i = min(dist(xᵢ, xⱼ) for j ≠ i)
   ```
   Very low distances indicate near-duplicates.

**Non-IID Detection**:

1. **Sorting by Feature**: Sort data by a feature and check if nearby indices have similar labels/features.

2. **Statistical Test**: Compare observed patterns to IID expectation.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Datalab Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────┐     ┌────────────────────────────────────┐ │
│  │ California     │────▶│ StandardScaler → Feature Matrix    │ │
│  │ Housing (20K)  │     │ (8 features: lat, lon, income, etc)│ │
│  └────────────────┘     └────────────────────────────────────┘ │
│                                           │                     │
│                                           ▼                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Cleanlab: Datalab                     │  │
│  │                                                          │  │
│  │  datalab = Datalab(data=df, label_name='price',          │  │
│  │                    task='regression')                    │  │
│  │                                                          │  │
│  │  datalab.find_issues(features=X_scaled, issue_types={    │  │
│  │      'outlier': {},                                      │  │
│  │      'near_duplicate': {},                               │  │
│  │      'non_iid': {}                                       │  │
│  │  })                                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                           │                     │
│       ┌───────────────────────────────────┼───────────────┐    │
│       │                                   │               │    │
│       ▼                                   ▼               ▼    │
│  ┌──────────┐                      ┌───────────┐   ┌─────────┐ │
│  │ Outlier  │                      │ Near-Dup  │   │ Non-IID │ │
│  │ Detection│                      │ Detection │   │ Check   │ │
│  │          │                      │           │   │         │ │
│  │ • kNN    │                      │ • Min     │   │ • Order │ │
│  │   dist   │                      │   dist    │   │   stats │ │
│  └──────────┘                      └───────────┘   └─────────┘ │
│       │                                   │               │    │
│       └───────────────────────────────────┴───────────────┘    │
│                                           │                     │
│                                           ▼                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Output:                                                │    │
│  │ • get_issues(): DataFrame with all issue scores        │    │
│  │ • get_issue_summary(): Counts per issue type           │    │
│  │ • Columns: is_outlier_issue, outlier_score,            │    │
│  │   is_near_duplicate_issue, near_duplicate_score, etc.  │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

#### Synthetic Outlier Injection

```python
def inject_synthetic_outliers(X, outlier_frac, outlier_scale, seed):
    """Multiply one feature by outlier_scale for outlier_frac% of examples."""
    n_out = round(outlier_frac * len(X))
    outlier_idx = random.choice(len(X), n_out)
    X.loc[outlier_idx, first_numeric_col] *= outlier_scale
    return X, outlier_idx
```

#### How to Run

```bash
jupyter notebook notebooks/10_outlier_detection_datalab_california_housing.ipynb
```

---

### 7. Multi-annotator + Active Learning (MovieLens 100K)

**Dataset**: MovieLens 100K (ratings from multiple users)  
**Task**: Find consensus labels and identify informative examples for labeling  
**Notebook**: `notebooks/09_multiannotator_active_learning_movielens_100k.ipynb`

#### Problem Statement

When multiple annotators label data:
- Annotators disagree
- Some annotators are more reliable
- Some examples are ambiguous

Cleanlab estimates consensus labels and annotator quality, and prioritizes which examples need more labels.

#### Mathematical Foundation

**Consensus Label Estimation**:

1. **Majority Vote**: Simple baseline:
   ```
   consensus_i = mode(labels from all annotators for example i)
   ```

2. **Weighted by Annotator Quality**: Better annotators get more weight:
   ```
   consensus_i = argmax_k Σⱼ qⱼ × I(annotator j labeled example i as k)
   ```
   where `qⱼ` is annotator j's quality score.

3. **Annotator Quality Estimation**: Based on agreement with model predictions:
   ```
   qⱼ = E[agreement between annotator j and model predictions]
   ```

**Label Quality for Multi-annotator**:

1. **Agreement Score**: How much do annotators agree on this example?
   ```
   agreement_i = (max label count) / (total labels for example i)
   ```

2. **Model Confidence**: How confident is the model?
   ```
   confidence_i = max_k p̂(y=k|xᵢ)
   ```

3. **Combined Score**: Low agreement + low confidence = low quality.

**Active Learning Score**:

1. **Uncertainty Sampling**: Prioritize examples where model is uncertain:
   ```
   uncertainty_i = 1 - max_k p̂(y=k|xᵢ)
   ```

2. **Disagreement**: Prioritize examples where annotators disagree:
   ```
   disagreement_i = entropy of annotator labels
   ```

3. **Active Learning Score**: Combines both:
   ```
   al_score_i = f(uncertainty_i, disagreement_i)
   ```
   Higher scores = more informative to label next.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Multi-annotator + Active Learning Pipeline         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ MovieLens 100K                                         │    │
│  │ • 100K ratings from 943 users on 1682 movies           │    │
│  │ • Treat users as annotators, movies as examples        │    │
│  │ • Labels: discretized ratings (e.g., 1-5 → classes)    │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         labels_multiannotator: DataFrame                 │  │
│  │         Shape: (n_examples, n_annotators)                │  │
│  │         Values: label or NaN if annotator didn't label   │  │
│  │                                                          │  │
│  │         Example:                                         │  │
│  │         │ user_1 │ user_2 │ user_3 │ ... │               │  │
│  │         │   3    │  NaN   │   4    │     │               │  │
│  │         │  NaN   │   2    │   2    │     │               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│              ┌───────────────┴───────────────┐                 │
│              ▼                               ▼                 │
│  ┌─────────────────────────┐    ┌─────────────────────────┐   │
│  │ get_majority_vote_label │    │ Train model on consensus│   │
│  │ → consensus labels      │    │ → pred_probs via CV     │   │
│  └─────────────────────────┘    └─────────────────────────┘   │
│              │                               │                 │
│              └───────────────┬───────────────┘                 │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │       get_label_quality_multiannotator                   │  │
│  │                                                          │  │
│  │  Input: labels_multiannotator, pred_probs                │  │
│  │  Output:                                                 │  │
│  │    • label_quality: per-example quality scores           │  │
│  │    • annotator_quality: per-annotator reliability        │  │
│  │    • consensus_label: best estimate of true label        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │       get_active_learning_scores                         │  │
│  │                                                          │  │
│  │  Input: labels_multiannotator, pred_probs                │  │
│  │  Output: al_scores (higher = more informative)           │  │
│  │                                                          │  │
│  │  Use case: Prioritize which examples to send for         │  │
│  │  additional labeling to maximize information gain        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Output:                                                │    │
│  │ • top_worst_quality_examples: need label correction    │    │
│  │ • top_active_learning_examples: need more labels       │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

#### How to Run

```bash
jupyter notebook notebooks/09_multiannotator_active_learning_movielens_100k.ipynb
```

---

### 8. Object Detection + Image Segmentation (PennFudanPed)

**Dataset**: PennFudan Pedestrian (170 images)  
**Task**: Detect pedestrians (bounding boxes) + segment pedestrians (pixel masks)  
**Notebook**: `notebooks/11_vision_detection_segmentation_pennfudan.ipynb`

#### Problem Statement

Computer vision annotation errors:
- **Object Detection**: Wrong bounding box coordinates, missing objects, wrong class
- **Segmentation**: Incorrect pixel labels, boundary errors

Cleanlab detects these using model predictions.

#### Mathematical Foundation

**Object Detection Label Issues**:

1. **IoU (Intersection over Union)**: Measures box overlap:
   ```
   IoU(box_a, box_b) = Area(A ∩ B) / Area(A ∪ B)
   ```

2. **Matching Predictions to Ground Truth**: For each predicted box with confidence > threshold, find best matching GT box by IoU.

3. **Issue Types**:
   - **Swapped**: Predicted class differs from GT class
   - **Overlooked**: High-confidence prediction has no matching GT
   - **Poorly Located**: IoU between prediction and GT is low
   - **Missing**: GT box has no matching prediction

4. **Image-Level Score**: Aggregate box-level issues to flag problematic images.

**Segmentation Label Issues**:

1. **Per-Pixel Predicted Probabilities**: Model outputs `p̂(y=k|pixel)` for each pixel.

2. **Pixel-Level Label Quality**:
   ```
   quality_pixel = p̂(given_label | pixel)
   ```

3. **Image-Level Aggregation** (softmin):
   ```
   quality_image = softmin(quality_pixel for all pixels)
   ```
   Softmin emphasizes the worst pixels:
   ```
   softmin(x) = Σᵢ xᵢ × exp(-xᵢ/T) / Σᵢ exp(-xᵢ/T)
   ```

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│         Vision Detection + Segmentation Pipeline                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ PennFudanPed Dataset                                   │    │
│  │ • 170 images with pedestrian annotations               │    │
│  │ • GT boxes: [x1, y1, x2, y2] per pedestrian            │    │
│  │ • GT masks: binary (H, W) per pedestrian → merged      │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │     Pretrained Model: Mask R-CNN (COCO weights)          │  │
│  │                                                          │  │
│  │  model = maskrcnn_resnet50_fpn(pretrained=True)          │  │
│  │  model.eval()                                            │  │
│  │                                                          │  │
│  │  For each image:                                         │  │
│  │    output = model(image_tensor)                          │  │
│  │    • boxes: predicted bounding boxes                     │  │
│  │    • labels: predicted class IDs                         │  │
│  │    • scores: confidence scores                           │  │
│  │    • masks: predicted segmentation masks                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│          ┌───────────────────┴────────────────────┐            │
│          ▼                                        ▼            │
│  ┌──────────────────────────┐     ┌──────────────────────────┐ │
│  │  Object Detection        │     │  Segmentation            │ │
│  │  Label Issue Detection   │     │  Label Issue Detection   │ │
│  │                          │     │                          │ │
│  │  find_label_issues(      │     │  get_label_quality_scores│ │
│  │    labels=[              │     │    (labels=masks_arr,    │ │
│  │      {'bboxes': GT,      │     │     pred_probs=pred_arr, │ │
│  │       'labels': cls}     │     │     method='softmin')    │ │
│  │    ],                    │     │                          │ │
│  │    predictions=[         │     │  Output: per-image       │ │
│  │      [pred_boxes]        │     │  quality scores          │ │
│  │    ]                     │     │                          │ │
│  │  )                       │     │                          │ │
│  │                          │     │                          │ │
│  │  Output: bool mask       │     │                          │ │
│  │  per image               │     │                          │ │
│  └──────────────────────────┘     └──────────────────────────┘ │
│          │                                        │            │
│          └────────────────────┬───────────────────┘            │
│                               ▼                                │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Output:                                                │    │
│  │ • Object Detection: n_flagged, precision, recall       │    │
│  │ • Segmentation: top-k worst images, precision@k        │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

#### Synthetic Corruption

```python
def corrupt_boxes(boxes, img_w, img_h, seed):
    """Shift, delete, or class-swap random boxes."""
    # 50% shift location by ~10%
    # 30% delete a box
    # 20% swap class (if multi-class)
    return corrupted_boxes

def corrupt_binary_mask(mask, seed):
    """Apply random morphological operations (dilate/erode)."""
    return morphology_op(mask)
```

#### How to Run

```bash
# Requires torch/torchvision
jupyter notebook notebooks/11_vision_detection_segmentation_pennfudan.ipynb
```

---

### 9. Anomaly Detection (Synthetic + Real-World)

**Datasets**: Synthetic, California Housing, Forest Cover, KDD Cup 99  
**Task**: Detect anomalous samples using multiple strategies  
**Notebook**: `notebooks/12_anomaly_detection_cleanlab.ipynb`

#### Problem Statement

**Anomaly detection** (outlier detection, fraud detection) identifies data points that deviate significantly from the expected pattern. Unlike the outlier detection in Section 6 which focuses on data quality, this task addresses:

- 🔒 **Fraud Detection**: Credit card fraud, insurance fraud
- 🌐 **Network Security**: Intrusion detection, DDoS attacks
- 🏭 **Manufacturing**: Defect detection, equipment failure prediction
- 🏥 **Healthcare**: Disease outbreak detection, abnormal test results

Cleanlab's `Datalab` combined with traditional methods (Isolation Forest, LOF) provides a robust anomaly detection toolkit.

#### Mathematical Foundation

**1. k-Nearest Neighbors Distance (Cleanlab Datalab)**

For each sample $x_i$, compute average distance to k nearest neighbors:

$$\text{score}(x_i) = \frac{1}{k} \sum_{j \in \text{kNN}(x_i)} \|x_i - x_j\|_2$$

Higher scores indicate more isolated (anomalous) points. Cleanlab normalizes to [0, 1] where **lower = more anomalous**.

**2. Isolation Forest**

Anomalies are "few and different" — easier to isolate with random splits.

For each sample, compute average path length $h(x)$ across trees:

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

where:
- $h(x)$ = path length to isolate sample $x$
- $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$ ≈ average path in BST

Score interpretation:
- $s \to 1$: Definite anomaly (short path)
- $s \to 0.5$: Normal point (average path)

**3. Local Outlier Factor (LOF)**

Compares local density of a point to its neighbors:

$$\text{LOF}_k(x) = \frac{1}{k} \sum_{o \in N_k(x)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(x)}$$

where $\text{lrd}_k(x)$ is local reachability density.

- LOF ≈ 1: Similar density to neighbors (normal)
- LOF >> 1: Lower density than neighbors (anomaly)

**4. Contamination Rate**

The expected fraction of anomalies $\varepsilon = \frac{n_{\text{anomaly}}}{n_{\text{total}}}$:

$$\text{threshold} = \text{percentile}(\text{scores}, \varepsilon \times 100)$$

Samples below threshold (lower score = more anomalous) are flagged.

#### Architecture (Object-Oriented Design)

```
┌─────────────────────────────────────────────────────────────────┐
│                   AnomalyDetectionTask                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌───────────────┐    ┌─────────────┐  │
│  │  DataProvider    │───▶│  Preprocessor │───▶│  Strategy   │  │
│  │  (Dependency     │    │  (Scaler)     │    │  (Factory   │  │
│  │   Injection)     │    │               │    │   Pattern)  │  │
│  └──────────────────┘    └───────────────┘    └─────────────┘  │
│         ▲                                           │          │
│         │                                           ▼          │
│  ┌──────────────────┐                      ┌─────────────────┐ │
│  │ • Synthetic      │                      │ • DataLabKNN    │ │
│  │ • California     │                      │ • IsolationForest│ │
│  │ • ForestCover    │                      │ • LOF           │ │
│  │ • KDDCup99       │                      │ • Ensemble      │ │
│  └──────────────────┘                      └─────────────────┘ │
│                                                     │          │
│                                                     ▼          │
│                                         ┌───────────────────┐  │
│                                         │  Result Builder   │  │
│                                         │  (Pydantic)       │  │
│                                         └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Design Patterns Used:**

1. **Strategy Pattern**: Interchangeable detection algorithms (`DataLabKNNStrategy`, `IsolationForestStrategy`, `LocalOutlierFactorStrategy`, `EnsembleStrategy`)
2. **Factory Pattern**: `get_strategy("isolation_forest")` creates instances
3. **Dependency Injection**: DataProvider is injected into Task
4. **Template Method**: `Task.run()` defines algorithm skeleton

#### Code Example

```python
from cleanlab_demo.tasks.anomaly import (
    AnomalyDetectionTask,
    AnomalyDetectionConfig,
    run_anomaly_detection,
)
from cleanlab_demo.data.providers.anomaly import SyntheticAnomalyProvider

# Object-Oriented API
provider = SyntheticAnomalyProvider(n_samples=2000, contamination=0.05)
task = AnomalyDetectionTask(provider)

config = AnomalyDetectionConfig(
    strategy="ensemble",      # datalab_knn, isolation_forest, local_outlier_factor, ensemble
    contamination=0.05,       # Expected anomaly fraction
    n_neighbors=20,           # For kNN-based methods
    n_estimators=100,         # For Isolation Forest
    seed=42,
)

result = task.run(config)

print(f"Detected: {result.summary.n_anomalies_detected}")
print(f"Precision: {result.summary.precision:.2%}")
print(f"Recall: {result.summary.recall:.2%}")
print(f"F1 Score: {result.summary.f1_score:.2%}")

# Functional API (one-liner)
result = run_anomaly_detection(
    SyntheticAnomalyProvider(),
    strategy="isolation_forest",
    contamination=0.03,
)
```

#### Available Data Providers

| Provider | Description | Has Ground Truth |
|----------|-------------|------------------|
| `SyntheticAnomalyProvider` | Gaussian normal + uniform anomalies | ✓ |
| `CaliforniaHousingAnomalyProvider` | Housing data with injected anomalies | ✓ |
| `ForestCoverAnomalyProvider` | Rare forest types as anomalies | ✓ |
| `KDDCup99Provider` | Network intrusion detection | ✓ |

#### Strategy Comparison

| Strategy | Best For | Speed | Interpretability |
|----------|----------|-------|------------------|
| `datalab_knn` | General purpose, data quality | Medium | High |
| `isolation_forest` | High-dimensional data | Fast | Medium |
| `local_outlier_factor` | Varying local densities | Medium | High |
| `ensemble` | Production, robustness | Slow | Medium |

#### How to Run

```bash
# Notebook (recommended for learning)
jupyter notebook notebooks/12_anomaly_detection_cleanlab.ipynb

# Python script
python -c "
from cleanlab_demo.tasks.anomaly import run_anomaly_detection
from cleanlab_demo.data.providers.anomaly import SyntheticAnomalyProvider

result = run_anomaly_detection(
    SyntheticAnomalyProvider(n_samples=5000),
    strategy='ensemble',
    contamination=0.05,
)
print(f'F1: {result.summary.f1_score:.2%}')
"
```

#### Metrics Reported

| Metric | Description |
|--------|-------------|
| Precision | TP / (TP + FP) — of detected, how many are actual anomalies |
| Recall | TP / (TP + FN) — of actual anomalies, how many were detected |
| F1 Score | Harmonic mean of precision and recall |
| True Positives | Correctly identified anomalies |
| False Positives | Normal samples incorrectly flagged |
| False Negatives | Anomalies missed |

---

## Architecture

### Project Structure

```
cleanlab_demo/
├── src/cleanlab_demo/
│   ├── __init__.py          # Package initialization
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Pydantic configuration models
│   ├── metrics.py           # Evaluation metrics
│   ├── settings.py          # Global settings and logging
│   │
│   ├── ai/                  # AI report generation
│   │   └── ...              # LLM-powered analysis (pydantic-ai)
│   │
│   ├── core/                # Core utilities
│   │   ├── constants.py     # Magic numbers and defaults
│   │   ├── exceptions.py    # Custom exceptions
│   │   └── types.py         # Type definitions
│   │
│   ├── data/                # Dataset loading
│   │   ├── hub.py           # DatasetHub for loading datasets
│   │   └── schemas.py       # Data schemas
│   │
│   ├── tasks/               # Task implementations
│   │   ├── base.py          # Abstract base classes
│   │   ├── anomaly/         # Anomaly detection (NEW)
│   │   ├── multiclass/      # Multi-class classification
│   │   ├── multilabel/      # Multi-label classification
│   │   ├── token/           # Token classification (NER/POS)
│   │   ├── regression/      # Regression with CleanLearning
│   │   ├── outlier/         # Outlier detection with Datalab
│   │   ├── multiannotator/  # Multi-annotator + active learning
│   │   └── vision/          # Object detection + segmentation
│   │
│   ├── experiments/         # Experiment orchestration
│   │   ├── runner.py        # Single experiment runner
│   │   └── sweep.py         # Model comparison sweeps
│   │
│   ├── features/            # Feature preprocessing
│   │   └── preprocessor.py  # Numeric + categorical pipelines
│   │
│   ├── models/              # Model factory
│   │   └── factory.py       # Create sklearn models by name
│   │
│   ├── ui/                  # Streamlit web interface
│   │   └── streamlit_app.py # Main UI application
│   │
│   └── utils/               # Utilities
│       ├── download.py      # Dataset downloading
│       ├── filesystem.py    # File operations
│       └── ml.py            # ML utilities
│
├── notebooks/               # Jupyter tutorials (01-11)
├── tests/                   # Test suite
├── data/                    # Downloaded datasets
├── artifacts/               # Experiment results
└── docker-compose.yml       # Container orchestration
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       High-Level Data Flow                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Input                                                     │
│  (CLI / UI / Notebook)                                          │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                            │
│  │  Configuration  │ ◄── RunConfig, TaskConfig (Pydantic)       │
│  │  Validation     │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │  DatasetHub /   │────▶│ DataProvider    │                    │
│  │  Download       │     │ (task-specific) │                    │
│  └─────────────────┘     └────────┬────────┘                    │
│                                   │                             │
│                                   ▼                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Task Implementation                    │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │  │
│  │  │Preprocessor│─▶│   Model    │─▶│  Cross-Validation  │  │  │
│  │  │(features)  │  │(sklearn/   │  │  (out-of-sample    │  │  │
│  │  │            │  │ torch)     │  │   predictions)     │  │  │
│  │  └────────────┘  └────────────┘  └─────────┬──────────┘  │  │
│  │                                            │             │  │
│  │                                            ▼             │  │
│  │                              ┌────────────────────────┐  │  │
│  │                              │  Cleanlab API          │  │  │
│  │                              │  • find_label_issues   │  │  │
│  │                              │  • CleanLearning       │  │  │
│  │                              │  • Datalab             │  │  │
│  │                              └────────────┬───────────┘  │  │
│  │                                           │              │  │
│  │                                           ▼              │  │
│  │                              ┌────────────────────────┐  │  │
│  │                              │  Prune & Retrain       │  │  │
│  │                              │  (optional)            │  │  │
│  │                              └────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                           │                     │
│                                           ▼                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Result (Pydantic)                    │   │
│  │  • Metrics (baseline vs pruned)                         │   │
│  │  • Label issue summary                                  │   │
│  │  • Precision/Recall of detection                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                           │                     │
│                         ┌─────────────────┼─────────────────┐  │
│                         ▼                 ▼                 ▼  │
│                  ┌───────────┐     ┌───────────┐     ┌────────┐│
│                  │ JSON File │     │ Streamlit │     │   AI   ││
│                  │ Artifact  │     │ Dashboard │     │ Report ││
│                  └───────────┘     └───────────┘     └────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Component Interactions

```
┌─────────────────────────────────────────────────────────────────┐
│                    Component Dependency Graph                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│              ┌──────────────────────────────────┐               │
│              │           cli.py                 │               │
│              │  (user entry point)              │               │
│              └───────────────┬──────────────────┘               │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ experiments/│     │   tasks/    │     │    ai/      │        │
│  │  runner.py  │     │  (all task  │     │ (reports)   │        │
│  │  sweep.py   │     │   modules)  │     └─────────────┘        │
│  └──────┬──────┘     └──────┬──────┘                            │
│         │                   │                                   │
│         └─────────┬─────────┘                                   │
│                   │                                             │
│         ┌─────────┴─────────┐                                   │
│         ▼                   ▼                                   │
│  ┌─────────────┐     ┌─────────────┐                            │
│  │   data/     │     │  features/  │                            │
│  │  hub.py     │     │preprocessor │                            │
│  └──────┬──────┘     └─────────────┘                            │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │   utils/    │     │   models/   │     │   config.py │        │
│  │ download.py │     │  factory.py │     │ (Pydantic)  │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
│         │                   │                   │               │
│         └───────────────────┴───────────────────┘               │
│                             │                                   │
│                             ▼                                   │
│                      ┌─────────────┐                            │
│                      │   core/     │                            │
│                      │ constants   │                            │
│                      │ exceptions  │                            │
│                      │ types       │                            │
│                      └─────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

Environment variables (prefix: `CLEANLAB_DEMO_`):

| Variable | Description | Default |
|----------|-------------|---------|
| `CLEANLAB_DEMO_DATA_DIR` | Directory for dataset cache | `data` |
| `CLEANLAB_DEMO_ARTIFACTS_DIR` | Directory for results/artifacts | `artifacts` |
| `CLEANLAB_DEMO_LOG_LEVEL` | Logging level | `INFO` |
| `CLEANLAB_DEMO_OPENAI_API_KEY` | OpenAI key (propagated to `OPENAI_API_KEY`) | - |
| `CLEANLAB_DEMO_ANTHROPIC_API_KEY` | Anthropic key (propagated to `ANTHROPIC_API_KEY`) | - |

The project also loads a project-root `.env` file (ignored by git) if present.
Use `.env.example` as a template.

## AI Report (optional)

Generate AI-powered analysis reports using an LLM:

```bash
# Set API key
export OPENAI_API_KEY="..."
# or
export ANTHROPIC_API_KEY="..."

# or using the app prefix (also supports `.env`)
export CLEANLAB_DEMO_OPENAI_API_KEY="..."

# Run experiment and generate report
cleanlab-demo run -d adult_income -o artifacts/last_result.json
cleanlab-demo ai-report

# Use deterministic report (no LLM)
cleanlab-demo ai-report --no-ai
```

Or use the notebook: `notebooks/03_pydantic_ai_report.ipynb`

## Repo Hygiene

This project follows production-grade best practices for code quality, security, and maintainability.

### Badges

```markdown
[![CI](https://github.com/your-org/cleanlab-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/cleanlab-demo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/your-org/cleanlab-demo/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/cleanlab-demo)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: ruff-format](https://img.shields.io/badge/code%20style-ruff--format-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

### Quality Standards

| Tool | Purpose | Threshold |
|------|---------|-----------|
| **pytest** | Unit & integration testing | 45% coverage minimum |
| **ruff** | Linting (replaces flake8, isort, pyupgrade, etc.) | Zero violations |
| **ruff format** | Code formatting (replaces black) | Auto-enforced |
| **mypy** | Static type checking | Strict mode, no errors |
| **bandit** | Security vulnerability scanning | No high/medium issues |
| **pre-commit** | Git hooks for automated checks | All hooks pass |

### Project Structure Best Practices

```
cleanlab_demo/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── .pre-commit-config.yaml     # Pre-commit hook configuration
├── .gitignore                  # Git ignore patterns
├── .dockerignore               # Docker ignore patterns
├── .env.example                # Environment variable template
├── pyproject.toml              # Single source of truth for config
├── README.md                   # Project documentation
├── LICENSE                     # MIT license
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-container orchestration
├── src/
│   └── cleanlab_demo/          # Source code (src layout)
│       ├── __init__.py         # Package version
│       ├── py.typed            # PEP 561 marker (typed package)
│       └── ...
├── tests/
│   ├── conftest.py             # Shared fixtures
│   └── test_*.py               # Test modules
├── notebooks/                  # Jupyter notebooks
├── data/                       # Data directory (git-ignored)
└── artifacts/                  # Output artifacts (git-ignored)
```

### Pre-commit Hooks

The project uses [pre-commit](https://pre-commit.com/) to enforce quality on every commit:

```yaml
# .pre-commit-config.yaml includes:
repos:
  - ruff (lint + format)           # Fast Python linter
  - pre-commit-hooks               # File hygiene (trailing whitespace, etc.)
  - mypy                           # Type checking
  - bandit                         # Security scanning
  - conventional-pre-commit        # Commit message format
```

**Setup:**

```bash
# Install pre-commit
pip install pre-commit

# Install hooks (run once after cloning)
pre-commit install
pre-commit install --hook-type commit-msg

# Run all hooks manually
pre-commit run --all-files

# Update hooks to latest versions
pre-commit autoupdate
```

### Commit Message Convention

This project enforces [Conventional Commits](https://www.conventionalcommits.org/) via pre-commit hooks:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(tasks): add vision detection task` |
| `fix` | Bug fix | `fix(cli): handle missing config file` |
| `docs` | Documentation | `docs: update installation guide` |
| `style` | Code style (formatting) | `style: apply ruff formatting` |
| `refactor` | Code refactoring | `refactor(core): simplify data loading` |
| `perf` | Performance improvement | `perf: optimize cross-validation loop` |
| `test` | Adding/updating tests | `test: add regression task tests` |
| `build` | Build system changes | `build: update dependencies` |
| `ci` | CI/CD changes | `ci: add Python 3.12 to matrix` |
| `chore` | Maintenance | `chore: clean up unused imports` |

**Breaking Changes:**

```bash
feat(api)!: redesign configuration schema

BREAKING CHANGE: RunConfig now requires `task_type` field
```

### CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):

```
┌─────────────────────────────────────────────────────────────────┐
│                        CI Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Push/PR to main                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Lint   │  │  Test   │  │ Security │  │ Build & Verify   │  │
│  │         │  │ Matrix  │  │   Scan   │  │    Package       │  │
│  │ • ruff  │  │ • 3.11  │  │ • bandit │  │ • python -m build│  │
│  │ • mypy  │  │ • 3.12  │  │ • safety │  │ • pip install    │  │
│  │         │  │ • 45%   │  │          │  │ • import check   │  │
│  │         │  │   cov   │  │          │  │                  │  │
│  └────┬────┘  └────┬────┘  └────┬─────┘  └────────┬─────────┘  │
│       │            │            │                  │            │
│       └────────────┴────────────┴──────────────────┘            │
│                             │                                   │
│                             ▼                                   │
│                    ┌─────────────────┐                          │
│                    │  Docker Build   │  (main branch only)      │
│                    │  + Upload       │                          │
│                    │  Artifacts      │                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

**Jobs:**

| Job | Trigger | Description |
|-----|---------|-------------|
| `lint` | All pushes/PRs | Ruff linting + mypy type checking |
| `test` | All pushes/PRs | Pytest with coverage (Python 3.11, 3.12) |
| `security` | All pushes/PRs | Bandit security scan |
| `build` | After lint+test pass | Build wheel, verify installation |
| `docker` | Push to main | Build Docker image, cache layers |

### Running Quality Checks Locally

```bash
# ============================================================
# Quick check (recommended before committing)
# ============================================================
pre-commit run --all-files

# ============================================================
# Individual tools
# ============================================================

# Linting (with auto-fix)
ruff check src/ tests/ --fix
ruff format src/ tests/

# Type checking
mypy src/cleanlab_demo

# Security scan
bandit -r src/cleanlab_demo -ll

# Dependency vulnerability check
pip install safety
safety check

# ============================================================
# Testing
# ============================================================

# Run all tests
pytest

# Run with coverage report
pytest --cov=cleanlab_demo --cov-report=term-missing --cov-fail-under=45

# Run specific test file
pytest tests/test_core.py -v

# Run tests matching pattern
pytest -k "test_multiclass" -v

# Run tests in parallel (faster)
pytest -n auto

# Skip slow tests
pytest -m "not slow"

# ============================================================
# Full CI simulation
# ============================================================
ruff check src/ tests/ && \
ruff format --check src/ tests/ && \
mypy src/cleanlab_demo && \
bandit -r src/cleanlab_demo -ll && \
pytest --cov=cleanlab_demo --cov-fail-under=45
```

### Git Ignore Patterns

The `.gitignore` is organized into sections for easy maintenance:

```gitignore
# ==============================================================================
# Main Categories in .gitignore
# ==============================================================================

# Python: __pycache__/, *.py[cod], *.egg-info/, dist/, build/, *.so
# Virtual Envs: .venv/, venv/, .python-version
# IDE: .idea/, .vscode/, *.swp
# Jupyter: .ipynb_checkpoints/
# Type/Lint: .mypy_cache/, .ruff_cache/, .pytest_cache/
# Environment: .env, .env.* (except .env.example)
# OS: .DS_Store, Thumbs.db
# Data: /data/, /artifacts/, /notebooks/data/, /notebooks/artifacts/
# Models: *.pkl, *.h5, *.pt, *.pth, *.ckpt (large binary files)
# Secrets: *_secret*, *.pem, *.key
```

**Key exclusions:**
- All Python bytecode and cache files
- Virtual environments (any naming convention)
- IDE-specific files (PyCharm, VS Code, Vim, etc.)
- Environment files containing secrets
- Large data files and model artifacts
- OS-generated metadata files

### Docker Ignore Patterns

The `.dockerignore` minimizes build context for faster builds:

```dockerignore
# ==============================================================================
# Main Categories in .dockerignore
# ==============================================================================

# Version Control: .git/, .github/
# IDE: .idea/, .vscode/
# Python: __pycache__/, *.egg-info/, dist/, build/
# Virtual Envs: .venv/, venv/
# Testing/Lint: .pytest_cache/, .mypy_cache/, .ruff_cache/, .coverage
# Jupyter: .ipynb_checkpoints/
# Environment: .env (except .env.example)
# Data: data/, artifacts/, *.pkl, *.h5, *.pt, *.zip
# Docs: *.md (except README.md), docs/
```

**Benefits:**
- Faster `docker build` (smaller context transfer)
- Smaller final images (no test files, docs in prod)
- No secrets accidentally copied into images

### Environment Variables

Use `.env.example` as a template:

```bash
# Copy template
cp .env.example .env

# Edit with your values
vim .env

```

**Never commit `.env` files** — they're git-ignored by default.

### Dependency Management

Dependencies are managed in `pyproject.toml` with optional groups:

```bash
# Core dependencies only
pip install -e .

# Development (testing, linting)
pip install -e ".[dev]"

# UI (Streamlit)
pip install -e ".[ui]"

# Notebooks (JupyterLab)
pip install -e ".[notebooks]"

# AI reports (pydantic-ai)
pip install -e ".[ai]"

# Everything
pip install -e ".[all]"
```

**Updating dependencies:**

```bash
# Check for outdated packages
pip list --outdated

# Update pre-commit hooks
pre-commit autoupdate

# Regenerate lock file (if using uv)
uv lock
```

### Release Checklist

Before releasing a new version:

- [ ] All tests pass (`pytest --cov-fail-under=45`)
- [ ] No linting errors (`ruff check src/ tests/`)
- [ ] No type errors (`mypy src/cleanlab_demo`)
- [ ] No security issues (`bandit -r src/cleanlab_demo -ll`)
- [ ] Update version in `src/cleanlab_demo/__init__.py`
- [ ] Update `CHANGELOG.md` (if present)
- [ ] Create git tag (`git tag -a v0.2.0 -m "Release v0.2.0"`)
- [ ] Push tag (`git push origin v0.2.0`)
- [ ] Verify CI passes on the tag
- [ ] Build and upload to PyPI (if applicable)

---

## Development

### Getting Started

```bash
# Clone the repository
git clone https://github.com/your-org/cleanlab-demo.git
cd cleanlab-demo

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev,ui,notebooks]"

# Set up pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Run tests to verify setup
pytest -v
```

### Adding a New Task

1. Create a new directory under `src/cleanlab_demo/tasks/`:
   ```
   tasks/
   └── new_task/
       ├── __init__.py
       ├── provider.py    # Data loading
       ├── schemas.py     # Pydantic models
       └── task.py        # Task implementation
   ```

2. Implement the `DataProvider` and `Task` classes following the base classes in `tasks/base.py`

3. Add tests in `tests/test_tasks/test_new_task.py`

4. Add a notebook in `notebooks/XX_new_task.ipynb`

5. Update the README with documentation

### Debugging Tips

```bash
# Run with verbose logging
CLEANLAB_DEMO_LOG_LEVEL=DEBUG cleanlab-demo run -d adult_income

# Run pytest with output
pytest -v -s --tb=long

# Run specific test with debugger
pytest tests/test_core.py::test_specific -v --pdb

# Profile memory usage
pip install memory_profiler
python -m memory_profiler src/cleanlab_demo/cli.py run -d adult_income
```

---

## License

MIT License

Copyright (c) 2024-2026 Cleanlab Demo Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.




