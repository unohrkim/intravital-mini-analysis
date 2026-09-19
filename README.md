# Intravital Cell Motility Analysis

A reproducible synthetic-analysis project for studying directed cell migration toward a simulated sinusoid, with classical trajectory statistics and sequence-based machine-learning baselines.

## Overview

This project asks a simple modeling question:

> Do cells in an injury condition show stronger migration toward a simulated sinusoid than cells in a control condition?

The project begins with a controlled synthetic trajectory simulator and conventional motility analysis, then extends the same underlying migration process into a sequence-classification benchmark.

The workflow was intentionally designed so that the ground-truth signal is known in advance. This makes it possible to test whether analysis and modeling pipelines can recover an encoded migration pattern without presenting the result as a biological discovery.

## Project Evolution

```text
Synthetic trajectory simulation
        ↓
Trajectory-level motility metrics
        ↓
Condition-level statistical comparison
        ↓
Synthetic TIFF / trajectory visualization
        ↓
Larger sequence-learning dataset
        ↓
Flattened Logistic Regression baseline
        ↓
TensorFlow 1D CNN
        ↓
PyTorch 1D CNN
        ↓
Frozen validation-stage configurations
        ↓
One-time held-out test comparison
```

## Synthetic Migration Model

The simulator generates control and injury trajectories in a 2D field containing a simulated sinusoid at:

```text
x = 100 µm
```

Each trajectory contains:

```text
30 frames
1-minute frame interval
```

Movement includes random displacement in both spatial dimensions.

For the injury condition, an additional directional x-axis bias is applied toward the simulated sinusoid:

```text
left of sinusoid  → positive x bias
right of sinusoid → negative x bias
```

The control condition contains only the random movement component.

This deliberately encoded rule provides a known signal that the downstream analyses are expected to recover.

## Classical Trajectory Analysis

The first stage quantifies each trajectory using predefined motility features, including:

- step speed
- mean speed
- path length
- net displacement
- persistence
- arrest coefficient
- distance to the simulated sinusoid
- approach distance

The primary condition-level analysis uses **approach distance** to test whether injury trajectories move more strongly toward the sinusoid.

The statistical workflow includes:

- two-sided permutation testing
- Hedges' g
- percentile bootstrap confidence intervals

In the original synthetic dataset, the injury condition showed a substantially larger approach distance than control, consistent with recovery of the directional bias built into the simulator.

This result is interpreted as **synthetic known-signal recovery**, not biological evidence.

## Deep-Learning Extension

The project was then extended to ask a different question:

> Can a model distinguish control and injury migration patterns directly from trajectory sequences rather than from a small set of predefined summary metrics?

A larger synthetic dataset was generated using the same migration logic:

```text
4,000 total trajectories
2,000 control
2,000 injury
```

Each track is represented as a 29-step sequence with three channels:

```text
[dx, dy, relative_x]
```

where:

```text
dx_t         = x_(t+1) - x_t
dy_t         = y_(t+1) - y_t
relative_x_t = (x_t - 100) / 100
```

The fixed, track-level stratified split is:

| Split | Control | Injury | Total |
|---|---:|---:|---:|
| Train | 1400 | 1400 | 2800 |
| Validation | 300 | 300 | 600 |
| Test | 300 | 300 | 600 |

All frames from one track remain in the same split.

## Models

Three frozen baselines were compared.

### Logistic Regression

The `(29, 3)` sequence is flattened to 87 features and standardized using statistics fit on the training split only.

This serves as a simple linear reference model.

### TensorFlow 1D CNN

```text
Input (29, 3)
→ Conv1D(16, kernel=3, ReLU)
→ Conv1D(16, kernel=3, ReLU)
→ GlobalAveragePooling1D
→ Dense(1, sigmoid)
```

Training configuration:

```text
Adam
learning rate = 1e-3
batch size = 32
epochs = 30
threshold = 0.5
random state = 2029
```

### PyTorch 1D CNN

The PyTorch model uses the same canonical `(29, 3)` input and internally converts it to the channels-first layout expected by `Conv1d`.

```text
Input (29, 3)
→ transpose to (3, 29)
→ Conv1d(3 → 16, kernel=3)
→ ReLU
→ Conv1d(16 → 16, kernel=3)
→ ReLU
→ global average pooling
→ Linear(16 → 1)
```

Training configuration:

```text
Adam
learning rate = 1e-3
BCEWithLogitsLoss
batch size = 32
epochs = 30
threshold = 0.5
random state = 2030
CPU only
```

## Final Held-Out Test Results

After all three validation-stage configurations were frozen, a single final comparison was performed on the held-out test split.

Test set:

```text
n = 600
control = 300
injury = 300
```

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.530 | 0.531 | 0.513 | 0.522 | 0.519 |
| TensorFlow CNN | 0.765 | 0.789 | 0.723 | 0.755 | 0.851 |
| PyTorch CNN | 0.768 | 0.742 | 0.823 | 0.780 | 0.852 |

The two CNN implementations preserved the discrimination observed during validation, while the current flattened Logistic Regression baseline remained close to chance.

The nearly identical ROC-AUC values of the TensorFlow and PyTorch models are consistent with the shared sequence-aware nonlinear architecture recovering the deliberately encoded synthetic migration signal across two framework implementations.

This should not be interpreted as evidence that CNNs are universally superior to linear models. A different linear representation, for example one with explicit interaction features, could behave differently.

## Validation and Test Discipline

The project keeps development and final evaluation separate.

The workflow was:

```text
define fixed split
→ develop on train
→ observe validation performance
→ freeze model configurations
→ implement and test final-comparison pipeline
→ commit pre-test code
→ record commit hash
→ evaluate held-out test once
→ no post-test tuning
```

The pre-test comparison pipeline was committed before the real test set was evaluated.

Pre-test commit:

```text
b26f0c1  Add final test comparison pipeline
```

No architecture, threshold, preprocessing rule, epoch count, or other model setting was changed after observing the final test results.

## Reproducibility

The project uses fixed seeds for:

- synthetic trajectory generation
- dataset splitting
- Logistic Regression
- TensorFlow training
- PyTorch training

Preprocessing statistics are estimated from the training split only.

The current test suite contains:

```text
219 passing tests
```

covering data validation, split integrity, preprocessing, model-shape contracts, deterministic behavior, leakage protection, and final-comparison plumbing.

## Repository Structure

```text
intravital-mini-analysis/
├── data/                  # synthetic trajectory data and persisted split
├── results/               # machine-readable analysis/model results
├── scripts/               # utility scripts
├── src/                   # simulation, analysis, and modeling code
├── tests/                 # pytest test suite
├── PROJECT_SPEC.md        # scientific / analytical project specification
├── CLAUDE.md              # coding-agent project instructions
├── pyproject.toml
└── uv.lock
```

Detailed step-by-step development notes are maintained separately in the companion notes repository:

```text
intravital-mini-analysis-notes
```

## Development Workflow

The implementation was developed iteratively with automated testing and code review at each stage. Claude Code was used as a coding and review assistant during development, while the scientific assumptions, analysis design, model configuration, and interpretation were explicitly defined and reviewed within the project workflow.

## Running the Project

Install dependencies:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest -v
```

Run the final comparison pipeline:

```bash
uv run python -m src.final_test_comparison
```

The final comparison output is written to:

```text
results/dl/final_test_comparison.csv
```

## Interpretation Boundary

This project uses simulated trajectories with a deliberately encoded migration rule.

Its results therefore demonstrate:

- reproducible synthetic trajectory analysis,
- recovery of a known directional migration signal,
- comparison of linear and sequence-aware model representations,
- train/validation/test discipline across TensorFlow and PyTorch implementations.

They do **not** establish:

- a biological migration mechanism,
- clinical relevance,
- biological validity of the simulator,
- performance on real intravital microscopy data.

The project is intended as a compact, reproducible example of scientific analysis, model evaluation, and software-engineering discipline for cell-motility data.
