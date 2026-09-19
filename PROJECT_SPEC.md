# Intravital Mini Analysis — Project Specification

## 1. Scientific Question

The project addresses the following synthetic research question:

> Do cells in an injury condition show stronger migration toward a simulated sinusoid than cells in a control condition?

This project uses synthetic data with a deliberately encoded directional migration signal.


## 2. Core Experimental Parameters

| Parameter | Value |
|---|---:|
| Conditions | `control`, `injury` |
| Cells per condition | 15 |
| Total cells | 30 |
| Frames per cell | 30 |
| Frame interval | 1 min |
| Field width | 200 µm |
| Field height | 200 µm |
| Spatial scale | 1 pixel = 1 µm |
| Simulated sinusoid | vertical line at `x = 100 µm` |
| Random seed | 42 |
| Arrest threshold | 2.0 µm/min |

These values are fixed for the original analysis unless a new experiment is explicitly defined.


## 3. Synthetic Movement Model

### 3.1 Initial positions

For each cell:

```text
x_initial ~ Uniform(0, 200 µm)
y_initial ~ Uniform(0, 200 µm)
```

The same initialization rule is used for both conditions.

### 3.2 Random movement

At each frame transition:

```text
dx_random ~ Normal(0, 3 µm)
dy_random ~ Normal(0, 3 µm)
```

### 3.3 Control condition

```text
dx = dx_random
dy = dy_random
```

No directional bias is added.

### 3.4 Injury condition

A constant directional x-axis bias toward the simulated sinusoid is added:

```text
dx_bias = 0.8 × sign(100 - x)

dx = dx_random + dx_bias
dy = dy_random
```

Bias behavior:

```text
x < 100 µm  → +0.8 µm/frame
x > 100 µm  → -0.8 µm/frame
x = 100 µm  →  0.0 µm/frame
```

The bias is not distance-dependent or speed-scaled.

### 3.5 Boundary handling

The valid field is:

```text
0 ≤ x ≤ 200 µm
0 ≤ y ≤ 200 µm
```

Reflective boundary handling is used whenever a proposed step leaves the field.


## 4. Simulated Sinusoid

The simulated sinusoid is represented as:

```text
x = 100 µm
```

For a position `(x, y)`:

```text
distance_to_sinusoid = |x - 100|
```


## 5. Ground-Truth Tracking Data

The coordinate table uses the following columns:

```text
track_id
condition
frame
time_min
x_um
y_um
```

The primary coordinate file is:

```text
data/tracks.csv
```

Each row represents one cell position at one time point.


## 6. Deterministic Migration Metrics

All metrics are calculated deterministically from the coordinate table.

### 6.1 Step distance

```text
step_distance =
sqrt((x[t+1] - x[t])² + (y[t+1] - y[t])²)
```

Unit: `µm`

### 6.2 Step speed

```text
step_speed = step_distance / frame_interval
```

Unit: `µm/min`

### 6.3 Mean speed

```text
mean_speed = mean(valid step speeds within a track)
```

### 6.4 Path length

```text
path_length = Σ step_distance
```

### 6.5 Displacement

```text
displacement =
sqrt((x_final - x_initial)² + (y_final - y_initial)²)
```

### 6.6 Persistence

```text
persistence = displacement / path_length
```

If `path_length = 0`, the implementation must handle the case explicitly.

### 6.7 Arrest coefficient

An interval is classified as arrested when:

```text
step_speed < 2.0 µm/min
```

Then:

```text
arrest_coefficient =
number of arrested intervals / number of valid intervals
```

### 6.8 Distance to sinusoid

```text
distance_to_sinusoid = |x - 100|
```

Unit: `µm`

### 6.9 Approach distance

```text
approach_distance =
initial_distance_to_sinusoid - final_distance_to_sinusoid
```

Interpretation:

```text
positive → net movement toward sinusoid
zero     → no net change
negative → net movement away from sinusoid
```

`approach_distance` is the primary spatial metric for the original condition-level comparison.


## 7. Classical Statistical Analysis Specification

Primary comparison:

```text
injury vs control
```

Primary outcome:

```text
approach_distance
```

Required analyses:

- two-sided permutation test,
- Hedges' g,
- percentile bootstrap confidence interval.

The analysis must not redefine responders or subsets using the same outcome and then test that same outcome as confirmatory evidence.


## 8. Synthetic TIFF Specification Boundary

Synthetic TIFF output may be generated from the ground-truth trajectories.

The coordinate table remains the authoritative source for analysis.

The TIFF is not used for segmentation, object detection, or track recovery in the current project.


## 9. Sequence-Modeling Dataset

The approved sequence-modeling extension uses a separate synthetic dataset:

```text
2,000 control trajectories
2,000 injury trajectories
4,000 total trajectories
30 frames per trajectory
```

The simulator logic remains unchanged.

The deep-learning generation seed is:

```text
DL_SEED = 2026
```


## 10. Sequence Representation

Each 30-frame trajectory is converted into 29 transitions.

Canonical input shape:

```text
(N, 29, 3)
```

Channel order:

```text
[dx, dy, relative_x]
```

Definitions:

```text
dx_t         = x_(t+1) - x_t
dy_t         = y_(t+1) - y_t
relative_x_t = (x_t - 100) / 100
```

`relative_x_t` uses the pre-step position `x_t`.

Class labels:

```text
control = 0
injury  = 1
```


## 11. Fixed Train / Validation / Test Split

Split seed:

```text
DL_SPLIT_SEED = 2027
```

Persisted split:

```text
data/dl_split.csv
```

Track-level stratified split:

| Split | Control | Injury | Total |
|---|---:|---:|---:|
| Train | 1400 | 1400 | 2800 |
| Validation | 300 | 300 | 600 |
| Test | 300 | 300 | 600 |

All frames from one trajectory must remain in the same split.

The persisted split must not be regenerated or reshuffled for the completed comparison.


## 12. Logistic Regression Baseline Specification

Input:

```text
(N, 29, 3)
→ flatten in C-order
→ 87 features
```

Preprocessing:

```text
StandardScaler
fit on training split only
```

Model:

```text
LogisticRegression(
    solver="lbfgs",
    l1_ratio=0,
    C=1.0,
    max_iter=1000,
    random_state=2028,
)
```

No handcrafted interaction features are included in this baseline.


## 13. TensorFlow 1D CNN Specification

Input:

```text
(N, 29, 3)
```

Per-channel scaling:

```text
reshape training data to (-1, 3)
compute one mean and one population standard deviation per channel
fit on training split only
zero standard deviation → 1.0
```

Architecture:

```text
Input (29, 3)
→ Conv1D(16, kernel_size=3, padding="same", activation="relu")
→ Conv1D(16, kernel_size=3, padding="same", activation="relu")
→ GlobalAveragePooling1D
→ Dense(1, activation="sigmoid")
```

Training configuration:

```text
optimizer = Adam
learning_rate = 1e-3
loss = binary crossentropy
batch_size = 32
epochs = 30
decision_threshold = 0.5
random_state = 2029
```

No early stopping or validation-driven hyperparameter adaptation is used.


## 14. PyTorch 1D CNN Specification

Canonical external input:

```text
(N, 29, 3)
```

Internal Conv1d layout:

```text
(N, 3, 29)
```

Per-channel scaling follows the same train-only rule as the TensorFlow model.

Architecture:

```text
Conv1d(3, 16, kernel_size=3, padding=1)
→ ReLU
→ Conv1d(16, 16, kernel_size=3, padding=1)
→ ReLU
→ mean over time dimension
→ Linear(16, 1)
```

Training configuration:

```text
optimizer = Adam
learning_rate = 1e-3
loss = BCEWithLogitsLoss
batch_size = 32
epochs = 30
decision_threshold = 0.5
random_state = 2030
device = CPU
```

Sigmoid is applied only when converting logits to probabilities.


## 15. Final Evaluation Protocol

The three model configurations are frozen before final test evaluation.

Final fitting rules:

- use the original training split only,
- fit preprocessing on training data only,
- do not combine train and validation,
- do not modify architecture or hyperparameters after seeing test results,
- evaluate each model on the same held-out test split.

Final metrics:

- accuracy,
- precision,
- recall,
- F1,
- ROC-AUC.

Final output:

```text
results/dl/final_test_comparison.csv
```

Pre-test implementation checkpoint:

```text
b26f0c1  Add final test comparison pipeline
```

The held-out test result is treated as final for this experiment.

Any later change to preprocessing, architecture, thresholds, epochs, features, or training data must be treated as a separate experiment.


## 16. Interpretation Constraints

The project supports interpretation only within the synthetic task.

Permitted interpretation:

- recovery of a deliberately encoded migration signal,
- held-out generalization within the synthetic data-generating process,
- comparison of fixed model representations under the same task.

Not permitted:

- biological discovery,
- clinical inference,
- biological validation of the simulator,
- claims about real intravital microscopy performance,
- universal claims about model-family superiority.
