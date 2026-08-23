# Intravital Mini Analysis — Project Specification

## Project Purpose

This project implements a small, reproducible analysis workflow using synthetic intravital microscopy-like cell tracking data.

The project has two complementary goals:

1. to practice deterministic analysis of cell migration and spatial behavior, and
2. to demonstrate a controlled AI-assisted coding workflow using Claude Code.

The scientific definitions and analysis rules are defined by the researcher. Claude Code may assist with implementation, testing, debugging, and code review, but it should not independently redefine the scientific analysis.



## Scientific Question

The project addresses the following synthetic research question:

> Do cells in an injury condition show stronger migration toward a simulated sinusoid than cells in a control condition?

This is a synthetic exercise rather than a biological experiment.

The expected directional effect is deliberately introduced into the simulated injury condition so that the analysis pipeline can be tested against known ground truth.


## Experimental Design

The initial simulation contains two conditions:

- `control`
- `injury`

The fixed parameters are:

| Parameter | Value |
| --- | --- |
| Conditions | Control and Injury |
| Cells per condition | 15 |
| Total cells | 30 |
| Frames per cell | 30 |
| Frame interval | 1 minute |
| Field size | 200 × 200 µm |
| Spatial scale | 1 pixel = 1 µm |
| Simulated sinusoid | Vertical line at x = 100 µm |
| Random seed | 42 |
| Arrest threshold | 2.0 µm/min |

These values should remain fixed unless they are deliberately revised by the researcher.


## Synthetic Cell Behavior

### Control Condition

Control cells will undergo two-dimensional random movement without an intentionally introduced directional preference toward the sinusoid.

Conceptually:

```text
random movement
```

### Injury Condition

Injury cells will undergo similar random movement but will also receive a weak directional bias toward the simulated sinusoid.

Conceptually:

```text
random movement
        +
directional bias toward x = 100 µm
```

The magnitude of this directional bias has not yet been fixed numerically.

It must be explicitly defined by the researcher before implementation and must not be chosen silently by the coding assistant.


## Simulated Sinusoid

The initial sinusoid geometry is intentionally simple.

It is represented by the vertical line:

```text
x = 100 µm
```

For a cell at coordinate `(x, y)`, distance to the sinusoid is therefore:

```text
distance_to_sinusoid = |x - 100|
```

This simplified geometry allows the spatial calculation to be inspected and validated directly.


## Tracking Data Structure

The synthetic tracking data will use a tidy tabular structure.

Expected columns:

```text
track_id
condition
frame
time_min
x_um
y_um
```

Example:

```text
track_id,condition,frame,time_min,x_um,y_um
C01,control,0,0,32.1,48.5
C01,control,1,1,33.0,49.2
C01,control,2,2,32.7,51.0
```

Each row represents one cell position at one time point.


## Deterministic Analysis Metrics

All primary scientific metrics must be calculated using deterministic Python functions.

No LLM call should be involved in calculating these values.

### Step Distance

For consecutive positions:

```text
step_distance =
sqrt((x[t+1] - x[t])² + (y[t+1] - y[t])²)
```

Unit:

```text
µm
```


### Step Speed

```text
step_speed =
step_distance / frame_interval
```

Because the frame interval is 1 minute, the resulting unit is:

```text
µm/min
```


### Mean Speed

Mean speed is the arithmetic mean of valid step speeds within a track.


### Path Length

Path length is the sum of consecutive step distances:

```text
path_length = Σ step_distance
```


### Displacement

Displacement is the Euclidean distance between the first and final cell positions:

```text
displacement =
sqrt((x_final - x_initial)² + (y_final - y_initial)²)
```


### Persistence

Persistence is defined as:

```text
persistence = displacement / path_length
```

A value closer to 1 represents a more direct trajectory.

A lower value represents a more wandering trajectory.

If path length is zero, the implementation must handle the case explicitly rather than performing an undefined division.


### Arrest Coefficient

For this synthetic project, an interval is defined as arrested when:

```text
step_speed < 2.0 µm/min
```

The arrest coefficient is:

```text
number of arrested intervals
────────────────────────────
number of valid intervals
```

The `2.0 µm/min` threshold is a project-specific analysis parameter.

It should not be interpreted as a universally validated biological threshold.


### Distance to Sinusoid

For every cell position:

```text
distance_to_sinusoid = |x - 100|
```

Unit:

```text
µm
```


### Approach Distance

Net approach toward the sinusoid is defined as:

```text
approach_distance =
initial_distance_to_sinusoid
-
final_distance_to_sinusoid
```

Interpretation:

```text
positive
→ net movement toward the sinusoid

zero
→ no net change

negative
→ net movement away from the sinusoid
```

Approach distance will be the primary spatial metric used to compare the control and injury conditions.


## Planned Outputs

The initial workflow is expected to generate:

```text
data/
├── tracks.csv
└── synthetic_movie.tif
```

and analysis outputs under:

```text
results/
└── figures/
```

The coordinate table in `tracks.csv` will represent the ground-truth tracks.

The initial project will not attempt to recover tracks from the TIFF movie through segmentation or automated tracking.


## Initial Scope

The first project version will include:

- synthetic cell-track generation,
- synthetic microscopy-style TIFF generation,
- deterministic migration metrics,
- distance-to-sinusoid analysis,
- automated tests,
- control-versus-injury comparisons,
- trajectory visualization,
- basic quality-control plots, and
- AI-assisted implementation with human review.

The first version will not include:

- real biological data,
- patient data,
- image segmentation,
- automated object detection,
- automated cell tracking,
- deep learning,
- LangGraph,
- LLM-based scientific classification, or
- autonomous biological interpretation.


## Scientific Responsibility

The researcher is responsible for:

- defining the scientific question,
- selecting metrics,
- defining thresholds,
- defining simulation parameters,
- validating calculations,
- deciding whether outputs are scientifically meaningful, and
- interpreting the results.

The coding assistant may help with:

- implementation,
- test design,
- debugging,
- refactoring,
- code organization,
- documentation, and
- identification of possible edge cases.

The coding assistant must not silently modify scientific definitions.

## Design Principle

The intended workflow is:

```text
Scientific question
        ↓
Human-defined specification
        ↓
AI-assisted implementation
        ↓
Deterministic Python calculations
        ↓
Automated tests
        ↓
Human validation
        ↓
Interpretation
```

The LLM is therefore used as a coding and review assistant rather than as the scientific analysis engine.


## Open Design Decision

One simulation parameter remains intentionally unresolved:

```text
Magnitude of the directional bias
for injury-condition cells
```

This value must be selected explicitly before the synthetic data generator is implemented.

The coding assistant may identify this as an unresolved design parameter but should not choose a value without researcher approval.