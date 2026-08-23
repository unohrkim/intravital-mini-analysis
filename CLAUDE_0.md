# Claude Code Project Instructions

## Project Purpose

This repository contains a small synthetic intravital imaging analysis project.

The project has two goals:

1. implement a reproducible and deterministic cell-migration analysis workflow, and
2. demonstrate a controlled AI-assisted coding workflow.

The authoritative scientific specification is defined in:

```text
PROJECT_SPEC.md
```

Read `PROJECT_SPEC.md` before proposing or implementing scientific analysis code.

---

## Core Principle

Scientific decisions remain human-defined.

Claude Code may assist with:

- implementation,
- testing,
- debugging,
- refactoring,
- code organization,
- documentation, and
- identification of edge cases.

Claude Code must not independently redefine:

- the scientific question,
- metric definitions,
- analysis thresholds,
- simulation assumptions,
- expected biological interpretation, or
- project scope.

If a scientific rule is missing or ambiguous, identify the ambiguity rather than silently choosing a value.

---

## Current Scientific Question

The project asks:

> Do cells in an injury condition show stronger migration toward a simulated sinusoid than cells in a control condition?

This is a synthetic analysis exercise.

The generated results must not be described as experimental biological evidence.

---

## Fixed Simulation Parameters

Unless explicitly changed by the researcher, use:

```text
conditions              = control, injury
cells_per_condition     = 15
frames_per_cell         = 30
frame_interval_min      = 1
field_width_um          = 200
field_height_um         = 200
pixel_size_um           = 1
sinusoid_x_um           = 100
random_seed             = 42
arrest_threshold_um_min = 2.0
```

Do not change these values simply to improve an output, test result, or visualization.

---

## Unresolved Scientific Parameter

The following parameter has intentionally not yet been assigned a numerical value:

```text
injury directional-bias magnitude
```

Do not select a value for this parameter without explicit researcher approval.

It is acceptable—and expected—to identify this as an unresolved requirement before implementation.

---

## Deterministic Analysis Requirements

The following quantities must be calculated using explicit deterministic Python functions:

- step distance,
- step speed,
- mean speed,
- path length,
- displacement,
- persistence,
- arrest coefficient,
- distance to sinusoid, and
- approach distance.

Do not use an LLM call to calculate, classify, estimate, or replace these metrics.

The definitions in `PROJECT_SPEC.md` are authoritative.

---

## Scientific Boundaries

Do not add any of the following unless explicitly requested:

- real biological datasets,
- patient data,
- image segmentation,
- automated object detection,
- automated cell tracking,
- deep-learning models,
- LangGraph,
- LLM-based cell classification,
- autonomous biological interpretation,
- external APIs, or
- unnecessary infrastructure.

Keep the initial implementation intentionally small.

---

## Data Safety

The project uses synthetic data only.

Do not introduce:

- patient identifiers,
- clinical records,
- confidential experimental data,
- credentials,
- API keys, or
- secrets

into the repository.

---

## Project Structure

Preserve the existing high-level structure:

```text
data/
    Synthetic input data

src/
    Python source code

tests/
    Deterministic tests

results/
    Generated analysis outputs

results/figures/
    Generated figures
```

Do not create new top-level directories unless there is a clear reason.

---

## Python Environment

The project uses:

```text
Python 3.12
uv
```

Current scientific dependencies include:

- NumPy,
- Pandas,
- Matplotlib,
- SciPy, and
- tifffile.

Testing uses:

- pytest.

Use `uv` for dependency management.

Do not introduce another environment or dependency-management system.

---

## Implementation Style

Prefer:

- small functions,
- descriptive names,
- explicit parameters,
- type hints where useful,
- deterministic behavior,
- simple control flow,
- reproducible random-number generation,
- separation of simulation and analysis logic, and
- code that can be tested independently.

Avoid unnecessary abstractions and frameworks.

---

## Testing Requirements

Scientific calculations should be independently testable.

When implementing or modifying a metric:

1. identify relevant edge cases,
2. construct small test cases with manually verifiable expected values,
3. implement or update the tests,
4. run the relevant tests, and
5. report the result.

Do not modify a scientific definition merely to make a test pass.

If a test reveals a conflict between the implementation and `PROJECT_SPEC.md`, report the inconsistency.

---

## Change Discipline

Before changing an existing scientific calculation:

1. describe the proposed change,
2. explain why it is needed,
3. identify whether it changes the scientific specification, and
4. obtain researcher approval if the specification would change.

Implementation improvements that preserve the scientific definition may be proposed normally.

Scientific-definition changes require explicit approval.

---

## Git Discipline

Before making substantial changes:

- inspect the existing repository,
- keep modifications focused on the requested task, and
- avoid unrelated cleanup.

After implementation, summarize which files were changed and why.

Do not rewrite Git history, force-push, delete branches, or modify Git configuration unless explicitly requested.

---

## Current Development Stage

The project is currently in the specification stage.

Do not implement the synthetic-data generator or analysis pipeline yet.

For the first Claude Code review:

1. read `CLAUDE.md`,
2. read `PROJECT_SPEC.md`,
3. inspect the repository structure,
4. summarize your understanding of the project,
5. identify unresolved parameters or inconsistencies, and
6. do not modify files.

The first review should focus on understanding the specification rather than proposing implementation details.