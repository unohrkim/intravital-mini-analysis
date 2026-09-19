# Claude Code Project Instructions

## Purpose

This repository contains a reproducible synthetic intravital cell-motility analysis project.

Claude Code may assist with implementation, testing, debugging, refactoring, documentation, and code review.

Scientific and evaluation decisions remain researcher-defined.

Read `PROJECT_SPEC.md` before modifying scientific analysis logic.


## Scientific Boundaries

Do not independently change:

- the scientific question,
- simulation assumptions,
- metric definitions,
- statistical methods,
- data-split rules,
- model configurations,
- interpretation boundaries.

If a scientific requirement is ambiguous, report the ambiguity rather than choosing a value silently.

This project uses synthetic data. Results must not be described as biological or clinical evidence.


## Reproducibility

Preserve deterministic and reproducible behavior.

Use the existing fixed random seeds, persisted data split, and train-only preprocessing rules defined by the project.

Do not regenerate or reshuffle the final train/validation/test split unless explicitly requested as a new experiment.


## Frozen Final Evaluation

The Logistic Regression, TensorFlow CNN, and PyTorch CNN configurations have already been frozen and evaluated on the held-out test set.

Do not modify architecture, preprocessing, thresholds, epochs, or other model settings in response to final test results.

Any future model change must be treated as a separate experiment.

The final comparison result is stored in:

`results/dl/final_test_comparison.csv`


## Implementation and Testing

Prefer:

- small, testable functions,
- explicit parameters,
- deterministic behavior,
- minimal changes,
- separation of simulation, analysis, and modeling logic.

When changing scientific or modeling logic:

1. explain the proposed change,
2. identify whether it affects the scientific specification,
3. update or add tests,
4. run relevant tests,
5. report changed files and test results.

Do not change a scientific definition merely to make a test pass.


## Git Discipline

Keep changes focused.

Do not rewrite Git history, force-push, delete branches, or modify Git configuration unless explicitly requested.

Preserve the distinction between frozen pre-test code and post-test results.


## Current Project Status

The core analysis and modeling workflow is complete.

Current work should focus primarily on:

- documentation,
- reproducibility,
- repository cleanup,
- portfolio presentation.

For future work, read:

1. `CLAUDE.md`
2. `PROJECT_SPEC.md`

before making substantial changes.
