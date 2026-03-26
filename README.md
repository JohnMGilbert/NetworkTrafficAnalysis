# Network Traffic Analysis

Scaffold for CSCI 5840 Homework 2 using the FLNET2023 dataset.

## Structure

- `task1/`: topology reconstruction work and task-specific notes.
- `task2/`: unsupervised discovery work and task-specific notes.
- `task3/`: supervised modeling work and task-specific notes.
- `src/common/`: shared Python utilities used across all tasks.
- `data/raw/`: original router CSV files from the assignment.
- `data/interim/`: cached samples, intermediate aggregates, and transformed data.
- `data/processed/`: final task-ready datasets.
- `outputs/`: generated figures, tables, and trained models.
- `docs/`: report planning notes and reference material.
- `scripts/`: entry points for repeatable workflows.

## Getting Started

1. Create a Python 3.11 environment.
2. Install dependencies from `requirements.txt`.
3. Place the FLNET2023 router files in `data/raw/`.
4. Build task-specific work on separate branches from this scaffold.

## Reproducibility Conventions

- Use a shared random seed from `src/common/config.py`.
- Keep generated artifacts out of git; write them under `outputs/` or `data/interim/`.
- Put task-specific code in the corresponding task directory, but move reusable logic into `src/common/`.

