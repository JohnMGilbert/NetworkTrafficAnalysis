"""Path helpers for common project directories."""

from __future__ import annotations

from pathlib import Path

from .config import CONFIG


def ensure_directory(path: Path) -> Path:
    """Create a directory when needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def task_output_dir(task_name: str, artifact_type: str) -> Path:
    """Return and create a standard output subdirectory for a task."""

    return ensure_directory(CONFIG.outputs_dir / task_name / artifact_type)

