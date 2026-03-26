"""I/O helpers for loading FLNET-style router files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def list_router_files(data_dir: Path, patterns: Iterable[str] = ("*.csv", "*.parquet")) -> list[Path]:
    """Return sorted router data files under a directory."""

    files: list[Path] = []
    for pattern in patterns:
        files.extend(data_dir.glob(pattern))
    return sorted(path for path in files if path.is_file())


def load_router_csv(path: Path, dtype_overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Load one router CSV with conservative defaults suitable for large tabular data."""

    return pd.read_csv(path, low_memory=False, dtype=dtype_overrides)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize incoming column names to a consistent snake_case-like form."""

    renamed = {
        column: column.strip().lower().replace(" ", "_").replace("/", "_")
        for column in frame.columns
    }
    return frame.rename(columns=renamed)


def validate_required_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a helpful error when expected columns are missing."""

    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

