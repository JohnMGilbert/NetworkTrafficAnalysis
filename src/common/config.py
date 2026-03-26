"""Shared project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RANDOM_SEED = 5840


@dataclass(frozen=True)
class ProjectConfig:
    """Small container for shared runtime configuration."""

    project_root: Path = PROJECT_ROOT
    random_seed: int = DEFAULT_RANDOM_SEED
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    interim_data_dir: Path = PROJECT_ROOT / "data" / "interim"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    outputs_dir: Path = PROJECT_ROOT / "outputs"


CONFIG = ProjectConfig()

