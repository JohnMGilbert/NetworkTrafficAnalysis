"""Simple environment bootstrap check."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.io import list_router_files
from src.common.logging_utils import configure_logging
from src.common.seed import set_global_seed


def main() -> None:
    configure_logging()
    set_global_seed(CONFIG.random_seed)

    files = list_router_files(CONFIG.raw_data_dir)
    print(f"Project root: {CONFIG.project_root}")
    print(f"Raw data directory: {CONFIG.raw_data_dir}")
    print(f"Detected router files: {len(files)}")


if __name__ == "__main__":
    main()
