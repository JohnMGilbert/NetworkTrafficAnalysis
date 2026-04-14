"""Question 1.2(a): router-to-router similarity matrix and heatmap."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory
from task1.q1_1a_summary import group_router_files, infer_router_name, iter_router_frames


EXCLUDED_COLUMNS = {"src_ip", "dst_ip", "timestamp"}


@dataclass
class RouterProfileAccumulator:
    feature_names: list[str]

    def __post_init__(self) -> None:
        width = len(self.feature_names)
        self.sums = np.zeros(width, dtype=np.float64)
        self.counts = np.zeros(width, dtype=np.int64)

    def update(self, frame: pd.DataFrame) -> None:
        for idx, feature in enumerate(self.feature_names):
            values = pd.to_numeric(frame[feature], errors="coerce")
            valid = values.notna()
            if not valid.any():
                continue
            self.sums[idx] += float(values[valid].sum())
            self.counts[idx] += int(valid.sum())

    def means(self) -> np.ndarray:
        result = np.full(len(self.feature_names), np.nan, dtype=np.float64)
        nonzero = self.counts > 0
        result[nonzero] = self.sums[nonzero] / self.counts[nonzero]
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=CONFIG.raw_data_dir,
        help="Directory containing per-router CSV or parquet files.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "tables",
        help="Directory for generated Q1.2(a) tables.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "figures",
        help="Directory for generated Q1.2(a) figures.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="CSV chunk size used to keep memory bounded.",
    )
    return parser.parse_args()


def discover_feature_columns(data_dir: Path) -> list[str]:
    sample_file = next(iter(sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.parquet"))), None)
    if sample_file is None:
        raise FileNotFoundError(f"No router files were found under {data_dir}.")

    for frame in iter_router_frames(sample_file, chunksize=50_000):
        feature_columns = []
        for column in frame.columns:
            if column in EXCLUDED_COLUMNS:
                continue
            if pd.to_numeric(frame[column], errors="coerce").notna().any():
                feature_columns.append(column)
        return feature_columns

    raise ValueError(f"Unable to read sample rows from {sample_file}.")


def build_router_profiles(grouped_files: dict[str, list[Path]], feature_names: list[str], chunksize: int) -> pd.DataFrame:
    rows = []
    for router, paths in sorted(grouped_files.items(), key=lambda item: int(item[0][1:])):
        accumulator = RouterProfileAccumulator(feature_names)
        for path in sorted(paths):
            for frame in iter_router_frames(path, chunksize):
                accumulator.update(frame)
        row = {"router": router}
        row.update(dict(zip(feature_names, accumulator.means(), strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    scaled = frame.copy()
    for column in scaled.columns:
        series = scaled[column].astype(float)
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        if std == 0 or np.isnan(std):
            scaled[column] = 0.0
        else:
            scaled[column] = (series - mean) / std
    return scaled


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    return normalized @ normalized.T


def build_methodology(feature_count: int) -> str:
    return "\n".join(
        [
            "# Q1.2(a) Methodology",
            "",
            "Chosen metric: cosine similarity on z-scored per-router mean feature vectors.",
            "",
            "Why this metric:",
            "- Each router is represented as a high-dimensional traffic profile across numeric CICFlowMeter features.",
            "- Cosine similarity compares profile shape rather than raw magnitude, which is useful because router traffic volume varies substantially.",
            "- Z-scoring each feature across routers prevents large-scale features from dominating the similarity matrix.",
            "- Per-router means are scalable to compute exactly over the full dataset with chunked aggregation.",
            "- The heatmap uses a diverging scale from -1 to 1 because cosine similarity can be negative after z-scoring.",
            "",
            "Aggregation method:",
            f"- Used {feature_count} numeric CICFlowMeter features after excluding `src_ip`, `dst_ip`, and `timestamp`.",
            "- For each router, aggregated each feature by the mean over all flows observed at that router.",
            "- Standardized features across routers before computing pairwise cosine similarity.",
        ]
    ) + "\n"


def make_heatmap(similarity: pd.DataFrame, figure_dir: Path) -> Path:
    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        similarity,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Cosine Similarity"},
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title("Task 1.2(a): Router-to-Router Similarity Heatmap")
    ax.set_xlabel("Router")
    ax.set_ylabel("Router")
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()

    output_path = figure_dir / "q1_2a_router_similarity_heatmap.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    configure_logging()

    router_files = [
        path for path in sorted(args.data_dir.glob("*.csv")) + sorted(args.data_dir.glob("*.parquet"))
        if path.is_file()
    ]
    if not router_files:
        raise FileNotFoundError(f"No router files were found under {args.data_dir}.")

    grouped = group_router_files(router_files)
    feature_names = discover_feature_columns(args.data_dir)
    profiles = build_router_profiles(grouped, feature_names, args.chunksize)

    router_labels = profiles["router"].tolist()
    feature_frame = profiles.drop(columns=["router"])
    feature_frame = feature_frame.fillna(feature_frame.mean())
    scaled = zscore_frame(feature_frame)

    similarity_values = cosine_similarity_matrix(scaled.to_numpy(dtype=np.float64))
    similarity = pd.DataFrame(similarity_values, index=router_labels, columns=router_labels)

    table_dir = ensure_directory(args.table_dir)
    figure_dir = ensure_directory(args.figure_dir)

    profiles_path = table_dir / "q1_2a_router_feature_profiles.csv"
    similarity_path = table_dir / "q1_2a_router_similarity_matrix.csv"
    methodology_path = table_dir / "q1_2a_methodology.md"

    profiles.to_csv(profiles_path, index=False)
    similarity.to_csv(similarity_path, index=True)
    methodology_path.write_text(build_methodology(len(feature_names)), encoding="utf-8")
    figure_path = make_heatmap(similarity, figure_dir)

    print(f"Wrote router profiles to {profiles_path}")
    print(f"Wrote similarity matrix to {similarity_path}")
    print(f"Wrote methodology notes to {methodology_path}")
    print(f"Wrote heatmap to {figure_path}")


if __name__ == "__main__":
    main()
