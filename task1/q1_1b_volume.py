"""Question 1.1(b): router-level traffic volume distributions."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from array import array
from dataclasses import dataclass, field
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
from task1.q1_1a_summary import group_router_files, iter_router_frames


FWD_BYTES_CANDIDATES = (
    "totlen_fwd_pkts",
    "total_length_of_fwd_packets",
)
BWD_BYTES_CANDIDATES = (
    "totlen_bwd_pkts",
    "total_length_of_bwd_packets",
)
PLOT_SAMPLE_SIZE = 25_000


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
        help="Directory for generated Q1.1(b) tables.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "figures",
        help="Directory for generated Q1.1(b) figures.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="CSV chunk size used to keep memory bounded.",
    )
    return parser.parse_args()


def find_first_available(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


@dataclass
class RouterVolumeAccumulator:
    router_name: str
    volumes: array = field(default_factory=lambda: array("d"))

    def update(self, frame: pd.DataFrame, fwd_col: str, bwd_col: str) -> None:
        fwd = pd.to_numeric(frame[fwd_col], errors="coerce").fillna(0.0)
        bwd = pd.to_numeric(frame[bwd_col], errors="coerce").fillna(0.0)
        flow_volume = (fwd + bwd).astype(float).to_numpy()
        self.volumes.extend(flow_volume)

    def summary_row(self) -> dict[str, object]:
        series = pd.Series(self.volumes, copy=False)
        return {
            "router": self.router_name,
            "flow_records": int(series.shape[0]),
            "volume_min": round(float(series.min()), 6),
            "volume_mean": round(float(series.mean()), 6),
            "volume_median": round(float(series.median()), 6),
            "volume_std": round(float(series.std(ddof=0)), 6),
            "volume_p90": round(float(series.quantile(0.90)), 6),
            "volume_p99": round(float(series.quantile(0.99)), 6),
            "volume_max": round(float(series.max()), 6),
        }


def summarize_router_group(paths: list[Path], chunksize: int) -> RouterVolumeAccumulator:
    router_name = f"D{int(paths[0].stem.rsplit('-', 1)[-1])}"
    accumulator = RouterVolumeAccumulator(router_name=router_name)
    fwd_col = None
    bwd_col = None

    for path in sorted(paths):
        for frame in iter_router_frames(path, chunksize):
            if fwd_col is None:
                fwd_col = find_first_available(frame.columns, FWD_BYTES_CANDIDATES)
                bwd_col = find_first_available(frame.columns, BWD_BYTES_CANDIDATES)
                if fwd_col is None or bwd_col is None:
                    raise ValueError(
                        f"Could not find forward/backward byte columns in {path.name}. "
                        f"Expected one of {FWD_BYTES_CANDIDATES} and {BWD_BYTES_CANDIDATES}."
                    )
            accumulator.update(frame, fwd_col, bwd_col)

    return accumulator


def build_discussion(summary: pd.DataFrame) -> str:
    highest_median = summary.loc[summary["volume_median"].idxmax()]
    highest_tail = summary.loc[summary["volume_p99"].idxmax()]
    lowest_median = summary.loc[summary["volume_median"].idxmin()]

    lines = [
        "# Q1.1(b) Discussion",
        "",
        (
            f"{highest_median['router']} has the highest median per-flow volume "
            f"({highest_median['volume_median']} bytes), which points to consistently heavier flows rather than just rare spikes."
        ),
        (
            f"{highest_tail['router']} has the heaviest extreme tail with the largest 99th percentile "
            f"({highest_tail['volume_p99']} bytes), suggesting occasional very large transfers or concentrated bulk traffic."
        ),
        (
            f"{lowest_median['router']} has the lowest median per-flow volume "
            f"({lowest_median['volume_median']} bytes), indicating that it sees predominantly small flows and likely lighter-weight application traffic."
        ),
        (
            "Routers with high medians and high upper quantiles are stronger candidates for aggregation or transit roles, "
            "while routers with low medians but long right tails may be edge-like vantage points that occasionally observe bulk transfers."
        ),
    ]
    return "\n\n".join(lines) + "\n"


def sample_plot_frame(accumulators: list[RouterVolumeAccumulator]) -> pd.DataFrame:
    rng = np.random.default_rng(CONFIG.random_seed)
    frames = []
    for accumulator in accumulators:
        values = np.frombuffer(accumulator.volumes, dtype=np.float64)
        if values.shape[0] > PLOT_SAMPLE_SIZE:
            index = rng.choice(values.shape[0], size=PLOT_SAMPLE_SIZE, replace=False)
            values = values[index]
        frames.append(
            pd.DataFrame(
                {
                    "router": accumulator.router_name,
                    "flow_volume_bytes": values,
                    "log10_flow_volume": np.log10(values + 1.0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_plot(plot_frame: pd.DataFrame, figure_dir: Path) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Task 1.1(b): Per-Flow Traffic Volume by Router", fontsize=20, y=1.02)

    sns.histplot(
        data=plot_frame,
        x="log10_flow_volume",
        hue="router",
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        bins=45,
        ax=axes[0],
    )
    axes[0].set_title("Overlay Histogram of log10(Flow Volume + 1)")
    axes[0].set_xlabel("log10(Bytes per Flow + 1)")
    axes[0].set_ylabel("Density")

    sns.violinplot(
        data=plot_frame,
        x="router",
        y="log10_flow_volume",
        hue="router",
        dodge=False,
        palette="viridis",
        legend=False,
        inner="quartile",
        cut=0,
        ax=axes[1],
    )
    axes[1].set_title("Violin Plot of log10(Flow Volume + 1)")
    axes[1].set_xlabel("Router")
    axes[1].set_ylabel("log10(Bytes per Flow + 1)")

    fig.tight_layout()
    output_path = figure_dir / "q1_1b_volume_distributions.png"
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
    ordered_groups = [paths for _, paths in sorted(grouped.items(), key=lambda item: int(item[0][1:]))]
    accumulators = [summarize_router_group(paths, args.chunksize) for paths in ordered_groups]

    table_dir = ensure_directory(args.table_dir)
    figure_dir = ensure_directory(args.figure_dir)

    summary = pd.DataFrame([acc.summary_row() for acc in accumulators])
    summary = summary.sort_values("router", key=lambda s: s.str[1:].astype(int)).reset_index(drop=True)

    summary_path = table_dir / "q1_1b_volume_summary.csv"
    discussion_path = table_dir / "q1_1b_discussion.md"
    summary.to_csv(summary_path, index=False)
    discussion_path.write_text(build_discussion(summary), encoding="utf-8")

    plot_frame = sample_plot_frame(accumulators)
    figure_path = make_plot(plot_frame, figure_dir)

    print(f"Wrote summary table to {summary_path}")
    print(f"Wrote discussion notes to {discussion_path}")
    print(f"Wrote figure to {figure_path}")


if __name__ == "__main__":
    main()
