"""Question 1.1(c): rank features by between-router vs within-router variance."""

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
from task1.q1_1a_summary import group_router_files, iter_router_frames


EXCLUDED_COLUMNS = {"src_ip", "dst_ip", "timestamp"}


@dataclass
class FeatureStats:
    count: int = 0
    mean_value: float = 0.0
    m2: float = 0.0

    def update(self, values: pd.Series) -> None:
        clean = pd.to_numeric(values, errors="coerce").dropna()
        if clean.empty:
            return

        chunk_count = int(clean.shape[0])
        chunk_mean = float(clean.mean())
        chunk_var = float(clean.var(ddof=0))
        chunk_m2 = chunk_var * chunk_count

        if self.count == 0:
            self.count = chunk_count
            self.mean_value = chunk_mean
            self.m2 = chunk_m2
            return

        combined_count = self.count + chunk_count
        delta = chunk_mean - self.mean_value
        self.m2 = (
            self.m2
            + chunk_m2
            + (delta * delta) * self.count * chunk_count / combined_count
        )
        self.mean_value += delta * chunk_count / combined_count
        self.count = combined_count

    @property
    def variance(self) -> float | None:
        if self.count == 0:
            return None
        return max(self.m2 / self.count, 0.0)


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
        help="Directory for generated Q1.1(c) tables.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "figures",
        help="Directory for generated Q1.1(c) figures.",
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
        columns = [
            column for column in frame.columns
            if column not in EXCLUDED_COLUMNS
        ]
        numeric_columns = [
            column
            for column in columns
            if pd.to_numeric(frame[column], errors="coerce").notna().any()
        ]
        return numeric_columns

    raise ValueError(f"Unable to read rows from {sample_file}.")


def summarize_router_group(paths: list[Path], feature_columns: list[str], chunksize: int) -> dict[str, FeatureStats]:
    stats = {feature: FeatureStats() for feature in feature_columns}
    for path in sorted(paths):
        for frame in iter_router_frames(path, chunksize):
            for feature in feature_columns:
                stats[feature].update(frame[feature])
    return stats


def compute_anova_rows(router_feature_stats: dict[str, dict[str, FeatureStats]]) -> pd.DataFrame:
    feature_rows = []
    features = list(next(iter(router_feature_stats.values())).keys())

    for feature in features:
        router_rows = []
        total_count = 0
        weighted_mean_sum = 0.0
        ss_within = 0.0

        for router, feature_stats in router_feature_stats.items():
            stats = feature_stats[feature]
            if stats.count == 0:
                continue
            variance = stats.variance or 0.0
            router_rows.append((router, stats.count, stats.mean_value, variance, stats.m2))
            total_count += stats.count
            weighted_mean_sum += stats.count * stats.mean_value
            ss_within += stats.m2

        if len(router_rows) < 2 or total_count == 0:
            continue

        grand_mean = weighted_mean_sum / total_count
        ss_between = sum(count * (mean - grand_mean) ** 2 for _, count, mean, _, _ in router_rows)
        df_between = len(router_rows) - 1
        df_within = total_count - len(router_rows)
        ms_between = ss_between / df_between if df_between > 0 else np.nan
        ms_within = ss_within / df_within if df_within > 0 else np.nan
        f_ratio = np.inf if ms_within == 0 and ms_between > 0 else (ms_between / ms_within if ms_within > 0 else np.nan)

        feature_rows.append(
            {
                "feature": feature,
                "router_count": len(router_rows),
                "total_observations": total_count,
                "grand_mean": grand_mean,
                "between_router_ms": ms_between,
                "within_router_ms": ms_within,
                "f_ratio": f_ratio,
            }
        )

    frame = pd.DataFrame(feature_rows)
    frame = frame.sort_values(["f_ratio", "between_router_ms"], ascending=[False, False]).reset_index(drop=True)
    return frame


def explain_feature(feature: str) -> str:
    if "port" in feature:
        return "Port-related behavior; strong separation suggests routers observe different service mixes or application roles."
    if "protocol" in feature:
        return "Protocol composition; variation indicates routers are exposed to different traffic types or encapsulations."
    if "duration" in feature:
        return "Flow longevity; separation points to differences in session persistence across router vantage points."
    if "pkts_s" in feature or "byts_s" in feature:
        return "Traffic rate intensity; high router separation suggests distinct throughput or burst patterns by location."
    if "totlen" in feature or "pkt_len" in feature or "seg_size" in feature:
        return "Packet-size structure; variation reflects different application payload sizes or transport behavior."
    if "iat" in feature:
        return "Inter-arrival timing; separation indicates different pacing, congestion, or traffic burstiness across routers."
    if "tcp" in feature or "flag" in feature:
        return "Transport-control behavior; differences often reflect different connection states or application interaction styles."
    if "idle" in feature or "active" in feature:
        return "Flow activity cadence; separation suggests different transaction rhythms or long-idle behaviors by vantage point."
    if "window" in feature or "bulk" in feature:
        return "Transport buffering or transfer pattern behavior; variation may mark transit-heavy versus edge-like observations."
    return "This feature separates routers through distinct traffic composition or flow-structure patterns."


def build_discussion(top10: pd.DataFrame, excluded: set[str]) -> str:
    lines = [
        "# Q1.1(c) Discussion",
        "",
        (
            "The ranking below uses a one-way ANOVA-style F-ratio: mean square variation between routers divided by "
            "mean square variation within routers. Larger values indicate features whose typical values change more "
            "across routers than they fluctuate inside any single router."
        ),
        (
            f"Excluded columns: {', '.join(sorted(excluded))}. These fields are identifiers or timestamps rather than stable numeric flow features."
        ),
        "",
        "## Top 10 Feature Interpretations",
        "",
    ]

    for _, row in top10.iterrows():
        lines.append(
            f"- `{row['feature']}`: {explain_feature(row['feature'])}"
        )

    return "\n".join(lines) + "\n"


def make_plot(top10: pd.DataFrame, figure_dir: Path) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_frame = top10.sort_values("f_ratio", ascending=True)
    sns.barplot(
        data=plot_frame,
        x="f_ratio",
        y="feature",
        hue="feature",
        dodge=False,
        palette="rocket",
        legend=False,
        ax=ax,
    )
    ax.set_title("Task 1.1(c): Top 10 Features by Between/Within Router F-Ratio")
    ax.set_xlabel("ANOVA-Style F-Ratio")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    output_path = figure_dir / "q1_1c_top10_feature_variance.png"
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

    feature_columns = discover_feature_columns(args.data_dir)
    grouped = group_router_files(router_files)
    ordered_items = sorted(grouped.items(), key=lambda item: int(item[0][1:]))

    router_feature_stats = {
        router: summarize_router_group(paths, feature_columns, args.chunksize)
        for router, paths in ordered_items
    }

    ranking = compute_anova_rows(router_feature_stats)
    top10 = ranking.head(10).copy()
    top10["interpretation"] = top10["feature"].map(explain_feature)

    table_dir = ensure_directory(args.table_dir)
    figure_dir = ensure_directory(args.figure_dir)

    full_path = table_dir / "q1_1c_feature_variance_ranking.csv"
    top10_path = table_dir / "q1_1c_top10_features.csv"
    discussion_path = table_dir / "q1_1c_discussion.md"

    ranking.to_csv(full_path, index=False)
    top10.to_csv(top10_path, index=False)
    discussion_path.write_text(build_discussion(top10, EXCLUDED_COLUMNS), encoding="utf-8")
    figure_path = make_plot(top10, figure_dir)

    print(f"Wrote full ranking to {full_path}")
    print(f"Wrote top-10 table to {top10_path}")
    print(f"Wrote discussion notes to {discussion_path}")
    print(f"Wrote figure to {figure_path}")


if __name__ == "__main__":
    main()
