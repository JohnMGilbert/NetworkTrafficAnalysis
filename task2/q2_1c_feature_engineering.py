"""Question 2.1(c): engineered attack-sensitive flow features."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.io import list_router_files, normalize_columns
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory
from src.common.seed import set_global_seed


LOGGER = logging.getLogger("task2.q2_1c")
EPSILON = 1.0
ROUTER_SUFFIX_PATTERN = re.compile(r"(?:^|[-_])D?(\d+)$", flags=re.IGNORECASE)
ROUTER_LABEL_PATTERN = re.compile(r"^D(\d+)$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class EngineeredFeatureSpec:
    name: str
    intuition: str
    computation: str


FEATURE_SPECS = [
    EngineeredFeatureSpec(
        name="directional_byte_imbalance",
        intuition="Flooding and scanning traffic is often highly one-sided, so strong byte asymmetry can reveal attack flows that do not receive proportionate responses.",
        computation="(totlen_fwd_pkts - totlen_bwd_pkts) / (totlen_fwd_pkts + totlen_bwd_pkts + 1)",
    ),
    EngineeredFeatureSpec(
        name="bytes_per_packet",
        intuition="Attack campaigns often generate many small packets or a distinctive payload profile, so average bytes per packet helps separate bursty control traffic from bulk transfers.",
        computation="(totlen_fwd_pkts + totlen_bwd_pkts) / (tot_fwd_pkts + tot_bwd_pkts + 1)",
    ),
    EngineeredFeatureSpec(
        name="burst_idle_log_ratio",
        intuition="Denial-of-service and scripted attacks tend to send traffic in concentrated bursts with limited idle time, unlike steadier benign application flows.",
        computation="log1p(max(active_mean, 0)) - log1p(max(idle_mean, 0))",
    ),
    EngineeredFeatureSpec(
        name="packet_size_asymmetry",
        intuition="Request-heavy or response-heavy attack behaviors can produce sharply different forward versus backward mean packet sizes.",
        computation="abs(fwd_pkt_len_mean - bwd_pkt_len_mean) / (fwd_pkt_len_mean + bwd_pkt_len_mean + 1)",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=CONFIG.raw_data_dir,
        help="Directory containing the original router CSV/parquet files.",
    )
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=CONFIG.processed_data_dir / "task2_preprocessed_standardized.parquet",
        help="Processed parquet from Question 2.1(a), used for deduplicated router counts.",
    )
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=CONFIG.interim_data_dir / "task2_q2_1c_engineered_sample.parquet",
        help="Output parquet for the engineered-feature sample.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "figures",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables",
        help="Directory for generated tables and reports.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Chunk size for raw CSV processing.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Exact stratified sample size for visualization and preview clustering.",
    )
    parser.add_argument(
        "--preview-clusters",
        type=int,
        default=11,
        help="Preview MiniBatchKMeans cluster count used for 2.1(c) distribution plots.",
    )
    return parser.parse_args()


def parquet_batches(path: Path, columns: list[str], batch_size: int):
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def count_router_rows(path: Path, batch_size: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for batch in parquet_batches(path, ["router_id"], batch_size):
        counts.update(batch["router_id"].astype(str).tolist())
    return counts


def target_sample_counts(router_counts: Counter[str], sample_size: int) -> dict[str, int]:
    total_rows = sum(router_counts.values())
    if sample_size >= total_rows:
        return dict(router_counts)

    proportional = {
        router_id: (count * sample_size) / total_rows
        for router_id, count in router_counts.items()
    }
    targets = {
        router_id: min(count, int(math.floor(proportional[router_id])))
        for router_id, count in router_counts.items()
    }
    remainder = sample_size - sum(targets.values())
    ranked = sorted(
        router_counts,
        key=lambda router_id: (proportional[router_id] - targets[router_id], router_counts[router_id]),
        reverse=True,
    )
    for router_id in ranked:
        if remainder <= 0:
            break
        if targets[router_id] < router_counts[router_id]:
            targets[router_id] += 1
            remainder -= 1
    return targets


def router_sort_key(router_id: object) -> tuple[int, int | str]:
    text = str(router_id)
    match = ROUTER_LABEL_PATTERN.match(text)
    if match:
        return (0, int(match.group(1)))
    return (1, text)


def iter_router_frames(path: Path, chunksize: int):
    if path.suffix.lower() == ".parquet":
        yield normalize_columns(pd.read_parquet(path))
        return

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        yield normalize_columns(chunk)


def duplicate_mask_from_hashes(hashes: np.ndarray, seen_hashes: set[int]) -> np.ndarray:
    duplicate_mask = np.zeros(len(hashes), dtype=bool)
    local_hashes: set[int] = set()
    for index, value in enumerate(hashes):
        hashed_value = int(value)
        if hashed_value in seen_hashes or hashed_value in local_hashes:
            duplicate_mask[index] = True
        else:
            local_hashes.add(hashed_value)
    seen_hashes.update(local_hashes)
    return duplicate_mask


def infer_router_id(path: Path) -> str:
    match = ROUTER_SUFFIX_PATTERN.search(path.stem)
    if match:
        return f"D{int(match.group(1))}"
    return path.stem


def compute_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    totlen_fwd = pd.to_numeric(frame["totlen_fwd_pkts"], errors="coerce").fillna(0.0)
    totlen_bwd = pd.to_numeric(frame["totlen_bwd_pkts"], errors="coerce").fillna(0.0)
    tot_fwd = pd.to_numeric(frame["tot_fwd_pkts"], errors="coerce").fillna(0.0)
    tot_bwd = pd.to_numeric(frame["tot_bwd_pkts"], errors="coerce").fillna(0.0)
    active_mean = pd.to_numeric(frame["active_mean"], errors="coerce").fillna(0.0).clip(lower=0.0)
    idle_mean = pd.to_numeric(frame["idle_mean"], errors="coerce").fillna(0.0).clip(lower=0.0)
    fwd_pkt_len_mean = pd.to_numeric(frame["fwd_pkt_len_mean"], errors="coerce").fillna(0.0).clip(lower=0.0)
    bwd_pkt_len_mean = pd.to_numeric(frame["bwd_pkt_len_mean"], errors="coerce").fillna(0.0).clip(lower=0.0)

    total_bytes = totlen_fwd + totlen_bwd
    total_packets = tot_fwd + tot_bwd

    engineered = pd.DataFrame(
        {
            "router_id": frame["router_id"].astype(str),
            "directional_byte_imbalance": (totlen_fwd - totlen_bwd) / (total_bytes + EPSILON),
            "bytes_per_packet": total_bytes / (total_packets + EPSILON),
            "burst_idle_log_ratio": np.log1p(active_mean) - np.log1p(idle_mean),
            "packet_size_asymmetry": np.abs(fwd_pkt_len_mean - bwd_pkt_len_mean) / (
                fwd_pkt_len_mean + bwd_pkt_len_mean + EPSILON
            ),
        }
    )
    return engineered.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def draw_router_batch_sample(
    group: pd.DataFrame,
    router_id: str,
    remaining_needed: dict[str, int],
    remaining_available: dict[str, int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    needed = remaining_needed[router_id]
    available = remaining_available[router_id]
    group_size = len(group)
    if needed <= 0:
        remaining_available[router_id] -= group_size
        return group.iloc[0:0]

    if group_size >= available:
        take_n = needed
    else:
        take_n = int(round(group_size * needed / available))
        take_n = min(max(take_n, 0), group_size, needed)

    remaining_available[router_id] -= group_size

    if take_n == 0:
        return group.iloc[0:0]
    if take_n == group_size:
        remaining_needed[router_id] -= take_n
        return group

    chosen = rng.choice(group.index.to_numpy(), size=take_n, replace=False)
    sampled = group.loc[np.sort(chosen)]
    remaining_needed[router_id] -= len(sampled)
    return sampled


def build_engineered_feature_sample(
    raw_data_dir: Path,
    processed_path: Path,
    chunksize: int,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    router_counts = count_router_rows(processed_path, batch_size=chunksize)
    targets = target_sample_counts(router_counts, sample_size)
    remaining_needed = dict(targets)
    remaining_available = dict(router_counts)
    seen_hashes: set[int] = set()
    rng = np.random.default_rng(seed)
    sampled_parts: list[pd.DataFrame] = []
    unique_rows = 0
    duplicate_rows = 0

    for path in list_router_files(raw_data_dir):
        router_id = infer_router_id(path)
        LOGGER.info("Engineering features from %s", path.name)
        for frame in iter_router_frames(path, chunksize):
            frame = frame.copy()
            frame.insert(0, "router_id", router_id)
            hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64, copy=False)
            duplicate_mask = duplicate_mask_from_hashes(hashes, seen_hashes)
            duplicate_rows += int(duplicate_mask.sum())

            unique_frame = frame.loc[~duplicate_mask].reset_index(drop=True)
            if unique_frame.empty:
                continue
            unique_rows += len(unique_frame)
            engineered = compute_engineered_features(unique_frame)
            sampled = draw_router_batch_sample(
                group=engineered,
                router_id=router_id,
                remaining_needed=remaining_needed,
                remaining_available=remaining_available,
                rng=rng,
            )
            if not sampled.empty:
                sampled_parts.append(sampled)

    sample = pd.concat(sampled_parts, ignore_index=True)
    allocation = pd.DataFrame(
        {
            "router_id": list(router_counts.keys()),
            "population_rows": [router_counts[router_id] for router_id in router_counts],
            "sample_rows": [targets[router_id] for router_id in router_counts],
        }
    ).sort_values("router_id", key=lambda series: series.map(router_sort_key), ignore_index=True)
    counts = {
        "unique_rows_seen": unique_rows,
        "duplicate_rows_skipped": duplicate_rows,
        "sample_rows": int(len(sample)),
    }
    return sample, allocation, counts


def summarize_engineered_features(sample: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for spec in FEATURE_SPECS:
        series = sample[spec.name]
        summaries.append(
            {
                "feature": spec.name,
                "mean": round(float(series.mean()), 6),
                "std": round(float(series.std(ddof=0)), 6),
                "p05": round(float(series.quantile(0.05)), 6),
                "median": round(float(series.quantile(0.50)), 6),
                "p95": round(float(series.quantile(0.95)), 6),
            }
        )
    return pd.DataFrame(summaries)


def preview_cluster_engineered_features(sample: pd.DataFrame, n_clusters: int, seed: int) -> pd.DataFrame:
    feature_columns = [spec.name for spec in FEATURE_SPECS]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sample[feature_columns].to_numpy(dtype=np.float32, copy=False))
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=4096,
        n_init=10,
    )
    labels = model.fit_predict(scaled)
    clustered = sample.copy()
    clustered["preview_cluster"] = labels
    return clustered


def save_cluster_distribution_plot(sample: pd.DataFrame, figure_path: Path) -> None:
    ordered_clusters = sorted(sample["preview_cluster"].unique().tolist())
    cluster_labels = [str(cluster_id) for cluster_id in ordered_clusters]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    palette = sns.color_palette("tab20", n_colors=len(ordered_clusters))
    for axis, spec in zip(axes, FEATURE_SPECS):
        subset = sample[["preview_cluster", spec.name]].copy()
        subset["preview_cluster_label"] = subset["preview_cluster"].astype(str)
        sns.boxplot(
            data=subset,
            x="preview_cluster_label",
            y=spec.name,
            hue="preview_cluster_label",
            ax=axis,
            order=cluster_labels,
            hue_order=cluster_labels,
            palette=palette,
            fliersize=1,
            linewidth=0.7,
            legend=False,
        )
        axis.set_title(spec.name.replace("_", " ").title())
        axis.set_xlabel("Preview Cluster")
        axis.set_ylabel("Value")
    fig.suptitle("Task 2.1(c): Engineered Feature Distributions Across Preview Clusters", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def cluster_summary_table(sample: pd.DataFrame) -> pd.DataFrame:
    grouped = sample.groupby("preview_cluster")[[spec.name for spec in FEATURE_SPECS]]
    means = grouped.mean().add_suffix("_mean")
    sizes = sample.groupby("preview_cluster").size().rename("cluster_size")
    summary = pd.concat([sizes, means], axis=1).reset_index().sort_values("preview_cluster")
    return summary


def feature_definition_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [spec.name for spec in FEATURE_SPECS],
            "networking_intuition": [spec.intuition for spec in FEATURE_SPECS],
            "computed_from_existing_features": [spec.computation for spec in FEATURE_SPECS],
        }
    )


def write_report(
    destination: Path,
    sample_counts: dict[str, int],
    preview_clusters: int,
) -> None:
    lines = [
        "# Task 2.1(c) Engineered Features",
        "",
        "## Engineered features",
    ]
    for spec in FEATURE_SPECS:
        lines.extend(
            [
                f"### {spec.name}",
                f"- Intuition: {spec.intuition}",
                f"- Computation: `{spec.computation}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Sampling and visualization",
            f"- Engineered features were computed from raw deduplicated flows after normalizing filename shards to `D1` through `D10`, skipping {sample_counts['duplicate_rows_skipped']:,} duplicates detected in Question 2.1(a).",
            f"- A stratified sample of {sample_counts['sample_rows']:,} flows was drawn for visualization, preserving the deduplicated router distribution.",
            f"- A provisional MiniBatchKMeans preview with k={preview_clusters} is kept here as an early visualization of engineered-feature behavior.",
            "- The final distribution-across-discovered-clusters artifact is produced by Question 2.2(c) after HDBSCAN cluster assignments are available.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_global_seed(CONFIG.random_seed)

    ensure_directory(args.sample_path.parent)
    ensure_directory(args.figure_dir)
    ensure_directory(args.table_dir)

    sample, allocation, sample_counts = build_engineered_feature_sample(
        raw_data_dir=args.raw_data_dir,
        processed_path=args.processed_path,
        chunksize=args.chunksize,
        sample_size=args.sample_size,
        seed=CONFIG.random_seed,
    )
    clustered = preview_cluster_engineered_features(
        sample=sample,
        n_clusters=args.preview_clusters,
        seed=CONFIG.random_seed,
    )
    clustered.to_parquet(args.sample_path, index=False)

    feature_definition_table().to_csv(args.table_dir / "q2_1c_engineered_feature_definitions.csv", index=False)
    summarize_engineered_features(clustered).to_csv(args.table_dir / "q2_1c_engineered_feature_summary.csv", index=False)
    allocation.to_csv(args.table_dir / "q2_1c_sample_allocation.csv", index=False)
    cluster_summary_table(clustered).to_csv(args.table_dir / "q2_1c_preview_cluster_feature_means.csv", index=False)
    save_cluster_distribution_plot(clustered, args.figure_dir / "q2_1c_engineered_feature_distributions.png")

    summary_payload = {
        **sample_counts,
        "preview_clusters": args.preview_clusters,
        "engineered_features": [spec.name for spec in FEATURE_SPECS],
        "sample_path": str(args.sample_path),
    }
    (args.table_dir / "q2_1c_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(
        destination=args.table_dir / "q2_1c_report.md",
        sample_counts=sample_counts,
        preview_clusters=args.preview_clusters,
    )

    LOGGER.info("Wrote engineered feature sample to %s", args.sample_path)
    LOGGER.info("Generated %s engineered features", len(FEATURE_SPECS))


if __name__ == "__main__":
    main()
