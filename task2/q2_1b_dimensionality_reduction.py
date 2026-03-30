"""Question 2.1(b): PCA variance analysis and nonlinear 2D embedding."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
import umap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory
from src.common.seed import set_global_seed


LOGGER = logging.getLogger("task2.q2_1b")
METADATA_COLUMNS = ("router_id", "src_ip", "dst_ip", "timestamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=CONFIG.processed_data_dir / "task2_preprocessed_standardized.parquet",
        help="Preprocessed parquet produced by Question 2.1(a).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "figures",
        help="Directory for generated Task 2.1(b) figures.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables",
        help="Directory for generated Task 2.1(b) tables and reports.",
    )
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=CONFIG.interim_data_dir / "task2_q2_1b_umap_sample.parquet",
        help="Cached stratified sample used for the nonlinear embedding.",
    )
    parser.add_argument(
        "--pca-batch-size",
        type=int,
        default=100_000,
        help="Batch size for streaming IncrementalPCA fitting.",
    )
    parser.add_argument(
        "--umap-sample-size",
        type=int,
        default=100_000,
        help="Stratified sample size for the UMAP embedding.",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors hyperparameter.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist hyperparameter.",
    )
    return parser.parse_args()


def parquet_batches(path: Path, columns: list[str] | None, batch_size: int):
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def discover_feature_columns(path: Path) -> list[str]:
    schema = pq.ParquetFile(path).schema_arrow
    return [name for name in schema.names if name not in METADATA_COLUMNS]


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
        router_id: min(count, int(math.floor(value)))
        for router_id, count in router_counts.items()
        for value in [proportional[router_id]]
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


def stratified_sample(
    path: Path,
    columns: list[str],
    router_counts: Counter[str],
    sample_size: int,
    batch_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = target_sample_counts(router_counts, sample_size)
    remaining_needed = dict(targets)
    remaining_available = dict(router_counts)
    rng = np.random.default_rng(seed)

    sampled_parts: list[pd.DataFrame] = []
    for batch in parquet_batches(path, columns, batch_size):
        grouped = batch.groupby("router_id", sort=False, observed=True)
        for router_id, group in grouped:
            sampled = draw_router_batch_sample(
                group=group,
                router_id=str(router_id),
                remaining_needed=remaining_needed,
                remaining_available=remaining_available,
                rng=rng,
            )
            if not sampled.empty:
                sampled_parts.append(sampled)

    sample = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame(columns=columns)
    allocation = pd.DataFrame(
        {
            "router_id": list(router_counts.keys()),
            "population_rows": [router_counts[router_id] for router_id in router_counts],
            "sample_rows": [targets[router_id] for router_id in router_counts],
        }
    ).sort_values("router_id", ignore_index=True)
    return sample, allocation


def fit_incremental_pca(path: Path, feature_columns: list[str], batch_size: int) -> IncrementalPCA:
    n_components = len(feature_columns)
    ipca = IncrementalPCA(n_components=n_components)
    for batch in parquet_batches(path, feature_columns, batch_size):
        ipca.partial_fit(batch.to_numpy(dtype=np.float32, copy=False))
    return ipca


def save_variance_curve(variance_table: pd.DataFrame, figure_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        variance_table["component"],
        variance_table["cumulative_explained_variance"],
        linewidth=2,
        color="#176087",
    )
    plt.axhline(0.95, color="#c04b32", linestyle="--", linewidth=1.5, label="95% variance")
    cutoff = variance_table.loc[variance_table["cumulative_explained_variance"] >= 0.95].iloc[0]
    plt.axvline(cutoff["component"], color="#c04b32", linestyle=":", linewidth=1.5)
    plt.scatter([cutoff["component"]], [cutoff["cumulative_explained_variance"]], color="#c04b32", zorder=3)
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Task 2.1(b): PCA Cumulative Explained Variance")
    plt.ylim(0, 1.01)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def color_map_for_categories(categories: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap_a = plt.get_cmap("tab20")
    cmap_b = plt.get_cmap("tab20b")
    cmap_c = plt.get_cmap("tab20c")
    palette = [cmap_a(i) for i in range(cmap_a.N)] + [cmap_b(i) for i in range(cmap_b.N)] + [cmap_c(i) for i in range(cmap_c.N)]
    return {
        category: palette[index % len(palette)]
        for index, category in enumerate(categories)
    }


def save_umap_plot(embedding: pd.DataFrame, figure_path: Path) -> None:
    routers = sorted(embedding["router_id"].astype(str).unique().tolist())
    color_map = color_map_for_categories(routers)

    plt.figure(figsize=(12, 8))
    for router_id in routers:
        subset = embedding.loc[embedding["router_id"] == router_id]
        plt.scatter(
            subset["umap_1"],
            subset["umap_2"],
            s=4,
            alpha=0.5,
            label=router_id,
            color=color_map[router_id],
            linewidths=0,
        )
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("Task 2.1(b): UMAP Embedding Colored by router_id")
    plt.grid(alpha=0.15)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        markerscale=2.5,
        ncol=2,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close()


def nearest_neighbor_router_purity(embedding: pd.DataFrame, neighbors: int = 15) -> dict[str, float]:
    coords = embedding[["umap_1", "umap_2"]].to_numpy(dtype=np.float32, copy=False)
    labels = embedding["router_id"].astype(str).to_numpy()
    k = min(neighbors + 1, len(embedding))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(coords)
    indices = nn.kneighbors(return_distance=False)
    neighbor_labels = labels[indices[:, 1:]]
    same_router_fraction = float((neighbor_labels == labels[:, None]).mean())
    largest_router_fraction = float(pd.Series(labels).value_counts(normalize=True).iloc[0])
    return {
        "same_router_neighbor_fraction": same_router_fraction,
        "largest_router_baseline_fraction": largest_router_fraction,
    }


def write_report(
    destination: Path,
    total_rows: int,
    sample_rows: int,
    pca_components_95: int,
    purity_stats: dict[str, float],
    router_count: int,
    umap_neighbors: int,
    umap_min_dist: float,
) -> None:
    same_router = purity_stats["same_router_neighbor_fraction"]
    baseline = purity_stats["largest_router_baseline_fraction"]
    if same_router >= baseline * 3:
        interpretation = (
            "The UMAP space shows meaningful local grouping by router_id, but not perfectly separated islands; "
            "there is structure worth clustering, although substantial overlap remains across routers."
        )
    elif same_router > baseline:
        interpretation = (
            "The embedding shows only weak-to-moderate router-driven grouping, suggesting partial structure but no clean natural partition by router_id alone."
        )
    else:
        interpretation = (
            "The embedding does not separate strongly by router_id, indicating that any natural clusters are likely driven by traffic behavior more than collection point."
        )

    lines = [
        "# Task 2.1(b) Dimensionality Reduction Summary",
        "",
        "## PCA",
        f"- Streaming IncrementalPCA was fit on all {total_rows:,} preprocessed rows.",
        f"- {pca_components_95} principal components were required to retain at least 95% of the variance.",
        "",
        "## Nonlinear embedding",
        f"- UMAP was run on a stratified sample of {sample_rows:,} rows preserving the observed router_id distribution across {router_count} routers.",
        f"- UMAP hyperparameters: n_neighbors={umap_neighbors}, min_dist={umap_min_dist}.",
        f"- Same-router {15}-nearest-neighbor purity in the UMAP space: {same_router:.4f}.",
        f"- Largest-router baseline share in the sampled data: {baseline:.4f}.",
        "",
        "## Interpretation",
        interpretation,
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_global_seed(CONFIG.random_seed)

    ensure_directory(args.figure_dir)
    ensure_directory(args.table_dir)
    ensure_directory(args.sample_path.parent)

    feature_columns = discover_feature_columns(args.input_path)
    router_counts = count_router_rows(args.input_path, batch_size=args.pca_batch_size)
    total_rows = sum(router_counts.values())

    LOGGER.info("Fitting IncrementalPCA on %s rows and %s features", total_rows, len(feature_columns))
    ipca = fit_incremental_pca(args.input_path, feature_columns, batch_size=args.pca_batch_size)
    explained = ipca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    pca_components_95 = int(np.argmax(cumulative >= 0.95) + 1)

    variance_table = pd.DataFrame(
        {
            "component": np.arange(1, len(feature_columns) + 1),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": cumulative,
        }
    )
    variance_table.to_csv(args.table_dir / "q2_1b_pca_explained_variance.csv", index=False)
    save_variance_curve(variance_table, args.figure_dir / "q2_1b_pca_cumulative_variance.png")

    sample_columns = ["router_id"] + feature_columns
    LOGGER.info("Drawing stratified UMAP sample of %s rows", args.umap_sample_size)
    sample, sample_allocation = stratified_sample(
        path=args.input_path,
        columns=sample_columns,
        router_counts=router_counts,
        sample_size=args.umap_sample_size,
        batch_size=args.pca_batch_size,
        seed=CONFIG.random_seed,
    )
    sample.to_parquet(args.sample_path, index=False)
    sample_allocation.to_csv(args.table_dir / "q2_1b_umap_sample_allocation.csv", index=False)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=CONFIG.random_seed,
        low_memory=True,
    )
    LOGGER.info("Running UMAP on sampled data")
    embedding_array = reducer.fit_transform(sample[feature_columns].to_numpy(dtype=np.float32, copy=False))
    embedding = pd.DataFrame(
        {
            "router_id": sample["router_id"].astype(str).to_numpy(),
            "umap_1": embedding_array[:, 0],
            "umap_2": embedding_array[:, 1],
        }
    )
    embedding.to_csv(args.table_dir / "q2_1b_umap_embedding.csv", index=False)
    save_umap_plot(embedding, args.figure_dir / "q2_1b_umap_router_embedding.png")

    purity_stats = nearest_neighbor_router_purity(embedding, neighbors=15)
    summary_payload = {
        "total_rows_used_for_pca": total_rows,
        "feature_count": len(feature_columns),
        "pca_components_for_95_variance": pca_components_95,
        "umap_sample_rows": int(len(sample)),
        "router_count": int(len(router_counts)),
        "umap_neighbors": args.umap_neighbors,
        "umap_min_dist": args.umap_min_dist,
        **purity_stats,
    }
    (args.table_dir / "q2_1b_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(
        destination=args.table_dir / "q2_1b_report.md",
        total_rows=total_rows,
        sample_rows=int(len(sample)),
        pca_components_95=pca_components_95,
        purity_stats=purity_stats,
        router_count=int(len(router_counts)),
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )

    LOGGER.info("PCA 95%% variance cutoff: %s components", pca_components_95)
    LOGGER.info("UMAP sample size: %s", len(sample))


if __name__ == "__main__":
    main()
