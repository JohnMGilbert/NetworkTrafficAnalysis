"""Question 2.2(a): unsupervised clustering experiments on the Q2.1(b) sample."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
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
from hdbscan import HDBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory
from src.common.seed import set_global_seed


LOGGER = logging.getLogger("task2.q2_2a")
EMBEDDING_COLUMNS = ("umap_1", "umap_2")


@dataclass(frozen=True)
class ClusteringResult:
    algorithm: str
    family: str
    labels: np.ndarray
    selected_hyperparameters: dict[str, object]
    selection_notes: str
    selection_score_name: str
    selection_score_value: float | None
    runtime_seconds: float
    cluster_count_excluding_noise: int
    noise_points: int
    noise_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=CONFIG.interim_data_dir / "task2_q2_1b_umap_sample.parquet",
        help="Parquet sample produced by Question 2.1(b).",
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_1b_umap_embedding.csv",
        help="UMAP coordinates produced by Question 2.1(b).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "figures",
        help="Directory for generated Task 2.2(a) figures.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables",
        help="Directory for generated Task 2.2(a) tables and reports.",
    )
    parser.add_argument(
        "--fit-size",
        type=int,
        default=20_000,
        help="Stratified subset size used for the final clustering fits.",
    )
    parser.add_argument(
        "--selection-size",
        type=int,
        default=8_000,
        help="Random subset size used for hyperparameter selection metrics.",
    )
    parser.add_argument(
        "--plot-size",
        type=int,
        default=10_000,
        help="Random subset size used for the cluster visualization figure.",
    )
    return parser.parse_args()


def load_sample(sample_path: Path, embedding_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sample = pd.read_parquet(sample_path).reset_index(drop=True)
    embedding = pd.read_csv(embedding_path).reset_index(drop=True)

    if len(sample) != len(embedding):
        raise ValueError(
            f"Sample row count ({len(sample)}) does not match embedding row count ({len(embedding)})."
        )
    if not sample["router_id"].astype(str).equals(embedding["router_id"].astype(str)):
        raise ValueError("Router ordering differs between the sample and the saved embedding.")

    feature_columns = [column for column in sample.columns if column != "router_id"]
    return sample, embedding, feature_columns


def build_random_subset(total_rows: int, subset_size: int, seed: int) -> np.ndarray:
    if subset_size >= total_rows:
        return np.arange(total_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_rows, size=subset_size, replace=False))


def proportional_targets(counts: pd.Series, subset_size: int) -> dict[str, int]:
    total = int(counts.sum())
    if subset_size >= total:
        return {str(label): int(count) for label, count in counts.items()}

    raw = {str(label): count * subset_size / total for label, count in counts.items()}
    targets = {label: min(int(counts[label]), int(math.floor(value))) for label, value in raw.items()}
    remainder = subset_size - sum(targets.values())
    ranked = sorted(
        raw,
        key=lambda label: (raw[label] - targets[label], counts[label]),
        reverse=True,
    )
    for label in ranked:
        if remainder <= 0:
            break
        if targets[label] < int(counts[label]):
            targets[label] += 1
            remainder -= 1
    return targets


def stratified_indices_by_router(router_ids: pd.Series, subset_size: int, seed: int) -> np.ndarray:
    if subset_size >= len(router_ids):
        return np.arange(len(router_ids))

    counts = router_ids.astype(str).value_counts().sort_index()
    targets = proportional_targets(counts, subset_size)
    rng = np.random.default_rng(seed)
    chosen_parts: list[np.ndarray] = []

    for router_id, target in targets.items():
        group_indices = np.flatnonzero(router_ids.to_numpy(dtype=str) == router_id)
        if target >= len(group_indices):
            chosen_parts.append(group_indices)
            continue
        chosen_parts.append(np.sort(rng.choice(group_indices, size=target, replace=False)))

    return np.sort(np.concatenate(chosen_parts))


def safe_silhouette_score(features: np.ndarray, labels: np.ndarray) -> float | None:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return None
    try:
        return float(silhouette_score(features, labels, metric="euclidean"))
    except ValueError:
        return None


def summarize_labels(labels: np.ndarray) -> tuple[int, int, float]:
    unique_labels = sorted(set(int(value) for value in labels))
    noise_points = int(np.sum(labels == -1))
    cluster_labels = [label for label in unique_labels if label != -1]
    noise_fraction = noise_points / len(labels) if len(labels) else 0.0
    return len(cluster_labels), noise_points, noise_fraction


def select_kmeans(
    full_features: np.ndarray,
    subset_features: np.ndarray,
    random_seed: int,
) -> tuple[ClusteringResult, pd.DataFrame]:
    LOGGER.info("Selecting MiniBatchKMeans hyperparameters on %s rows.", len(subset_features))
    candidate_rows: list[dict[str, object]] = []
    best_score = -math.inf
    best_k = None

    for n_clusters in range(8, 15):
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=4096,
            max_iter=200,
            n_init=10,
            random_state=random_seed,
        )
        labels = model.fit_predict(subset_features)
        silhouette = safe_silhouette_score(subset_features, labels)
        candidate_rows.append(
            {
                "algorithm": "MiniBatchKMeans",
                "family": "partitional",
                "n_clusters": n_clusters,
                "batch_size": 4096,
                "max_iter": 200,
                "n_init": 10,
                "selection_metric": "silhouette",
                "selection_metric_value": silhouette,
                "inertia": float(model.inertia_),
            }
        )
        if silhouette is not None and silhouette > best_score:
            best_score = silhouette
            best_k = n_clusters

    if best_k is None:
        raise RuntimeError("KMeans hyperparameter selection failed to produce a valid candidate.")

    LOGGER.info("Fitting MiniBatchKMeans with k=%s on %s rows.", best_k, len(full_features))
    start = time.perf_counter()
    final_model = MiniBatchKMeans(
        n_clusters=best_k,
        batch_size=4096,
        max_iter=200,
        n_init=10,
        random_state=random_seed,
    )
    final_labels = final_model.fit_predict(full_features)
    runtime_seconds = time.perf_counter() - start
    cluster_count, noise_points, noise_fraction = summarize_labels(final_labels)

    result = ClusteringResult(
        algorithm="MiniBatchKMeans",
        family="partitional",
        labels=final_labels,
        selected_hyperparameters={
            "n_clusters": best_k,
            "batch_size": 4096,
            "max_iter": 200,
            "n_init": 10,
        },
        selection_notes="Selected the cluster count with the highest silhouette score over k=8..14 on a random subset.",
        selection_score_name="silhouette",
        selection_score_value=best_score,
        runtime_seconds=runtime_seconds,
        cluster_count_excluding_noise=cluster_count,
        noise_points=noise_points,
        noise_fraction=noise_fraction,
    )
    return result, pd.DataFrame(candidate_rows)


def hdbscan_selection_score(features: np.ndarray, labels: np.ndarray) -> float | None:
    non_noise_mask = labels != -1
    if non_noise_mask.sum() < 2:
        return None
    filtered_labels = labels[non_noise_mask]
    if np.unique(filtered_labels).size < 2:
        return None
    silhouette = safe_silhouette_score(features[non_noise_mask], filtered_labels)
    if silhouette is None:
        return None
    coverage = float(non_noise_mask.mean())
    return silhouette * coverage


def select_hdbscan(
    full_features: np.ndarray,
    subset_features: np.ndarray,
) -> tuple[ClusteringResult, pd.DataFrame]:
    LOGGER.info("Selecting HDBSCAN hyperparameters on %s rows.", len(subset_features))
    candidate_rows: list[dict[str, object]] = []
    best_score = -math.inf
    best_params: dict[str, int] | None = None

    for min_cluster_size in (250, 500, 1000):
        for min_samples in (10, 25):
            model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method="eom",
            )
            labels = model.fit_predict(subset_features)
            score = hdbscan_selection_score(subset_features, labels)
            cluster_count, noise_points, noise_fraction = summarize_labels(labels)
            candidate_rows.append(
                {
                    "algorithm": "HDBSCAN",
                    "family": "density_based",
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "cluster_selection_method": "eom",
                    "selection_metric": "silhouette_x_coverage",
                    "selection_metric_value": score,
                    "cluster_count_excluding_noise": cluster_count,
                    "noise_fraction": noise_fraction,
                    "noise_points": noise_points,
                }
            )
            if score is not None and score > best_score:
                best_score = score
                best_params = {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                }

    if best_params is None:
        raise RuntimeError("HDBSCAN hyperparameter selection failed to produce a valid candidate.")

    LOGGER.info(
        "Fitting HDBSCAN with min_cluster_size=%s and min_samples=%s on %s rows.",
        best_params["min_cluster_size"],
        best_params["min_samples"],
        len(full_features),
    )
    start = time.perf_counter()
    final_model = HDBSCAN(
        min_cluster_size=best_params["min_cluster_size"],
        min_samples=best_params["min_samples"],
        cluster_selection_method="eom",
    )
    final_labels = final_model.fit_predict(full_features)
    runtime_seconds = time.perf_counter() - start
    cluster_count, noise_points, noise_fraction = summarize_labels(final_labels)

    result = ClusteringResult(
        algorithm="HDBSCAN",
        family="density_based",
        labels=final_labels,
        selected_hyperparameters={
            **best_params,
            "cluster_selection_method": "eom",
        },
        selection_notes=(
            "Selected the min_cluster_size/min_samples pair with the highest silhouette score multiplied by "
            "non-noise coverage on the random subset."
        ),
        selection_score_name="silhouette_x_coverage",
        selection_score_value=best_score,
        runtime_seconds=runtime_seconds,
        cluster_count_excluding_noise=cluster_count,
        noise_points=noise_points,
        noise_fraction=noise_fraction,
    )
    return result, pd.DataFrame(candidate_rows)


def select_gmm(
    full_features: np.ndarray,
    subset_features: np.ndarray,
    random_seed: int,
) -> tuple[ClusteringResult, pd.DataFrame]:
    LOGGER.info("Selecting GaussianMixture hyperparameters on %s rows.", len(subset_features))
    candidate_rows: list[dict[str, object]] = []
    best_bic = math.inf
    best_n_components = None
    subset_features = subset_features.astype(np.float64, copy=False)
    full_features = full_features.astype(np.float64, copy=False)

    for n_components in range(8, 15):
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            reg_covar=1e-4,
            max_iter=200,
            init_params="kmeans",
            random_state=random_seed,
        )
        try:
            model.fit(subset_features)
            labels = model.predict(subset_features)
            bic = float(model.bic(subset_features))
            silhouette = safe_silhouette_score(subset_features, labels)
        except ValueError:
            bic = None
            silhouette = None
        candidate_rows.append(
            {
                "algorithm": "GaussianMixture",
                "family": "model_based",
                "n_components": n_components,
                "covariance_type": "diag",
                "max_iter": 200,
                "reg_covar": 1e-4,
                "init_params": "kmeans",
                "selection_metric": "bic",
                "selection_metric_value": bic,
                "silhouette_reference": silhouette,
            }
        )
        if bic is not None and bic < best_bic:
            best_bic = bic
            best_n_components = n_components

    if best_n_components is None:
        raise RuntimeError("Gaussian Mixture hyperparameter selection failed to produce a valid candidate.")

    LOGGER.info("Fitting GaussianMixture with %s components on %s rows.", best_n_components, len(full_features))
    start = time.perf_counter()
    final_model = GaussianMixture(
        n_components=best_n_components,
        covariance_type="diag",
        reg_covar=1e-4,
        max_iter=200,
        init_params="kmeans",
        random_state=random_seed,
    )
    final_model.fit(full_features)
    final_labels = final_model.predict(full_features)
    runtime_seconds = time.perf_counter() - start
    cluster_count, noise_points, noise_fraction = summarize_labels(final_labels)

    result = ClusteringResult(
        algorithm="GaussianMixture",
        family="model_based",
        labels=final_labels,
        selected_hyperparameters={
            "n_components": best_n_components,
            "covariance_type": "diag",
            "reg_covar": 1e-4,
            "max_iter": 200,
            "init_params": "kmeans",
        },
        selection_notes="Selected the component count with the lowest Bayesian Information Criterion over k=8..14.",
        selection_score_name="bic",
        selection_score_value=best_bic,
        runtime_seconds=runtime_seconds,
        cluster_count_excluding_noise=cluster_count,
        noise_points=noise_points,
        noise_fraction=noise_fraction,
    )
    return result, pd.DataFrame(candidate_rows)


def build_assignments_table(
    sample: pd.DataFrame,
    embedding: pd.DataFrame,
    results: list[ClusteringResult],
) -> pd.DataFrame:
    assignments = pd.DataFrame(
        {
            "router_id": sample["router_id"].astype(str),
            "umap_1": embedding["umap_1"],
            "umap_2": embedding["umap_2"],
        }
    )
    for result in results:
        assignments[f"{result.algorithm.lower()}_cluster"] = result.labels
    return assignments


def build_summary_table(results: list[ClusteringResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        rows.append(
            {
                "algorithm": result.algorithm,
                "family": result.family,
                "selected_hyperparameters": json.dumps(result.selected_hyperparameters, sort_keys=True),
                "selection_metric": result.selection_score_name,
                "selection_metric_value": result.selection_score_value,
                "clusters_discovered_excluding_noise": result.cluster_count_excluding_noise,
                "noise_points": result.noise_points,
                "noise_fraction": round(result.noise_fraction, 6),
                "runtime_seconds": round(result.runtime_seconds, 3),
                "selection_notes": result.selection_notes,
            }
        )
    return pd.DataFrame(rows)


def cluster_size_table(results: list[ClusteringResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        counts = pd.Series(result.labels).value_counts().sort_index()
        for cluster_label, cluster_size in counts.items():
            rows.append(
                {
                    "algorithm": result.algorithm,
                    "cluster_label": int(cluster_label),
                    "cluster_size": int(cluster_size),
                    "is_noise": int(cluster_label) == -1,
                }
            )
    return pd.DataFrame(rows)


def color_map_for_labels(labels: np.ndarray) -> dict[int, tuple[float, float, float, float]]:
    unique_labels = sorted(set(int(value) for value in labels if int(value) != -1))
    palette = plt.get_cmap("tab20")
    mapping = {
        label: palette(index % palette.N)
        for index, label in enumerate(unique_labels)
    }
    mapping[-1] = (0.55, 0.55, 0.55, 0.55)
    return mapping


def figure_slug(algorithm: str) -> str:
    return algorithm.strip().lower().replace(" ", "_")


def save_cluster_figures(assignments: pd.DataFrame, results: list[ClusteringResult], output_dir: Path) -> list[str]:
    output_names: list[str] = []
    figure_data = assignments

    for result in results:
        plt.figure(figsize=(8, 6))
        label_column = f"{result.algorithm.lower()}_cluster"
        colors = color_map_for_labels(figure_data[label_column].to_numpy())
        color_values = [colors[int(label)] for label in figure_data[label_column]]
        plt.scatter(
            figure_data["umap_1"],
            figure_data["umap_2"],
            c=color_values,
            s=6,
            alpha=0.72,
            linewidths=0,
        )
        plt.title(
            f"Task 2.2(a): {result.algorithm} Clusters on the Q2.1(b) UMAP Embedding\n"
            f"clusters={result.cluster_count_excluding_noise}"
            + (f", noise={result.noise_fraction:.1%}" if result.noise_points else "")
        )
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.grid(alpha=0.15)
        plt.tight_layout()

        output_name = f"q2_2a_{figure_slug(result.algorithm)}_clusters.png"
        plt.savefig(output_dir / output_name, dpi=220, bbox_inches="tight")
        plt.close()
        output_names.append(output_name)

    return output_names


def write_report(
    output_path: Path,
    sample_size: int,
    fit_size: int,
    selection_size: int,
    summary_table: pd.DataFrame,
    figure_names: list[str],
) -> None:
    lines = [
        "# Task 2.2(a) Clustering Report",
        "",
        "## Sampling Strategy",
        (
            f"The clustering experiments reuse the stratified {sample_size:,}-row sample created in Question 2.1(b), "
            "which preserves the router distribution while keeping computation tractable on the full preprocessed "
            f"dataset. The final clustering models are fitted on a stratified {fit_size:,}-row subset of "
            f"that sample, and hyperparameter selection uses a random {selection_size:,}-row subset of the fit set so "
            "that silhouette- and BIC-based comparisons remain practical."
        ),
        "",
        "## Algorithm Coverage",
        "- MiniBatchKMeans as the required partitional method.",
        "- HDBSCAN as the required density-based method.",
        "- GaussianMixture as the required hierarchical/model-based family representative.",
        "",
        "## Selected Configurations",
    ]

    for row in summary_table.to_dict(orient="records"):
        params = row["selected_hyperparameters"]
        lines.append(
            f"- {row['algorithm']}: {params}; discovered {row['clusters_discovered_excluding_noise']} clusters"
            + (
                f" with {row['noise_fraction']:.1%} noise."
                if row["noise_points"]
                else "."
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            (
                "These outputs satisfy Question 2.2(a) by documenting the hyperparameters, showing the number of "
                "clusters discovered by each algorithm, and plotting each algorithm's cluster assignments in its own "
                "standalone graph on the saved Question 2.1(b) UMAP embedding."
            ),
            "",
            "Generated figures:",
        ]
    )
    lines.extend([f"- {name}" for name in figure_names])
    lines.extend(
        [
            "",
            f"Question 2.1(b) sample available: {sample_size:,} rows.",
            f"Rows clustered for Question 2.2(a): {fit_size:,} rows.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_global_seed(CONFIG.random_seed)

    ensure_directory(args.figure_dir)
    ensure_directory(args.table_dir)

    sample, embedding, feature_columns = load_sample(args.sample_path, args.embedding_path)
    fit_indices = stratified_indices_by_router(sample["router_id"], args.fit_size, CONFIG.random_seed)
    fit_sample = sample.iloc[fit_indices].reset_index(drop=True)
    fit_embedding = embedding.iloc[fit_indices].reset_index(drop=True)
    features = fit_sample[feature_columns].to_numpy(dtype=np.float32, copy=False)

    selection_indices = build_random_subset(len(fit_sample), args.selection_size, CONFIG.random_seed)
    plot_indices = build_random_subset(len(fit_sample), args.plot_size, CONFIG.random_seed + 1)
    selection_features = features[selection_indices]

    LOGGER.info(
        "Loaded %s sample rows with %s numeric features; fitting on a stratified subset of %s rows.",
        len(sample),
        len(feature_columns),
        len(fit_sample),
    )
    LOGGER.info("Using %s rows for hyperparameter selection.", len(selection_indices))

    results: list[ClusteringResult] = []
    search_tables: list[pd.DataFrame] = []

    kmeans_result, kmeans_search = select_kmeans(features, selection_features, CONFIG.random_seed)
    results.append(kmeans_result)
    search_tables.append(kmeans_search)

    hdbscan_result, hdbscan_search = select_hdbscan(features, selection_features)
    results.append(hdbscan_result)
    search_tables.append(hdbscan_search)

    gmm_result, gmm_search = select_gmm(features, selection_features, CONFIG.random_seed)
    results.append(gmm_result)
    search_tables.append(gmm_search)

    assignments = build_assignments_table(fit_sample, fit_embedding, results)
    summary_table = build_summary_table(results)
    sizes_table = cluster_size_table(results)
    search_table = pd.concat(search_tables, ignore_index=True)

    figure_assignments = assignments.iloc[plot_indices].reset_index(drop=True)

    assignments.to_csv(args.table_dir / "q2_2a_cluster_assignments.csv", index=False)
    summary_table.to_csv(args.table_dir / "q2_2a_clustering_summary.csv", index=False)
    sizes_table.to_csv(args.table_dir / "q2_2a_cluster_sizes.csv", index=False)
    search_table.to_csv(args.table_dir / "q2_2a_hyperparameter_search.csv", index=False)

    summary_payload = {
        "sample_path": str(args.sample_path),
        "embedding_path": str(args.embedding_path),
        "sample_rows": len(sample),
        "fit_rows": len(fit_sample),
        "feature_count": len(feature_columns),
        "selection_rows": len(selection_indices),
        "plot_rows": len(plot_indices),
        "algorithms": [
            {
                "algorithm": result.algorithm,
                "family": result.family,
                "selected_hyperparameters": result.selected_hyperparameters,
                "selection_metric": result.selection_score_name,
                "selection_metric_value": result.selection_score_value,
                "clusters_discovered_excluding_noise": result.cluster_count_excluding_noise,
                "noise_points": result.noise_points,
                "noise_fraction": result.noise_fraction,
                "runtime_seconds": result.runtime_seconds,
            }
            for result in results
        ],
    }
    (args.table_dir / "q2_2a_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    figure_names = save_cluster_figures(
        assignments=figure_assignments,
        results=results,
        output_dir=args.figure_dir,
    )
    write_report(
        output_path=args.table_dir / "q2_2a_report.md",
        sample_size=len(sample),
        fit_size=len(fit_sample),
        selection_size=len(selection_indices),
        summary_table=summary_table,
        figure_names=figure_names,
    )

    LOGGER.info("Task 2.2(a) artifacts written to %s and %s.", args.table_dir, args.figure_dir)


if __name__ == "__main__":
    main()
