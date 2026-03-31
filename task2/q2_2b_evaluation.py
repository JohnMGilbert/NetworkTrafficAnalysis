"""Question 2.2(b): internal validation and comparison of clustering results."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory
from task2.q2_2a_clustering import load_sample
from task2.q2_2a_clustering import stratified_indices_by_router


LOGGER = logging.getLogger("task2.q2_2b")
ALGORITHM_COLUMNS = {
    "MiniBatchKMeans": "minibatchkmeans_cluster",
    "HDBSCAN": "hdbscan_cluster",
    "GaussianMixture": "gaussianmixture_cluster",
}


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
        "--q2-2a-summary-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_2a_summary.json",
        help="Summary JSON produced by Question 2.2(a).",
    )
    parser.add_argument(
        "--assignments-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_2a_cluster_assignments.csv",
        help="Cluster assignments produced by Question 2.2(a).",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables",
        help="Directory for generated Task 2.2(b) tables and reports.",
    )
    return parser.parse_args()


def parse_q2_2a_summary(path: Path) -> tuple[int, dict[str, dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    algorithms = {
        item["algorithm"]: item
        for item in payload["algorithms"]
    }
    return int(payload["fit_rows"]), algorithms


def reconstruct_fit_sample(
    sample_path: Path,
    embedding_path: Path,
    fit_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sample, embedding, feature_columns = load_sample(sample_path, embedding_path)
    fit_indices = stratified_indices_by_router(sample["router_id"], fit_rows, CONFIG.random_seed)
    fit_sample = sample.iloc[fit_indices].reset_index(drop=True)
    fit_embedding = embedding.iloc[fit_indices].reset_index(drop=True)
    return fit_sample, fit_embedding, feature_columns


def validate_alignment(
    fit_sample: pd.DataFrame,
    fit_embedding: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    if len(fit_sample) != len(assignments):
        raise ValueError(
            f"Reconstructed fit sample has {len(fit_sample)} rows, but assignments contain {len(assignments)} rows."
        )
    if not fit_sample["router_id"].astype(str).equals(assignments["router_id"].astype(str)):
        raise ValueError("Router ordering differs between the reconstructed fit sample and the Q2.2(a) assignments.")
    if not np.allclose(fit_embedding["umap_1"].to_numpy(), assignments["umap_1"].to_numpy()):
        raise ValueError("UMAP 1 coordinates differ between the reconstructed fit sample and the Q2.2(a) assignments.")
    if not np.allclose(fit_embedding["umap_2"].to_numpy(), assignments["umap_2"].to_numpy()):
        raise ValueError("UMAP 2 coordinates differ between the reconstructed fit sample and the Q2.2(a) assignments.")


def safe_metric(function, features: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        return float(function(features, labels))
    except ValueError:
        return None


def rank_descending(series: pd.Series) -> pd.Series:
    return series.rank(method="min", ascending=False)


def rank_ascending(series: pd.Series) -> pd.Series:
    return series.rank(method="min", ascending=True)


def build_evaluation_table(
    features: np.ndarray,
    assignments: pd.DataFrame,
    q2_2a_algorithms: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for algorithm, column_name in ALGORITHM_COLUMNS.items():
        labels = assignments[column_name].to_numpy()
        evaluation_mask = np.ones(len(labels), dtype=bool)
        metric_scope = "full_sample"
        excluded_noise_rows = 0

        if np.any(labels == -1):
            evaluation_mask = labels != -1
            metric_scope = "non_noise_only"
            excluded_noise_rows = int(np.sum(labels == -1))

        evaluated_features = features[evaluation_mask]
        evaluated_labels = labels[evaluation_mask]
        unique_labels = np.unique(evaluated_labels)
        if unique_labels.size < 2:
            silhouette = None
            davies_bouldin = None
            calinski_harabasz = None
        else:
            silhouette = safe_metric(silhouette_score, evaluated_features, evaluated_labels)
            davies_bouldin = safe_metric(davies_bouldin_score, evaluated_features, evaluated_labels)
            calinski_harabasz = safe_metric(calinski_harabasz_score, evaluated_features, evaluated_labels)

        algorithm_summary = q2_2a_algorithms[algorithm]
        rows.append(
            {
                "algorithm": algorithm,
                "family": algorithm_summary["family"],
                "clusters_discovered_excluding_noise": int(
                    algorithm_summary["clusters_discovered_excluding_noise"]
                ),
                "noise_points": int(algorithm_summary["noise_points"]),
                "noise_fraction": float(algorithm_summary["noise_fraction"]),
                "evaluated_rows": int(evaluation_mask.sum()),
                "excluded_noise_rows": excluded_noise_rows,
                "metric_scope": metric_scope,
                "silhouette_score": silhouette,
                "davies_bouldin_index": davies_bouldin,
                "calinski_harabasz_index": calinski_harabasz,
            }
        )

    table = pd.DataFrame(rows)
    table["silhouette_rank"] = rank_descending(table["silhouette_score"])
    table["davies_bouldin_rank"] = rank_ascending(table["davies_bouldin_index"])
    table["calinski_harabasz_rank"] = rank_descending(table["calinski_harabasz_index"])
    table["mean_rank"] = table[
        ["silhouette_rank", "davies_bouldin_rank", "calinski_harabasz_rank"]
    ].mean(axis=1)
    table["overall_rank"] = rank_ascending(table["mean_rank"])
    return table.sort_values(["overall_rank", "algorithm"]).reset_index(drop=True)


def algorithm_assessment(table: pd.DataFrame) -> tuple[str, str]:
    best_row = table.sort_values(["overall_rank", "algorithm"]).iloc[0]
    best_algorithm = str(best_row["algorithm"])

    cluster_notes = []
    for row in table.to_dict(orient="records"):
        approx_eleven = abs(int(row["clusters_discovered_excluding_noise"]) - 11)
        cluster_notes.append((approx_eleven, row["algorithm"]))
    closest_to_eleven = sorted(cluster_notes)[0][1]

    summary = (
        f"{best_algorithm} achieved the strongest overall internal validation profile based on the mean rank across "
        "Silhouette, Davies-Bouldin, and Calinski-Harabasz."
    )
    eleven_note = (
        f"{closest_to_eleven} came closest to the assignment's expected 11 traffic classes, based on the discovered "
        "cluster count."
    )
    return summary, eleven_note


def strengths_and_limitations(table: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    for row in table.to_dict(orient="records"):
        algorithm = row["algorithm"]
        if algorithm == "MiniBatchKMeans":
            notes.append(
                "MiniBatchKMeans scales well and is fast on large flow samples, but it forces every point into a cluster "
                "and can miss irregular or low-density attack structure."
            )
        elif algorithm == "HDBSCAN":
            notes.append(
                "HDBSCAN is better at handling noise and imbalanced cluster densities, but it can label many rare or "
                "borderline flows as noise and its scores were computed on the non-noise subset only."
            )
        elif algorithm == "GaussianMixture":
            notes.append(
                "GaussianMixture can model softer decision boundaries and finer substructure, but it is more sensitive to "
                "distributional assumptions and can over-split the traffic into more components than the expected class count."
            )
    return notes


def write_report(output_path: Path, table: pd.DataFrame) -> None:
    best_summary, eleven_note = algorithm_assessment(table)
    lines = [
        "# Task 2.2(b) Internal Validation Report",
        "",
        "## Metric Scope",
        (
            "Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index were computed on the same fitted sample "
            "used in Question 2.2(a). For HDBSCAN, noise points were excluded from metric computation because these "
            "internal validation metrics assume regular cluster assignments rather than an explicit noise label."
        ),
        "",
        "## Comparison Summary",
        f"- {best_summary}",
        f"- {eleven_note}",
        "",
        "## Strengths And Limitations",
    ]
    lines.extend(f"- {note}" for note in strengths_and_limitations(table))
    lines.extend(
        [
            "",
            "## Table Interpretation",
            (
                "Higher Silhouette and Calinski-Harabasz values are better, while lower Davies-Bouldin values are better. "
                "The overall rank averages those three rankings to provide one concise comparison."
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    ensure_directory(args.table_dir)

    fit_rows, q2_2a_algorithms = parse_q2_2a_summary(args.q2_2a_summary_path)
    assignments = pd.read_csv(args.assignments_path)
    fit_sample, fit_embedding, feature_columns = reconstruct_fit_sample(
        sample_path=args.sample_path,
        embedding_path=args.embedding_path,
        fit_rows=fit_rows,
    )
    validate_alignment(fit_sample, fit_embedding, assignments)
    features = fit_sample[feature_columns].to_numpy(dtype=np.float32, copy=False)

    LOGGER.info("Evaluating %s clustering assignments on %s rows.", len(ALGORITHM_COLUMNS), len(features))
    evaluation_table = build_evaluation_table(
        features=features,
        assignments=assignments,
        q2_2a_algorithms=q2_2a_algorithms,
    )

    comparison_path = args.table_dir / "q2_2b_internal_validation.csv"
    evaluation_table.to_csv(comparison_path, index=False)

    summary_payload = {
        "fit_rows": fit_rows,
        "feature_count": len(feature_columns),
        "comparison_table_path": str(comparison_path),
        "best_algorithm": evaluation_table.iloc[0]["algorithm"],
        "metrics": evaluation_table.to_dict(orient="records"),
    }
    (args.table_dir / "q2_2b_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    write_report(args.table_dir / "q2_2b_report.md", evaluation_table)
    LOGGER.info("Task 2.2(b) artifacts written to %s.", args.table_dir)


if __name__ == "__main__":
    main()
