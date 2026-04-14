"""Question 2.3(b): analyze how class imbalance affects clustering behavior."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.paths import ensure_directory


ALGORITHM_INTERPRETATIONS = {
    "HDBSCAN": {
        "rare_behavior": "noise_or_outliers",
    },
    "GaussianMixture": {
        "rare_behavior": "fragments_and_microclusters",
    },
    "MiniBatchKMeans": {
        "rare_behavior": "absorbed_into_larger_clusters",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q2-2a-summary-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_2a_summary.json",
        help="Summary JSON produced by Question 2.2(a).",
    )
    parser.add_argument(
        "--cluster-sizes-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_2a_cluster_sizes.csv",
        help="Cluster size table produced by Question 2.2(a).",
    )
    parser.add_argument(
        "--q2-2b-summary-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_2b_summary.json",
        help="Summary JSON produced by Question 2.2(b).",
    )
    parser.add_argument(
        "--q2-2c-summary-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_2c_hdbscan_summary.json",
        help="Summary JSON produced by Question 2.2(c).",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables",
        help="Directory for generated Task 2.3(b) outputs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cluster_concentration(cluster_sizes: pd.Series) -> float:
    shares = cluster_sizes / cluster_sizes.sum()
    return float((shares**2).sum())


def format_microcluster_sizes(sizes: pd.Series, *, threshold: int = 100) -> str:
    microcluster_sizes = sorted(int(size) for size in sizes[sizes <= threshold].tolist())
    if not microcluster_sizes:
        return f"no clusters of size <= {threshold}"
    if len(microcluster_sizes) <= 6:
        return "sizes " + ", ".join(f"{size:,}" for size in microcluster_sizes)
    shown = ", ".join(f"{size:,}" for size in microcluster_sizes[:6])
    return f"sizes {shown}, and {len(microcluster_sizes) - 6} more"


def build_interpretation(
    algorithm: str,
    *,
    sizes: pd.Series,
    non_noise: pd.Series,
    noise_rows: int,
    total_rows: int,
    largest_cluster_fraction: float,
    microcluster_count: int,
) -> str:
    if algorithm == "HDBSCAN":
        return (
            f"HDBSCAN discovered {len(non_noise)} dense non-noise clusters and assigned "
            f"{noise_rows:,} of {total_rows:,} fitted rows to the noise bucket "
            f"({noise_rows / total_rows:.2%}). This is the clearest sign that minority or ambiguous "
            "behaviors are more often rejected as outliers than given their own stable cluster."
        )
    if algorithm == "GaussianMixture":
        return (
            "GaussianMixture forced every point into a component and produced "
            f"{microcluster_count} microclusters ({format_microcluster_sizes(sizes)}). "
            "That pattern is consistent with rare classes being split into small fragments instead of "
            "forming one clean minority cluster."
        )
    if algorithm == "MiniBatchKMeans":
        return (
            f"MiniBatchKMeans also forced every point into a cluster, but with only {len(sizes)} partitions "
            f"and a largest-cluster share of {largest_cluster_fraction:.2%}, it mainly created a few "
            "medium-to-large groups. That behavior suggests minority classes are more likely to be absorbed "
            "into the nearest majority cluster than isolated cleanly."
        )
    raise KeyError(f"Unsupported clustering algorithm for imbalance interpretation: {algorithm}")


def build_algorithm_table(cluster_sizes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for algorithm, subset in cluster_sizes.groupby("algorithm", sort=False):
        sizes = subset["cluster_size"].astype(int).sort_values(ascending=False).reset_index(drop=True)
        non_noise = subset.loc[~subset["is_noise"].astype(bool), "cluster_size"].astype(int)
        noise_rows = int(subset.loc[subset["is_noise"].astype(bool), "cluster_size"].sum())
        total_rows = int(sizes.sum())
        interpretation = ALGORITHM_INTERPRETATIONS[algorithm]
        largest_cluster_fraction = float(sizes.iloc[0] / total_rows)
        microcluster_count = int((sizes <= 100).sum())

        rows.append(
            {
                "algorithm": algorithm,
                "total_rows": total_rows,
                "cluster_count": int(len(sizes)),
                "non_noise_cluster_count": int(len(non_noise)),
                "noise_rows": noise_rows,
                "noise_fraction": round(noise_rows / total_rows, 4),
                "largest_cluster_size": int(sizes.iloc[0]),
                "largest_cluster_fraction": round(largest_cluster_fraction, 4),
                "smallest_cluster_size": int(sizes.iloc[-1]),
                "microcluster_count_leq_100": microcluster_count,
                "microcluster_fraction_leq_100": round(float(sizes[sizes <= 100].sum() / total_rows), 4),
                "cluster_concentration_hhi": round(cluster_concentration(sizes), 4),
                "rare_class_tendency": interpretation["rare_behavior"],
                "interpretation": build_interpretation(
                    algorithm,
                    sizes=sizes,
                    non_noise=non_noise,
                    noise_rows=noise_rows,
                    total_rows=total_rows,
                    largest_cluster_fraction=largest_cluster_fraction,
                    microcluster_count=microcluster_count,
                ),
            }
        )

    return pd.DataFrame(rows)


def render_report(
    q2_2a_summary: dict,
    q2_2b_summary: dict,
    q2_2c_summary: dict,
    algorithm_table: pd.DataFrame,
) -> str:
    fit_rows = int(q2_2a_summary["fit_rows"])
    hdbscan_row = algorithm_table.loc[algorithm_table["algorithm"] == "HDBSCAN"].iloc[0]
    gmm_row = algorithm_table.loc[algorithm_table["algorithm"] == "GaussianMixture"].iloc[0]
    kmeans_row = algorithm_table.loc[algorithm_table["algorithm"] == "MiniBatchKMeans"].iloc[0]
    known_clusters = [cluster for cluster in q2_2c_summary["clusters"] if cluster["predicted_label"] != "Unknown"]
    known_cluster_fraction = sum(float(cluster["cluster_fraction"]) for cluster in known_clusters)
    unknown_fraction = 1.0 - known_cluster_fraction

    lines = [
        "# Task 2.3(b) Class Imbalance And Clustering Performance",
        "",
        "## Direct Answer",
        f"- The imbalance strongly shapes cluster formation: the largest classes dominate the geometry of the embedding, so the best-performing unsupervised method (`{q2_2b_summary['best_algorithm']}`) recovers only a few broad dense groups instead of 11 clean classes.",
        "- Majority classes create stable high-density regions because they contribute enough repeated examples to define the local neighborhood structure.",
        "- Minority classes do not have enough support to anchor their own regions, so they are either pulled into a nearby majority cluster, broken into tiny fragments, or rejected as noise depending on the clustering algorithm.",
        "",
        "## Evidence From The Existing Results",
        f"- HDBSCAN found only 5 non-noise clusters on {fit_rows:,} fitted rows and sent {int(hdbscan_row['noise_rows']):,} rows ({float(hdbscan_row['noise_fraction']):.2%}) to noise.",
        f"- The semantically mapped HDBSCAN clusters cover only {known_cluster_fraction:.2%} of the fitted sample, leaving {unknown_fraction:.2%} in the mixed `Unknown` bucket.",
        f"- GaussianMixture created {int(gmm_row['cluster_count'])} components, including {int(gmm_row['microcluster_count_leq_100'])} microclusters of size <= 100, which is the clearest evidence of fragmentation under imbalance.",
        f"- MiniBatchKMeans produced 8 forced partitions with no noise label and a largest-cluster share of {float(kmeans_row['largest_cluster_fraction']):.2%}, which is the signature of absorption into broader majority clusters.",
        "",
        "## Assignment Questions",
        "- `Web Command Injection` versus `DDoS TCP`: this imbalance means the dense DDoS TCP region heavily influences distance-based structure, while the Web Command Injection examples are too sparse to form a stable basin of attraction. In practice, the dominant class gets a coherent cluster; the rare class usually does not.",
        "- Do rare attack types get absorbed, fragmented, or marked as noise? In these results, all three happen depending on the model, but the strongest pattern is `noise/outliers` for HDBSCAN, `fragmentation` for GaussianMixture, and `absorption` for MiniBatchKMeans.",
        "",
        "## Interpretation",
        "- The best internal-validation scores came from HDBSCAN because it can protect dense majority clusters by refusing to force every ambiguous point into one of them.",
        "- That same strength has a tradeoff: many rare attacks are effectively treated as insufficiently dense and move into the noise bucket instead of being discovered as their own cluster.",
        "- A model that forces every point into a cluster hides this failure mode, but it does not solve it; it simply redistributes minority examples into large clusters or microclusters.",
        "",
        "## Conclusion",
        "- Severe class imbalance does not just reduce clustering quality uniformly. It specifically makes unsupervised discovery biased toward high-volume attack families and against rare, application-layer, or context-dependent behaviors.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_directory(args.table_dir)

    q2_2a_summary = load_json(args.q2_2a_summary_path)
    q2_2b_summary = load_json(args.q2_2b_summary_path)
    q2_2c_summary = load_json(args.q2_2c_summary_path)
    cluster_sizes = pd.read_csv(args.cluster_sizes_path)

    algorithm_table = build_algorithm_table(cluster_sizes)
    report = render_report(q2_2a_summary, q2_2b_summary, q2_2c_summary, algorithm_table)

    algorithm_table_path = args.table_dir / "q2_3b_cluster_imbalance_summary.csv"
    report_path = args.table_dir / "q2_3b_report.md"
    summary_path = args.table_dir / "q2_3b_summary.json"

    algorithm_table.to_csv(algorithm_table_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "best_algorithm": q2_2b_summary["best_algorithm"],
                "fit_rows": int(q2_2a_summary["fit_rows"]),
                "conclusion": (
                    "Rare classes are primarily labeled as noise by HDBSCAN, fragmented by GaussianMixture, and "
                    "absorbed into larger clusters by MiniBatchKMeans."
                ),
                "algorithm_comparison": algorithm_table.to_dict(orient="records"),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
