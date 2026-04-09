"""Prepare a short Task 2 prediction report and row-level prediction file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.paths import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assignments-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_2a_cluster_assignments.csv",
        help="Cluster assignments produced by Question 2.2(a).",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_2c_hdbscan_cluster_mapping.csv",
        help="Cluster-to-label mapping produced by Question 2.2(c).",
    )
    parser.add_argument(
        "--q2-2a-summary-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_2a_summary.json",
        help="Summary JSON produced by Question 2.2(a).",
    )
    parser.add_argument(
        "--q2-2b-summary-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_2b_summary.json",
        help="Summary JSON produced by Question 2.2(b).",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables",
        help="Directory for generated Task 2 submission outputs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_report(
    fit_rows: int,
    best_algorithm: str,
    predictions: pd.DataFrame,
    mapping: pd.DataFrame,
) -> str:
    label_counts = predictions["predicted_label"].value_counts().sort_values(ascending=False)
    lines = [
        "# Task 2 Prediction Submission Report",
        "",
        "## Scope",
        f"This report summarizes the unlabeled traffic predictions produced for Task 2 using `{best_algorithm}`, the best-performing unsupervised model from Question 2.2(b).",
        f"The attached row-level prediction file covers the {fit_rows:,}-row fitted sample used for clustering in Question 2.2(a), because those are the rows for which the repository currently stores final HDBSCAN assignments.",
        "",
        "## Predicted Cluster-To-Class Mapping",
    ]

    for row in mapping.itertuples(index=False):
        lines.append(
            f"- Cluster `{row.cluster_label}` -> **{row.predicted_label}** ({row.confidence} confidence): {row.evidence}"
        )

    lines.extend(
        [
            "",
            "## Prediction Counts",
        ]
    )
    for label, count in label_counts.items():
        lines.append(f"- **{label}**: {int(count):,} rows")

    lines.extend(
        [
            "",
            "## Submission Files",
            "- `q2_task2_predictions.csv`: row-level predicted labels for the clustered sample.",
            "- `q2_task2_prediction_report.md`: short narrative summary of the predictions.",
            "",
            "## Note",
            "Several of the assignment's 11 named classes do not appear as dedicated clusters in the unsupervised output. Those behaviors are conservatively left inside the `Unknown` bucket rather than over-claiming a specific attack label.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_directory(args.table_dir)

    assignments = pd.read_csv(args.assignments_path)
    mapping = pd.read_csv(args.mapping_path)
    q2_2a_summary = load_json(args.q2_2a_summary_path)
    q2_2b_summary = load_json(args.q2_2b_summary_path)

    mapping_lookup = mapping[["cluster_label", "predicted_label", "confidence"]].copy()
    predictions = assignments.merge(
        mapping_lookup,
        left_on="hdbscan_cluster",
        right_on="cluster_label",
        how="left",
    ).drop(columns=["cluster_label"])

    predictions["predicted_label"] = predictions["predicted_label"].fillna("Unknown")
    predictions["confidence"] = predictions["confidence"].fillna("low")

    prediction_columns = [
        "router_id",
        "umap_1",
        "umap_2",
        "hdbscan_cluster",
        "predicted_label",
        "confidence",
        "minibatchkmeans_cluster",
        "gaussianmixture_cluster",
    ]
    predictions = predictions[prediction_columns]

    report = render_report(
        fit_rows=int(q2_2a_summary["fit_rows"]),
        best_algorithm=str(q2_2b_summary["best_algorithm"]),
        predictions=predictions,
        mapping=mapping,
    )

    predictions_path = args.table_dir / "q2_task2_predictions.csv"
    report_path = args.table_dir / "q2_task2_prediction_report.md"
    summary_path = args.table_dir / "q2_task2_prediction_summary.json"

    predictions.to_csv(predictions_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "best_algorithm": q2_2b_summary["best_algorithm"],
                "fit_rows": int(q2_2a_summary["fit_rows"]),
                "prediction_counts": predictions["predicted_label"].value_counts().to_dict(),
                "mapping": mapping.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
