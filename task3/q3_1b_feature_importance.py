"""Question 3.1(b): extract top-20 feature importances from the strongest tree baseline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory
from src.common.seed import set_global_seed
from q3_1a_baselines import (
    DEFAULT_LABELED_DATA_DIR,
    DEFAULT_SPLIT_STRATEGY,
    build_gradient_boosted_spec,
    build_preprocessor,
    drop_missing_labels,
    encode_labels,
    infer_label_column,
    load_datasets_from_args,
    render_table,
    select_feature_columns,
)


LOGGER = logging.getLogger("task3.q3_1b")
TREE_MODELS = ("RandomForest", "LightGBM", "XGBoost")
MODEL_FILENAME_MAP = {
    "RandomForest": "q3_1a_randomforest_baseline.joblib",
    "LightGBM": "q3_1a_lightgbm_baseline.joblib",
    "XGBoost": "q3_1a_xgboost_baseline.joblib",
}
ENGINEERED_ALIGNMENT_MAP = {
    "directional_byte_imbalance": {"totlen_fwd_pkts", "totlen_bwd_pkts", "down_up_ratio"},
    "bytes_per_packet": {
        "totlen_fwd_pkts",
        "totlen_bwd_pkts",
        "tot_fwd_pkts",
        "tot_bwd_pkts",
        "pkt_size_avg",
        "fwd_seg_size_avg",
        "bwd_seg_size_avg",
    },
    "burst_idle_log_ratio": {"active_mean", "active_max", "active_min", "idle_mean", "idle_max", "idle_min"},
    "packet_size_asymmetry": {
        "fwd_pkt_len_mean",
        "bwd_pkt_len_mean",
        "fwd_pkt_len_std",
        "bwd_pkt_len_std",
        "pkt_len_mean",
        "pkt_len_std",
        "pkt_len_var",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        type=Path,
        default=None,
        help="Optional explicit path to the labeled training dataset.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=None,
        help="Optional explicit path to the labeled test dataset.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=CONFIG.processed_data_dir,
        help="Directory searched for labeled train/test files when explicit paths are omitted.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Optional explicit label column name.",
    )
    parser.add_argument(
        "--labeled-data-dir",
        type=Path,
        default=DEFAULT_LABELED_DATA_DIR,
        help=(
            "Directory searched recursively for labeled CSV/parquet files when an explicit or "
            "pre-split train/test pair is unavailable."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test-set fraction used when auto-splitting a labeled corpus directory.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("row", "source_file", "router", "hybrid"),
        default=DEFAULT_SPLIT_STRATEGY,
        help="How to split an auto-discovered labeled corpus before feature-importance analysis.",
    )
    parser.add_argument(
        "--baseline-summary-path",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_1a_baseline_summary.csv",
        help="Baseline summary CSV from Question 3.1(a), used to choose the strongest tree model.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=CONFIG.outputs_dir / "models" / "task3",
        help="Directory containing serialized Question 3.1(a) models.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit path to a fitted RandomForest/LightGBM/XGBoost pipeline.",
    )
    parser.add_argument(
        "--preferred-model",
        choices=("auto", "RandomForest", "LightGBM", "XGBoost"),
        default="auto",
        help="Tree model to analyze. 'auto' chooses the best tree baseline from the 3.1(a) summary when available.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top-ranked features to include in the table and figure.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "figures",
        help="Directory for generated Task 3.1(b) figures.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables",
        help="Directory for generated Task 3.1(b) tables and report files.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=CONFIG.random_seed,
        help="Random seed used if the selected tree model must be retrained.",
    )
    return parser.parse_args()


def infer_model_name_from_pipeline(pipeline: Pipeline) -> str:
    classifier_name = pipeline.named_steps["classifier"].__class__.__name__.lower()
    if "randomforest" in classifier_name:
        return "RandomForest"
    if "lgbm" in classifier_name or "lightgbm" in classifier_name:
        return "LightGBM"
    if "xgb" in classifier_name or "xgboost" in classifier_name:
        return "XGBoost"
    raise ValueError(f"Unsupported classifier type for Question 3.1(b): {pipeline.named_steps['classifier'].__class__.__name__}")


def choose_tree_model_name(preferred_model: str, baseline_summary_path: Path) -> str:
    if preferred_model != "auto":
        return preferred_model

    if baseline_summary_path.exists():
        summary = pd.read_csv(baseline_summary_path)
        candidates = summary[summary["model"].isin(TREE_MODELS)].copy()
        if not candidates.empty:
            candidates = candidates.sort_values(["f1_macro", "accuracy"], ascending=False)
            return str(candidates.iloc[0]["model"])

    return "RandomForest"


def fit_tree_pipeline(
    model_name: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    label_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    random_seed: int,
) -> Pipeline:
    encoder, encoded_train_labels, encoded_test_labels = encode_labels(
        train_frame[label_column],
        test_frame[label_column],
    )
    train_encoded = train_frame.copy()
    test_encoded = test_frame.copy()
    train_encoded[label_column] = encoded_train_labels
    test_encoded[label_column] = encoded_test_labels

    if model_name == "RandomForest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_seed,
        )
    else:
        gradient_spec = build_gradient_boosted_spec(random_seed, len(encoder.classes_))
        if gradient_spec.name != model_name:
            raise ImportError(
                f"Requested {model_name}, but the available gradient-boosting backend resolves to {gradient_spec.name}."
            )
        classifier = gradient_spec.factory()

    feature_columns = numeric_columns + categorical_columns
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                    scale_numeric=False,
                ),
            ),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(train_encoded[feature_columns], train_encoded[label_column])
    return pipeline


def load_or_train_tree_pipeline(
    *,
    model_name: str,
    model_path: Path | None,
    model_dir: Path,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    label_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    random_seed: int,
) -> tuple[Pipeline, str, str]:
    if model_path is not None:
        LOGGER.info("Loading tree model from %s", model_path)
        pipeline = joblib.load(model_path)
        return pipeline, infer_model_name_from_pipeline(pipeline), str(model_path)

    candidate_path = model_dir / MODEL_FILENAME_MAP[model_name]
    if candidate_path.exists():
        LOGGER.info("Loading %s baseline model from %s", model_name, candidate_path)
        pipeline = joblib.load(candidate_path)
        return pipeline, infer_model_name_from_pipeline(pipeline), str(candidate_path)

    LOGGER.info("Serialized %s baseline not found; retraining from the labeled training split.", model_name)
    pipeline = fit_tree_pipeline(
        model_name=model_name,
        train_frame=train_frame,
        test_frame=test_frame,
        label_column=label_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        random_seed=random_seed,
    )
    return pipeline, model_name, "retrained_in_memory"


def feature_family(feature_name: str) -> str:
    if feature_name.startswith("router_id_"):
        return "router"
    if "flag" in feature_name:
        return "tcp_flags"
    if "port" in feature_name or "protocol" in feature_name:
        return "endpoint_protocol"
    if "iat" in feature_name or "duration" in feature_name or "idle" in feature_name or "active" in feature_name:
        return "timing"
    if "pkt" in feature_name or "packet" in feature_name or "seg_size" in feature_name:
        return "packets"
    if "byt" in feature_name or "byte" in feature_name or "win" in feature_name:
        return "bytes_windows"
    return "other"


def aligned_engineered_features(feature_name: str) -> str:
    alignments = [
        engineered_name
        for engineered_name, supporting_features in ENGINEERED_ALIGNMENT_MAP.items()
        if feature_name in supporting_features
    ]
    return ", ".join(alignments)


def extract_feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        raise ValueError(f"{classifier.__class__.__name__} does not expose feature_importances_.")

    feature_names = list(preprocessor.get_feature_names_out())
    importances = np.asarray(classifier.feature_importances_, dtype=float)
    if len(feature_names) != len(importances):
        raise ValueError("Transformed feature count does not match the importance vector length.")

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    total_importance = float(table["importance"].sum())
    if total_importance > 0:
        table["importance_share"] = table["importance"] / total_importance
    else:
        table["importance_share"] = 0.0
    table["rank"] = np.arange(1, len(table) + 1)
    table["feature_family"] = table["feature"].map(feature_family)
    table["engineered_alignment"] = table["feature"].map(aligned_engineered_features)
    return table


def feature_series_for_analysis(train_frame: pd.DataFrame, feature_name: str) -> pd.Series | None:
    if feature_name in train_frame.columns:
        return pd.to_numeric(train_frame[feature_name], errors="coerce").fillna(0.0)
    if feature_name.startswith("router_id_") and "router_id" in train_frame.columns:
        router_value = feature_name.removeprefix("router_id_")
        return (train_frame["router_id"].astype(str) == router_value).astype(float)
    return None


def strongest_pair_for_feature(series: pd.Series, labels: pd.Series) -> dict[str, object]:
    data = pd.DataFrame({"label": labels.astype(str), "value": series.astype(float)})
    grouped = data.groupby("label")["value"].agg(["mean", "std", "count"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    best_result: dict[str, object] | None = None

    for (_, row_a), (_, row_b) in combinations(grouped.iterrows(), 2):
        pooled_std = np.sqrt((float(row_a["std"]) ** 2 + float(row_b["std"]) ** 2) / 2.0)
        standardized_gap = abs(float(row_a["mean"]) - float(row_b["mean"])) / max(pooled_std, 1e-9)
        candidate = {
            "class_a": str(row_a["label"]),
            "class_b": str(row_b["label"]),
            "class_a_mean": round(float(row_a["mean"]), 6),
            "class_b_mean": round(float(row_b["mean"]), 6),
            "standardized_gap": round(float(standardized_gap), 6),
        }
        if best_result is None or candidate["standardized_gap"] > best_result["standardized_gap"]:
            best_result = candidate

    if best_result is None:
        return {
            "class_a": "",
            "class_b": "",
            "class_a_mean": 0.0,
            "class_b_mean": 0.0,
            "standardized_gap": 0.0,
        }
    return best_result


def build_pair_separation_table(
    train_frame: pd.DataFrame,
    label_column: str,
    top_features: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = train_frame[label_column].astype(str)
    for record in top_features.to_dict(orient="records"):
        feature_name = str(record["feature"])
        feature_values = feature_series_for_analysis(train_frame, feature_name)
        if feature_values is None:
            rows.append(
                {
                    "feature": feature_name,
                    "most_separated_class_pair": "",
                    "standardized_gap": 0.0,
                    "class_a_mean": 0.0,
                    "class_b_mean": 0.0,
                    "analysis_available": False,
                }
            )
            continue

        strongest = strongest_pair_for_feature(feature_values, labels)
        rows.append(
            {
                "feature": feature_name,
                "most_separated_class_pair": f"{strongest['class_a']} vs {strongest['class_b']}",
                "standardized_gap": strongest["standardized_gap"],
                "class_a_mean": strongest["class_a_mean"],
                "class_b_mean": strongest["class_b_mean"],
                "analysis_available": True,
            }
        )

    return pd.DataFrame(rows)


def save_importance_figure(top_features: pd.DataFrame, figure_path: Path, model_name: str) -> None:
    ordered = top_features.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    colors = ["#1f5aa6" if family != "router" else "#d67c2c" for family in ordered["feature_family"]]
    ax.barh(ordered["feature"], ordered["importance"], color=colors)
    ax.set_title(f"Task 3.1(b): Top {len(top_features)} Feature Importances ({model_name})")
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    *,
    report_path: Path,
    model_name: str,
    model_source: str,
    top_features: pd.DataFrame,
    pair_table: pd.DataFrame,
) -> None:
    family_summary = (
        top_features["feature_family"]
        .value_counts()
        .rename_axis("feature_family")
        .reset_index(name="count")
    )
    engineered_hits = top_features[top_features["engineered_alignment"] != ""].copy()
    pair_highlights = pair_table.sort_values("standardized_gap", ascending=False).head(8)

    lines = [
        "# Task 3.1(b) Feature Importance Report",
        "",
        "## Model Used",
        f"- Feature importances were extracted from the `{model_name}` baseline.",
        f"- Model source: `{model_source}`",
        "",
        "## Top-20 Features",
        render_table(top_features),
        "",
        "## Dominant Feature Families",
        render_table(family_summary),
        "",
        "## Alignment With Task 2 Engineered Features",
    ]
    if engineered_hits.empty:
        lines.append("- None of the top-ranked features directly overlapped the Task 2 engineered-feature ingredients.")
    else:
        lines.extend(
            [
                "- Several high-importance raw features line up with the Task 2 engineering themes around directionality, packet size, and burst/idle timing.",
                render_table(engineered_hits[["rank", "feature", "engineered_alignment", "importance_share"]]),
            ]
        )

    lines.extend(
        [
            "",
            "## Likely Attack-Pair Separators",
            (
                "The table below highlights which class pair each top feature separates most strongly "
                "based on standardized mean gaps in the labeled training data."
            ),
            render_table(pair_highlights),
            "",
            "## Discussion",
            "- The strongest features indicate whether traffic classes separate mainly through timing, packet-shape, byte-volume, or router-locality signals.",
            "- High overlap with the Task 2 engineered-feature ingredients suggests those engineered features were directionally sensible even before labels were available.",
            "- Pair-specific separation results help identify which raw CICFlowMeter fields are especially useful for confusing attack families such as DDoS versus DoS variants.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_global_seed(args.random_seed)

    ensure_directory(args.figure_dir)
    ensure_directory(args.table_dir)
    ensure_directory(args.model_dir)

    datasets = load_datasets_from_args(args)
    train_frame = datasets.train_frame
    test_frame = datasets.test_frame

    label_column = infer_label_column(train_frame, test_frame, args.label_column)
    train_frame = drop_missing_labels(train_frame, label_column, "train")
    test_frame = drop_missing_labels(test_frame, label_column, "test")
    numeric_columns, categorical_columns, dropped_columns = select_feature_columns(
        train_frame,
        test_frame,
        label_column,
    )

    selected_model_name = choose_tree_model_name(args.preferred_model, args.baseline_summary_path)
    pipeline, resolved_model_name, model_source = load_or_train_tree_pipeline(
        model_name=selected_model_name,
        model_path=args.model_path,
        model_dir=args.model_dir,
        train_frame=train_frame,
        test_frame=test_frame,
        label_column=label_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        random_seed=args.random_seed,
    )

    importance_table = extract_feature_importances(pipeline)
    top_features = importance_table.head(args.top_k).copy()
    top_features["importance"] = top_features["importance"].round(6)
    top_features["importance_share"] = top_features["importance_share"].round(6)

    pair_table = build_pair_separation_table(train_frame, label_column, top_features)
    top_features = top_features.merge(pair_table, on="feature", how="left")

    importance_path = args.table_dir / "q3_1b_top20_feature_importance.csv"
    pair_path = args.table_dir / "q3_1b_feature_pair_separation.csv"
    summary_path = args.table_dir / "q3_1b_summary.json"
    report_path = args.table_dir / "q3_1b_report.md"
    figure_path = args.figure_dir / "q3_1b_top20_feature_importance.png"

    top_features.to_csv(importance_path, index=False)
    pair_table.to_csv(pair_path, index=False)
    save_importance_figure(top_features, figure_path, resolved_model_name)
    write_report(
        report_path=report_path,
        model_name=resolved_model_name,
        model_source=model_source,
        top_features=top_features,
        pair_table=pair_table,
    )

    summary_payload = {
        "train_path": datasets.train_source,
        "test_path": datasets.test_source,
        "split_strategy": datasets.split_strategy,
        "train_groups": datasets.train_groups,
        "test_groups": datasets.test_groups,
        "label_column": label_column,
        "model_used": resolved_model_name,
        "model_source": model_source,
        "top_k": int(args.top_k),
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "dropped_columns": dropped_columns,
        "top_features": top_features.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    LOGGER.info("Wrote feature-importance table to %s", importance_path)
    LOGGER.info("Wrote feature-pair separation table to %s", pair_path)
    LOGGER.info("Wrote feature-importance figure to %s", figure_path)
    LOGGER.info("Wrote feature-importance report to %s", report_path)


if __name__ == "__main__":
    main()
