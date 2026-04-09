"""Question 3.1(c): generate a confusion matrix for the best baseline model."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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
from sklearn.metrics import confusion_matrix
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
    build_model_specs,
    build_preprocessor,
    drop_missing_labels,
    encode_labels,
    infer_label_column,
    load_datasets_from_args,
    render_table,
    select_feature_columns,
)


LOGGER = logging.getLogger("task3.q3_1c")
MODEL_FILENAME_MAP = {
    "RandomForest": "q3_1a_randomforest_baseline.joblib",
    "HistGradientBoosting": "q3_1a_histgradientboosting_baseline.joblib",
    "LightGBM": "q3_1a_lightgbm_baseline.joblib",
    "XGBoost": "q3_1a_xgboost_baseline.joblib",
    "MLP": "q3_1a_mlp_baseline.joblib",
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
        help="How to split an auto-discovered labeled corpus before confusion-matrix analysis.",
    )
    parser.add_argument(
        "--baseline-summary-path",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_1a_baseline_summary.csv",
        help="Baseline summary CSV from Question 3.1(a), used to choose the best baseline model.",
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
        help="Optional explicit path to a fitted baseline pipeline.",
    )
    parser.add_argument(
        "--preferred-model",
        choices=("auto", "RandomForest", "HistGradientBoosting", "LightGBM", "XGBoost", "MLP"),
        default="auto",
        help="Baseline model to analyze. 'auto' selects the best baseline from the 3.1(a) summary when available.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "figures",
        help="Directory for generated Task 3.1(c) figures.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables",
        help="Directory for generated Task 3.1(c) tables and reports.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of off-diagonal confusion pairs to highlight in the report.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=CONFIG.random_seed,
        help="Random seed used if the selected baseline must be retrained.",
    )
    return parser.parse_args()


def infer_model_name_from_pipeline(pipeline: Pipeline) -> str:
    classifier_name = pipeline.named_steps["classifier"].__class__.__name__.lower()
    if "randomforest" in classifier_name:
        return "RandomForest"
    if "histgradientboosting" in classifier_name:
        return "HistGradientBoosting"
    if "lgbm" in classifier_name or "lightgbm" in classifier_name:
        return "LightGBM"
    if "xgb" in classifier_name or "xgboost" in classifier_name:
        return "XGBoost"
    if "mlp" in classifier_name:
        return "MLP"
    raise ValueError(f"Unsupported classifier type for Question 3.1(c): {pipeline.named_steps['classifier'].__class__.__name__}")


def choose_best_model_name(preferred_model: str, baseline_summary_path: Path) -> str:
    if preferred_model != "auto":
        return preferred_model

    if baseline_summary_path.exists():
        summary = pd.read_csv(baseline_summary_path)
        if not summary.empty:
            ranked = summary.sort_values(["f1_macro", "accuracy"], ascending=False)
            return str(ranked.iloc[0]["model"])

    return "RandomForest"


def fit_baseline_pipeline(
    *,
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

    specs = build_model_specs(random_seed, len(encoder.classes_))
    matches = [spec for spec in specs if spec.name == model_name]
    if not matches:
        raise ValueError(f"Could not find a baseline configuration for model '{model_name}'.")
    spec = matches[0]

    feature_columns = numeric_columns + categorical_columns
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                    scale_numeric=spec.name == "MLP",
                ),
            ),
            ("classifier", spec.factory()),
        ]
    )
    pipeline.fit(train_encoded[feature_columns], train_encoded[label_column])
    return pipeline


def load_or_train_baseline_pipeline(
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
        LOGGER.info("Loading baseline model from %s", model_path)
        pipeline = joblib.load(model_path)
        return pipeline, infer_model_name_from_pipeline(pipeline), str(model_path)

    candidate_filename = MODEL_FILENAME_MAP.get(model_name)
    if candidate_filename is not None:
        candidate_path = model_dir / candidate_filename
        if candidate_path.exists():
            LOGGER.info("Loading %s baseline model from %s", model_name, candidate_path)
            pipeline = joblib.load(candidate_path)
            return pipeline, infer_model_name_from_pipeline(pipeline), str(candidate_path)

    LOGGER.info("Serialized %s baseline not found; retraining from the labeled training split.", model_name)
    pipeline = fit_baseline_pipeline(
        model_name=model_name,
        train_frame=train_frame,
        test_frame=test_frame,
        label_column=label_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        random_seed=random_seed,
    )
    return pipeline, model_name, "retrained_in_memory"


def build_confusion_tables(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    confusion_table = pd.DataFrame(matrix, index=class_names, columns=class_names)

    normalized = confusion_table.div(confusion_table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    normalized.index.name = "true_label"
    normalized.columns.name = "predicted_label"

    rows: list[dict[str, object]] = []
    for true_index, true_name in enumerate(class_names):
        for pred_index, pred_name in enumerate(class_names):
            if true_index == pred_index:
                continue
            count = int(matrix[true_index, pred_index])
            if count == 0:
                continue
            row_total = int(matrix[true_index].sum())
            rows.append(
                {
                    "true_label": true_name,
                    "predicted_label": pred_name,
                    "count": count,
                    "row_share": round(count / row_total, 6) if row_total else 0.0,
                }
            )

    top_confusions = pd.DataFrame(rows).sort_values(["count", "row_share"], ascending=False, ignore_index=True)
    return confusion_table, normalized, top_confusions


def infer_family(label: str) -> str:
    return label.split("-", maxsplit=1)[0].lower()


def infer_confusion_hypothesis(true_label: str, predicted_label: str) -> str:
    true_family = infer_family(true_label)
    pred_family = infer_family(predicted_label)

    if true_family == pred_family:
        if true_family == "ddos":
            return "These two DDoS variants are both high-volume flood behaviors, so per-flow timing and packet-shape features can overlap heavily."
        if true_family == "dos":
            return "These DoS variants both concentrate on application or transport exhaustion, so CICFlowMeter summary features may not cleanly separate them."
        if true_family == "web":
            return "These web attacks are typically low-rate and payload-driven, while the available features are mostly flow summaries rather than deep content features."
        return "These classes belong to the same broad family, so they likely share similar flow-level signatures."

    families = {true_family, pred_family}
    if families == {"ddos", "dos"}:
        return "Both classes are denial-of-service traffic, so rate, burstiness, and packet-size statistics can look similar when the model only sees flow summaries."
    if "normal" in families and "infiltration" in families:
        return "Infiltration traffic can be stealthy and low-volume, which makes it look closer to benign traffic than loud flooding attacks do."
    if "web" in families and "normal" in families:
        return "Low-volume web attacks may resemble ordinary application traffic when payload semantics are not directly visible in the feature set."
    if "web" in families and "infiltration" in families:
        return "Both behaviors can be subtle and low-rate, so timing and packet-size summaries may not capture the higher-level intent difference."
    if "normal" in families:
        return "This confusion suggests the attack sometimes resembles ordinary traffic at the flow-summary level, especially when it is short or low-rate."

    return "These classes likely share overlapping timing, volume, or packet-shape statistics, which makes them difficult to separate using only CICFlowMeter flow summaries."


def save_confusion_figure(
    confusion_normalized: pd.DataFrame,
    *,
    figure_path: Path,
    model_name: str,
) -> None:
    matrix = confusion_normalized.to_numpy(dtype=float)
    labels = confusion_normalized.index.tolist()

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=max(float(matrix.max()), 1e-9))
    ax.set_title(f"Task 3.1(c): Normalized Confusion Matrix ({model_name})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            text_color = "white" if value >= 0.5 * matrix.max(initial=0.0) and value > 0 else "#16324f"
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized share")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    *,
    report_path: Path,
    model_name: str,
    model_source: str,
    top_confusions: pd.DataFrame,
    confusion_counts: pd.DataFrame,
    accuracy: float,
) -> None:
    highlighted = top_confusions.copy()
    if not highlighted.empty:
        highlighted["networking_hypothesis"] = [
            infer_confusion_hypothesis(row["true_label"], row["predicted_label"])
            for _, row in highlighted.iterrows()
        ]

    lines = [
        "# Task 3.1(c) Confusion Matrix Report",
        "",
        "## Model Used",
        f"- Best baseline selected: `{model_name}`",
        f"- Model source: `{model_source}`",
        f"- Test accuracy: {accuracy:.6f}",
        "",
        "## Most Common Confusions",
    ]
    if highlighted.empty:
        lines.append("- No off-diagonal confusions were observed on the evaluated test split.")
    else:
        lines.extend(
            [
                "The table below highlights the most frequent off-diagonal confusion pairs for the selected baseline.",
                render_table(highlighted),
            ]
        )

    lines.extend(
        [
            "",
            "## Full Confusion Matrix",
            render_table(confusion_counts.reset_index().rename(columns={"index": "true_label"})),
            "",
            "## Discussion",
        ]
    )

    if highlighted.empty:
        lines.append("- The evaluated model classified every test instance correctly, so there were no confusion patterns to analyze.")
    else:
        lines.extend(
            [
                "- Confusions within the same attack family usually indicate that the flows share similar rate, burstiness, or packet-size distributions.",
                "- Confusions involving `Normal` or `Infiltration-mitm` are often driven by lower-volume traffic that looks less distinct at the flow-summary level.",
                "- These networking explanations are hypotheses drawn from the confusion matrix and the available CICFlowMeter-style features.",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_global_seed(args.random_seed)

    figure_dir = ensure_directory(args.figure_dir)
    table_dir = ensure_directory(args.table_dir)
    model_dir = ensure_directory(args.model_dir)

    datasets = load_datasets_from_args(args)
    train_frame = datasets.train_frame
    test_frame = datasets.test_frame

    label_column = infer_label_column(train_frame, test_frame, args.label_column)
    train_frame = drop_missing_labels(train_frame, label_column, "train")
    test_frame = drop_missing_labels(test_frame, label_column, "test")

    encoder, encoded_train_labels, encoded_test_labels = encode_labels(
        train_frame[label_column],
        test_frame[label_column],
    )
    class_names = [str(label) for label in encoder.classes_]

    train_encoded = train_frame.copy()
    test_encoded = test_frame.copy()
    train_encoded[label_column] = encoded_train_labels
    test_encoded[label_column] = encoded_test_labels

    numeric_columns, categorical_columns, _ = select_feature_columns(
        train_encoded,
        test_encoded,
        label_column,
    )
    feature_columns = numeric_columns + categorical_columns

    selected_model_name = choose_best_model_name(args.preferred_model, args.baseline_summary_path)
    pipeline, resolved_model_name, model_source = load_or_train_baseline_pipeline(
        model_name=selected_model_name,
        model_path=args.model_path,
        model_dir=model_dir,
        train_frame=train_frame,
        test_frame=test_frame,
        label_column=label_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        random_seed=args.random_seed,
    )

    LOGGER.info("Generating predictions for the %s baseline.", resolved_model_name)
    predictions = pipeline.predict(test_encoded[feature_columns])
    confusion_counts, confusion_normalized, top_confusions = build_confusion_tables(
        encoded_test_labels,
        predictions,
        class_names,
    )

    top_confusions = top_confusions.head(args.top_k).copy()
    if not top_confusions.empty:
        top_confusions["networking_hypothesis"] = [
            infer_confusion_hypothesis(row["true_label"], row["predicted_label"])
            for _, row in top_confusions.iterrows()
        ]

    figure_path = figure_dir / "q3_1c_confusion_matrix.png"
    counts_path = table_dir / "q3_1c_confusion_matrix.csv"
    normalized_path = table_dir / "q3_1c_confusion_matrix_normalized.csv"
    confusions_path = table_dir / "q3_1c_top_confusions.csv"
    report_path = table_dir / "q3_1c_report.md"
    summary_path = table_dir / "q3_1c_summary.json"

    confusion_counts.to_csv(counts_path, index=True)
    confusion_normalized.to_csv(normalized_path, index=True)
    top_confusions.to_csv(confusions_path, index=False)
    save_confusion_figure(confusion_normalized, figure_path=figure_path, model_name=resolved_model_name)

    accuracy = float(np.trace(confusion_counts.to_numpy())) / max(float(confusion_counts.to_numpy().sum()), 1.0)
    write_report(
        report_path=report_path,
        model_name=resolved_model_name,
        model_source=model_source,
        top_confusions=top_confusions,
        confusion_counts=confusion_counts,
        accuracy=accuracy,
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
        "accuracy": accuracy,
        "class_names": class_names,
        "top_confusions": top_confusions.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    LOGGER.info("Wrote confusion matrix counts to %s", counts_path)
    LOGGER.info("Wrote normalized confusion matrix to %s", normalized_path)
    LOGGER.info("Wrote top confusion pairs to %s", confusions_path)
    LOGGER.info("Wrote confusion matrix figure to %s", figure_path)
    LOGGER.info("Wrote confusion matrix report to %s", report_path)


if __name__ == "__main__":
    main()
