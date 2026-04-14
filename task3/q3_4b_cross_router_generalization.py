"""Question 3.4(b): train on routers D1-D7 and test on routers D8-D10."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

try:
    import joblib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing optional dependency 'joblib'. Install project dependencies with "
        "'python3 -m pip install -r requirements.txt' and rerun q3_4b_cross_router_generalization.py."
    ) from exc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory
from src.common.seed import set_global_seed
from task3.q3_1a_baselines import (
    DEFAULT_LABELED_DATA_DIR,
    SOURCE_ROUTER_COLUMN,
    drop_missing_labels,
    load_labeled_corpus,
    render_table,
)
from task3.q3_2a_imbalance_strategies import (
    StrategySpec,
    build_sampling_strategy,
    build_strategy_specs,
    cap_rows,
    prepare_training_data,
    resolve_smote_neighbors,
)
from task3.q3_3_advanced_model import (
    BaseConfig,
    WebDetectorConfig,
    WebSubtypeConfig,
    add_engineered_features,
    compute_per_class_metrics,
    compute_summary_metrics,
    predict_hybrid_model,
    train_hybrid_model,
)
from task3.q3_4a_router_level_analysis import (
    SelectedModel,
    build_class_distribution,
    build_router_mix_summary,
    choose_best_available_model,
    infer_router_labels,
    summarize_router_metrics,
)


LOGGER = logging.getLogger("task3.q3_4b")
TRAIN_ROUTER_DEFAULT = "D1,D2,D3,D4,D5,D6,D7"
TEST_ROUTER_DEFAULT = "D8,D9,D10"
METRIC_ORDER = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1 (macro)",
    "f1_weighted": "F1 (weighted)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Optional explicit label column for the labeled router corpus.",
    )
    parser.add_argument(
        "--labeled-data-dir",
        type=Path,
        default=DEFAULT_LABELED_DATA_DIR,
        help="Directory containing the labeled per-router CSV/parquet files.",
    )
    parser.add_argument(
        "--train-routers",
        type=str,
        default=TRAIN_ROUTER_DEFAULT,
        help="Comma-separated router ids used for training, e.g. D1,D2,D3,D4,D5,D6,D7.",
    )
    parser.add_argument(
        "--test-routers",
        type=str,
        default=TEST_ROUTER_DEFAULT,
        help="Comma-separated router ids used for testing, e.g. D8,D9,D10.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap for training rows, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional cap for test rows, useful for smoke tests.",
    )
    parser.add_argument(
        "--q3-1a-summary-json",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_1a_summary.json",
        help="Task 3.1(a) summary JSON used during best-model selection fallback.",
    )
    parser.add_argument(
        "--q3-1a-summary-csv",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_1a_baseline_summary.csv",
        help="Task 3.1(a) summary CSV used during best-model selection fallback.",
    )
    parser.add_argument(
        "--q3-2a-summary-csv",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_2a_imbalance_summary.csv",
        help="Task 3.2(a) summary CSV used during best-model selection fallback.",
    )
    parser.add_argument(
        "--q3-2a-per-class-csv",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_2a_per_class_metrics.csv",
        help="Task 3.2(a) per-class CSV used during best-model selection fallback.",
    )
    parser.add_argument(
        "--q3-3-summary-json",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_3_summary.json",
        help="Task 3.3 summary JSON used during best-model selection fallback.",
    )
    parser.add_argument(
        "--q3-3-summary-csv",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_3_advanced_summary.csv",
        help="Task 3.3 summary CSV used during best-model selection fallback.",
    )
    parser.add_argument(
        "--q3-3-per-class-csv",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_3_per_class_metrics.csv",
        help="Task 3.3 per-class metrics CSV used during best-model selection fallback.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=CONFIG.outputs_dir / "models" / "task3",
        help="Directory containing serialized Task 3 model artifacts.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables",
        help="Directory for generated Task 3.4(b) tables and reports.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "figures",
        help="Directory for generated Task 3.4(b) figures.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=CONFIG.random_seed,
        help="Random seed used across the cross-router experiment.",
    )
    return parser.parse_args()


def parse_router_ids(value: str) -> tuple[int, ...]:
    routers: list[int] = []
    for chunk in value.split(","):
        token = chunk.strip()
        if not token:
            continue
        upper = token.upper()
        if upper.startswith("D"):
            upper = upper[1:]
        if not upper.isdigit():
            raise ValueError(f"Invalid router id '{token}'. Use values like D1,D2,D3.")
        routers.append(int(upper))

    unique = tuple(sorted(set(routers)))
    if not unique:
        raise ValueError("At least one router id must be provided.")
    return unique


def router_labels(router_ids: tuple[int, ...]) -> list[str]:
    return [f"D{router_id}" for router_id in router_ids]


def ordered_class_names(
    preferred: tuple[str, ...],
    y_true: pd.Series,
    predictions: np.ndarray,
) -> tuple[str, ...]:
    observed = sorted(
        set(map(str, y_true.astype(str).tolist()))
        | set(map(str, np.asarray(predictions, dtype=object).tolist()))
    )
    ordered = [class_name for class_name in preferred if class_name in observed]
    ordered.extend(class_name for class_name in observed if class_name not in ordered)
    return tuple(ordered)


def load_cross_router_split(
    labeled_data_dir: Path,
    *,
    label_column: str | None,
    train_router_ids: tuple[int, ...],
    test_router_ids: tuple[int, ...],
    max_train_rows: int | None,
    max_test_rows: int | None,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    combined, _, inferred_label_column = load_labeled_corpus(
        labeled_data_dir,
        label_column=label_column,
    )
    combined = drop_missing_labels(combined, inferred_label_column, "combined")

    train_mask = combined[SOURCE_ROUTER_COLUMN].isin(train_router_ids)
    test_mask = combined[SOURCE_ROUTER_COLUMN].isin(test_router_ids)
    train_frame = combined.loc[train_mask].reset_index(drop=True)
    test_frame = combined.loc[test_mask].reset_index(drop=True)

    if train_frame.empty:
        raise ValueError(f"No training rows were found for routers {router_labels(train_router_ids)}.")
    if test_frame.empty:
        raise ValueError(f"No test rows were found for routers {router_labels(test_router_ids)}.")

    train_frame = cap_rows(
        train_frame,
        label_column=inferred_label_column,
        max_rows=max_train_rows,
        random_seed=random_seed,
        split_name="cross_router_train",
    )
    test_frame = cap_rows(
        test_frame,
        label_column=inferred_label_column,
        max_rows=max_test_rows,
        random_seed=random_seed,
        split_name="cross_router_test",
    )

    train_frame = infer_router_labels(train_frame).reset_index(drop=True)
    test_frame = infer_router_labels(test_frame).reset_index(drop=True)
    return train_frame, test_frame, inferred_label_column


def build_class_coverage_table(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    train_counts = train_frame[label_column].astype(str).value_counts().rename("train_count")
    test_counts = test_frame[label_column].astype(str).value_counts().rename("test_count")
    coverage = pd.concat([train_counts, test_counts], axis=1).fillna(0).rename_axis("class_name").reset_index()
    coverage["train_count"] = coverage["train_count"].astype(int)
    coverage["test_count"] = coverage["test_count"].astype(int)
    coverage["present_in_train"] = coverage["train_count"] > 0
    coverage["present_in_test"] = coverage["test_count"] > 0
    coverage["test_only_class"] = coverage["present_in_test"] & ~coverage["present_in_train"]
    coverage["train_only_class"] = coverage["present_in_train"] & ~coverage["present_in_test"]
    total_test_rows = max(int(coverage["test_count"].sum()), 1)
    coverage["test_share"] = (coverage["test_count"] / total_test_rows).round(6)
    return coverage.sort_values(
        ["present_in_test", "test_count", "class_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def load_standard_reference(
    selected_model: SelectedModel,
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame]:
    if selected_model.artifact_kind == "advanced_hybrid":
        summary_table = pd.read_csv(args.q3_3_summary_csv)
        if summary_table.empty:
            raise ValueError(f"Standard reference summary is empty: {args.q3_3_summary_csv}")
        row = summary_table.iloc[0].to_dict()
        standard_summary = {
            "model_name": selected_model.display_name,
            "source_tag": selected_model.source_tag,
            "training_time_seconds": float(row["total_training_time_seconds"]),
            **{metric: float(row[metric]) for metric in METRIC_ORDER},
        }
        per_class = pd.read_csv(args.q3_3_per_class_csv).copy()
        if "variant" in per_class.columns:
            per_class = per_class.drop(columns=["variant"])
        return standard_summary, per_class

    if selected_model.artifact_kind == "imbalance_random_forest":
        strategy = selected_model.display_name.removeprefix("RandomForest (").removesuffix(")")
        summary_table = pd.read_csv(args.q3_2a_summary_csv)
        matched = summary_table[summary_table["strategy"] == strategy]
        if matched.empty:
            raise ValueError(f"Could not find the selected Task 3.2(a) strategy in {args.q3_2a_summary_csv}.")
        row = matched.iloc[0].to_dict()
        standard_summary = {
            "model_name": selected_model.display_name,
            "source_tag": selected_model.source_tag,
            "training_time_seconds": float(row["training_time_seconds"]),
            **{metric: float(row[metric]) for metric in METRIC_ORDER},
        }
        per_class = pd.read_csv(args.q3_2a_per_class_csv)
        per_class = per_class[per_class["strategy"] == strategy].drop(columns=["strategy"]).reset_index(drop=True)
        return standard_summary, per_class

    if selected_model.artifact_kind == "baseline_pipeline":
        model_name = selected_model.display_name.removesuffix(" baseline")
        summary_table = pd.read_csv(args.q3_1a_summary_csv)
        matched = summary_table[summary_table["model"] == model_name]
        if matched.empty:
            raise ValueError(f"Could not find the selected Task 3.1(a) model in {args.q3_1a_summary_csv}.")
        row = matched.iloc[0].to_dict()
        standard_summary = {
            "model_name": selected_model.display_name,
            "source_tag": selected_model.source_tag,
            "training_time_seconds": float(row["training_time_seconds"]),
            **{metric: float(row[metric]) for metric in METRIC_ORDER},
        }
        per_class = pd.read_csv(CONFIG.outputs_dir / "task3" / "tables" / "q3_1a_per_class_metrics.csv")
        per_class = per_class[per_class["model"] == model_name].drop(columns=["model"]).reset_index(drop=True)
        return standard_summary, per_class

    raise ValueError(f"Unsupported selected model kind: {selected_model.artifact_kind}")


def build_metric_comparison_table(
    standard_summary: dict[str, object],
    cross_router_summary: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRIC_ORDER:
        standard_value = float(standard_summary[metric])
        cross_value = float(cross_router_summary[metric])
        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "standard_result": round(standard_value, 6),
                "cross_router_result": round(cross_value, 6),
                "delta": round(cross_value - standard_value, 6),
            }
        )
    return pd.DataFrame(rows)


def build_per_class_comparison_table(
    cross_router_per_class: pd.DataFrame,
    standard_per_class: pd.DataFrame,
) -> pd.DataFrame:
    merged = cross_router_per_class.merge(
        standard_per_class,
        on="class_name",
        how="left",
        suffixes=("_cross_router", "_standard"),
    )
    merged["precision_delta"] = (
        merged["precision_cross_router"] - merged["precision_standard"]
    ).round(6)
    merged["recall_delta"] = (merged["recall_cross_router"] - merged["recall_standard"]).round(6)
    merged["f1_delta"] = (merged["f1_score_cross_router"] - merged["f1_score_standard"]).round(6)
    return merged.sort_values(
        ["f1_delta", "support_cross_router", "class_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def select_strategy_spec(strategy_name: str, artifact: dict[str, object]) -> StrategySpec:
    for spec in build_strategy_specs():
        if spec.name == strategy_name:
            return spec
    return StrategySpec(
        name=strategy_name,
        slug="cross_router",
        description="Recovered from a serialized Task 3.2(a) artifact.",
        class_weight=artifact.get("class_weight"),
        resampling_mode=str(artifact.get("resampling_mode", "none")),
    )


def infer_minority_target_count(artifact: dict[str, object]) -> int:
    sampling_strategy = artifact.get("sampling_strategy", {})
    if isinstance(sampling_strategy, dict) and sampling_strategy:
        return int(max(sampling_strategy.values()))
    return 25_000


def retrain_advanced_hybrid(
    selected_model: SelectedModel,
    serialized_artifact: dict[str, object],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    label_column: str,
    random_seed: int,
) -> tuple[dict[str, object], np.ndarray, dict[str, object], dict[str, object]]:
    train_ready = add_engineered_features(train_frame.copy())
    test_ready = add_engineered_features(test_frame.copy())

    base_config = BaseConfig(**serialized_artifact["base_config"])
    web_subtype_config = (
        WebSubtypeConfig(**serialized_artifact["web_subtype_config"])
        if bool(serialized_artifact.get("use_web_specialist"))
        else None
    )
    web_detector_config = (
        WebDetectorConfig(**serialized_artifact["web_detector_config"])
        if bool(serialized_artifact.get("use_web_specialist"))
        else None
    )

    model_artifact, training_times = train_hybrid_model(
        train_ready,
        label_column=label_column,
        feature_columns=list(serialized_artifact["feature_columns"]),
        base_config=base_config,
        class_score_weights=serialized_artifact.get("class_score_weights"),
        web_subtype_config=web_subtype_config,
        web_detector_config=web_detector_config,
        use_web_specialist=bool(serialized_artifact.get("use_web_specialist", False)),
        random_seed=random_seed,
    )
    predictions, extras = predict_hybrid_model(model_artifact, test_ready)
    y_test = test_ready[label_column].astype(str)
    class_names = ordered_class_names(selected_model.class_names, y_test, predictions)
    summary = {
        "selected_model": selected_model.display_name,
        "selected_source": selected_model.source_tag,
        "training_time_seconds": float(training_times["total_training_time_seconds"]),
        **training_times,
        **compute_summary_metrics(y_test, predictions),
        "override_rows": int(np.asarray(extras["override_mask"]).sum()),
    }
    model_payload = {
        "model_type": "cross_router_advanced_model",
        "label_column": label_column,
        "feature_columns": list(serialized_artifact["feature_columns"]),
        "train_hybrid_artifact": model_artifact,
        "base_config": serialized_artifact["base_config"],
        "class_score_weights": serialized_artifact.get("class_score_weights"),
        "web_subtype_config": serialized_artifact.get("web_subtype_config"),
        "web_detector_config": serialized_artifact.get("web_detector_config"),
        "use_web_specialist": bool(serialized_artifact.get("use_web_specialist", False)),
    }
    return summary, np.asarray(predictions, dtype=object), {"override_rows": summary["override_rows"]}, {
        "class_names": class_names,
        "model_payload": model_payload,
    }


def retrain_imbalance_random_forest(
    selected_model: SelectedModel,
    serialized_artifact: dict[str, object],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    label_column: str,
    random_seed: int,
) -> tuple[dict[str, object], np.ndarray, dict[str, object], dict[str, object]]:
    feature_columns = list(serialized_artifact["feature_columns"])
    y_train = train_frame[label_column].astype(str)
    y_test = test_frame[label_column].astype(str)

    imputer = SimpleImputer(strategy="median")
    x_train_imputed = imputer.fit_transform(train_frame[feature_columns]).astype(np.float32, copy=False)
    x_test_imputed = imputer.transform(test_frame[feature_columns]).astype(np.float32, copy=False)

    strategy_name = str(serialized_artifact["strategy"])
    strategy_spec = select_strategy_spec(strategy_name, serialized_artifact)
    minority_target_count = infer_minority_target_count(serialized_artifact)
    sampling_strategy = build_sampling_strategy(y_train, minority_target_count)
    smote_k_neighbors = resolve_smote_neighbors(
        y_train,
        sampling_strategy,
        int(serialized_artifact.get("smote_k_neighbors", 5)),
    )

    x_train_prepared, y_train_prepared = prepare_training_data(
        x_train_imputed,
        y_train,
        spec=strategy_spec,
        sampling_strategy=sampling_strategy,
        smote_k_neighbors=smote_k_neighbors,
        hybrid_majority_cap=int(serialized_artifact.get("hybrid_majority_cap", 250_000)),
        random_seed=random_seed,
    )

    classifier = clone(serialized_artifact["classifier"])
    start_time = time.perf_counter()
    classifier.fit(x_train_prepared, y_train_prepared)
    training_time = time.perf_counter() - start_time

    predictions = classifier.predict(x_test_imputed)
    class_names = ordered_class_names(selected_model.class_names, y_test, predictions)
    summary = {
        "selected_model": selected_model.display_name,
        "selected_source": selected_model.source_tag,
        "training_time_seconds": round(training_time, 3),
        **compute_summary_metrics(y_test, predictions),
        "train_rows_after_sampling": int(len(y_train_prepared)),
        "class_weight": strategy_spec.class_weight if strategy_spec.class_weight is not None else "none",
        "sampler": strategy_spec.resampling_mode,
    }
    model_payload = {
        "model_type": "cross_router_imbalance_random_forest",
        "label_column": label_column,
        "feature_columns": feature_columns,
        "strategy": strategy_name,
        "class_weight": strategy_spec.class_weight,
        "resampling_mode": strategy_spec.resampling_mode,
        "sampling_strategy": sampling_strategy,
        "smote_k_neighbors": smote_k_neighbors,
        "hybrid_majority_cap": int(serialized_artifact.get("hybrid_majority_cap", 250_000)),
        "imputer": imputer,
        "classifier": classifier,
    }
    return summary, np.asarray(predictions, dtype=object), {}, {
        "class_names": class_names,
        "model_payload": model_payload,
    }


def retrain_baseline_pipeline(
    selected_model: SelectedModel,
    serialized_artifact: object,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    label_column: str,
) -> tuple[dict[str, object], np.ndarray, dict[str, object], dict[str, object]]:
    y_train = train_frame[label_column].astype(str)
    y_test = test_frame[label_column].astype(str)

    pipeline = clone(serialized_artifact)
    start_time = time.perf_counter()
    pipeline.fit(train_frame, y_train)
    training_time = time.perf_counter() - start_time

    predictions = pipeline.predict(test_frame)
    class_names = ordered_class_names(selected_model.class_names, y_test, predictions)
    summary = {
        "selected_model": selected_model.display_name,
        "selected_source": selected_model.source_tag,
        "training_time_seconds": round(training_time, 3),
        **compute_summary_metrics(y_test, predictions),
    }
    model_payload = {
        "model_type": "cross_router_baseline_pipeline",
        "label_column": label_column,
        "pipeline": pipeline,
    }
    return summary, np.asarray(predictions, dtype=object), {}, {
        "class_names": class_names,
        "model_payload": model_payload,
    }


def evaluate_cross_router_generalization(
    selected_model: SelectedModel,
    serialized_artifact: object,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    label_column: str,
    random_seed: int,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray, dict[str, object], dict[str, object]]:
    if selected_model.artifact_kind == "advanced_hybrid":
        summary, predictions, extras, details = retrain_advanced_hybrid(
            selected_model,
            serialized_artifact,
            train_frame,
            test_frame,
            label_column=label_column,
            random_seed=random_seed,
        )
    elif selected_model.artifact_kind == "imbalance_random_forest":
        summary, predictions, extras, details = retrain_imbalance_random_forest(
            selected_model,
            serialized_artifact,
            train_frame,
            test_frame,
            label_column=label_column,
            random_seed=random_seed,
        )
    elif selected_model.artifact_kind == "baseline_pipeline":
        summary, predictions, extras, details = retrain_baseline_pipeline(
            selected_model,
            serialized_artifact,
            train_frame,
            test_frame,
            label_column=label_column,
        )
    else:
        raise ValueError(f"Unsupported model type: {selected_model.artifact_kind}")

    y_test = test_frame[label_column].astype(str)
    per_class = compute_per_class_metrics(y_test, predictions, list(details["class_names"]))
    return summary, per_class, predictions, extras, details


def create_metric_comparison_chart(comparison_table: pd.DataFrame, figure_path: Path) -> None:
    x = np.arange(len(comparison_table))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    standard_values = comparison_table["standard_result"].to_numpy(dtype=float)
    cross_values = comparison_table["cross_router_result"].to_numpy(dtype=float)

    ax.bar(
        x - width / 2,
        standard_values,
        width=width,
        label="Standard split",
        color="#4C6A92",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar(
        x + width / 2,
        cross_values,
        width=width,
        label="Cross-router",
        color="#D97D54",
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_table["metric_label"].tolist(), rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Question 3.4(b): Standard vs Cross-Router Generalization", fontsize=14, pad=14)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    for index, value in enumerate(standard_values):
        ax.text(index - width / 2, min(1.03, value + 0.015), f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for index, value in enumerate(cross_values):
        ax.text(index + width / 2, min(1.03, value + 0.015), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(
    *,
    selected_model: SelectedModel,
    train_router_ids: tuple[int, ...],
    test_router_ids: tuple[int, ...],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    cross_router_summary: dict[str, object],
    standard_summary: dict[str, object],
    comparison_table: pd.DataFrame,
    per_class_comparison: pd.DataFrame,
    class_coverage: pd.DataFrame,
    router_summary: pd.DataFrame,
    router_mix_summary: pd.DataFrame,
    comparison_figure_path: Path,
    report_path: Path,
) -> None:
    test_only = class_coverage[class_coverage["test_only_class"]].copy()
    test_only_classes = test_only["class_name"].tolist()
    unseen_test_share = float(test_only["test_share"].sum()) if not test_only.empty else 0.0
    hardest_router = router_summary.sort_values(
        ["macro_f1_present_classes", "weighted_f1", "accuracy"],
        ascending=True,
    ).iloc[0]
    easiest_router = router_summary.sort_values(
        ["macro_f1_present_classes", "weighted_f1", "accuracy"],
        ascending=False,
    ).iloc[0]
    standard_macro = float(standard_summary["f1_macro"])
    cross_macro = float(cross_router_summary["f1_macro"])
    macro_delta = cross_macro - standard_macro
    per_class_present = per_class_comparison.sort_values(
        ["f1_delta", "support_cross_router", "class_name"],
        ascending=[True, False, True],
    )

    lines = [
        "# Task 3.4(b) Cross-Router Generalization",
        "",
        "## Experiment Setup",
        f"- Selected model family: `{selected_model.display_name}` from {selected_model.source_tag}",
        f"- Training routers: {', '.join(router_labels(train_router_ids))}",
        f"- Test routers: {', '.join(router_labels(test_router_ids))}",
        f"- Training rows: {len(train_frame):,}",
        f"- Test rows: {len(test_frame):,}",
        f"- Standard reference macro F1: {standard_macro:.6f}",
        f"- Cross-router macro F1: {cross_macro:.6f}",
        f"- Cross-router macro F1 delta vs standard: {macro_delta:.6f}",
        (
            f"- Test-only classes absent from training: {', '.join(test_only_classes)} "
            f"({unseen_test_share * 100:.1f}% of the cross-router test rows)"
            if test_only_classes
            else "- All cross-router test classes are represented in the training routers."
        ),
        (
            "- The standard and cross-router evaluations use different test distributions, so the comparison below "
            "should be interpreted as a deployment stress test rather than an apples-to-apples benchmark."
        ),
        "",
        "## Overall Comparison",
        render_table(comparison_table[["metric_label", "standard_result", "cross_router_result", "delta"]]),
        "",
        "## Class Coverage",
        render_table(
            class_coverage[
                [
                    "class_name",
                    "train_count",
                    "test_count",
                    "present_in_train",
                    "present_in_test",
                    "test_only_class",
                    "test_share",
                ]
            ]
        ),
        "",
        "## Cross-Router Router Breakdown",
        render_table(
            router_summary[
                [
                    "router_label",
                    "rows",
                    "observed_class_count",
                    "accuracy",
                    "weighted_f1",
                    "macro_f1_present_classes",
                    "dominant_class",
                    "dominant_class_share",
                    "worst_observed_class",
                    "worst_observed_class_f1",
                ]
            ]
        ),
        "",
        "## Cross-Router Test Mix",
        render_table(router_mix_summary[["router_label", "rows", "observed_class_count", "top_classes"]]),
        "",
        "## Per-Class Delta on Cross-Router Test Classes",
        render_table(
            per_class_present[
                [
                    "class_name",
                    "support_cross_router",
                    "f1_score_standard",
                    "f1_score_cross_router",
                    "f1_delta",
                    "recall_standard",
                    "recall_cross_router",
                ]
            ]
        ),
        "",
        "## Interpretation",
        (
            f"- The best unseen-router performer is `{easiest_router['router_label']}` "
            f"with macro F1 {float(easiest_router['macro_f1_present_classes']):.6f}, "
            f"while the hardest is `{hardest_router['router_label']}` at "
            f"{float(hardest_router['macro_f1_present_classes']):.6f}."
        ),
        (
            f"- The biggest cross-router failures are concentrated in classes that never appear in the training routers, "
            f"especially {', '.join(test_only_classes)}."
            if test_only_classes
            else "- The main cross-router errors come from distribution shift within classes that are still present in train."
        ),
        (
            "- Models trained on one subset of routers do generalize reasonably to shared traffic patterns, "
            "but they do not generalize reliably to unseen router-specific attack classes or mixes."
        ),
        (
            "- For deployment in distributed networks, that means IDS models should be trained on traffic collected "
            "from multiple routers, refreshed as new routers come online, and monitored for domain shift rather than "
            "assuming a single centralized training view will transfer cleanly everywhere."
        ),
        "",
        "## Figure",
        f"- Overall metric comparison: `{comparison_figure_path}`",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_global_seed(args.random_seed)

    train_router_ids = parse_router_ids(args.train_routers)
    test_router_ids = parse_router_ids(args.test_routers)
    overlap = set(train_router_ids) & set(test_router_ids)
    if overlap:
        raise ValueError(f"Train/test router sets must be disjoint, but overlap was found: {sorted(overlap)}")

    table_dir = ensure_directory(args.table_dir)
    figure_dir = ensure_directory(args.figure_dir)
    model_dir = ensure_directory(args.model_dir)

    selected_model = choose_best_available_model(args)
    LOGGER.info(
        "Selected %s from %s (macro F1 %.6f, accuracy %.6f).",
        selected_model.display_name,
        selected_model.source_tag,
        selected_model.f1_macro,
        selected_model.accuracy,
    )

    train_frame, test_frame, inferred_label_column = load_cross_router_split(
        args.labeled_data_dir,
        label_column=args.label_column,
        train_router_ids=train_router_ids,
        test_router_ids=test_router_ids,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        random_seed=args.random_seed,
    )
    LOGGER.info(
        "Loaded cross-router split: %s training rows from %s; %s test rows from %s.",
        len(train_frame),
        ", ".join(router_labels(train_router_ids)),
        len(test_frame),
        ", ".join(router_labels(test_router_ids)),
    )
    label_column = selected_model.label_column or inferred_label_column
    if label_column != inferred_label_column and label_column not in train_frame.columns:
        raise ValueError(
            f"Selected model expects label column '{label_column}', but the cross-router corpus exposes "
            f"'{inferred_label_column}'."
        )

    serialized_artifact = joblib.load(selected_model.model_path)
    cross_router_summary, cross_router_per_class, predictions, extras, details = evaluate_cross_router_generalization(
        selected_model,
        serialized_artifact,
        train_frame,
        test_frame,
        label_column=label_column,
        random_seed=args.random_seed,
    )
    LOGGER.info(
        "Finished cross-router evaluation for %s. Macro F1 is %.6f.",
        selected_model.display_name,
        float(cross_router_summary["f1_macro"]),
    )

    class_coverage = build_class_coverage_table(train_frame, test_frame, label_column)
    standard_summary, standard_per_class = load_standard_reference(selected_model, args)
    comparison_table = build_metric_comparison_table(standard_summary, cross_router_summary)
    per_class_comparison = build_per_class_comparison_table(cross_router_per_class, standard_per_class)

    test_distribution = build_class_distribution(test_frame, label_column)
    router_summary, router_class_metrics = summarize_router_metrics(
        test_frame,
        predictions,
        tuple(details["class_names"]),
        label_column,
    )
    router_mix_summary = build_router_mix_summary(test_distribution)

    comparison_figure_path = figure_dir / "q3_4b_overall_metric_comparison.png"
    create_metric_comparison_chart(comparison_table, comparison_figure_path)

    summary_path = table_dir / "q3_4b_cross_router_summary.csv"
    per_class_path = table_dir / "q3_4b_cross_router_per_class_metrics.csv"
    comparison_path = table_dir / "q3_4b_comparison_vs_standard.csv"
    per_class_comparison_path = table_dir / "q3_4b_per_class_comparison_vs_standard.csv"
    class_coverage_path = table_dir / "q3_4b_class_coverage.csv"
    router_summary_path = table_dir / "q3_4b_router_summary.csv"
    router_class_metrics_path = table_dir / "q3_4b_router_class_metrics.csv"
    router_mix_path = table_dir / "q3_4b_router_mix_summary.csv"
    report_path = table_dir / "q3_4b_report.md"
    summary_json_path = table_dir / "q3_4b_summary.json"
    model_path = model_dir / "q3_4b_cross_router_model.joblib"

    pd.DataFrame([cross_router_summary]).to_csv(summary_path, index=False)
    cross_router_per_class.to_csv(per_class_path, index=False)
    comparison_table.to_csv(comparison_path, index=False)
    per_class_comparison.to_csv(per_class_comparison_path, index=False)
    class_coverage.to_csv(class_coverage_path, index=False)
    router_summary.to_csv(router_summary_path, index=False)
    router_class_metrics.to_csv(router_class_metrics_path, index=False)
    router_mix_summary.to_csv(router_mix_path, index=False)
    write_report(
        selected_model=selected_model,
        train_router_ids=train_router_ids,
        test_router_ids=test_router_ids,
        train_frame=train_frame,
        test_frame=test_frame,
        cross_router_summary=cross_router_summary,
        standard_summary=standard_summary,
        comparison_table=comparison_table,
        per_class_comparison=per_class_comparison,
        class_coverage=class_coverage,
        router_summary=router_summary,
        router_mix_summary=router_mix_summary,
        comparison_figure_path=comparison_figure_path,
        report_path=report_path,
    )

    model_payload = {
        "selected_model_display_name": selected_model.display_name,
        "selected_model_source": selected_model.source_tag,
        "selected_model_kind": selected_model.artifact_kind,
        "train_router_ids": list(train_router_ids),
        "test_router_ids": list(test_router_ids),
        "label_column": label_column,
        "prediction_extras": extras,
        "model": details["model_payload"],
    }
    joblib.dump(model_payload, model_path)

    summary_payload = {
        "selected_model": {
            "display_name": selected_model.display_name,
            "source_tag": selected_model.source_tag,
            "artifact_kind": selected_model.artifact_kind,
            "reference_f1_macro": selected_model.f1_macro,
            "reference_accuracy": selected_model.accuracy,
        },
        "train_router_ids": list(train_router_ids),
        "test_router_ids": list(test_router_ids),
        "label_column": label_column,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "cross_router_summary": cross_router_summary,
        "standard_summary": standard_summary,
        "comparison_rows": comparison_table.to_dict(orient="records"),
        "class_coverage": class_coverage.to_dict(orient="records"),
        "router_rows": router_summary.to_dict(orient="records"),
        "prediction_extras": extras,
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    LOGGER.info("Wrote cross-router summary to %s", summary_path)
    LOGGER.info("Wrote cross-router per-class metrics to %s", per_class_path)
    LOGGER.info("Wrote comparison table to %s", comparison_path)
    LOGGER.info("Wrote per-class comparison table to %s", per_class_comparison_path)
    LOGGER.info("Wrote class coverage table to %s", class_coverage_path)
    LOGGER.info("Wrote cross-router router summary to %s", router_summary_path)
    LOGGER.info("Wrote cross-router router/class metrics to %s", router_class_metrics_path)
    LOGGER.info("Wrote cross-router router mix summary to %s", router_mix_path)
    LOGGER.info("Wrote Task 3.4(b) figure to %s", comparison_figure_path)
    LOGGER.info("Wrote Task 3.4(b) report to %s", report_path)
    LOGGER.info("Wrote Task 3.4(b) summary JSON to %s", summary_json_path)
    LOGGER.info("Serialized the cross-router model to %s", model_path)


if __name__ == "__main__":
    main()
