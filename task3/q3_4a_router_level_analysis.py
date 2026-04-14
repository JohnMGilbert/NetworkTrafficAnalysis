"""Question 3.4(a): evaluate the best Task 3 model per router and per class."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

try:
    import joblib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing optional dependency 'joblib'. Install project dependencies with "
        "'python3 -m pip install -r requirements.txt' and rerun q3_4a_router_level_analysis.py."
    ) from exc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory
from src.common.seed import set_global_seed
from task3.q3_1a_baselines import (
    DEFAULT_EVALUATION_CORPUS_DIR,
    SOURCE_FILE_COLUMN,
    SOURCE_ROUTER_COLUMN,
    DEFAULT_LABELED_DATA_DIR,
    DEFAULT_TRAINING_CORPUS_DIR,
    drop_missing_labels,
    infer_label_column,
    load_datasets_from_args,
    parse_router_id_from_path,
    read_dataset,
    render_table,
)
from task3.q3_3_advanced_model import add_engineered_features, predict_hybrid_model


LOGGER = logging.getLogger("task3.q3_4a")
ROW_SPLIT_SUFFIX = " [row-split]"
ROUTER_PATTERN = re.compile(r"dataset-(\d+)", flags=re.IGNORECASE)
ROUTER_ORDER = [f"D{router_id}" for router_id in range(1, 11)]
CLASS_COLORMAP = plt.get_cmap("tab20")
Q3_2_STRATEGY_TO_SLUG = {
    "No Balancing": "baseline",
    "SMOTE Oversampling": "smote",
    "Class Weighting": "class_weight",
    "SMOTE + Undersampling Hybrid": "hybrid",
}


@dataclass(frozen=True)
class SelectedModel:
    artifact_kind: str
    display_name: str
    source_tag: str
    model_path: Path
    f1_macro: float
    accuracy: float
    label_column: str | None = None
    class_names: tuple[str, ...] = ()


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
        help=(
            "Directory searched for labeled train/test files when explicit paths are omitted "
            "and dedicated training/evaluation corpora are unavailable."
        ),
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Optional explicit label column name.",
    )
    parser.add_argument(
        "--training-corpus-dir",
        type=Path,
        default=DEFAULT_TRAINING_CORPUS_DIR,
        help="Directory containing the labeled training corpus used when explicit train/test paths are omitted.",
    )
    parser.add_argument(
        "--evaluation-corpus-dir",
        type=Path,
        default=DEFAULT_EVALUATION_CORPUS_DIR,
        help="Directory containing the labeled evaluation corpus used when explicit train/test paths are omitted.",
    )
    parser.add_argument(
        "--labeled-data-dir",
        type=Path,
        default=DEFAULT_LABELED_DATA_DIR,
        help=(
            "Legacy fallback directory searched recursively for labeled CSV/parquet files when "
            "explicit paths, dedicated corpora, and saved split metadata are unavailable."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test-set fraction used only when reconstructing legacy row-split singleton files.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("row", "source_file", "router", "hybrid"),
        default="hybrid",
        help="How to split the legacy auto-discovered labeled corpus when a saved split manifest is unavailable.",
    )
    parser.add_argument(
        "--q3-1a-summary-json",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_1a_summary.json",
        help="Legacy Task 3.1(a) summary JSON used to reconstruct the saved hybrid test split when available.",
    )
    parser.add_argument(
        "--q3-1a-summary-csv",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_1a_baseline_summary.csv",
        help="Task 3.1(a) baseline summary CSV used as a model-selection fallback.",
    )
    parser.add_argument(
        "--q3-2a-summary-csv",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_2a_imbalance_summary.csv",
        help="Task 3.2(a) summary CSV used as a model-selection fallback.",
    )
    parser.add_argument(
        "--q3-2a-per-class-csv",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_2a_per_class_metrics.csv",
        help="Task 3.2(a) per-class CSV used to recover class names when needed.",
    )
    parser.add_argument(
        "--q3-3-summary-json",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_3_summary.json",
        help="Task 3.3 summary JSON used to recover label metadata for the advanced model.",
    )
    parser.add_argument(
        "--q3-3-summary-csv",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_3_advanced_summary.csv",
        help="Task 3.3 summary CSV used for best-model selection.",
    )
    parser.add_argument(
        "--q3-3-per-class-csv",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_3_per_class_metrics.csv",
        help="Task 3.3 per-class metrics CSV used to recover class names when needed.",
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
        help="Directory for generated Task 3.4(a) tables and reports.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "figures",
        help="Directory for generated Task 3.4(a) figures.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=CONFIG.random_seed,
        help="Random seed used for deterministic split reconstruction.",
    )
    return parser.parse_args()


def strip_row_split_suffix(entry: str) -> tuple[str, bool]:
    if entry.endswith(ROW_SPLIT_SUFFIX):
        return entry[: -len(ROW_SPLIT_SUFFIX)], True
    return entry, False


def summarize_top_classes(distribution: pd.DataFrame, router_label: str, top_n: int = 3) -> str:
    router_rows = distribution[distribution["router_label"] == router_label].sort_values(
        ["router_share", "support"],
        ascending=[False, False],
    )
    top_rows = router_rows.head(top_n)
    if top_rows.empty:
        return "No labeled flows in the evaluation slice."
    return ", ".join(
        f"{row['class_name']} ({row['router_share'] * 100:.1f}%)"
        for row in top_rows.to_dict(orient="records")
    )


def router_label_from_value(value: object) -> str:
    if pd.isna(value):
        return "Unknown"

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "Unknown"
        if stripped.upper().startswith("D") and stripped[1:].isdigit():
            return f"D{int(stripped[1:])}"
        if stripped.isdigit():
            return f"D{int(stripped)}"
        match = ROUTER_PATTERN.search(stripped)
        if match:
            return f"D{int(match.group(1))}"
        return stripped

    try:
        return f"D{int(value)}"
    except (TypeError, ValueError):
        return str(value)


def infer_router_labels(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()

    if "router_id" in enriched.columns:
        router_source = enriched["router_id"]
    elif SOURCE_ROUTER_COLUMN in enriched.columns:
        router_source = enriched[SOURCE_ROUTER_COLUMN]
    elif SOURCE_FILE_COLUMN in enriched.columns:
        router_source = enriched[SOURCE_FILE_COLUMN].astype(str).str.extract(ROUTER_PATTERN)[0]
    else:
        raise ValueError(
            "Router-level analysis requires router metadata, but no usable router column was found. "
            "Use the full labeled evaluation corpus or include `router_id` / `_source_router_id` in the test data."
        )

    enriched["router_label"] = router_source.map(router_label_from_value)
    enriched["router_sort_key"] = (
        enriched["router_label"]
        .str.extract(r"^D(\d+)$")[0]
        .fillna("999")
        .astype(int)
    )
    return enriched


def attach_source_metadata(frame: pd.DataFrame, *, relative_name: str, path: Path) -> pd.DataFrame:
    enriched = frame.copy()
    enriched[SOURCE_FILE_COLUMN] = relative_name
    enriched[SOURCE_ROUTER_COLUMN] = parse_router_id_from_path(path)
    return enriched


def can_use_saved_split_manifest(args: argparse.Namespace) -> bool:
    if not (
        args.train_path is None
        and args.test_path is None
        and args.random_seed == CONFIG.random_seed
        and abs(args.test_size - 0.2) < 1e-12
        and args.split_strategy == "hybrid"
        and args.q3_1a_summary_json.exists()
    ):
        return False

    try:
        summary_payload = json.loads(args.q3_1a_summary_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False

    test_groups = summary_payload.get("test_groups")
    return (
        str(summary_payload.get("split_strategy") or "") == "hybrid"
        and isinstance(test_groups, list)
        and bool(test_groups)
    )


def load_test_frame_from_saved_manifest(args: argparse.Namespace) -> tuple[pd.DataFrame, str, str]:
    summary_payload = json.loads(args.q3_1a_summary_json.read_text(encoding="utf-8"))
    test_groups = summary_payload.get("test_groups")
    if not isinstance(test_groups, list) or not test_groups:
        raise ValueError(
            f"Saved split manifest {args.q3_1a_summary_json} does not contain `test_groups`."
        )

    held_out_files: list[str] = []
    row_split_files: list[str] = []
    for entry in map(str, test_groups):
        relative_name, is_row_split = strip_row_split_suffix(entry)
        if is_row_split:
            row_split_files.append(relative_name)
        else:
            held_out_files.append(relative_name)

    frames: list[pd.DataFrame] = []

    for relative_name in sorted(set(held_out_files)):
        path = args.labeled_data_dir / relative_name
        file_frame = read_dataset(path)
        frames.append(attach_source_metadata(file_frame, relative_name=relative_name, path=path))

    for offset, relative_name in enumerate(sorted(set(row_split_files)), start=1):
        path = args.labeled_data_dir / relative_name
        file_frame = attach_source_metadata(
            read_dataset(path),
            relative_name=relative_name,
            path=path,
        )
        if len(file_frame) < 2:
            raise ValueError(
                f"Saved split reconstruction needs at least 2 rows in {relative_name}, found {len(file_frame)}."
            )
        test_rows = max(1, round(len(file_frame) * args.test_size))
        test_rows = min(test_rows, len(file_frame) - 1)
        sampled_test = file_frame.sample(n=test_rows, random_state=args.random_seed + offset)
        frames.append(sampled_test.reset_index(drop=True))

    if not frames:
        raise ValueError("Saved split manifest reconstruction produced an empty test frame.")

    label_column = str(summary_payload.get("label_column") or "label")
    test_source = str(summary_payload.get("test_path") or args.labeled_data_dir)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = drop_missing_labels(combined, label_column, "test")
    return combined.reset_index(drop=True), label_column, test_source


def load_test_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, str, str]:
    if can_use_saved_split_manifest(args):
        LOGGER.info("Reconstructing the legacy Task 3 hybrid test split from %s.", args.q3_1a_summary_json)
        return load_test_frame_from_saved_manifest(args)

    LOGGER.info(
        "Saved legacy split metadata is unavailable or incompatible with the requested settings. "
        "Falling back to the standard Task 3 dataset loader."
    )
    datasets = load_datasets_from_args(args)
    label_column = infer_label_column(datasets.train_frame, datasets.test_frame, args.label_column)
    test_frame = drop_missing_labels(datasets.test_frame, label_column, "test")
    return test_frame.reset_index(drop=True), label_column, datasets.test_source


def load_q3_1a_class_names(summary_json_path: Path) -> tuple[str, ...]:
    if not summary_json_path.exists():
        return ()
    payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    class_names = payload.get("class_names", [])
    if not isinstance(class_names, list):
        return ()
    return tuple(map(str, class_names))


def load_q3_3_class_names(per_class_path: Path) -> tuple[str, ...]:
    if not per_class_path.exists():
        return ()
    table = pd.read_csv(per_class_path)
    if "class_name" not in table.columns:
        return ()
    return tuple(map(str, table["class_name"].dropna().unique().tolist()))


def load_q3_2_class_names(per_class_path: Path) -> tuple[str, ...]:
    if not per_class_path.exists():
        return ()
    table = pd.read_csv(per_class_path)
    if "class_name" not in table.columns:
        return ()
    return tuple(sorted(map(str, table["class_name"].dropna().unique().tolist())))


def choose_best_available_model(args: argparse.Namespace) -> SelectedModel:
    candidates: list[SelectedModel] = []

    advanced_model_path = args.model_dir / "q3_3_hybrid_advanced_model.joblib"
    if args.q3_3_summary_csv.exists() and advanced_model_path.exists():
        advanced_summary = pd.read_csv(args.q3_3_summary_csv)
        if not advanced_summary.empty:
            row = advanced_summary.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]
            label_column = None
            if args.q3_3_summary_json.exists():
                payload = json.loads(args.q3_3_summary_json.read_text(encoding="utf-8"))
                label_column = payload.get("label_column")
            candidates.append(
                SelectedModel(
                    artifact_kind="advanced_hybrid",
                    display_name="Advanced Model",
                    source_tag="Question 3.3",
                    model_path=advanced_model_path,
                    f1_macro=float(row["f1_macro"]),
                    accuracy=float(row["accuracy"]),
                    label_column=str(label_column) if label_column is not None else None,
                    class_names=load_q3_3_class_names(args.q3_3_per_class_csv),
                )
            )

    if args.q3_2a_summary_csv.exists():
        q3_2_summary = pd.read_csv(args.q3_2a_summary_csv)
        if not q3_2_summary.empty:
            row = q3_2_summary.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]
            strategy = str(row["strategy"])
            slug = Q3_2_STRATEGY_TO_SLUG.get(strategy)
            if slug is not None:
                model_path = args.model_dir / f"q3_2a_randomforest_{slug}.joblib"
                if model_path.exists():
                    candidates.append(
                        SelectedModel(
                            artifact_kind="imbalance_random_forest",
                            display_name=f"RandomForest ({strategy})",
                            source_tag="Question 3.2(a)",
                            model_path=model_path,
                            f1_macro=float(row["f1_macro"]),
                            accuracy=float(row["accuracy"]),
                            class_names=load_q3_2_class_names(args.q3_2a_per_class_csv),
                        )
                    )

    if args.q3_1a_summary_csv.exists():
        q3_1_summary = pd.read_csv(args.q3_1a_summary_csv)
        if not q3_1_summary.empty:
            row = q3_1_summary.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]
            model_name = str(row["model"])
            slug = model_name.lower()
            model_path = args.model_dir / f"q3_1a_{slug}_baseline.joblib"
            if model_path.exists():
                candidates.append(
                    SelectedModel(
                        artifact_kind="baseline_pipeline",
                        display_name=f"{model_name} baseline",
                        source_tag="Question 3.1(a)",
                        model_path=model_path,
                        f1_macro=float(row["f1_macro"]),
                        accuracy=float(row["accuracy"]),
                        class_names=load_q3_1a_class_names(args.q3_1a_summary_json),
                    )
                )

    if not candidates:
        raise FileNotFoundError(
            "No usable Task 3 model artifacts were found. Run Question 3.1(a), 3.2(a), or 3.3 first."
        )

    return max(candidates, key=lambda candidate: (candidate.f1_macro, candidate.accuracy))


def ensure_model_ready_frame(frame: pd.DataFrame, selected_model: SelectedModel, artifact: object) -> pd.DataFrame:
    if selected_model.artifact_kind != "advanced_hybrid":
        return frame

    prepared = frame.copy()
    feature_columns = list(artifact["feature_columns"])
    if any(column not in prepared.columns for column in feature_columns):
        prepared = add_engineered_features(prepared)
    missing_columns = [column for column in feature_columns if column not in prepared.columns]
    if missing_columns:
        raise ValueError(
            "The advanced-model evaluation frame is missing required feature columns: "
            f"{missing_columns}."
        )
    return prepared


def resolve_label_column(
    test_frame: pd.DataFrame,
    inferred_label_column: str,
    selected_model: SelectedModel,
) -> str:
    if selected_model.label_column is not None:
        requested = selected_model.label_column.strip().lower().replace(" ", "_").replace("/", "_")
        if requested in test_frame.columns:
            return requested
    return inferred_label_column


def predict_with_selected_model(
    selected_model: SelectedModel,
    artifact: object,
    frame: pd.DataFrame,
    label_column: str,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, object]]:
    if selected_model.artifact_kind == "advanced_hybrid":
        predictions, extras = predict_hybrid_model(artifact, frame)
        class_names = selected_model.class_names or tuple(
            sorted(
                set(map(str, frame[label_column].dropna().astype(str).unique().tolist()))
                | set(map(str, np.unique(predictions).tolist()))
            )
        )
        return np.asarray(predictions, dtype=object), tuple(map(str, class_names)), extras

    if selected_model.artifact_kind == "imbalance_random_forest":
        feature_columns = list(artifact["feature_columns"])
        transformed = artifact["imputer"].transform(frame[feature_columns]).astype(np.float32, copy=False)
        predictions = artifact["classifier"].predict(transformed)
        class_names = selected_model.class_names or tuple(
            sorted(
                set(map(str, frame[label_column].dropna().astype(str).unique().tolist()))
                | set(map(str, np.unique(predictions).tolist()))
            )
        )
        return np.asarray(predictions, dtype=object), tuple(map(str, class_names)), {}

    if selected_model.artifact_kind == "baseline_pipeline":
        raw_predictions = artifact.predict(frame)
        if not selected_model.class_names:
            raise ValueError(
                "The Task 3.1(a) fallback needs class names from q3_1a_summary.json to decode predictions."
            )
        class_names = tuple(map(str, selected_model.class_names))
        decoded_predictions = np.asarray(
            [class_names[int(prediction)] for prediction in raw_predictions],
            dtype=object,
        )
        return decoded_predictions, class_names, {}

    raise ValueError(f"Unsupported model artifact kind: {selected_model.artifact_kind}")


def build_class_distribution(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    distribution = (
        frame.groupby(["router_label", "router_sort_key", label_column], dropna=False)
        .size()
        .reset_index(name="support")
        .rename(columns={label_column: "class_name"})
    )
    distribution["router_total_rows"] = distribution.groupby("router_label")["support"].transform("sum")
    distribution["router_share"] = distribution["support"] / distribution["router_total_rows"]
    return distribution.sort_values(
        ["router_sort_key", "support", "class_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def summarize_router_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    class_names: tuple[str, ...],
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    router_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []

    for router_label, group in frame.groupby("router_label", sort=False):
        router_index = group.index.to_numpy()
        y_true = group[label_column].astype(str)
        y_pred = predictions[router_index]
        accuracy = float(accuracy_score(y_true, y_pred))
        _, _, weighted_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        precision, recall, f1_scores, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(class_names),
            zero_division=0,
        )
        observed_mask = support > 0
        observed_f1 = f1_scores[observed_mask]
        macro_f1_present = float(observed_f1.mean()) if observed_f1.size else 0.0

        observed_pairs = [
            {
                "class_name": class_names[index],
                "f1_score": float(f1_scores[index]),
                "support": int(support[index]),
            }
            for index in range(len(class_names))
            if int(support[index]) > 0
        ]
        hardest_pair = min(observed_pairs, key=lambda row: (row["f1_score"], row["support"], row["class_name"]))
        easiest_pair = max(observed_pairs, key=lambda row: (row["f1_score"], row["support"], row["class_name"]))
        class_counts = y_true.value_counts()
        dominant_class = str(class_counts.index[0])
        dominant_share = float(class_counts.iloc[0] / len(group))

        router_rows.append(
            {
                "router_label": router_label,
                "router_sort_key": int(group["router_sort_key"].iloc[0]),
                "rows": int(len(group)),
                "observed_class_count": int(observed_mask.sum()),
                "accuracy": round(accuracy, 6),
                "weighted_f1": round(float(weighted_f1), 6),
                "macro_f1_present_classes": round(macro_f1_present, 6),
                "dominant_class": dominant_class,
                "dominant_class_share": round(dominant_share, 6),
                "best_observed_class": easiest_pair["class_name"],
                "best_observed_class_f1": round(easiest_pair["f1_score"], 6),
                "worst_observed_class": hardest_pair["class_name"],
                "worst_observed_class_f1": round(hardest_pair["f1_score"], 6),
            }
        )

        for class_index, class_name in enumerate(class_names):
            per_class_rows.append(
                {
                    "router_label": router_label,
                    "router_sort_key": int(group["router_sort_key"].iloc[0]),
                    "class_name": class_name,
                    "precision": round(float(precision[class_index]), 6),
                    "recall": round(float(recall[class_index]), 6),
                    "f1_score": round(float(f1_scores[class_index]), 6),
                    "support": int(support[class_index]),
                    "present_in_router": bool(support[class_index] > 0),
                }
            )

    summary = pd.DataFrame(router_rows).sort_values("router_sort_key").reset_index(drop=True)
    per_class = pd.DataFrame(per_class_rows).sort_values(
        ["router_sort_key", "class_name"],
        ascending=[True, True],
    ).reset_index(drop=True)
    return summary, per_class


def build_f1_matrix(per_class_metrics: pd.DataFrame) -> pd.DataFrame:
    observed = per_class_metrics.copy()
    observed.loc[~observed["present_in_router"], "f1_score"] = np.nan
    matrix = observed.pivot(
        index="router_label",
        columns="class_name",
        values="f1_score",
    )
    router_index = list(ROUTER_ORDER) + [router for router in matrix.index if router not in ROUTER_ORDER]
    matrix = matrix.reindex(router_index)
    matrix.index.name = "router_label"
    return matrix


def build_router_mix_summary(distribution: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for router_label, group in distribution.groupby("router_label", sort=False):
        top_classes = summarize_top_classes(distribution, router_label)
        rows.append(
            {
                "router_label": router_label,
                "router_sort_key": int(group["router_sort_key"].iloc[0]),
                "rows": int(group["support"].sum()),
                "observed_class_count": int((group["support"] > 0).sum()),
                "top_classes": top_classes,
            }
        )
    return pd.DataFrame(rows).sort_values("router_sort_key").reset_index(drop=True)


def build_hardest_pairs_table(per_class_metrics: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    observed = per_class_metrics[per_class_metrics["present_in_router"]].copy()
    return observed.sort_values(
        ["f1_score", "support", "router_sort_key", "class_name"],
        ascending=[True, True, True, True],
    ).head(limit).reset_index(drop=True)


def create_router_class_heatmap(matrix: pd.DataFrame, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 7))
    display_values = matrix.to_numpy(dtype=float)
    heatmap_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "router_f1_yellow_green",
        ["#fff7bc", "#c7e9b4", "#31a354"],
    )
    heatmap_cmap.set_bad(color="#f2f2f2")
    image = ax.imshow(
        np.ma.masked_invalid(display_values),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap=heatmap_cmap,
    )

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=10)
    ax.set_xlabel("Class")
    ax.set_ylabel("Router")
    ax.set_title("Question 3.4(a): Router-by-Class F1 Heatmap", fontsize=14, pad=14)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = display_values[row_index, column_index]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            text_color = "white" if not np.isnan(value) and value >= 0.82 else "#1f1f1f"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.02)
    colorbar.set_label("F1-score", rotation=270, labelpad=16)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_router_distribution_chart(distribution: pd.DataFrame, figure_path: Path) -> None:
    pivot = (
        distribution.pivot(index="router_label", columns="class_name", values="router_share")
        .fillna(0.0)
    )
    router_index = list(ROUTER_ORDER) + [router for router in pivot.index if router not in ROUTER_ORDER]
    pivot = pivot.reindex(router_index)

    fig, ax = plt.subplots(figsize=(15, 7))
    bottoms = np.zeros(len(pivot), dtype=float)
    colors = [CLASS_COLORMAP(index % CLASS_COLORMAP.N) for index in range(len(pivot.columns))]

    for color, class_name in zip(colors, pivot.columns.tolist(), strict=True):
        values = pivot[class_name].to_numpy(dtype=float)
        ax.bar(
            pivot.index.tolist(),
            values,
            bottom=bottoms,
            label=class_name,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms += values

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Share of router test flows", fontsize=12)
    ax.set_xlabel("Router", fontsize=12)
    ax.set_title("Question 3.4(a): Test-Set Class Distribution per Router", fontsize=14, pad=14)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(
    *,
    selected_model: SelectedModel,
    test_source: str,
    label_column: str,
    router_summary: pd.DataFrame,
    router_mix_summary: pd.DataFrame,
    hardest_pairs: pd.DataFrame,
    f1_matrix: pd.DataFrame,
    distribution: pd.DataFrame,
    heatmap_path: Path,
    distribution_figure_path: Path,
    report_path: Path,
    missing_routers: list[str],
) -> None:
    easiest_router = router_summary.sort_values(
        ["macro_f1_present_classes", "weighted_f1", "accuracy"],
        ascending=False,
    ).iloc[0]
    hardest_router = router_summary.sort_values(
        ["macro_f1_present_classes", "weighted_f1", "accuracy"],
        ascending=True,
    ).iloc[0]

    easiest_top_classes = summarize_top_classes(distribution, str(easiest_router["router_label"]))
    hardest_top_classes = summarize_top_classes(distribution, str(hardest_router["router_label"]))
    hardest_router_pairs = hardest_pairs[hardest_pairs["router_label"] == hardest_router["router_label"]].head(3)
    hardest_router_pair_summary = ", ".join(
        f"{row['class_name']} ({row['f1_score']:.3f}, n={row['support']})"
        for row in hardest_router_pairs.to_dict(orient="records")
    )

    lines = [
        "# Task 3.4(a) Router-Level Analysis",
        "",
        "## Evaluation Setup",
        f"- Selected model: `{selected_model.display_name}` from {selected_model.source_tag}",
        f"- Reference macro F1 used for selection: {selected_model.f1_macro:.6f}",
        f"- Reference accuracy used for selection: {selected_model.accuracy:.6f}",
        f"- Test dataset source: `{test_source}`",
        f"- Label column: `{label_column}`",
        f"- Routers evaluated: {', '.join(router_summary['router_label'].tolist())}",
    ]
    if missing_routers:
        lines.append(
            "- Routers absent from the current evaluation set: "
            + ", ".join(missing_routers)
            + ". Their matrix rows are shown as `NA`."
        )
    lines.extend(
        [
            "",
            "## Router Summary",
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
            "## Interpretation",
            (
                f"- Easiest router: `{easiest_router['router_label']}` with macro F1 across present classes "
                f"of {float(easiest_router['macro_f1_present_classes']):.6f}. "
                f"Its test mix is concentrated in {easiest_top_classes}."
            ),
            (
                f"- Hardest router: `{hardest_router['router_label']}` with macro F1 across present classes "
                f"of {float(hardest_router['macro_f1_present_classes']):.6f}. "
                f"Its test mix is {hardest_top_classes}."
            ),
            (
                f"- The lowest-scoring router/class combinations on `{hardest_router['router_label']}` are "
                f"{hardest_router_pair_summary}."
            ),
        ]
    )
    if missing_routers:
        lines.append(
            "- Because the current evaluation set omits some routers entirely, the easiest/hardest ranking above "
            "is relative to the routers that actually appear in that evaluation slice."
        )
    lines.extend(
        [
            (
                "- Across routers, the easiest cases are the ones dominated by high-volume classes such as "
                "Normal, DDoS, or DoS variants, while lower macro F1 is driven by routers that contain rare "
                "web-injection traffic or a more heterogeneous attack mix."
            ),
            "",
            "## Router Mix Summary",
            render_table(router_mix_summary[["router_label", "rows", "observed_class_count", "top_classes"]]),
            "",
            "## Hardest Router/Class Cells",
            render_table(hardest_pairs[["router_label", "class_name", "support", "f1_score", "precision", "recall"]]),
            "",
            "## Router-by-Class F1 Matrix",
            render_table(f1_matrix.reset_index().rename(columns={"router_label": "router"})),
            "",
            "## Figures",
            f"- Router/class F1 heatmap: `{heatmap_path}`",
            f"- Router class-distribution chart: `{distribution_figure_path}`",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_global_seed(args.random_seed)

    table_dir = ensure_directory(args.table_dir)
    figure_dir = ensure_directory(args.figure_dir)

    test_frame, inferred_label_column, test_source = load_test_frame(args)
    selected_model = choose_best_available_model(args)
    LOGGER.info(
        "Selected %s from %s (macro F1 %.6f, accuracy %.6f).",
        selected_model.display_name,
        selected_model.source_tag,
        selected_model.f1_macro,
        selected_model.accuracy,
    )

    artifact = joblib.load(selected_model.model_path)
    test_frame = ensure_model_ready_frame(test_frame, selected_model, artifact)
    label_column = resolve_label_column(test_frame, inferred_label_column, selected_model)
    test_frame = drop_missing_labels(test_frame, label_column, "test")
    test_frame = infer_router_labels(test_frame).reset_index(drop=True)

    predictions, class_names, extras = predict_with_selected_model(
        selected_model,
        artifact,
        test_frame,
        label_column,
    )
    predictions = np.asarray(predictions, dtype=object)

    router_distribution = build_class_distribution(test_frame, label_column)
    router_summary, router_class_metrics = summarize_router_metrics(
        test_frame,
        predictions,
        class_names,
        label_column,
    )
    router_mix_summary = build_router_mix_summary(router_distribution)
    hardest_pairs = build_hardest_pairs_table(router_class_metrics)
    f1_matrix = build_f1_matrix(router_class_metrics)
    missing_routers = [router for router in ROUTER_ORDER if router not in router_summary["router_label"].tolist()]
    if missing_routers:
        LOGGER.warning(
            "The current evaluation split contains no rows for routers: %s",
            ", ".join(missing_routers),
        )

    router_summary_path = table_dir / "q3_4a_router_summary.csv"
    router_class_metrics_path = table_dir / "q3_4a_router_class_metrics.csv"
    router_f1_matrix_path = table_dir / "q3_4a_router_class_f1_matrix.csv"
    router_distribution_path = table_dir / "q3_4a_router_class_distribution.csv"
    router_mix_summary_path = table_dir / "q3_4a_router_mix_summary.csv"
    hardest_pairs_path = table_dir / "q3_4a_hardest_router_class_pairs.csv"
    report_path = table_dir / "q3_4a_report.md"
    summary_json_path = table_dir / "q3_4a_summary.json"
    heatmap_path = figure_dir / "q3_4a_router_class_f1_heatmap.png"
    distribution_figure_path = figure_dir / "q3_4a_router_class_distribution.png"

    create_router_class_heatmap(f1_matrix, heatmap_path)
    create_router_distribution_chart(router_distribution, distribution_figure_path)

    router_summary.to_csv(router_summary_path, index=False)
    router_class_metrics.to_csv(router_class_metrics_path, index=False)
    f1_matrix.to_csv(router_f1_matrix_path, index=True)
    router_distribution.to_csv(router_distribution_path, index=False)
    router_mix_summary.to_csv(router_mix_summary_path, index=False)
    hardest_pairs.to_csv(hardest_pairs_path, index=False)

    write_report(
        selected_model=selected_model,
        test_source=test_source,
        label_column=label_column,
        router_summary=router_summary,
        router_mix_summary=router_mix_summary,
        hardest_pairs=hardest_pairs,
        f1_matrix=f1_matrix,
        distribution=router_distribution,
        heatmap_path=heatmap_path,
        distribution_figure_path=distribution_figure_path,
        report_path=report_path,
        missing_routers=missing_routers,
    )

    summary_payload = {
        "selected_model": {
            "display_name": selected_model.display_name,
            "source_tag": selected_model.source_tag,
            "artifact_kind": selected_model.artifact_kind,
            "model_path": str(selected_model.model_path),
            "reference_f1_macro": selected_model.f1_macro,
            "reference_accuracy": selected_model.accuracy,
        },
        "test_source": test_source,
        "label_column": label_column,
        "routers": router_summary.to_dict(orient="records"),
        "hardest_pairs": hardest_pairs.to_dict(orient="records"),
        "class_names": list(class_names),
        "missing_routers": missing_routers,
        "prediction_extras": {
            "override_rows": int(np.asarray(extras.get("override_mask", np.zeros(len(test_frame), dtype=bool))).sum())
            if extras
            else 0,
        },
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    LOGGER.info("Wrote router summary to %s", router_summary_path)
    LOGGER.info("Wrote router/class metrics to %s", router_class_metrics_path)
    LOGGER.info("Wrote router/class F1 matrix to %s", router_f1_matrix_path)
    LOGGER.info("Wrote router class distribution to %s", router_distribution_path)
    LOGGER.info("Wrote router mix summary to %s", router_mix_summary_path)
    LOGGER.info("Wrote hardest router/class pairs to %s", hardest_pairs_path)
    LOGGER.info("Wrote Task 3.4(a) figures to %s and %s", heatmap_path, distribution_figure_path)
    LOGGER.info("Wrote Task 3.4(a) report to %s", report_path)
    LOGGER.info("Wrote Task 3.4(a) summary JSON to %s", summary_json_path)


if __name__ == "__main__":
    main()
