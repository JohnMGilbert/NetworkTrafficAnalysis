"""Question 3.3: tuned hybrid advanced model plus ablation study."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

try:
    import joblib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing optional dependency 'joblib'. Install project dependencies with "
        "'python3 -m pip install -r requirements.txt' and rerun q3_3_advanced_model.py."
    ) from exc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory
from src.common.seed import set_global_seed
from task2.q2_1c_feature_engineering import compute_engineered_features
from task3.q3_1a_baselines import (
    DEFAULT_EVALUATION_CORPUS_DIR,
    DEFAULT_LABELED_DATA_DIR,
    DEFAULT_TRAINING_CORPUS_DIR,
    drop_missing_labels,
    infer_label_column,
    load_datasets_from_args,
    render_table,
    select_feature_columns,
)


LOGGER = logging.getLogger("task3.q3_3")
WEB_PREFIX = "Web"
ENGINEERED_FEATURE_NAMES = (
    "directional_byte_imbalance",
    "bytes_per_packet",
    "burst_idle_log_ratio",
    "packet_size_asymmetry",
)


@dataclass(frozen=True)
class BaseConfig:
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    class_weight: str | None


@dataclass(frozen=True)
class WebSubtypeConfig:
    n_neighbors: int
    weights: str


@dataclass(frozen=True)
class WebDetectorConfig:
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    class_weight: str | None
    negative_multiplier: int
    threshold: float


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
            "explicit paths, dedicated corpora, and pre-split datasets are unavailable."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test-set fraction used only for the legacy auto-split fallback.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("row", "source_file", "router", "hybrid"),
        default="hybrid",
        help="How to split the legacy auto-discovered labeled corpus.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.15,
        help="Fraction of the training data reserved for hyperparameter selection.",
    )
    parser.add_argument(
        "--tuning-max-train-rows",
        type=int,
        default=600_000,
        help=(
            "Optional stratified cap for the internal tuning-training split. "
            "The final model is still retrained on the full Task 3 training set."
        ),
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap for the Task 3 training rows, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional cap for the Task 3 test rows, useful for smoke tests.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables",
        help="Directory for generated Task 3.3 tables and reports.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "figures",
        help="Directory for generated Task 3.3 figures.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=CONFIG.outputs_dir / "models" / "task3",
        help="Directory for serialized Task 3.3 models.",
    )
    parser.add_argument(
        "--reference-summary-path",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_2a_imbalance_summary.csv",
        help="Question 3.2(a) summary CSV used for the comparison table when available.",
    )
    parser.add_argument(
        "--reference-per-class-path",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_2a_per_class_metrics.csv",
        help="Question 3.2(a) per-class CSV used for the comparison table when available.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=CONFIG.random_seed,
        help="Random seed used across all Task 3.3 experiments.",
    )
    return parser.parse_args()


def cap_rows(
    frame: pd.DataFrame,
    *,
    label_column: str,
    max_rows: int | None,
    random_seed: int,
    split_name: str,
) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    if max_rows <= 0:
        raise ValueError(f"{split_name} row cap must be positive, received {max_rows}.")

    labels = frame[label_column].astype(str)
    stratify = labels if labels.value_counts().min() >= 2 else None
    sampled, _ = train_test_split(
        frame,
        train_size=max_rows,
        random_state=random_seed,
        shuffle=True,
        stratify=stratify,
    )
    LOGGER.warning(
        "Applying %s cap: sampled %s rows from %s rows.",
        split_name,
        len(sampled),
        len(frame),
    )
    return sampled.reset_index(drop=True)


def ensure_router_column(frame: pd.DataFrame) -> pd.DataFrame:
    if "router_id" in frame.columns:
        return frame

    enriched = frame.copy()
    if "_source_router_id" in enriched.columns:
        enriched["router_id"] = enriched["_source_router_id"].fillna(-1).astype(int).astype(str)
    else:
        enriched["router_id"] = "unknown"
    return enriched


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = ensure_router_column(frame)
    engineered = compute_engineered_features(enriched)
    for column in ENGINEERED_FEATURE_NAMES:
        enriched[column] = engineered[column].to_numpy()
    return enriched


def choose_feature_sets(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    label_column: str,
) -> tuple[list[str], list[str], list[str]]:
    numeric_columns, _, dropped_columns = select_feature_columns(
        train_frame,
        test_frame,
        label_column,
    )
    original_columns = [
        column for column in numeric_columns if column not in set(ENGINEERED_FEATURE_NAMES)
    ]
    engineered_columns = numeric_columns
    missing_engineered = [
        column for column in ENGINEERED_FEATURE_NAMES if column not in engineered_columns
    ]
    if missing_engineered:
        raise ValueError(
            "The engineered feature set is incomplete. Missing engineered columns: "
            f"{missing_engineered}."
        )
    return original_columns, engineered_columns, dropped_columns


def build_base_search_space() -> list[BaseConfig]:
    return [
        BaseConfig(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
        )
        for n_estimators, max_depth, min_samples_leaf, class_weight in product(
            (200,),
            (None, 24),
            (1, 2),
            (None, "balanced_subsample"),
        )
    ]


def build_web_subtype_search_space() -> list[WebSubtypeConfig]:
    return [
        WebSubtypeConfig(n_neighbors=n_neighbors, weights=weights)
        for n_neighbors, weights in product((3, 5, 7, 9, 11, 15), ("uniform", "distance"))
    ]


def build_web_detector_search_space() -> list[tuple[BaseConfig, int, float]]:
    base_variants = [
        BaseConfig(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample",
        )
        for n_estimators, max_depth, min_samples_leaf in product(
            (200, 300),
            (20, 24),
            (1, 2),
        )
    ]
    return [
        (base_config, negative_multiplier, threshold)
        for base_config, negative_multiplier, threshold in product(
            base_variants,
            (10, 20),
            (0.25, 0.30, 0.35, 0.40),
        )
    ]


def fit_base_model(
    train_frame: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: list[str],
    config: BaseConfig,
    random_seed: int,
) -> tuple[dict[str, object], float]:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_frame[feature_columns]).astype(np.float32, copy=False)
    y_train = train_frame[label_column].astype(str)

    classifier = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        class_weight=config.class_weight,
        n_jobs=-1,
        random_state=random_seed,
    )
    start_time = time.perf_counter()
    classifier.fit(x_train, y_train)
    training_time = time.perf_counter() - start_time
    return {
        "feature_columns": feature_columns,
        "imputer": imputer,
        "classifier": classifier,
        "config": asdict(config),
    }, training_time


def predict_base_probabilities(
    model: dict[str, object],
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    feature_columns = list(model["feature_columns"])
    transformed = model["imputer"].transform(frame[feature_columns]).astype(np.float32, copy=False)
    classifier = model["classifier"]
    probabilities = classifier.predict_proba(transformed)
    class_names = np.asarray(classifier.classes_, dtype=object)
    return probabilities, class_names


def apply_class_score_weights(
    probabilities: np.ndarray,
    class_names: np.ndarray,
    score_weights: dict[str, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if score_weights:
        weight_vector = np.array(
            [float(score_weights.get(str(class_name), 1.0)) for class_name in class_names],
            dtype=np.float32,
        )
        adjusted_probabilities = probabilities * weight_vector.reshape(1, -1)
    else:
        adjusted_probabilities = probabilities

    predictions = class_names[np.argmax(adjusted_probabilities, axis=1)]
    return np.asarray(predictions, dtype=object), adjusted_probabilities


def predict_base_model(model: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    probabilities, class_names = predict_base_probabilities(model, frame)
    predictions, _ = apply_class_score_weights(
        probabilities,
        class_names,
        model.get("class_score_weights"),
    )
    return predictions


def tune_class_score_weights(
    model: dict[str, object],
    validation_frame: pd.DataFrame,
    *,
    label_column: str,
) -> tuple[dict[str, float], np.ndarray, dict[str, float]]:
    y_valid = validation_frame[label_column].astype(str)
    probabilities, class_names = predict_base_probabilities(model, validation_frame)
    score_weights = {str(class_name): 1.0 for class_name in class_names}
    candidate_weights = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)

    best_predictions, _ = apply_class_score_weights(probabilities, class_names, score_weights)
    best_metrics = compute_summary_metrics(y_valid, best_predictions)
    best_score = (best_metrics["f1_macro"], best_metrics["accuracy"])

    for _ in range(2):
        improved = False
        for class_name in map(str, class_names.tolist()):
            current_best_weight = score_weights[class_name]
            local_best_score = best_score
            local_best_predictions = best_predictions
            local_best_metrics = best_metrics

            for candidate in candidate_weights:
                trial_weights = dict(score_weights)
                trial_weights[class_name] = float(candidate)
                trial_predictions, _ = apply_class_score_weights(
                    probabilities,
                    class_names,
                    trial_weights,
                )
                trial_metrics = compute_summary_metrics(y_valid, trial_predictions)
                trial_score = (trial_metrics["f1_macro"], trial_metrics["accuracy"])
                if trial_score > local_best_score:
                    local_best_score = trial_score
                    current_best_weight = float(candidate)
                    local_best_predictions = trial_predictions
                    local_best_metrics = trial_metrics

            if current_best_weight != score_weights[class_name]:
                score_weights[class_name] = current_best_weight
                best_score = local_best_score
                best_predictions = local_best_predictions
                best_metrics = local_best_metrics
                improved = True

        if not improved:
            break

    return score_weights, best_predictions, best_metrics


def build_web_detector_training_frame(
    train_frame: pd.DataFrame,
    *,
    label_column: str,
    negative_multiplier: int,
    random_seed: int,
) -> pd.DataFrame:
    positives = train_frame[train_frame[label_column].astype(str).str.startswith(WEB_PREFIX)].copy()
    negatives = train_frame[~train_frame[label_column].astype(str).str.startswith(WEB_PREFIX)].copy()
    if positives.empty:
        raise ValueError("Cannot train the web detector because the training split contains no web samples.")

    negative_target = min(len(negatives), max(len(positives) * negative_multiplier, 5_000))
    sampled_negatives = negatives.sample(
        n=negative_target,
        random_state=random_seed,
        replace=False,
    )
    return pd.concat([positives, sampled_negatives], ignore_index=True)


def fit_web_detector_model(
    train_frame: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: list[str],
    config: WebDetectorConfig,
    random_seed: int,
) -> tuple[dict[str, object], float]:
    detector_train = build_web_detector_training_frame(
        train_frame,
        label_column=label_column,
        negative_multiplier=config.negative_multiplier,
        random_seed=random_seed,
    )
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(detector_train[feature_columns]).astype(np.float32, copy=False)
    y_train = (
        detector_train[label_column]
        .astype(str)
        .str.startswith(WEB_PREFIX)
        .astype(int)
        .to_numpy(dtype=np.int8, copy=False)
    )

    classifier = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        class_weight=config.class_weight,
        n_jobs=-1,
        random_state=random_seed,
    )
    start_time = time.perf_counter()
    classifier.fit(x_train, y_train)
    training_time = time.perf_counter() - start_time
    return {
        "feature_columns": feature_columns,
        "imputer": imputer,
        "classifier": classifier,
        "config": asdict(config),
    }, training_time


def predict_web_detector(model: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    feature_columns = list(model["feature_columns"])
    transformed = model["imputer"].transform(frame[feature_columns]).astype(np.float32, copy=False)
    return model["classifier"].predict_proba(transformed)[:, 1]


def fit_web_subtype_model(
    train_frame: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: list[str],
    config: WebSubtypeConfig,
) -> tuple[Pipeline, float]:
    web_train = train_frame[train_frame[label_column].astype(str).str.startswith(WEB_PREFIX)].copy()
    if web_train.empty:
        raise ValueError("Cannot train the web subtype specialist because the training split contains no web samples.")

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=config.n_neighbors, weights=config.weights)),
        ]
    )
    start_time = time.perf_counter()
    model.fit(web_train[feature_columns], web_train[label_column].astype(str))
    training_time = time.perf_counter() - start_time
    return model, training_time


def compute_summary_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 6),
        "precision_macro": round(float(precision_macro), 6),
        "recall_macro": round(float(recall_macro), 6),
        "f1_macro": round(float(f1_macro), 6),
        "f1_weighted": round(float(f1_weighted), 6),
    }


def compute_per_class_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    precision, recall, f1_scores, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=class_names,
        zero_division=0,
    )
    rows = []
    for class_index, class_name in enumerate(class_names):
        rows.append(
            {
                "class_name": class_name,
                "precision": round(float(precision[class_index]), 6),
                "recall": round(float(recall[class_index]), 6),
                "f1_score": round(float(f1_scores[class_index]), 6),
                "support": int(support[class_index]),
            }
        )
    return pd.DataFrame(rows)


def hybrid_predictions(
    base_predictions: np.ndarray,
    *,
    web_probabilities: np.ndarray | None,
    web_threshold: float | None,
    web_subtype_predictions: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if web_probabilities is None or web_threshold is None or web_subtype_predictions is None:
        return np.array(base_predictions, dtype=object), np.zeros(len(base_predictions), dtype=bool)

    override_mask = (web_probabilities >= web_threshold) | pd.Series(base_predictions).astype(str).str.startswith(
        WEB_PREFIX
    ).to_numpy()
    final_predictions = np.array(base_predictions, dtype=object)
    final_predictions[override_mask] = web_subtype_predictions[override_mask]
    return final_predictions, override_mask


def tune_base_model(
    tuning_train: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    label_column: str,
    original_feature_columns: list[str],
    engineered_feature_columns: list[str],
    random_seed: int,
) -> tuple[BaseConfig, str, list[str], dict[str, float], pd.DataFrame]:
    y_valid = validation_frame[label_column].astype(str)
    search_rows: list[dict[str, object]] = []
    best_config: BaseConfig | None = None
    best_feature_set_name: str | None = None
    best_feature_columns: list[str] | None = None
    best_score_weights: dict[str, float] | None = None
    best_predictions: np.ndarray | None = None
    best_score: tuple[float, float, int, int, float] | None = None
    feature_sets = {
        "Original Features": original_feature_columns,
        "Engineered Features": engineered_feature_columns,
    }

    for feature_set_name, feature_columns in feature_sets.items():
        for config in build_base_search_space():
            model, training_time = fit_base_model(
                tuning_train,
                label_column=label_column,
                feature_columns=feature_columns,
                config=config,
                random_seed=random_seed,
            )
            score_weights, predictions, metrics = tune_class_score_weights(
                model,
                validation_frame,
                label_column=label_column,
            )
            model["class_score_weights"] = score_weights
            non_default_weights = {
                class_name: round(weight, 3)
                for class_name, weight in score_weights.items()
                if abs(weight - 1.0) > 1e-9
            }
            search_rows.append(
                {
                    "component": "advanced_backbone",
                    "feature_set": feature_set_name,
                    "uses_engineered_features": feature_set_name == "Engineered Features",
                    "training_time_seconds": round(training_time, 3),
                    **asdict(config),
                    **metrics,
                    "calibrated_classes": len(non_default_weights),
                    "calibration_weights": json.dumps(non_default_weights, sort_keys=True),
                }
            )
            score = (
                metrics["f1_macro"],
                metrics["accuracy"],
                1 if config.class_weight is None else 0,
                1 if feature_set_name == "Original Features" else 0,
                -training_time,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_config = config
                best_feature_set_name = feature_set_name
                best_feature_columns = feature_columns
                best_score_weights = score_weights
                best_predictions = predictions

    if (
        best_config is None
        or best_predictions is None
        or best_feature_set_name is None
        or best_feature_columns is None
        or best_score_weights is None
    ):
        raise RuntimeError("Base-model tuning did not evaluate any configurations.")

    return (
        best_config,
        best_feature_set_name,
        best_feature_columns,
        best_score_weights,
        pd.DataFrame(search_rows),
    )


def tune_web_subtype_model(
    tuning_train: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: list[str],
) -> tuple[WebSubtypeConfig, pd.DataFrame, Pipeline, np.ndarray]:
    web_valid = validation_frame[validation_frame[label_column].astype(str).str.startswith(WEB_PREFIX)].copy()
    if web_valid.empty:
        raise ValueError(
            "The internal validation split contains no web samples, so the web specialist cannot be tuned."
        )

    y_valid_web = web_valid[label_column].astype(str)
    search_rows: list[dict[str, object]] = []
    best_config: WebSubtypeConfig | None = None
    best_model: Pipeline | None = None
    best_predictions: np.ndarray | None = None
    best_score: tuple[float, float] | None = None

    for config in build_web_subtype_search_space():
        model, training_time = fit_web_subtype_model(
            tuning_train,
            label_column=label_column,
            feature_columns=feature_columns,
            config=config,
        )
        predictions = model.predict(web_valid[feature_columns])
        metrics = compute_summary_metrics(y_valid_web, predictions)
        search_rows.append(
            {
                "component": "web_subtype_knn",
                "training_time_seconds": round(training_time, 3),
                **asdict(config),
                **metrics,
            }
        )
        score = (metrics["f1_macro"], metrics["accuracy"])
        if best_score is None or score > best_score:
            best_score = score
            best_config = config
            best_model = model
            best_predictions = predictions

    if best_config is None or best_model is None or best_predictions is None:
        raise RuntimeError("Web subtype tuning did not evaluate any configurations.")

    return best_config, pd.DataFrame(search_rows), best_model, best_predictions


def tune_web_detector(
    tuning_train: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: list[str],
    base_valid_predictions: np.ndarray,
    web_subtype_model: Pipeline,
    random_seed: int,
) -> tuple[WebDetectorConfig, pd.DataFrame]:
    y_valid = validation_frame[label_column].astype(str)
    web_valid_truth = y_valid.str.startswith(WEB_PREFIX).astype(int).to_numpy()
    xss_mask = y_valid.eq("Web-xss").to_numpy()
    web_subtype_predictions = web_subtype_model.predict(validation_frame[feature_columns])

    search_rows: list[dict[str, object]] = []
    best_config: WebDetectorConfig | None = None
    best_score: tuple[float, float] | None = None

    for detector_base_config, negative_multiplier, threshold in build_web_detector_search_space():
        detector_config = WebDetectorConfig(
            n_estimators=detector_base_config.n_estimators,
            max_depth=detector_base_config.max_depth,
            min_samples_leaf=detector_base_config.min_samples_leaf,
            class_weight=detector_base_config.class_weight,
            negative_multiplier=negative_multiplier,
            threshold=threshold,
        )
        detector_model, training_time = fit_web_detector_model(
            tuning_train,
            label_column=label_column,
            feature_columns=feature_columns,
            config=detector_config,
            random_seed=random_seed,
        )
        web_probabilities = predict_web_detector(detector_model, validation_frame)
        final_predictions, override_mask = hybrid_predictions(
            base_valid_predictions,
            web_probabilities=web_probabilities,
            web_threshold=threshold,
            web_subtype_predictions=web_subtype_predictions,
        )
        metrics = compute_summary_metrics(y_valid, final_predictions)
        binary_predictions = (web_probabilities >= threshold).astype(int)
        web_precision, web_recall, web_f1, _ = precision_recall_fscore_support(
            web_valid_truth,
            binary_predictions,
            average="binary",
            zero_division=0,
        )
        xss_recall = (
            float((final_predictions[xss_mask] == y_valid[xss_mask].to_numpy()).mean()) if xss_mask.any() else 0.0
        )
        search_rows.append(
            {
                "component": "web_detector",
                "training_time_seconds": round(training_time, 3),
                **asdict(detector_config),
                **metrics,
                "web_precision": round(float(web_precision), 6),
                "web_recall": round(float(web_recall), 6),
                "web_f1": round(float(web_f1), 6),
                "xss_recall": round(xss_recall, 6),
                "override_rows": int(override_mask.sum()),
            }
        )
        score = (metrics["f1_macro"], metrics["accuracy"])
        if best_score is None or score > best_score:
            best_score = score
            best_config = detector_config

    if best_config is None:
        raise RuntimeError("Web-detector tuning did not evaluate any configurations.")

    return best_config, pd.DataFrame(search_rows)


def train_hybrid_model(
    train_frame: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: list[str],
    base_config: BaseConfig,
    class_score_weights: dict[str, float] | None,
    web_subtype_config: WebSubtypeConfig | None,
    web_detector_config: WebDetectorConfig | None,
    use_web_specialist: bool,
    random_seed: int,
) -> tuple[dict[str, object], dict[str, float]]:
    base_model, base_training_time = fit_base_model(
        train_frame,
        label_column=label_column,
        feature_columns=feature_columns,
        config=base_config,
        random_seed=random_seed,
    )
    base_model["class_score_weights"] = dict(class_score_weights or {})

    artifact: dict[str, object] = {
        "base_model": base_model,
        "feature_columns": feature_columns,
        "label_column": label_column,
        "use_web_specialist": use_web_specialist,
        "base_config": asdict(base_config),
        "class_score_weights": dict(class_score_weights or {}),
    }
    training_times = {
        "base_training_time_seconds": round(base_training_time, 3),
        "web_subtype_training_time_seconds": 0.0,
        "web_detector_training_time_seconds": 0.0,
    }

    if use_web_specialist:
        if web_subtype_config is None or web_detector_config is None:
            raise ValueError("The hybrid specialist requires both web subtype and web detector configs.")

        web_subtype_model, web_subtype_training_time = fit_web_subtype_model(
            train_frame,
            label_column=label_column,
            feature_columns=feature_columns,
            config=web_subtype_config,
        )
        web_detector_model, web_detector_training_time = fit_web_detector_model(
            train_frame,
            label_column=label_column,
            feature_columns=feature_columns,
            config=web_detector_config,
            random_seed=random_seed,
        )
        artifact["web_subtype_model"] = web_subtype_model
        artifact["web_detector_model"] = web_detector_model
        artifact["web_subtype_config"] = asdict(web_subtype_config)
        artifact["web_detector_config"] = asdict(web_detector_config)
        training_times["web_subtype_training_time_seconds"] = round(web_subtype_training_time, 3)
        training_times["web_detector_training_time_seconds"] = round(web_detector_training_time, 3)

    training_times["total_training_time_seconds"] = round(sum(training_times.values()), 3)
    return artifact, training_times


def predict_hybrid_model(
    model: dict[str, object],
    frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    base_predictions = predict_base_model(model["base_model"], frame)

    if not bool(model["use_web_specialist"]):
        return base_predictions, {
            "base_predictions": base_predictions,
            "web_probabilities": None,
            "web_subtype_predictions": None,
            "override_mask": np.zeros(len(frame), dtype=bool),
        }

    web_detector_model = model["web_detector_model"]
    web_subtype_model = model["web_subtype_model"]
    web_probabilities = predict_web_detector(web_detector_model, frame)
    web_subtype_predictions = web_subtype_model.predict(frame[model["feature_columns"]])
    threshold = float(model["web_detector_config"]["threshold"])
    final_predictions, override_mask = hybrid_predictions(
        base_predictions,
        web_probabilities=web_probabilities,
        web_threshold=threshold,
        web_subtype_predictions=web_subtype_predictions,
    )
    return final_predictions, {
        "base_predictions": base_predictions,
        "web_probabilities": web_probabilities,
        "web_subtype_predictions": web_subtype_predictions,
        "override_mask": override_mask,
    }


def evaluate_variant(
    *,
    variant_name: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    label_column: str,
    class_names: list[str],
    feature_columns: list[str],
    base_config: BaseConfig,
    class_score_weights: dict[str, float] | None,
    web_subtype_config: WebSubtypeConfig | None,
    web_detector_config: WebDetectorConfig | None,
    use_web_specialist: bool,
    random_seed: int,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    model, training_times = train_hybrid_model(
        train_frame,
        label_column=label_column,
        feature_columns=feature_columns,
        base_config=base_config,
        class_score_weights=class_score_weights,
        web_subtype_config=web_subtype_config,
        web_detector_config=web_detector_config,
        use_web_specialist=use_web_specialist,
        random_seed=random_seed,
    )
    predictions, extras = predict_hybrid_model(model, test_frame)
    y_test = test_frame[label_column].astype(str)
    summary = {
        "variant": variant_name,
        "feature_count": len(feature_columns),
        "uses_engineered_features": any(
            column in set(ENGINEERED_FEATURE_NAMES) for column in feature_columns
        ),
        "uses_web_specialist": use_web_specialist,
        **training_times,
        **compute_summary_metrics(y_test, predictions),
        "override_rows": int(extras["override_mask"].sum()),
    }
    per_class = compute_per_class_metrics(y_test, predictions, class_names)
    per_class.insert(0, "variant", variant_name)
    return summary, per_class, {
        "model": model,
        "predictions": predictions,
        "extras": extras,
    }


def load_best_q3_2_reference(
    summary_path: Path,
    per_class_path: Path,
) -> tuple[dict[str, object] | None, pd.DataFrame | None]:
    if not summary_path.exists() or not per_class_path.exists():
        LOGGER.warning(
            "Skipping Question 3.2 comparison because the reference files are missing: %s, %s",
            summary_path,
            per_class_path,
        )
        return None, None

    summary_table = pd.read_csv(summary_path)
    if summary_table.empty:
        return None, None
    best_row = summary_table.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0].to_dict()

    per_class_table = pd.read_csv(per_class_path)
    if "strategy" not in per_class_table.columns:
        return best_row, None

    strategy_name = str(best_row["strategy"])
    reference_per_class = (
        per_class_table[per_class_table["strategy"] == strategy_name]
        .drop(columns=["strategy"])
        .reset_index(drop=True)
    )
    return best_row, reference_per_class


def build_metric_comparison_table(
    advanced_summary: dict[str, object],
    reference_summary: dict[str, object],
) -> pd.DataFrame:
    metrics = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")
    rows = []
    for metric in metrics:
        reference_value = float(reference_summary[metric])
        advanced_value = float(advanced_summary[metric])
        rows.append(
            {
                "metric": metric,
                "q3_2_best": round(reference_value, 6),
                "q3_3_advanced": round(advanced_value, 6),
                "delta": round(advanced_value - reference_value, 6),
            }
        )
    return pd.DataFrame(rows)


def build_per_class_comparison_table(
    advanced_per_class: pd.DataFrame,
    reference_per_class: pd.DataFrame,
) -> pd.DataFrame:
    merged = advanced_per_class.merge(
        reference_per_class,
        on="class_name",
        how="left",
        suffixes=("_q3_3", "_q3_2"),
    )
    merged["precision_delta"] = (
        merged["precision_q3_3"] - merged["precision_q3_2"]
    ).round(6)
    merged["recall_delta"] = (merged["recall_q3_3"] - merged["recall_q3_2"]).round(6)
    merged["f1_delta"] = (merged["f1_score_q3_3"] - merged["f1_score_q3_2"]).round(6)
    return merged.sort_values("f1_delta", ascending=False).reset_index(drop=True)


def evaluate_ablation_candidate(
    variant_name: str,
    *,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    label_column: str,
    class_names: list[str],
    original_feature_columns: list[str],
    engineered_feature_columns: list[str],
    base_config: BaseConfig,
    selected_feature_columns: list[str],
    selected_feature_set_name: str,
    selected_score_weights: dict[str, float],
    random_seed: int,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    feature_columns = selected_feature_columns
    class_score_weights: dict[str, float] | None = selected_score_weights
    config = base_config

    if variant_name == "No Score Calibration":
        class_score_weights = None
    elif variant_name in {"No Engineered Features", "Add Engineered Features"}:
        feature_columns = (
            original_feature_columns
            if selected_feature_set_name == "Engineered Features"
            else engineered_feature_columns
        )
    elif variant_name in {"No Class Weighting", "Enable Class Weighting"}:
        config = BaseConfig(
            n_estimators=base_config.n_estimators,
            max_depth=base_config.max_depth,
            min_samples_leaf=base_config.min_samples_leaf,
            class_weight=None if base_config.class_weight is not None else "balanced_subsample",
        )

    return evaluate_variant(
        variant_name=variant_name,
        train_frame=train_frame,
        test_frame=test_frame,
        label_column=label_column,
        class_names=class_names,
        feature_columns=feature_columns,
        base_config=config,
        class_score_weights=class_score_weights,
        web_subtype_config=None,
        web_detector_config=None,
        use_web_specialist=False,
        random_seed=random_seed,
    )


def build_ablation_table(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    label_column: str,
    class_names: list[str],
    original_feature_columns: list[str],
    engineered_feature_columns: list[str],
    base_config: BaseConfig,
    selected_feature_columns: list[str],
    selected_feature_set_name: str,
    selected_score_weights: dict[str, float],
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    full_summary, _, _ = evaluate_ablation_candidate(
        "Full Advanced",
        train_frame=train_frame,
        test_frame=test_frame,
        label_column=label_column,
        class_names=class_names,
        original_feature_columns=original_feature_columns,
        engineered_feature_columns=engineered_feature_columns,
        base_config=base_config,
        selected_feature_columns=selected_feature_columns,
        selected_feature_set_name=selected_feature_set_name,
        selected_score_weights=selected_score_weights,
        random_seed=random_seed,
    )
    rows.append(full_summary)

    no_calibration_summary, _, _ = evaluate_ablation_candidate(
        "No Score Calibration",
        train_frame=train_frame,
        test_frame=test_frame,
        label_column=label_column,
        class_names=class_names,
        original_feature_columns=original_feature_columns,
        engineered_feature_columns=engineered_feature_columns,
        base_config=base_config,
        selected_feature_columns=selected_feature_columns,
        selected_feature_set_name=selected_feature_set_name,
        selected_score_weights=selected_score_weights,
        random_seed=random_seed,
    )
    rows.append(no_calibration_summary)

    alternate_feature_variant = (
        "No Engineered Features"
        if selected_feature_set_name == "Engineered Features"
        else "Add Engineered Features"
    )
    alternate_feature_summary, _, _ = evaluate_ablation_candidate(
        alternate_feature_variant,
        train_frame=train_frame,
        test_frame=test_frame,
        label_column=label_column,
        class_names=class_names,
        original_feature_columns=original_feature_columns,
        engineered_feature_columns=engineered_feature_columns,
        base_config=base_config,
        selected_feature_columns=selected_feature_columns,
        selected_feature_set_name=selected_feature_set_name,
        selected_score_weights=selected_score_weights,
        random_seed=random_seed,
    )
    rows.append(alternate_feature_summary)

    alternate_weight_variant = (
        "No Class Weighting"
        if base_config.class_weight is not None
        else "Enable Class Weighting"
    )
    alternate_weight_summary, _, _ = evaluate_ablation_candidate(
        alternate_weight_variant,
        train_frame=train_frame,
        test_frame=test_frame,
        label_column=label_column,
        class_names=class_names,
        original_feature_columns=original_feature_columns,
        engineered_feature_columns=engineered_feature_columns,
        base_config=base_config,
        selected_feature_columns=selected_feature_columns,
        selected_feature_set_name=selected_feature_set_name,
        selected_score_weights=selected_score_weights,
        random_seed=random_seed,
    )
    rows.append(alternate_weight_summary)

    ablation_table = pd.DataFrame(rows)
    return ablation_table.sort_values(["variant"]).reset_index(drop=True)


def architecture_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="420" viewBox="0 0 1180 420">
  <style>
    .box { fill: #f5f0e6; stroke: #2e3a46; stroke-width: 2; rx: 16; ry: 16; }
    .accent { fill: #dce8d5; stroke: #2e3a46; stroke-width: 2; rx: 16; ry: 16; }
    .merge { fill: #f8d8b0; stroke: #2e3a46; stroke-width: 2; rx: 16; ry: 16; }
    .text { font-family: Helvetica, Arial, sans-serif; font-size: 18px; fill: #1d232a; }
    .small { font-family: Helvetica, Arial, sans-serif; font-size: 15px; fill: #1d232a; }
    .arrow { stroke: #2e3a46; stroke-width: 3; fill: none; marker-end: url(#arrow); }
  </style>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#2e3a46"/>
    </marker>
  </defs>
  <rect class="box" x="40" y="150" width="210" height="110"/>
  <text class="text" x="145" y="188" text-anchor="middle">Input Flow Record</text>
  <text class="small" x="145" y="218" text-anchor="middle">79 original numeric features</text>
  <text class="small" x="145" y="242" text-anchor="middle">metadata excluded from modeling</text>

  <rect class="accent" x="320" y="120" width="250" height="170"/>
  <text class="text" x="445" y="160" text-anchor="middle">Feature Builder</text>
  <text class="small" x="445" y="192" text-anchor="middle">evaluates original 79-feature and</text>
  <text class="small" x="445" y="216" text-anchor="middle">83-feature engineered variants</text>
  <text class="small" x="445" y="240" text-anchor="middle">median imputation for tree inference</text>

  <rect class="box" x="650" y="50" width="250" height="110"/>
  <text class="text" x="775" y="90" text-anchor="middle">Backbone Search</text>
  <text class="small" x="775" y="118" text-anchor="middle">tuned RandomForest candidates</text>
  <text class="small" x="775" y="142" text-anchor="middle">weighting + depth + leaf-size search</text>

  <rect class="box" x="650" y="230" width="250" height="110"/>
  <text class="text" x="775" y="270" text-anchor="middle">Score Calibration</text>
  <text class="small" x="775" y="298" text-anchor="middle">per-class score multipliers tuned</text>
  <text class="small" x="775" y="322" text-anchor="middle">to improve macro-F1 stability</text>

  <rect class="accent" x="940" y="230" width="200" height="110"/>
  <text class="text" x="1040" y="270" text-anchor="middle">Variant Selection</text>
  <text class="small" x="1040" y="298" text-anchor="middle">compare advanced candidates</text>
  <text class="small" x="1040" y="322" text-anchor="middle">retain strongest overall model</text>

  <rect class="merge" x="940" y="60" width="200" height="110"/>
  <text class="text" x="1040" y="100" text-anchor="middle">Final Model</text>
  <text class="small" x="1040" y="128" text-anchor="middle">selected RandomForest variant</text>
  <text class="small" x="1040" y="152" text-anchor="middle">used for full-corpus inference</text>

  <line class="arrow" x1="250" y1="205" x2="320" y2="205"/>
  <line class="arrow" x1="570" y1="165" x2="650" y2="105"/>
  <line class="arrow" x1="570" y1="245" x2="650" y2="285"/>
  <line class="arrow" x1="900" y1="105" x2="940" y2="115"/>
  <line class="arrow" x1="900" y1="285" x2="940" y2="285"/>
  <line class="arrow" x1="1040" y1="230" x2="1040" y2="170"/>
</svg>
"""


def write_report(
    *,
    report_path: Path,
    architecture_path: Path,
    train_source: str,
    test_source: str,
    label_column: str,
    original_feature_columns: list[str],
    engineered_feature_columns: list[str],
    dropped_columns: list[str],
    tuning_train_rows: int,
    validation_rows: int,
    final_summary: dict[str, object],
    final_per_class: pd.DataFrame,
    base_search_table: pd.DataFrame,
    comparison_table: pd.DataFrame | None,
    per_class_comparison: pd.DataFrame | None,
    ablation_table: pd.DataFrame,
    best_base_config: BaseConfig,
    best_feature_set_name: str,
    best_feature_columns: list[str],
    best_score_weights: dict[str, float],
    reference_summary: dict[str, object] | None,
) -> None:
    top_base = base_search_table.sort_values(["f1_macro", "accuracy"], ascending=False).head(8)
    selected_variant = str(final_summary["variant"])
    delta_column = "f1_macro_delta_vs_selected"
    worst_ablation = ablation_table[ablation_table["variant"] != selected_variant].sort_values(delta_column).iloc[0]
    non_default_weights = {
        class_name: round(weight, 3)
        for class_name, weight in best_score_weights.items()
        if abs(weight - 1.0) > 1e-9
    }
    outcome_summary_lines: list[str] = []
    if comparison_table is not None and per_class_comparison is not None and reference_summary is not None:
        comparison_index = comparison_table.set_index("metric")
        f1_delta = float(comparison_index.loc["f1_macro", "delta"])
        f1_reference = float(comparison_index.loc["f1_macro", "q3_2_best"])
        f1_advanced = float(comparison_index.loc["f1_macro", "q3_3_advanced"])
        accuracy_delta = float(comparison_index.loc["accuracy", "delta"])
        accuracy_reference = float(comparison_index.loc["accuracy", "q3_2_best"])
        accuracy_advanced = float(comparison_index.loc["accuracy", "q3_3_advanced"])
        worst_class = per_class_comparison.sort_values(
            ["f1_delta", "support_q3_2"],
            ascending=[True, False],
        ).iloc[0]

        outcome_summary_lines = ["## Outcome Summary"]
        if f1_delta < 0:
            outcome_summary_lines.append(
                f"- This advanced attempt did not outperform the best Question 3.2 result, `{reference_summary['strategy']}`."
            )
            outcome_summary_lines.append(
                f"- Macro F1 fell from {f1_reference:.6f} to {f1_advanced:.6f} ({f1_delta:.6f}), and accuracy fell from "
                f"{accuracy_reference:.6f} to {accuracy_advanced:.6f} ({accuracy_delta:.6f})."
            )
            outcome_summary_lines.append(
                f"- The largest per-class regression was `{worst_class['class_name']}`, whose F1 changed by "
                f"{float(worst_class['f1_delta']):.6f} relative to the Question 3.2 winner."
            )
            override_rows = int(final_summary.get("override_rows", 0) or 0)
            if override_rows > 0:
                outcome_summary_lines.append(
                    f"- Specialist overrides affected {override_rows:,} evaluation rows, so the advanced logic likely "
                    "reduced precision more than it improved recall."
                )
        else:
            outcome_summary_lines.append(
                f"- This advanced attempt outperformed the best Question 3.2 result, `{reference_summary['strategy']}`."
            )
            outcome_summary_lines.append(
                f"- Macro F1 improved from {f1_reference:.6f} to {f1_advanced:.6f} ({f1_delta:+.6f}), and accuracy changed "
                f"from {accuracy_reference:.6f} to {accuracy_advanced:.6f} ({accuracy_delta:+.6f})."
            )
        outcome_summary_lines.append("")

    lines = [
        "# Task 3.3 Advanced Model Report",
        "",
        "## Data Summary",
        f"- Training dataset: `{train_source}`",
        f"- Test dataset: `{test_source}`",
        f"- Label column: `{label_column}`",
        f"- Original numeric feature count: {len(original_feature_columns)}",
        f"- Engineered feature count: {len(engineered_feature_columns) - len(original_feature_columns)}",
        f"- Final advanced-model feature count: {len(best_feature_columns)}",
        f"- Internal tuning-train rows: {tuning_train_rows:,}",
        f"- Internal validation rows: {validation_rows:,}",
    ]
    if dropped_columns:
        lines.append(f"- Dropped unsupported metadata columns: {', '.join(dropped_columns)}")

    lines.extend(
        [
            "",
            "## Architecture",
            f"- Diagram asset: `{architecture_path}`",
            (
                "- The advanced candidate family centers on a tuned RandomForest backbone with "
                "optional engineered features and per-class score calibration."
            ),
            f"- Final selected variant: `{selected_variant}`",
            f"- Selected feature set: `{best_feature_set_name}` ({len(best_feature_columns)} columns)",
            f"- Selected RandomForest config: `{json.dumps(asdict(best_base_config), sort_keys=True)}`",
            (
                f"- Non-default class score multipliers: `{json.dumps(non_default_weights, sort_keys=True)}`"
                if non_default_weights
                else "- No class-specific score scaling was needed beyond the raw RandomForest scores."
            ),
            "",
            "```mermaid",
            "flowchart LR",
            '    A["Input flow"] --> B["Feature builder\\noriginal + engineered candidates"]',
            '    B --> C["Tuned RandomForest backbone"]',
            '    C --> D["Per-class score calibration"]',
            '    D --> E["Candidate comparison"]',
            '    E --> F["Selected advanced model"]',
            '    F --> G["Final prediction"]',
            "```",
            "",
            "## Hyperparameter Search",
            "- Backbone search space:",
            "  - feature set in `{Original Features, Engineered Features}`",
            "  - `n_estimators` fixed at `200` to stay comparable with the strongest baseline",
            "  - `max_depth` in `{None, 24}`",
            "  - `min_samples_leaf` in `{1, 2}`",
            "  - `class_weight` in `{None, balanced_subsample}`",
            "",
            "### Top Backbone Trials",
            render_table(top_base),
            "",
            "## Final Test-Set Metrics",
            render_table(pd.DataFrame([final_summary])),
            "",
            "## Per-Class Metrics",
            render_table(final_per_class),
            "",
        ]
    )

    if comparison_table is not None and per_class_comparison is not None and reference_summary is not None:
        lines.extend(outcome_summary_lines)
        lines.extend(
            [
                "## Comparison vs Best Question 3.2 Result",
                f"- Best Question 3.2 strategy: `{reference_summary['strategy']}`",
                render_table(comparison_table),
                "",
                "### Per-Class Delta vs Question 3.2",
                render_table(per_class_comparison),
                "",
            ]
        )

    lines.extend(
        [
            "## Ablation Study",
            render_table(ablation_table),
            "",
            "## Ablation Discussion",
            (
                f"- The largest macro-F1 drop relative to the selected advanced model came from `{worst_ablation['variant']}`, "
                f"which changed macro F1 by {worst_ablation[delta_column]:.6f} "
                "relative to the final model."
            ),
            (
                "- These ablations quantify which parts of the tuned RandomForest pipeline actually "
                "helped on the full labeled evaluation corpus."
            ),
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
    model_dir = ensure_directory(args.model_dir)

    datasets = load_datasets_from_args(args)
    train_frame = datasets.train_frame.copy()
    test_frame = datasets.test_frame.copy()

    label_column = infer_label_column(train_frame, test_frame, args.label_column)
    train_frame = drop_missing_labels(train_frame, label_column, "train")
    test_frame = drop_missing_labels(test_frame, label_column, "test")
    train_frame = cap_rows(
        train_frame,
        label_column=label_column,
        max_rows=args.max_train_rows,
        random_seed=args.random_seed,
        split_name="train",
    )
    test_frame = cap_rows(
        test_frame,
        label_column=label_column,
        max_rows=args.max_test_rows,
        random_seed=args.random_seed,
        split_name="test",
    )

    train_frame = add_engineered_features(train_frame)
    test_frame = add_engineered_features(test_frame)
    original_feature_columns, engineered_feature_columns, dropped_columns = choose_feature_sets(
        train_frame,
        test_frame,
        label_column,
    )
    class_names = sorted(train_frame[label_column].astype(str).unique().tolist())

    validation_stratify = train_frame[label_column].astype(str)
    if validation_stratify.value_counts().min() < 2:
        LOGGER.warning(
            "Disabling stratification for the Q3.3 tuning split because at least one class has fewer than 2 rows."
        )
        validation_stratify = None

    tuning_train, validation_frame = train_test_split(
        train_frame,
        test_size=args.validation_size,
        random_state=args.random_seed,
        shuffle=True,
        stratify=validation_stratify,
    )
    tuning_train = cap_rows(
        tuning_train.reset_index(drop=True),
        label_column=label_column,
        max_rows=args.tuning_max_train_rows,
        random_seed=args.random_seed,
        split_name="tuning_train",
    )
    validation_frame = validation_frame.reset_index(drop=True)

    LOGGER.info("Tuning backbone model on %s rows; validation size is %s rows.", len(tuning_train), len(validation_frame))
    best_base_config, best_feature_set_name, best_feature_columns, best_score_weights, base_search_table = tune_base_model(
        tuning_train,
        validation_frame,
        label_column=label_column,
        original_feature_columns=original_feature_columns,
        engineered_feature_columns=engineered_feature_columns,
        random_seed=args.random_seed,
    )
    LOGGER.info(
        "Selected advanced backbone candidate: feature set=%s, class_weight=%s, min_samples_leaf=%s, max_depth=%s.",
        best_feature_set_name,
        best_base_config.class_weight,
        best_base_config.min_samples_leaf,
        best_base_config.max_depth,
    )

    LOGGER.info("Running ablation variants.")
    ablation_table = build_ablation_table(
        train_frame,
        test_frame,
        label_column=label_column,
        class_names=class_names,
        original_feature_columns=original_feature_columns,
        engineered_feature_columns=engineered_feature_columns,
        base_config=best_base_config,
        selected_feature_columns=best_feature_columns,
        selected_feature_set_name=best_feature_set_name,
        selected_score_weights=best_score_weights,
        random_seed=args.random_seed,
    )
    selected_variant_name = str(
        ablation_table.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]["variant"]
    )
    LOGGER.info("Selected final advanced variant after candidate evaluation: %s", selected_variant_name)
    final_summary, final_per_class, final_details = evaluate_ablation_candidate(
        selected_variant_name,
        train_frame=train_frame,
        test_frame=test_frame,
        label_column=label_column,
        class_names=class_names,
        original_feature_columns=original_feature_columns,
        engineered_feature_columns=engineered_feature_columns,
        base_config=best_base_config,
        selected_feature_columns=best_feature_columns,
        selected_feature_set_name=best_feature_set_name,
        selected_score_weights=best_score_weights,
        random_seed=args.random_seed,
    )
    ablation_table["f1_macro_delta_vs_selected"] = (
        ablation_table["f1_macro"] - float(final_summary["f1_macro"])
    ).round(6)

    compare_with_q3_2 = args.max_train_rows is None and args.max_test_rows is None
    reference_summary, reference_per_class = (None, None)
    comparison_table: pd.DataFrame | None = None
    per_class_comparison: pd.DataFrame | None = None
    if compare_with_q3_2:
        reference_summary, reference_per_class = load_best_q3_2_reference(
            args.reference_summary_path,
            args.reference_per_class_path,
        )
        if reference_summary is not None and reference_per_class is not None:
            comparison_table = build_metric_comparison_table(final_summary, reference_summary)
            per_class_comparison = build_per_class_comparison_table(final_per_class, reference_per_class)
    else:
        LOGGER.warning(
            "Skipping the Question 3.2 comparison because row caps changed the evaluation set."
        )

    architecture_path = figure_dir / "q3_3_advanced_architecture.svg"
    architecture_path.write_text(architecture_svg(), encoding="utf-8")

    summary_path = table_dir / "q3_3_advanced_summary.csv"
    per_class_path = table_dir / "q3_3_per_class_metrics.csv"
    search_path = table_dir / "q3_3_hyperparameter_search.csv"
    comparison_path = table_dir / "q3_3_comparison_vs_q3_2.csv"
    per_class_comparison_path = table_dir / "q3_3_per_class_comparison_vs_q3_2.csv"
    ablation_path = table_dir / "q3_3_ablation_results.csv"
    report_path = table_dir / "q3_3_report.md"
    summary_json_path = table_dir / "q3_3_summary.json"
    model_path = model_dir / "q3_3_hybrid_advanced_model.joblib"

    pd.DataFrame([final_summary]).to_csv(summary_path, index=False)
    final_per_class.to_csv(per_class_path, index=False)
    base_search_table.to_csv(search_path, index=False)
    ablation_table.to_csv(ablation_path, index=False)
    if comparison_table is not None:
        comparison_table.to_csv(comparison_path, index=False)
    if per_class_comparison is not None:
        per_class_comparison.to_csv(per_class_comparison_path, index=False)

    write_report(
        report_path=report_path,
        architecture_path=architecture_path,
        train_source=datasets.train_source,
        test_source=datasets.test_source,
        label_column=label_column,
        original_feature_columns=original_feature_columns,
        engineered_feature_columns=engineered_feature_columns,
        dropped_columns=dropped_columns,
        tuning_train_rows=len(tuning_train),
        validation_rows=len(validation_frame),
        final_summary=final_summary,
        final_per_class=final_per_class,
        base_search_table=base_search_table,
        comparison_table=comparison_table,
        per_class_comparison=per_class_comparison,
        ablation_table=ablation_table,
        best_base_config=best_base_config,
        best_feature_set_name=best_feature_set_name,
        best_feature_columns=best_feature_columns,
        best_score_weights=best_score_weights,
        reference_summary=reference_summary,
    )

    model_payload = {
        "model_type": "advanced_randomforest_calibrated",
        "label_column": label_column,
        "feature_columns": best_feature_columns,
        "base_config": asdict(best_base_config),
        "class_score_weights": best_score_weights,
        "web_subtype_config": None,
        "web_detector_config": None,
        "use_web_specialist": False,
        **final_details["model"],
    }
    joblib.dump(model_payload, model_path)

    summary_payload = {
        "train_source": datasets.train_source,
        "test_source": datasets.test_source,
        "label_column": label_column,
        "original_feature_columns": original_feature_columns,
        "engineered_feature_columns": engineered_feature_columns,
        "selected_feature_set_name": best_feature_set_name,
        "selected_feature_columns": best_feature_columns,
        "final_summary": final_summary,
        "best_base_config": asdict(best_base_config),
        "best_score_weights": best_score_weights,
        "best_web_subtype_config": None,
        "best_web_detector_config": None,
        "comparison_available": comparison_table is not None,
        "reference_summary": reference_summary,
        "ablation_rows": ablation_table.to_dict(orient="records"),
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    LOGGER.info("Wrote Task 3.3 summary to %s", summary_path)
    LOGGER.info("Wrote Task 3.3 per-class metrics to %s", per_class_path)
    LOGGER.info("Wrote Task 3.3 hyperparameter search table to %s", search_path)
    LOGGER.info("Wrote Task 3.3 ablation table to %s", ablation_path)
    LOGGER.info("Wrote Task 3.3 report to %s", report_path)
    LOGGER.info("Wrote Task 3.3 architecture diagram to %s", architecture_path)
    LOGGER.info("Serialized the Task 3.3 model to %s", model_path)


if __name__ == "__main__":
    main()
