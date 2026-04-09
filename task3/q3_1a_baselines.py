"""Question 3.1(a): train baseline supervised classifiers on labeled FLNET2023 data."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

try:
    import joblib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing optional dependency 'joblib'. Install project dependencies with "
        "'python3 -m pip install -r requirements.txt' and rerun q3_1a_baselines.py."
    ) from exc
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.io import list_router_files, normalize_columns
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory
from src.common.seed import set_global_seed


LOGGER = logging.getLogger("task3.q3_1a")
LABEL_CANDIDATES = (
    "label",
    "class",
    "target",
    "attack_label",
    "attack_type",
    "traffic_label",
    "y",
)
TRAIN_KEYWORDS = ("train", "training")
TEST_KEYWORDS = ("test", "testing", "holdout")
METADATA_EXCLUSIONS = {"src_ip", "dst_ip", "timestamp", "_source_file", "_source_router_id"}
ALLOWED_CATEGORICAL_FEATURES = {"router_id"}
DEFAULT_LABELED_DATA_DIR = PROJECT_ROOT / "data" / "raw_labeled"
DEFAULT_SPLIT_STRATEGY = "hybrid"
SOURCE_FILE_COLUMN = "_source_file"
SOURCE_ROUTER_COLUMN = "_source_router_id"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hyperparameters: dict[str, object]
    factory: Callable[[], object]


@dataclass(frozen=True)
class DatasetBundle:
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    train_source: str
    test_source: str
    split_strategy: str
    train_groups: list[str]
    test_groups: list[str]


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
        help=(
            "How to split an auto-discovered labeled corpus. "
            "'hybrid' keeps whole labeled source files together when possible and only splits rows "
            "within singleton-class files so all classes remain represented."
        ),
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables",
        help="Directory for generated Task 3.1(a) tables and reports.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=CONFIG.outputs_dir / "models" / "task3",
        help="Directory for serialized baseline models.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=CONFIG.random_seed,
        help="Random seed used across all baseline experiments.",
    )
    return parser.parse_args()


def discover_default_split_paths(data_dir: Path) -> tuple[Path, Path]:
    files = list_router_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No CSV or parquet files were found in {data_dir}. "
            "Provide --train-path and --test-path once the labeled files are available."
        )

    train_candidates = [
        path for path in files if any(keyword in path.stem.lower() for keyword in TRAIN_KEYWORDS)
    ]
    test_candidates = [
        path for path in files if any(keyword in path.stem.lower() for keyword in TEST_KEYWORDS)
    ]

    if len(train_candidates) != 1 or len(test_candidates) != 1:
        raise FileNotFoundError(
            "Could not infer a unique labeled train/test split from data/processed. "
            "Pass --train-path and --test-path explicitly. "
            f"Train candidates: {[path.name for path in train_candidates]}; "
            f"test candidates: {[path.name for path in test_candidates]}."
        )

    return train_candidates[0], test_candidates[0]


def resolve_dataset_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.train_path and args.test_path:
        return args.train_path, args.test_path
    if args.train_path or args.test_path:
        raise ValueError("Provide both --train-path and --test-path, or omit both to use auto-discovery.")
    return discover_default_split_paths(args.data_dir)


def list_recursive_data_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.csv", "*.parquet"):
        files.extend(data_dir.rglob(pattern))
    return sorted(path for path in files if path.is_file())


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported dataset format for {path}. Use CSV or parquet.")

    return normalize_columns(frame)


def parse_router_id_from_path(path: Path) -> int | None:
    dataset_match = re.search(r"dataset-(\d+)", path.stem, flags=re.IGNORECASE)
    if dataset_match:
        return int(dataset_match.group(1))

    raw_match = re.search(r"-(\d+)$", path.stem)
    if raw_match:
        return int(raw_match.group(1))

    return None


def summarize_file_label(frame: pd.DataFrame, label_column: str, path: Path) -> str:
    labels = sorted(frame[label_column].astype(str).dropna().unique().tolist())
    if not labels:
        raise ValueError(f"No non-missing labels were found in {path}.")
    if len(labels) > 1:
        LOGGER.warning("Labeled source file %s contains multiple labels: %s", path, labels)
    return labels[0]


def load_labeled_corpus(
    labeled_data_dir: Path,
    *,
    label_column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    files = list_recursive_data_files(labeled_data_dir)
    if not files:
        raise FileNotFoundError(
            f"No labeled CSV or parquet files were found under {labeled_data_dir}."
        )

    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    resolved_label_column: str | None = None

    for path in files:
        frame = read_dataset(path)
        file_label_column = infer_label_column(frame, frame, label_column)
        if resolved_label_column is None:
            resolved_label_column = file_label_column
        elif file_label_column != resolved_label_column:
            raise ValueError(
                f"Inconsistent label columns detected while reading {path}: "
                f"expected {resolved_label_column}, found {file_label_column}."
            )

        frame = frame.dropna(subset=[file_label_column]).reset_index(drop=True)
        source_file = str(path.relative_to(labeled_data_dir))
        router_id = parse_router_id_from_path(path)
        frame[SOURCE_FILE_COLUMN] = source_file
        frame[SOURCE_ROUTER_COLUMN] = router_id
        frames.append(frame)
        manifest_rows.append(
            {
                SOURCE_FILE_COLUMN: source_file,
                SOURCE_ROUTER_COLUMN: router_id,
                "label": summarize_file_label(frame, file_label_column, path),
                "rows": int(len(frame)),
            }
        )

    if resolved_label_column is None:
        raise ValueError(f"Could not infer a label column from files under {labeled_data_dir}.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    manifest = pd.DataFrame(manifest_rows).sort_values(SOURCE_FILE_COLUMN).reset_index(drop=True)
    return combined, manifest, resolved_label_column


def choose_balanced_label_subsets(
    manifest: pd.DataFrame,
    *,
    test_size: float,
    random_seed: int,
) -> list[str]:
    selected_groups: list[str] = []

    for label_index, (label_name, group) in enumerate(manifest.groupby("label", sort=True), start=1):
        if len(group) <= 1:
            continue

        shuffled = group.sample(frac=1.0, random_state=random_seed + label_index).reset_index(drop=True)
        target_rows = float(shuffled["rows"].sum()) * test_size
        target_group_count = max(1, round(len(shuffled) * test_size))
        best_subset: tuple[str, ...] | None = None
        best_score: tuple[float, float, str] | None = None

        records = shuffled.to_dict(orient="records")
        for subset_size in range(1, len(records)):
            for subset in combinations(records, subset_size):
                subset_rows = float(sum(int(record["rows"]) for record in subset))
                subset_names = tuple(sorted(str(record[SOURCE_FILE_COLUMN]) for record in subset))
                score = (
                    abs(subset_rows - target_rows),
                    abs(subset_size - target_group_count),
                    "|".join(subset_names),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_subset = subset_names

        if best_subset is not None:
            selected_groups.extend(best_subset)

    return sorted(set(selected_groups))


def split_singleton_label_files(
    combined: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    singleton_files = (
        manifest.groupby("label")
        .filter(lambda group: len(group) == 1)[SOURCE_FILE_COLUMN]
        .astype(str)
        .tolist()
    )

    if not singleton_files:
        empty = combined.iloc[0:0].copy()
        return empty, empty, []

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for offset, source_file in enumerate(sorted(singleton_files), start=1):
        file_frame = combined[combined[SOURCE_FILE_COLUMN] == source_file].copy()
        if len(file_frame) < 2:
            raise ValueError(
                f"Hybrid split requires at least 2 rows in singleton source file {source_file}, "
                f"but only found {len(file_frame)}."
            )

        test_rows = max(1, round(len(file_frame) * test_size))
        test_rows = min(test_rows, len(file_frame) - 1)
        sampled_test = file_frame.sample(n=test_rows, random_state=random_seed + offset)
        sampled_train = file_frame.drop(index=sampled_test.index)
        train_parts.append(sampled_train)
        test_parts.append(sampled_test)

    train_frame = pd.concat(train_parts, ignore_index=True, sort=False)
    test_frame = pd.concat(test_parts, ignore_index=True, sort=False)
    return train_frame, test_frame, sorted(singleton_files)


def choose_router_test_groups(
    manifest: pd.DataFrame,
    *,
    test_size: float,
) -> list[int]:
    grouped = (
        manifest.groupby(SOURCE_ROUTER_COLUMN, dropna=False)
        .agg(
            rows=("rows", "sum"),
            labels=("label", lambda values: sorted(set(map(str, values)))),
            files=(SOURCE_FILE_COLUMN, lambda values: sorted(map(str, values))),
        )
        .reset_index()
    )

    records = grouped.to_dict(orient="records")
    if len(records) <= 1:
        return []

    total_rows = float(grouped["rows"].sum())
    target_rows = total_rows * test_size
    all_labels = set(manifest["label"].astype(str))

    best_router_ids: list[int] = []
    best_score: tuple[float, float, float, str] | None = None

    for subset_size in range(1, len(records)):
        for subset in combinations(records, subset_size):
            subset_labels = set().union(*(set(record["labels"]) for record in subset))
            complement_labels = all_labels - {label for record in subset for label in set(record["labels"])}
            shared_labels = len(subset_labels & complement_labels)
            subset_rows = float(sum(int(record["rows"]) for record in subset))
            subset_router_ids = sorted(int(record[SOURCE_ROUTER_COLUMN]) for record in subset if pd.notna(record[SOURCE_ROUTER_COLUMN]))
            score = (
                -float(shared_labels),
                abs(subset_rows - target_rows),
                abs(subset_size - max(1, round(len(records) * test_size))),
                "|".join(map(str, subset_router_ids)),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_router_ids = subset_router_ids

    return best_router_ids


def auto_split_labeled_directory(
    labeled_data_dir: Path,
    *,
    label_column: str | None,
    random_seed: int,
    test_size: float,
    split_strategy: str,
) -> DatasetBundle:
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"--test-size must be between 0 and 1, received {test_size}.")

    combined, manifest, inferred_label_column = load_labeled_corpus(
        labeled_data_dir,
        label_column=label_column,
    )

    LOGGER.info(
        "No explicit or pre-split Task 3 datasets were found. "
        "Building a reproducible %s train/test split from %s labeled files in %s.",
        split_strategy,
        len(manifest),
        labeled_data_dir,
    )
    combined = drop_missing_labels(combined, inferred_label_column, "combined_labeled_corpus")

    if split_strategy == "row":
        label_values = combined[inferred_label_column].astype(str)
        class_counts = label_values.value_counts()
        use_stratify = bool((class_counts >= 2).all())
        if not use_stratify:
            rare_labels = class_counts[class_counts < 2].index.tolist()
            LOGGER.warning(
                "Falling back to a non-stratified split because some labels have fewer than 2 rows: %s",
                rare_labels,
            )

        train_frame, test_frame = train_test_split(
            combined,
            test_size=test_size,
            random_state=random_seed,
            shuffle=True,
            stratify=label_values if use_stratify else None,
        )
        train_groups: list[str] = []
        test_groups: list[str] = []
    else:
        if split_strategy == "source_file":
            test_groups = choose_balanced_label_subsets(
                manifest,
                test_size=test_size,
                random_seed=random_seed,
            )
            singleton_split_files: list[str] = []
            mask = combined[SOURCE_FILE_COLUMN].isin(test_groups)
            train_frame = combined.loc[~mask].reset_index(drop=True)
            test_frame = combined.loc[mask].reset_index(drop=True)
        elif split_strategy == "hybrid":
            multifile_manifest = manifest.groupby("label").filter(lambda group: len(group) > 1).reset_index(drop=True)
            test_groups = choose_balanced_label_subsets(
                multifile_manifest,
                test_size=test_size,
                random_seed=random_seed,
            )
            singleton_train, singleton_test, singleton_split_files = split_singleton_label_files(
                combined,
                manifest,
                test_size=test_size,
                random_seed=random_seed,
            )
            singleton_mask = combined[SOURCE_FILE_COLUMN].isin(singleton_split_files)
            multifile_pool = combined.loc[~singleton_mask].copy()
            train_frame = pd.concat(
                [
                    multifile_pool.loc[~multifile_pool[SOURCE_FILE_COLUMN].isin(test_groups)],
                    singleton_train,
                ],
                ignore_index=True,
                sort=False,
            )
            test_frame = pd.concat(
                [
                    multifile_pool.loc[multifile_pool[SOURCE_FILE_COLUMN].isin(test_groups)],
                    singleton_test,
                ],
                ignore_index=True,
                sort=False,
            )
            test_groups = sorted(set(test_groups + [f"{file_name} [row-split]" for file_name in singleton_split_files]))
        elif split_strategy == "router":
            test_groups = choose_router_test_groups(
                manifest,
                test_size=test_size,
            )
            singleton_split_files = []
            if not test_groups:
                raise ValueError(
                    f"Could not construct a non-empty {split_strategy} test split from {labeled_data_dir}."
                )
            selected_router_ids = {int(value) for value in test_groups}
            mask = combined[SOURCE_ROUTER_COLUMN].isin(selected_router_ids)
            train_frame = combined.loc[~mask].reset_index(drop=True)
            test_frame = combined.loc[mask].reset_index(drop=True)
        else:
            raise ValueError(f"Unsupported split strategy: {split_strategy}")

        if split_strategy in {"source_file", "hybrid"} and test_frame.empty:
            raise ValueError(
                f"Could not construct a non-empty {split_strategy} test split from {labeled_data_dir}."
            )

        train_groups = sorted(set(train_frame[SOURCE_FILE_COLUMN].astype(str).tolist()))

        train_labels = set(train_frame[inferred_label_column].astype(str))
        test_labels = set(test_frame[inferred_label_column].astype(str))
        train_only_labels = sorted(train_labels - test_labels)
        test_only_labels = sorted(test_labels - train_labels)
        if train_only_labels:
            LOGGER.warning(
                "The stricter %s split leaves some labels only in train: %s",
                split_strategy,
                train_only_labels,
            )
        if test_only_labels:
            LOGGER.warning(
                "The stricter %s split leaves some labels only in test: %s",
                split_strategy,
                test_only_labels,
            )

    train_source = f"{labeled_data_dir} [auto-split train; strategy={split_strategy}; test_size={test_size}]"
    test_source = f"{labeled_data_dir} [auto-split test; strategy={split_strategy}; test_size={test_size}]"
    return DatasetBundle(
        train_frame=train_frame.reset_index(drop=True),
        test_frame=test_frame.reset_index(drop=True),
        train_source=train_source,
        test_source=test_source,
        split_strategy=split_strategy,
        train_groups=train_groups,
        test_groups=[str(group) for group in test_groups],
    )


def load_datasets_from_args(args: argparse.Namespace) -> DatasetBundle:
    try:
        train_path, test_path = resolve_dataset_paths(args)
    except FileNotFoundError:
        if args.train_path or args.test_path:
            raise
        return auto_split_labeled_directory(
            args.labeled_data_dir,
            label_column=args.label_column,
            random_seed=args.random_seed,
            test_size=args.test_size,
            split_strategy=getattr(args, "split_strategy", DEFAULT_SPLIT_STRATEGY),
        )

    LOGGER.info("Loading training data from %s", train_path)
    train_frame = read_dataset(train_path)
    LOGGER.info("Loading test data from %s", test_path)
    test_frame = read_dataset(test_path)
    return DatasetBundle(
        train_frame=train_frame,
        test_frame=test_frame,
        train_source=str(train_path),
        test_source=str(test_path),
        split_strategy="explicit_paths",
        train_groups=[],
        test_groups=[],
    )


def infer_label_column(train_frame: pd.DataFrame, test_frame: pd.DataFrame, label_column: str | None) -> str:
    if label_column is not None:
        candidate = label_column.strip().lower().replace(" ", "_").replace("/", "_")
        if candidate not in train_frame.columns or candidate not in test_frame.columns:
            raise ValueError(f"Requested label column '{candidate}' is missing from train or test data.")
        return candidate

    for candidate in LABEL_CANDIDATES:
        if candidate in train_frame.columns and candidate in test_frame.columns:
            return candidate

    raise ValueError(
        "Could not infer the label column. Pass --label-column explicitly once the labeled data files are available."
    )


def drop_missing_labels(frame: pd.DataFrame, label_column: str, split_name: str) -> pd.DataFrame:
    missing_labels = int(frame[label_column].isna().sum())
    if missing_labels:
        LOGGER.warning("Dropping %s rows with missing labels from %s split.", missing_labels, split_name)
    return frame.dropna(subset=[label_column]).reset_index(drop=True)


def select_feature_columns(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    label_column: str,
) -> tuple[list[str], list[str], list[str]]:
    shared_columns = [column for column in train_frame.columns if column in test_frame.columns]

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    dropped_columns: list[str] = []

    for column in shared_columns:
        if column == label_column or column in METADATA_EXCLUSIONS:
            continue
        if pd.api.types.is_numeric_dtype(train_frame[column]) and pd.api.types.is_numeric_dtype(test_frame[column]):
            numeric_columns.append(column)
        elif column in ALLOWED_CATEGORICAL_FEATURES:
            categorical_columns.append(column)
        else:
            dropped_columns.append(column)

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable feature columns were found after excluding labels and unsupported metadata.")

    return numeric_columns, categorical_columns, dropped_columns


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    transformers: list[tuple[str, object, list[str]]] = []

    if numeric_columns:
        numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def build_gradient_boosted_spec(random_seed: int, num_classes: int) -> ModelSpec:
    backend_errors: list[str] = []

    try:
        from lightgbm import LGBMClassifier

        hyperparameters = {
            "objective": "multiclass",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_seed,
            "n_jobs": -1,
            "verbosity": -1,
        }
        return ModelSpec(
            name="LightGBM",
            hyperparameters=hyperparameters,
            factory=lambda: LGBMClassifier(**hyperparameters),
        )
    except Exception as exc:
        backend_errors.append(f"LightGBM unavailable ({type(exc).__name__}: {exc})")

    try:
        from xgboost import XGBClassifier

        hyperparameters = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": random_seed,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
        }
        return ModelSpec(
            name="XGBoost",
            hyperparameters=hyperparameters,
            factory=lambda: XGBClassifier(**hyperparameters),
        )
    except Exception as exc:
        backend_errors.append(f"XGBoost unavailable ({type(exc).__name__}: {exc})")

    LOGGER.warning(
        "Neither LightGBM nor XGBoost is available in this environment. "
        "Falling back to scikit-learn HistGradientBoostingClassifier. Details: %s",
        " | ".join(backend_errors),
    )
    hyperparameters = {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "random_state": random_seed,
    }
    return ModelSpec(
        name="HistGradientBoosting",
        hyperparameters=hyperparameters,
        factory=lambda: HistGradientBoostingClassifier(**hyperparameters),
    )


def build_model_specs(random_seed: int, num_classes: int) -> list[ModelSpec]:
    random_forest_params = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "n_jobs": -1,
        "random_state": random_seed,
    }
    mlp_params = {
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": 4096,
        "learning_rate_init": 1e-3,
        "max_iter": 60,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 5,
        "random_state": random_seed,
    }

    return [
        ModelSpec(
            name="RandomForest",
            hyperparameters=random_forest_params,
            factory=lambda: RandomForestClassifier(**random_forest_params),
        ),
        build_gradient_boosted_spec(random_seed, num_classes),
        ModelSpec(
            name="MLP",
            hyperparameters=mlp_params,
            factory=lambda: MLPClassifier(**mlp_params),
        ),
    ]


def encode_labels(train_labels: pd.Series, test_labels: pd.Series) -> tuple[LabelEncoder, pd.Series, pd.Series]:
    encoder = LabelEncoder()
    y_train = pd.Series(encoder.fit_transform(train_labels.astype(str)), index=train_labels.index)

    unseen_test_labels = sorted(set(test_labels.astype(str)) - set(encoder.classes_))
    if unseen_test_labels:
        raise ValueError(f"Test labels contain unseen classes not present in training data: {unseen_test_labels}")

    y_test = pd.Series(encoder.transform(test_labels.astype(str)), index=test_labels.index)
    return encoder, y_train, y_test


def train_and_evaluate_models(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    label_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    model_specs: list[ModelSpec],
    model_dir: Path,
    class_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []

    feature_columns = numeric_columns + categorical_columns
    x_train = train_frame[feature_columns]
    x_test = test_frame[feature_columns]
    y_train = train_frame[label_column]
    y_test = test_frame[label_column]

    for spec in model_specs:
        scale_numeric = spec.name == "MLP"
        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(
                        numeric_columns,
                        categorical_columns,
                        scale_numeric=scale_numeric,
                    ),
                ),
                ("classifier", spec.factory()),
            ]
        )

        LOGGER.info("Training %s baseline.", spec.name)
        start_time = time.perf_counter()
        pipeline.fit(x_train, y_train)
        training_time = time.perf_counter() - start_time

        predictions = pipeline.predict(x_test)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        summary_rows.append(
            {
                "model": spec.name,
                "training_time_seconds": round(training_time, 3),
                "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
                "precision_macro": round(float(precision_macro), 6),
                "recall_macro": round(float(recall_macro), 6),
                "f1_macro": round(float(f1_macro), 6),
                "f1_weighted": round(float(f1_weighted), 6),
                "hyperparameters": json.dumps(spec.hyperparameters, sort_keys=True),
            }
        )

        precision, recall, f1_score_values, support = precision_recall_fscore_support(
            y_test,
            predictions,
            labels=list(range(len(class_names))),
            zero_division=0,
        )
        for class_index, class_name in enumerate(class_names):
            per_class_rows.append(
                {
                    "model": spec.name,
                    "class_name": class_name,
                    "precision": round(float(precision[class_index]), 6),
                    "recall": round(float(recall[class_index]), 6),
                    "f1_score": round(float(f1_score_values[class_index]), 6),
                    "support": int(support[class_index]),
                }
            )

        model_path = model_dir / f"q3_1a_{spec.name.lower()}_baseline.joblib"
        joblib.dump(pipeline, model_path)
        LOGGER.info("Saved %s model to %s", spec.name, model_path)

    return pd.DataFrame(summary_rows), pd.DataFrame(per_class_rows)


def render_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def write_report(
    summary_table: pd.DataFrame,
    per_class_table: pd.DataFrame,
    *,
    train_source: str,
    test_source: str,
    split_strategy: str,
    test_groups: list[str],
    label_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    dropped_columns: list[str],
    train_rows: int,
    test_rows: int,
    report_path: Path,
) -> None:
    best_row = summary_table.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]
    lines = [
        "# Task 3.1(a) Baseline Model Report",
        "",
        "## Data Summary",
        f"- Training dataset: `{train_source}` ({train_rows:,} rows)",
        f"- Test dataset: `{test_source}` ({test_rows:,} rows)",
        f"- Split strategy: `{split_strategy}`",
        f"- Label column: `{label_column}`",
        f"- Numeric feature count: {len(numeric_columns)}",
        f"- Categorical feature count: {len(categorical_columns)}",
    ]
    if test_groups:
        lines.append(f"- Held-out groups ({len(test_groups)}): {', '.join(test_groups)}")
    if categorical_columns:
        lines.append(f"- Included categorical features: {', '.join(categorical_columns)}")
    if dropped_columns:
        lines.append(f"- Dropped non-numeric/non-router metadata columns: {', '.join(dropped_columns)}")

    lines.extend(
        [
            "",
            "## Baseline Comparison",
            render_table(summary_table),
            "",
            "## Best Baseline",
            (
                f"- `{best_row['model']}` achieved the highest macro F1 score "
                f"({best_row['f1_macro']:.6f}) with accuracy {best_row['accuracy']:.6f}."
            ),
            (
                f"- Its training time was {best_row['training_time_seconds']:.3f} seconds "
                "on the provided training set."
            ),
            "",
            "## Per-Class Results",
            render_table(per_class_table),
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_global_seed(args.random_seed)

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

    train_frame = train_frame.copy()
    test_frame = test_frame.copy()
    train_frame[label_column] = encoded_train_labels
    test_frame[label_column] = encoded_test_labels

    numeric_columns, categorical_columns, dropped_columns = select_feature_columns(
        train_frame,
        test_frame,
        label_column,
    )

    model_specs = build_model_specs(args.random_seed, len(class_names))
    summary_table, per_class_table = train_and_evaluate_models(
        train_frame=train_frame,
        test_frame=test_frame,
        label_column=label_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        model_specs=model_specs,
        model_dir=model_dir,
        class_names=class_names,
    )

    summary_path = table_dir / "q3_1a_baseline_summary.csv"
    per_class_path = table_dir / "q3_1a_per_class_metrics.csv"
    report_path = table_dir / "q3_1a_report.md"
    summary_json_path = table_dir / "q3_1a_summary.json"

    summary_table.to_csv(summary_path, index=False)
    per_class_table.to_csv(per_class_path, index=False)
    write_report(
        summary_table,
        per_class_table,
        train_source=datasets.train_source,
        test_source=datasets.test_source,
        split_strategy=datasets.split_strategy,
        test_groups=datasets.test_groups,
        label_column=label_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        dropped_columns=dropped_columns,
        train_rows=len(train_frame),
        test_rows=len(test_frame),
        report_path=report_path,
    )

    summary_payload = {
        "train_path": datasets.train_source,
        "test_path": datasets.test_source,
        "split_strategy": datasets.split_strategy,
        "train_groups": datasets.train_groups,
        "test_groups": datasets.test_groups,
        "label_column": label_column,
        "num_classes": len(class_names),
        "class_names": class_names,
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "dropped_columns": dropped_columns,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "baseline_results": summary_table.to_dict(orient="records"),
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    LOGGER.info("Wrote baseline summary to %s", summary_path)
    LOGGER.info("Wrote per-class metrics to %s", per_class_path)
    LOGGER.info("Wrote markdown report to %s", report_path)
    LOGGER.info("Wrote JSON summary to %s", summary_json_path)


if __name__ == "__main__":
    main()
