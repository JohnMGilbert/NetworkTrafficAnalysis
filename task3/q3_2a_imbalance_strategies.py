"""Question 3.2(a): compare class-imbalance strategies for the best baseline model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import joblib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing optional dependency 'joblib'. Install project dependencies with "
        "'python3 -m pip install -r requirements.txt' and rerun q3_2a_imbalance_strategies.py."
    ) from exc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory
from src.common.seed import set_global_seed
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


LOGGER = logging.getLogger("task3.q3_2a")
RARE_CLASS_FOCUS = ("Web-command-injection", "Web-sql-injection", "Infiltration-mitm")


@dataclass(frozen=True)
class StrategySpec:
    name: str
    slug: str
    description: str
    class_weight: str | dict[str, float] | None = None
    resampling_mode: str = "none"


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
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables",
        help="Directory for generated Task 3.2(a) tables and reports.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=CONFIG.outputs_dir / "models" / "task3",
        help="Directory for serialized imbalance-aware models.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=CONFIG.random_seed,
        help="Random seed used across all imbalance experiments.",
    )
    parser.add_argument(
        "--minority-target-count",
        type=int,
        default=25000,
        help=(
            "Target training count used for selective SMOTE-based strategies. Only classes "
            "below this threshold are oversampled."
        ),
    )
    parser.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=5,
        help="Maximum SMOTE neighborhood size for synthetic sample generation.",
    )
    parser.add_argument(
        "--hybrid-majority-cap",
        type=int,
        default=250000,
        help=(
            "Maximum training count per majority class for the hybrid strategy after selective "
            "SMOTE has been applied."
        ),
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap for training rows, useful for quick smoke tests.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional cap for test rows, useful for quick smoke tests.",
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
        "Applying %s cap: sampled %s rows from %s rows for faster iteration.",
        split_name,
        len(sampled),
        len(frame),
    )
    return sampled.reset_index(drop=True)


def build_sampling_strategy(y_train: pd.Series, minority_target_count: int) -> dict[str, int]:
    counts = y_train.astype(str).value_counts().sort_index()
    return {
        class_name: int(minority_target_count)
        for class_name, count in counts.items()
        if 1 < int(count) < minority_target_count
    }


def resolve_smote_neighbors(
    y_train: pd.Series,
    sampling_strategy: dict[str, int],
    requested_k: int,
) -> int:
    if not sampling_strategy:
        return requested_k

    counts = y_train.astype(str).value_counts()
    smallest_targeted_class = min(int(counts[class_name]) for class_name in sampling_strategy)
    return max(1, min(requested_k, smallest_targeted_class - 1))


def selective_smote(
    x_train: np.ndarray,
    y_train: pd.Series,
    *,
    sampling_strategy: dict[str, int],
    smote_k_neighbors: int,
    random_seed: int,
) -> tuple[np.ndarray, pd.Series]:
    if not sampling_strategy:
        return x_train, y_train.reset_index(drop=True)

    rng = np.random.default_rng(random_seed)
    y_series = y_train.astype(str).reset_index(drop=True)
    x_blocks: list[np.ndarray] = [x_train]
    y_blocks: list[pd.Series] = [y_series]

    for class_name, target_count in sampling_strategy.items():
        class_mask = y_series == class_name
        x_class = x_train[class_mask.to_numpy()]
        current_count = len(x_class)
        additional_needed = int(target_count) - current_count
        if additional_needed <= 0 or current_count < 2:
            continue

        local_k = max(1, min(smote_k_neighbors, current_count - 1))
        neighbors = NearestNeighbors(n_neighbors=local_k + 1)
        neighbors.fit(x_class)
        neighbor_indices = neighbors.kneighbors(return_distance=False)

        base_indices = rng.integers(0, current_count, size=additional_needed)
        chosen_neighbors = np.empty(additional_needed, dtype=np.int64)
        for synthetic_index, base_index in enumerate(base_indices):
            neighbor_pool = neighbor_indices[base_index, 1:]
            chosen_neighbors[synthetic_index] = int(rng.choice(neighbor_pool))

        base_samples = x_class[base_indices]
        neighbor_samples = x_class[chosen_neighbors]
        interpolation = rng.random((additional_needed, 1), dtype=np.float32)
        synthetic_samples = base_samples + interpolation * (neighbor_samples - base_samples)

        x_blocks.append(synthetic_samples.astype(np.float32, copy=False))
        y_blocks.append(pd.Series([class_name] * additional_needed, dtype="string"))

    x_resampled = np.vstack(x_blocks).astype(np.float32, copy=False)
    y_resampled = pd.concat(y_blocks, ignore_index=True).astype(str)
    return x_resampled, y_resampled


def random_undersample_majorities(
    x_train: np.ndarray,
    y_train: pd.Series,
    *,
    majority_cap: int,
    random_seed: int,
) -> tuple[np.ndarray, pd.Series]:
    if majority_cap <= 0:
        raise ValueError(f"--hybrid-majority-cap must be positive, received {majority_cap}.")

    rng = np.random.default_rng(random_seed)
    y_series = y_train.astype(str).reset_index(drop=True)
    kept_indices: list[np.ndarray] = []

    for class_name in sorted(y_series.unique()):
        class_indices = np.flatnonzero(y_series.to_numpy() == class_name)
        if len(class_indices) > majority_cap:
            sampled_indices = np.sort(rng.choice(class_indices, size=majority_cap, replace=False))
            kept_indices.append(sampled_indices)
        else:
            kept_indices.append(class_indices)

    if not kept_indices:
        return x_train, y_series

    merged_indices = np.concatenate(kept_indices)
    merged_indices.sort()
    return x_train[merged_indices], y_series.iloc[merged_indices].reset_index(drop=True)


def prepare_training_data(
    x_train: np.ndarray,
    y_train: pd.Series,
    *,
    spec: StrategySpec,
    sampling_strategy: dict[str, int],
    smote_k_neighbors: int,
    hybrid_majority_cap: int,
    random_seed: int,
) -> tuple[np.ndarray, pd.Series]:
    if spec.resampling_mode == "none":
        return x_train, y_train.reset_index(drop=True)
    if spec.resampling_mode == "smote":
        return selective_smote(
            x_train,
            y_train,
            sampling_strategy=sampling_strategy,
            smote_k_neighbors=smote_k_neighbors,
            random_seed=random_seed,
        )
    if spec.resampling_mode == "hybrid":
        x_smote, y_smote = selective_smote(
            x_train,
            y_train,
            sampling_strategy=sampling_strategy,
            smote_k_neighbors=smote_k_neighbors,
            random_seed=random_seed,
        )
        return random_undersample_majorities(
            x_smote,
            y_smote,
            majority_cap=hybrid_majority_cap,
            random_seed=random_seed,
        )
    raise ValueError(f"Unsupported resampling mode: {spec.resampling_mode}")


def build_strategy_specs() -> list[StrategySpec]:
    return [
        StrategySpec(
            name="No Balancing",
            slug="baseline",
            description="Original RandomForest baseline with no resampling and no class weighting.",
        ),
        StrategySpec(
            name="SMOTE Oversampling",
            slug="smote",
            description="Selective SMOTE applied only to classes below the configured minority target.",
            resampling_mode="smote",
        ),
        StrategySpec(
            name="Class Weighting",
            slug="class_weight",
            description="RandomForest with class_weight='balanced' and no data resampling.",
            class_weight="balanced",
        ),
        StrategySpec(
            name="SMOTE + Undersampling Hybrid",
            slug="hybrid",
            description="Selective SMOTE followed by random undersampling of the largest classes.",
            resampling_mode="hybrid",
        ),
    ]


def evaluate_strategies(
    *,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    label_column: str,
    feature_columns: list[str],
    class_names: list[str],
    strategy_specs: list[StrategySpec],
    model_dir: Path,
    random_seed: int,
    sampling_strategy: dict[str, int],
    smote_k_neighbors: int,
    hybrid_majority_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    sampling_rows: list[dict[str, object]] = []

    x_train = train_frame[feature_columns]
    y_train = train_frame[label_column].astype(str)
    x_test = test_frame[feature_columns]
    y_test = test_frame[label_column].astype(str)
    imputer = SimpleImputer(strategy="median")
    x_train_imputed = imputer.fit_transform(x_train).astype(np.float32, copy=False)
    x_test_imputed = imputer.transform(x_test).astype(np.float32, copy=False)

    for spec in strategy_specs:
        LOGGER.info("Training RandomForest with strategy: %s", spec.name)
        x_train_prepared, y_train_prepared = prepare_training_data(
            x_train_imputed,
            y_train,
            spec=spec,
            sampling_strategy=sampling_strategy,
            smote_k_neighbors=smote_k_neighbors,
            hybrid_majority_cap=hybrid_majority_cap,
            random_seed=random_seed,
        )

        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_seed,
            class_weight=spec.class_weight,
        )

        start_time = time.perf_counter()
        classifier.fit(x_train_prepared, y_train_prepared)
        training_time = time.perf_counter() - start_time

        predictions = classifier.predict(x_test_imputed)
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
                "strategy": spec.name,
                "training_time_seconds": round(training_time, 3),
                "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
                "precision_macro": round(float(precision_macro), 6),
                "recall_macro": round(float(recall_macro), 6),
                "f1_macro": round(float(f1_macro), 6),
                "f1_weighted": round(float(f1_weighted), 6),
                "train_rows_after_sampling": int(len(y_train_prepared)),
                "class_weight": spec.class_weight if spec.class_weight is not None else "none",
                "sampler": spec.resampling_mode,
            }
        )

        precision, recall, f1_scores, support = precision_recall_fscore_support(
            y_test,
            predictions,
            labels=class_names,
            zero_division=0,
        )
        for class_index, class_name in enumerate(class_names):
            per_class_rows.append(
                {
                    "strategy": spec.name,
                    "class_name": class_name,
                    "precision": round(float(precision[class_index]), 6),
                    "recall": round(float(recall[class_index]), 6),
                    "f1_score": round(float(f1_scores[class_index]), 6),
                    "support": int(support[class_index]),
                }
            )

        resampled_counts = y_train_prepared.value_counts().sort_index()
        original_counts = y_train.value_counts().sort_index()
        for class_name in sorted(class_names):
            sampling_rows.append(
                {
                    "strategy": spec.name,
                    "class_name": class_name,
                    "original_train_count": int(original_counts.get(class_name, 0)),
                    "resampled_train_count": int(resampled_counts.get(class_name, 0)),
                    "changed_by_strategy": int(original_counts.get(class_name, 0))
                    != int(resampled_counts.get(class_name, 0)),
                    "rare_class_focus": class_name in RARE_CLASS_FOCUS,
                }
            )

        model_path = model_dir / f"q3_2a_randomforest_{spec.slug}.joblib"
        artifact = {
            "strategy": spec.name,
            "feature_columns": feature_columns,
            "imputer": imputer,
            "classifier": classifier,
            "class_weight": spec.class_weight,
            "resampling_mode": spec.resampling_mode,
            "sampling_strategy": sampling_strategy,
            "smote_k_neighbors": smote_k_neighbors,
            "hybrid_majority_cap": hybrid_majority_cap,
        }
        joblib.dump(artifact, model_path)
        LOGGER.info("Saved %s model to %s", spec.name, model_path)

    return pd.DataFrame(summary_rows), pd.DataFrame(per_class_rows), pd.DataFrame(sampling_rows)


def build_sampling_plan_table(
    train_frame: pd.DataFrame,
    *,
    label_column: str,
    sampling_strategy: dict[str, int],
    hybrid_majority_cap: int,
) -> pd.DataFrame:
    original_counts = train_frame[label_column].astype(str).value_counts().sort_index()
    rows: list[dict[str, object]] = []
    for class_name, original_count in original_counts.items():
        target_count = sampling_strategy.get(class_name, int(original_count))
        hybrid_target = min(int(target_count), int(hybrid_majority_cap))
        if class_name in sampling_strategy:
            hybrid_target = int(target_count)
        rows.append(
            {
                "class_name": class_name,
                "original_train_count": int(original_count),
                "smote_target_count": int(target_count),
                "hybrid_target_cap": int(hybrid_target),
                "touched_by_smote": class_name in sampling_strategy,
                "rare_class_focus": class_name in RARE_CLASS_FOCUS,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    summary_table: pd.DataFrame,
    per_class_table: pd.DataFrame,
    sampling_plan_table: pd.DataFrame,
    *,
    train_source: str,
    test_source: str,
    label_column: str,
    feature_columns: list[str],
    dropped_columns: list[str],
    minority_target_count: int,
    smote_k_neighbors: int,
    hybrid_majority_cap: int,
    max_train_rows: int | None,
    max_test_rows: int | None,
    report_path: Path,
) -> None:
    best_row = summary_table.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]
    rare_rows = per_class_table[per_class_table["class_name"].isin(RARE_CLASS_FOCUS)].copy()
    rare_rows = rare_rows.sort_values(["class_name", "strategy"]).reset_index(drop=True)

    lines = [
        "# Task 3.2(a) Imbalance Strategy Report",
        "",
        "## Data Summary",
        f"- Training dataset: `{train_source}`",
        f"- Test dataset: `{test_source}`",
        f"- Label column: `{label_column}`",
        f"- Feature count used by RandomForest: {len(feature_columns)}",
        f"- Minority target count for SMOTE-based strategies: {minority_target_count:,}",
        f"- SMOTE k-neighbors: {smote_k_neighbors}",
        f"- Hybrid majority-class cap: {hybrid_majority_cap:,}",
        (
            "- Runtime row caps: "
            f"train={max_train_rows if max_train_rows is not None else 'none'}, "
            f"test={max_test_rows if max_test_rows is not None else 'none'}"
        ),
    ]
    if dropped_columns:
        lines.append(f"- Dropped unsupported metadata columns: {', '.join(dropped_columns)}")

    lines.extend(
        [
            "",
            "## Why Selective SMOTE",
            (
                "- The assignment explicitly notes that naive SMOTE over the full 6.8M-flow corpus is "
                "computationally prohibitive. This workflow therefore applies selective SMOTE only to "
                "classes whose training counts fall below the configured minority target."
            ),
            (
                "- The hybrid strategy then randomly undersamples only the very largest classes, which "
                "still satisfies the required over+under sampling design without depending on the "
                "currently broken imbalanced-learn install in this environment."
            ),
            "",
            "## Strategy Comparison",
            render_table(summary_table),
            "",
            "## Rare-Class Snapshot",
            render_table(rare_rows),
            "",
            "## Sampling Plan",
            render_table(sampling_plan_table),
            "",
            "## Best Result",
            (
                f"- `{best_row['strategy']}` achieved the highest macro F1 "
                f"({best_row['f1_macro']:.6f}) with accuracy {best_row['accuracy']:.6f}."
            ),
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

    numeric_columns, categorical_columns, dropped_columns = select_feature_columns(
        train_frame,
        test_frame,
        label_column,
    )
    if categorical_columns:
        raise ValueError(
            "Question 3.2(a) currently expects a numeric-only feature matrix for SMOTE-based "
            f"strategies, but found categorical columns: {categorical_columns}"
        )

    feature_columns = numeric_columns
    class_names = sorted(train_frame[label_column].astype(str).unique().tolist())
    sampling_strategy = build_sampling_strategy(
        train_frame[label_column].astype(str),
        args.minority_target_count,
    )
    smote_k_neighbors = resolve_smote_neighbors(
        train_frame[label_column].astype(str),
        sampling_strategy,
        args.smote_k_neighbors,
    )

    if sampling_strategy:
        LOGGER.info("Selective SMOTE target classes: %s", sampling_strategy)
    else:
        LOGGER.warning(
            "No class fell below the minority target count (%s). SMOTE-based strategies will act like the baseline.",
            args.minority_target_count,
        )

    strategy_specs = build_strategy_specs()
    summary_table, per_class_table, strategy_sampling_table = evaluate_strategies(
        train_frame=train_frame,
        test_frame=test_frame,
        label_column=label_column,
        feature_columns=feature_columns,
        class_names=class_names,
        strategy_specs=strategy_specs,
        model_dir=model_dir,
        random_seed=args.random_seed,
        sampling_strategy=sampling_strategy,
        smote_k_neighbors=smote_k_neighbors,
        hybrid_majority_cap=args.hybrid_majority_cap,
    )

    sampling_plan_table = build_sampling_plan_table(
        train_frame,
        label_column=label_column,
        sampling_strategy=sampling_strategy,
        hybrid_majority_cap=args.hybrid_majority_cap,
    )

    summary_path = table_dir / "q3_2a_imbalance_summary.csv"
    per_class_path = table_dir / "q3_2a_per_class_metrics.csv"
    sampling_plan_path = table_dir / "q3_2a_sampling_plan.csv"
    strategy_sampling_path = table_dir / "q3_2a_strategy_sampling.csv"
    report_path = table_dir / "q3_2a_report.md"
    summary_json_path = table_dir / "q3_2a_summary.json"

    summary_table.to_csv(summary_path, index=False)
    per_class_table.to_csv(per_class_path, index=False)
    sampling_plan_table.to_csv(sampling_plan_path, index=False)
    strategy_sampling_table.to_csv(strategy_sampling_path, index=False)
    write_report(
        summary_table,
        per_class_table,
        sampling_plan_table,
        train_source=datasets.train_source,
        test_source=datasets.test_source,
        label_column=label_column,
        feature_columns=feature_columns,
        dropped_columns=dropped_columns,
        minority_target_count=args.minority_target_count,
        smote_k_neighbors=smote_k_neighbors,
        hybrid_majority_cap=args.hybrid_majority_cap,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        report_path=report_path,
    )

    summary_payload = {
        "train_path": datasets.train_source,
        "test_path": datasets.test_source,
        "label_column": label_column,
        "feature_columns": feature_columns,
        "dropped_columns": dropped_columns,
        "minority_target_count": args.minority_target_count,
        "smote_k_neighbors": smote_k_neighbors,
        "hybrid_majority_cap": args.hybrid_majority_cap,
        "max_train_rows": args.max_train_rows,
        "max_test_rows": args.max_test_rows,
        "sampling_strategy": sampling_strategy,
        "results": summary_table.to_dict(orient="records"),
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    LOGGER.info("Wrote imbalance summary to %s", summary_path)
    LOGGER.info("Wrote per-class metrics to %s", per_class_path)
    LOGGER.info("Wrote sampling plan to %s", sampling_plan_path)
    LOGGER.info("Wrote strategy sampling table to %s", strategy_sampling_path)
    LOGGER.info("Wrote markdown report to %s", report_path)
    LOGGER.info("Wrote JSON summary to %s", summary_json_path)


if __name__ == "__main__":
    main()
