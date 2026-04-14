"""Question 2.1(a): dataset cleaning, duplicate handling, and standardization."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.io import list_router_files, normalize_columns
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory


LOGGER = logging.getLogger("task2.q2_1a")
METADATA_COLUMNS = ("router_id", "src_ip", "dst_ip", "timestamp")
VARIANCE_EPSILON = 1e-12
ROUTER_SUFFIX_PATTERN = re.compile(r"(?:^|[-_])D?(\d+)$", flags=re.IGNORECASE)


@dataclass
class FeatureAccumulator:
    name: str
    total_rows: int = 0
    missing_count: int = 0
    infinite_count: int = 0
    valid_count: int = 0
    value_sum: float = 0.0
    value_sumsq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, series: pd.Series) -> None:
        numeric = pd.to_numeric(series, errors="coerce")
        self.total_rows += int(len(numeric))

        infinite_mask = np.isinf(numeric.to_numpy(dtype=np.float64, na_value=np.nan))
        self.infinite_count += int(infinite_mask.sum())

        missing_mask = numeric.isna().to_numpy() | infinite_mask
        self.missing_count += int(missing_mask.sum())

        clean = numeric.mask(infinite_mask, np.nan).dropna()
        if clean.empty:
            return

        values = clean.to_numpy(dtype=np.float64, copy=False)
        self.valid_count += int(values.size)
        self.value_sum += float(values.sum())
        self.value_sumsq += float(np.square(values).sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    @property
    def mean(self) -> float | None:
        if self.valid_count == 0:
            return None
        return self.value_sum / self.valid_count

    @property
    def variance(self) -> float | None:
        if self.valid_count == 0:
            return None
        mean = self.mean
        if mean is None:
            return None
        variance = (self.value_sumsq / self.valid_count) - (mean * mean)
        return max(float(variance), 0.0)

    @property
    def std(self) -> float | None:
        variance = self.variance
        if variance is None:
            return None
        return math.sqrt(variance)

    @property
    def missing_rate(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return self.missing_count / self.total_rows

    @property
    def removable_reason(self) -> str | None:
        if self.valid_count == 0:
            return "all_values_missing_or_infinite"
        variance = self.variance
        if variance is not None and variance <= VARIANCE_EPSILON:
            return "zero_variance"
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=CONFIG.raw_data_dir,
        help="Directory containing router CSV/parquet files.",
    )
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=CONFIG.processed_data_dir / "task2_preprocessed_standardized.parquet",
        help="Output parquet path for the cleaned standardized dataset.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables",
        help="Directory for generated Task 2.1(a) tables and report files.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="CSV chunk size used during streaming passes.",
    )
    return parser.parse_args()


def infer_router_id(path: Path) -> str:
    match = ROUTER_SUFFIX_PATTERN.search(path.stem)
    if match:
        return f"D{int(match.group(1))}"
    return path.stem


def iter_router_frames(path: Path, chunksize: int):
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        yield normalize_columns(frame)
        return

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        yield normalize_columns(chunk)


def discover_numeric_features(files: list[Path], chunksize: int) -> list[str]:
    for path in files:
        for frame in iter_router_frames(path, chunksize):
            numeric_columns: list[str] = []
            for column in frame.columns:
                if column in METADATA_COLUMNS:
                    continue
                if pd.api.types.is_numeric_dtype(frame[column]):
                    numeric_columns.append(column)
            return numeric_columns
    raise ValueError("No router files were found under the provided data directory.")


def initial_feature_accumulators(feature_columns: list[str]) -> dict[str, FeatureAccumulator]:
    return {
        column: FeatureAccumulator(name=column)
        for column in feature_columns
    }


def duplicate_mask_from_hashes(hashes: np.ndarray, seen_hashes: set[int]) -> np.ndarray:
    duplicate_mask = np.zeros(len(hashes), dtype=bool)
    local_hashes: set[int] = set()

    for index, value in enumerate(hashes):
        hashed_value = int(value)
        if hashed_value in seen_hashes or hashed_value in local_hashes:
            duplicate_mask[index] = True
        else:
            local_hashes.add(hashed_value)

    seen_hashes.update(local_hashes)
    return duplicate_mask


def scan_dataset(
    files: list[Path],
    feature_columns: list[str],
    chunksize: int,
) -> tuple[dict[str, FeatureAccumulator], dict[str, int]]:
    accumulators = initial_feature_accumulators(feature_columns)
    seen_hashes: set[int] = set()
    total_rows = 0
    duplicate_rows = 0

    for path in files:
        router_id = infer_router_id(path)
        LOGGER.info("Scanning %s", path.name)
        for frame in iter_router_frames(path, chunksize):
            frame = frame.copy()
            frame.insert(0, "router_id", router_id)
            total_rows += int(len(frame))

            dedupe_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64, copy=False)
            duplicate_mask = duplicate_mask_from_hashes(dedupe_hashes, seen_hashes)
            duplicate_rows += int(duplicate_mask.sum())

            for column in feature_columns:
                accumulators[column].update(frame[column])

    dataset_counts = {
        "total_rows_before_deduplication": total_rows,
        "duplicate_rows_removed": duplicate_rows,
        "rows_after_deduplication": total_rows - duplicate_rows,
    }
    return accumulators, dataset_counts


def feature_audit_table(accumulators: dict[str, FeatureAccumulator]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in sorted(accumulators):
        stats = accumulators[column]
        rows.append(
            {
                "feature": column,
                "valid_count": stats.valid_count,
                "missing_or_infinite_count": stats.missing_count,
                "missing_or_infinite_rate": round(stats.missing_rate, 6),
                "minimum_before_cleaning": normalize_number(stats.minimum),
                "maximum_before_cleaning": normalize_number(stats.maximum),
                "mean_before_cleaning": normalize_number(stats.mean),
                "std_before_cleaning": normalize_number(stats.std),
                "variance_before_cleaning": normalize_number(stats.variance),
                "removal_reason": stats.removable_reason or "",
            }
        )
    return pd.DataFrame(rows)


def normalize_number(value: float | None) -> float | None:
    if value is None or value in (math.inf, -math.inf):
        return None
    return round(float(value), 6)


def build_scaler_table(
    accumulators: dict[str, FeatureAccumulator],
    retained_features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in retained_features:
        stats = accumulators[column]
        mean = stats.mean if stats.mean is not None else 0.0
        std = stats.std if stats.std not in (None, 0.0) else 1.0
        rows.append(
            {
                "feature": column,
                "imputation_value_mean": normalize_number(mean),
                "scaling_mean": normalize_number(mean),
                "scaling_std": normalize_number(std),
            }
        )
    return pd.DataFrame(rows)


def write_processed_dataset(
    files: list[Path],
    retained_features: list[str],
    accumulators: dict[str, FeatureAccumulator],
    processed_path: Path,
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    ensure_directory(processed_path.parent)

    means = {
        column: (accumulators[column].mean if accumulators[column].mean is not None else 0.0)
        for column in retained_features
    }
    stds = {
        column: (accumulators[column].std if accumulators[column].std not in (None, 0.0) else 1.0)
        for column in retained_features
    }

    seen_hashes: set[int] = set()
    writer: pq.ParquetWriter | None = None
    transformed_accumulators = initial_feature_accumulators(retained_features)
    duplicate_rows = 0
    written_rows = 0

    try:
        if processed_path.exists():
            processed_path.unlink()

        for path in files:
            router_id = infer_router_id(path)
            LOGGER.info("Transforming %s", path.name)
            for frame in iter_router_frames(path, chunksize):
                frame = frame.copy()
                frame.insert(0, "router_id", router_id)

                dedupe_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64, copy=False)
                duplicate_mask = duplicate_mask_from_hashes(dedupe_hashes, seen_hashes)
                duplicate_rows += int(duplicate_mask.sum())

                unique_frame = frame.loc[~duplicate_mask].reset_index(drop=True)
                if unique_frame.empty:
                    continue

                output_columns: dict[str, pd.Series] = {}
                for column in METADATA_COLUMNS:
                    if column in unique_frame.columns:
                        output_columns[column] = unique_frame[column]

                for column in retained_features:
                    numeric = pd.to_numeric(unique_frame[column], errors="coerce").astype(np.float64)
                    numeric = numeric.mask(np.isinf(numeric), np.nan)
                    filled = numeric.fillna(means[column])
                    standardized = (filled - means[column]) / stds[column]
                    output_columns[column] = standardized.astype(np.float32)
                    transformed_accumulators[column].update(output_columns[column])

                output_frame = pd.DataFrame(output_columns)
                written_rows += int(len(output_frame))

                table = pa.Table.from_pandas(output_frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(processed_path, table.schema, compression="snappy")
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    after_rows: list[dict[str, object]] = []
    for column in retained_features:
        stats = transformed_accumulators[column]
        after_rows.append(
            {
                "feature": column,
                "rows_after_cleaning": stats.total_rows,
                "missing_or_infinite_after_cleaning": stats.missing_count,
                "minimum_after_standardization": normalize_number(stats.minimum),
                "maximum_after_standardization": normalize_number(stats.maximum),
                "mean_after_standardization": normalize_number(stats.mean),
                "std_after_standardization": normalize_number(stats.std),
            }
        )

    dataset_counts = {
        "duplicate_rows_removed": duplicate_rows,
        "rows_written": written_rows,
    }
    return pd.DataFrame(after_rows), dataset_counts


def write_markdown_report(
    destination: Path,
    dataset_counts: dict[str, int],
    removed_features: list[str],
    retained_features: list[str],
) -> None:
    lines = [
        "# Task 2.1(a) Preprocessing Summary",
        "",
        "## Cleaning decisions",
        "- Router file shards were normalized to assignment router IDs such as `D1` through `D10` based on the trailing filename suffix.",
        "- Duplicates were defined as rows that match on every original column within the same normalized router.",
        "- Missing values and infinite values were converted to NaN, then imputed with each feature mean before scaling.",
        "- Features with no usable finite values or zero variance were removed before standardization.",
        "- Remaining numeric flow features were z-score standardized.",
        "",
        "## Standardization choice",
        "Standardization was used instead of min-max scaling because Task 2.1(b) applies PCA, and PCA is sensitive to feature scale and benefits from centered unit-variance inputs.",
        "Standardization was preferred over robust scaling here because extreme rates and counts can be signal for attack traffic rather than noise, and z-score scaling preserves those deviations while still making heterogeneous units comparable.",
        "",
        "## Dataset counts",
        f"- Rows before deduplication: {dataset_counts['total_rows_before_deduplication']:,}",
        f"- Duplicate rows removed: {dataset_counts['duplicate_rows_removed']:,}",
        f"- Rows after deduplication: {dataset_counts['rows_after_deduplication']:,}",
        f"- Features retained: {len(retained_features)}",
        f"- Features removed: {len(removed_features)}",
        "",
        "## Removed features",
    ]
    lines.extend(["- None"] if not removed_features else [f"- {feature}" for feature in removed_features])
    lines.append("")
    report = "\n".join(lines)
    destination.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()

    files = list_router_files(args.data_dir)
    if not files:
        raise FileNotFoundError(f"No router files found under {args.data_dir}")

    ensure_directory(args.table_dir)
    feature_columns = discover_numeric_features(files, args.chunksize)

    accumulators, dataset_counts = scan_dataset(
        files=files,
        feature_columns=feature_columns,
        chunksize=args.chunksize,
    )
    audit = feature_audit_table(accumulators)

    removed_features = audit.loc[audit["removal_reason"] != "", "feature"].tolist()
    retained_features = audit.loc[audit["removal_reason"] == "", "feature"].tolist()

    scaler_table = build_scaler_table(accumulators, retained_features)
    after_stats, write_counts = write_processed_dataset(
        files=files,
        retained_features=retained_features,
        accumulators=accumulators,
        processed_path=args.processed_path,
        chunksize=args.chunksize,
    )

    if write_counts["duplicate_rows_removed"] != dataset_counts["duplicate_rows_removed"]:
        raise RuntimeError("Duplicate accounting mismatch between scan and transform passes.")

    audit.to_csv(args.table_dir / "q2_1a_feature_audit.csv", index=False)
    audit.loc[audit["removal_reason"] == "", [
        "feature",
        "valid_count",
        "missing_or_infinite_count",
        "missing_or_infinite_rate",
        "minimum_before_cleaning",
        "maximum_before_cleaning",
        "mean_before_cleaning",
        "std_before_cleaning",
    ]].to_csv(args.table_dir / "q2_1a_before_stats.csv", index=False)
    after_stats.to_csv(args.table_dir / "q2_1a_after_stats.csv", index=False)
    scaler_table.to_csv(args.table_dir / "q2_1a_scaler_parameters.csv", index=False)

    summary_payload = {
        **dataset_counts,
        "features_before_filtering": len(feature_columns),
        "features_removed": len(removed_features),
        "features_retained": len(retained_features),
        "processed_dataset_path": str(args.processed_path),
    }
    (args.table_dir / "q2_1a_dataset_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(
        destination=args.table_dir / "q2_1a_preprocessing_report.md",
        dataset_counts=dataset_counts,
        removed_features=removed_features,
        retained_features=retained_features,
    )

    LOGGER.info("Wrote processed dataset to %s", args.processed_path)
    LOGGER.info("Retained %s features and removed %s features", len(retained_features), len(removed_features))


if __name__ == "__main__":
    main()
