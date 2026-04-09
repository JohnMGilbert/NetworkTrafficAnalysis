"""Question 3.2(b): analyze rare-class impact of imbalance-handling strategies."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory
from task3.q3_1a_baselines import render_table


LOGGER = logging.getLogger("task3.q3_2b")
RARE_CLASSES = ("Web-command-injection", "Web-sql-injection", "Infiltration-mitm")
STRATEGY_ORDER = (
    "No Balancing",
    "SMOTE Oversampling",
    "Class Weighting",
    "SMOTE + Undersampling Hybrid",
)
STRATEGY_SHORT_LABELS = {
    "No Balancing": "No balancing",
    "SMOTE Oversampling": "Oversampling",
    "Class Weighting": "Class weighting",
    "SMOTE + Undersampling Hybrid": "Hybrid",
}
STRATEGY_COLORS = {
    "No Balancing": "#4C6A92",
    "SMOTE Oversampling": "#D97D54",
    "Class Weighting": "#3A9D5D",
    "SMOTE + Undersampling Hybrid": "#8C6BB1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-class-path",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_2a_per_class_metrics.csv",
        help="Per-class metrics CSV produced by Question 3.2(a).",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables" / "q3_2a_imbalance_summary.csv",
        help="Strategy summary CSV produced by Question 3.2(a).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "figures",
        help="Directory for generated Task 3.2(b) figures.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task3" / "tables",
        help="Directory for generated Task 3.2(b) tables and reports.",
    )
    return parser.parse_args()


def load_inputs(per_class_path: Path, summary_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not per_class_path.exists():
        raise FileNotFoundError(
            f"Missing per-class metrics file: {per_class_path}. Run Question 3.2(a) first."
        )
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing strategy summary file: {summary_path}. Run Question 3.2(a) first."
        )

    per_class = pd.read_csv(per_class_path)
    summary = pd.read_csv(summary_path)
    return per_class, summary


def build_rare_class_table(per_class: pd.DataFrame) -> pd.DataFrame:
    rare_table = per_class[per_class["class_name"].isin(RARE_CLASSES)].copy()
    rare_table["strategy"] = pd.Categorical(
        rare_table["strategy"],
        categories=STRATEGY_ORDER,
        ordered=True,
    )
    rare_table["class_name"] = pd.Categorical(
        rare_table["class_name"],
        categories=RARE_CLASSES,
        ordered=True,
    )
    rare_table = rare_table.sort_values(["class_name", "strategy"]).reset_index(drop=True)
    rare_table["strategy_label"] = rare_table["strategy"].map(STRATEGY_SHORT_LABELS)
    return rare_table


def build_rare_summary_table(rare_table: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for strategy in STRATEGY_ORDER:
        strategy_rows = rare_table[rare_table["strategy"] == strategy]
        if strategy_rows.empty:
            continue
        summary_rows.append(
            {
                "strategy": strategy,
                "strategy_label": STRATEGY_SHORT_LABELS[strategy],
                "mean_rare_f1": round(float(strategy_rows["f1_score"].mean()), 6),
                "min_rare_f1": round(float(strategy_rows["f1_score"].min()), 6),
                "max_rare_f1": round(float(strategy_rows["f1_score"].max()), 6),
            }
        )
    return pd.DataFrame(summary_rows)


def build_majority_tradeoff_table(per_class: pd.DataFrame) -> pd.DataFrame:
    majority = per_class[~per_class["class_name"].isin(RARE_CLASSES)].copy()
    baseline = majority[majority["strategy"] == "No Balancing"][
        ["class_name", "f1_score", "support"]
    ].rename(columns={"f1_score": "baseline_f1", "support": "baseline_support"})

    rows: list[dict[str, object]] = []
    for strategy in STRATEGY_ORDER:
        strategy_rows = majority[majority["strategy"] == strategy][["class_name", "f1_score", "support"]]
        merged = strategy_rows.merge(baseline, on="class_name", how="left")
        merged["delta_vs_baseline"] = merged["f1_score"] - merged["baseline_f1"]

        rows.append(
            {
                "strategy": strategy,
                "strategy_label": STRATEGY_SHORT_LABELS[strategy],
                "mean_majority_f1": round(float(merged["f1_score"].mean()), 6),
                "weighted_majority_f1": round(
                    float(np.average(merged["f1_score"], weights=merged["support"])),
                    6,
                ),
                "mean_majority_delta_vs_baseline": round(float(merged["delta_vs_baseline"].mean()), 6),
                "worst_majority_delta_vs_baseline": round(float(merged["delta_vs_baseline"].min()), 6),
                "best_majority_delta_vs_baseline": round(float(merged["delta_vs_baseline"].max()), 6),
            }
        )

    tradeoff = pd.DataFrame(rows)
    tradeoff["strategy"] = pd.Categorical(tradeoff["strategy"], categories=STRATEGY_ORDER, ordered=True)
    return tradeoff.sort_values("strategy").reset_index(drop=True)


def build_majority_deltas_table(per_class: pd.DataFrame) -> pd.DataFrame:
    majority = per_class[~per_class["class_name"].isin(RARE_CLASSES)].copy()
    baseline = majority[majority["strategy"] == "No Balancing"][["class_name", "f1_score"]].rename(
        columns={"f1_score": "baseline_f1"}
    )
    merged = majority.merge(baseline, on="class_name", how="left")
    merged["delta_vs_baseline"] = merged["f1_score"] - merged["baseline_f1"]
    merged["strategy"] = pd.Categorical(merged["strategy"], categories=STRATEGY_ORDER, ordered=True)
    return merged.sort_values(["strategy", "class_name"]).reset_index(drop=True)


def select_best_rare_strategy(rare_summary: pd.DataFrame) -> pd.Series:
    ranked = rare_summary.sort_values(["mean_rare_f1", "min_rare_f1"], ascending=False)
    return ranked.iloc[0]


def create_grouped_bar_chart(rare_table: pd.DataFrame, figure_path: Path) -> None:
    classes = list(RARE_CLASSES)
    strategies = list(STRATEGY_ORDER)
    x = np.arange(len(classes))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 7))
    for index, strategy in enumerate(strategies):
        strategy_rows = rare_table[rare_table["strategy"] == strategy]
        values = []
        for class_name in classes:
            match = strategy_rows[strategy_rows["class_name"] == class_name]
            values.append(float(match.iloc[0]["f1_score"]) if not match.empty else 0.0)

        offsets = x + (index - 1.5) * width
        bars = ax.bar(
            offsets,
            values,
            width=width,
            label=STRATEGY_SHORT_LABELS[strategy],
            color=STRATEGY_COLORS[strategy],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.02, value + 0.02),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Web Command\nInjection", "Web SQL\nInjection", "Infiltration\nMITM"],
        fontsize=11,
    )
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_title("Question 3.2(b): Rare-Class F1 by Imbalance Strategy", fontsize=14, pad=14)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, fontsize=10)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(
    rare_table: pd.DataFrame,
    rare_summary: pd.DataFrame,
    majority_tradeoff: pd.DataFrame,
    majority_deltas: pd.DataFrame,
    overall_summary: pd.DataFrame,
    *,
    report_path: Path,
    figure_path: Path,
) -> None:
    best_rare = select_best_rare_strategy(rare_summary)
    best_strategy = str(best_rare["strategy"])
    best_majority_row = majority_tradeoff[majority_tradeoff["strategy"] == best_strategy].iloc[0]

    largest_majority_drop = majority_deltas[
        (majority_deltas["strategy"] == best_strategy) & (majority_deltas["delta_vs_baseline"] < 0)
    ].sort_values("delta_vs_baseline").head(3)

    lines = [
        "# Task 3.2(b) Rare-Class Analysis Report",
        "",
        "## Rare-Class F1 Comparison",
        f"![Rare-Class F1 Chart]({figure_path})",
        "",
        render_table(
            rare_table[["strategy", "class_name", "f1_score", "support"]].rename(
                columns={"f1_score": "f1_score_test"}
            )
        ),
        "",
        "## Rare-Class Summary",
        render_table(rare_summary),
        "",
        "## Majority-Class Tradeoff",
        render_table(majority_tradeoff),
        "",
        "## Interpretation",
        (
            f"- `{best_strategy}` helps the rare classes the most overall, with mean rare-class F1 "
            f"{best_rare['mean_rare_f1']:.6f}."
        ),
        (
            f"- The biggest rare-class change is on `Web-sql-injection`, where `{best_strategy}` "
            "substantially outperforms the unbalanced baseline."
        ),
        (
            f"- For non-rare classes, `{best_strategy}` has weighted majority-class F1 "
            f"{best_majority_row['weighted_majority_f1']:.6f} and mean delta "
            f"{best_majority_row['mean_majority_delta_vs_baseline']:+.6f} versus no balancing."
        ),
    ]

    if largest_majority_drop.empty:
        lines.append(
            f"- `{best_strategy}` does not introduce any measurable majority-class F1 drop relative to no balancing."
        )
    else:
        formatted_drops = ", ".join(
            f"{row.class_name} ({row.delta_vs_baseline:+.6f})" for row in largest_majority_drop.itertuples()
        )
        lines.append(
            f"- The clearest majority-class costs under `{best_strategy}` are: {formatted_drops}."
        )

    lines.extend(
        [
            "",
            "## Overall Strategy Summary",
            render_table(overall_summary),
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()

    figure_dir = ensure_directory(args.figure_dir)
    table_dir = ensure_directory(args.table_dir)

    per_class, overall_summary = load_inputs(args.per_class_path, args.summary_path)
    rare_table = build_rare_class_table(per_class)
    rare_summary = build_rare_summary_table(rare_table)
    majority_tradeoff = build_majority_tradeoff_table(per_class)
    majority_deltas = build_majority_deltas_table(per_class)

    figure_path = figure_dir / "q3_2b_rare_class_f1.png"
    rare_table_path = table_dir / "q3_2b_rare_class_f1.csv"
    rare_summary_path = table_dir / "q3_2b_rare_class_summary.csv"
    majority_tradeoff_path = table_dir / "q3_2b_majority_tradeoff.csv"
    majority_deltas_path = table_dir / "q3_2b_majority_deltas.csv"
    report_path = table_dir / "q3_2b_report.md"
    summary_json_path = table_dir / "q3_2b_summary.json"

    create_grouped_bar_chart(rare_table, figure_path)
    rare_table.to_csv(rare_table_path, index=False)
    rare_summary.to_csv(rare_summary_path, index=False)
    majority_tradeoff.to_csv(majority_tradeoff_path, index=False)
    majority_deltas.to_csv(majority_deltas_path, index=False)
    write_report(
        rare_table,
        rare_summary,
        majority_tradeoff,
        majority_deltas,
        overall_summary,
        report_path=report_path,
        figure_path=figure_path,
    )

    best_rare = select_best_rare_strategy(rare_summary)
    summary_payload = {
        "rare_classes": list(RARE_CLASSES),
        "best_strategy_for_rare_classes": str(best_rare["strategy"]),
        "best_mean_rare_f1": float(best_rare["mean_rare_f1"]),
        "rare_summary": rare_summary.to_dict(orient="records"),
        "majority_tradeoff": majority_tradeoff.to_dict(orient="records"),
        "figure_path": str(figure_path),
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    LOGGER.info("Wrote rare-class figure to %s", figure_path)
    LOGGER.info("Wrote rare-class table to %s", rare_table_path)
    LOGGER.info("Wrote rare-class summary to %s", rare_summary_path)
    LOGGER.info("Wrote majority tradeoff table to %s", majority_tradeoff_path)
    LOGGER.info("Wrote majority delta table to %s", majority_deltas_path)
    LOGGER.info("Wrote markdown report to %s", report_path)
    LOGGER.info("Wrote JSON summary to %s", summary_json_path)


if __name__ == "__main__":
    main()
