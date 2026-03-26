"""Create graphs from the Question 1.1(a) router summary table."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.paths import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "tables" / "q1_1a_router_summary.csv",
        help="Path to the generated Q1.1(a) summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "figures",
        help="Directory where plots should be written.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.sort_values("router", key=lambda s: s.str[1:].astype(int)).reset_index(drop=True)


def make_dashboard(summary: pd.DataFrame, output_dir: Path) -> Path:
    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Task 1.1(a): Router-Level Summary Statistics", fontsize=20, y=0.98)

    sns.barplot(
        data=summary,
        x="router",
        y="flow_records",
        hue="router",
        dodge=False,
        palette="crest",
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Flow Records by Router")
    axes[0, 0].set_xlabel("Router")
    axes[0, 0].set_ylabel("Flow Records")
    axes[0, 0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    sns.barplot(
        data=summary,
        x="router",
        y="flow_duration_mean",
        hue="router",
        dodge=False,
        palette="flare",
        legend=False,
        ax=axes[0, 1],
    )
    axes[0, 1].errorbar(
        x=range(len(summary)),
        y=summary["flow_duration_mean"],
        yerr=summary["flow_duration_std"],
        fmt="none",
        ecolor="black",
        elinewidth=1.4,
        capsize=5,
    )
    axes[0, 1].scatter(
        range(len(summary)),
        summary["flow_duration_median"],
        color="black",
        marker="D",
        s=45,
        label="Median",
        zorder=4,
    )
    axes[0, 1].set_title("Mean Flow Duration with Std and Median")
    axes[0, 1].set_xlabel("Router")
    axes[0, 1].set_ylabel("Flow Duration")
    axes[0, 1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[0, 1].legend(frameon=True)

    sns.barplot(
        data=summary,
        x="router",
        y="forward_to_backward_packet_ratio",
        hue="router",
        dodge=False,
        palette="mako",
        legend=False,
        ax=axes[1, 0],
    )
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    axes[1, 0].set_title("Forward / Backward Packet Ratio")
    axes[1, 0].set_xlabel("Router")
    axes[1, 0].set_ylabel("Packet Ratio")

    port_rows = []
    for _, row in summary.iterrows():
        for rank, entry in enumerate(str(row["top_5_destination_ports"]).split("; "), start=1):
            if not entry:
                continue
            port, count_text = entry.rsplit(" (", 1)
            port_rows.append(
                {
                    "router": row["router"],
                    "rank": rank,
                    "port": port,
                    "count": int(count_text.rstrip(")")),
                }
            )

    port_frame = pd.DataFrame(port_rows)
    heatmap_frame = port_frame.pivot(index="router", columns="rank", values="count")
    annotations = port_frame.pivot(index="router", columns="rank", values="port")
    sns.heatmap(
        heatmap_frame,
        annot=annotations,
        fmt="",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Port Frequency"},
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Top-5 Destination Ports per Router")
    axes[1, 1].set_xlabel("Top-Port Rank")
    axes[1, 1].set_ylabel("Router")

    fig.tight_layout()
    output_path = output_dir / "q1_1a_router_summary_dashboard.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    summary = load_summary(args.summary_csv)
    output_path = make_dashboard(summary, output_dir)
    print(f"Wrote Q1.1(a) dashboard to {output_path}")


if __name__ == "__main__":
    main()
