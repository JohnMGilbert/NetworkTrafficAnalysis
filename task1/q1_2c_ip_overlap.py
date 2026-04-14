"""Question 1.2(c): IP overlap and flow-level cross-router correlation analysis."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.paths import ensure_directory
from task1.q1_1a_summary import group_router_files, iter_router_frames
from task1.q1_2b_graphs import build_maximum_spanning_tree, edge_table, load_similarity


@dataclass
class RouterOverlapAccumulator:
    router_name: str
    src_ips: set[str] = field(default_factory=set)
    dst_ips: set[str] = field(default_factory=set)
    tuples: set[tuple[int, int, int]] = field(default_factory=set)

    def update(self, frame: pd.DataFrame) -> None:
        self.src_ips.update(frame["src_ip"].dropna().astype(str).unique().tolist())
        self.dst_ips.update(frame["dst_ip"].dropna().astype(str).unique().tolist())

        tuple_frame = (
            frame[["src_port", "dst_port", "protocol"]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .astype(int)
            .drop_duplicates()
        )
        self.tuples.update(map(tuple, tuple_frame.to_records(index=False).tolist()))

    @property
    def all_ips(self) -> set[str]:
        return self.src_ips | self.dst_ips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=CONFIG.raw_data_dir,
        help="Directory containing per-router CSV or parquet files.",
    )
    parser.add_argument(
        "--similarity-csv",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "tables" / "q1_2a_router_similarity_matrix.csv",
        help="Path to the Q1.2(a) similarity matrix CSV.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "tables",
        help="Directory for generated Q1.2(c) tables.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "figures",
        help="Directory for generated Q1.2(c) figures.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="CSV chunk size used to keep memory bounded.",
    )
    return parser.parse_args()


def jaccard(left: set, right: set) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def summarize_router_group(router: str, paths: list[Path], chunksize: int) -> RouterOverlapAccumulator:
    accumulator = RouterOverlapAccumulator(router_name=router)
    for path in sorted(paths):
        for frame in iter_router_frames(path, chunksize):
            accumulator.update(frame)
    return accumulator


def build_pairwise_tables(accumulators: list[RouterOverlapAccumulator]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    routers = [acc.router_name for acc in accumulators]
    ip_matrix = pd.DataFrame(index=routers, columns=routers, dtype=float)
    tuple_matrix = pd.DataFrame(index=routers, columns=routers, dtype=float)
    pair_rows = []

    for left in accumulators:
        for right in accumulators:
            ip_score = jaccard(left.all_ips, right.all_ips)
            tuple_score = jaccard(left.tuples, right.tuples)
            ip_matrix.loc[left.router_name, right.router_name] = ip_score
            tuple_matrix.loc[left.router_name, right.router_name] = tuple_score

            if left.router_name >= right.router_name:
                continue
            pair_rows.append(
                {
                    "router_a": left.router_name,
                    "router_b": right.router_name,
                    "shared_ip_count": len(left.all_ips & right.all_ips),
                    "ip_jaccard": round(ip_score, 6),
                    "tuple_jaccard": round(tuple_score, 6),
                }
            )

    return ip_matrix, tuple_matrix, pd.DataFrame(pair_rows).sort_values(
        ["ip_jaccard", "tuple_jaccard", "shared_ip_count"],
        ascending=[False, False, False],
    )


def normalize_feature_similarity(similarity: pd.DataFrame) -> pd.DataFrame:
    return (similarity + 1.0) / 2.0


def refine_similarity(
    feature_similarity: pd.DataFrame,
    ip_jaccard: pd.DataFrame,
    tuple_jaccard: pd.DataFrame,
) -> pd.DataFrame:
    normalized_feature = normalize_feature_similarity(feature_similarity)
    refined = 0.5 * normalized_feature + 0.25 * ip_jaccard + 0.25 * tuple_jaccard
    for router in refined.index:
        refined.loc[router, router] = 1.0
    return refined


def compare_graphs(base_graph: nx.Graph, refined_graph: nx.Graph) -> dict[str, object]:
    base_edges = {tuple(sorted(edge[:2])) for edge in base_graph.edges()}
    refined_edges = {tuple(sorted(edge[:2])) for edge in refined_graph.edges()}
    retained = sorted(base_edges & refined_edges)
    added = sorted(refined_edges - base_edges)
    removed = sorted(base_edges - refined_edges)
    return {
        "retained_edges": retained,
        "added_edges": added,
        "removed_edges": removed,
    }


def build_discussion(pairwise: pd.DataFrame, comparison: dict[str, object]) -> str:
    top_pairs = pairwise.head(5)
    retained = ", ".join(f"{a}-{b}" for a, b in comparison["retained_edges"]) or "none"
    added = ", ".join(f"{a}-{b}" for a, b in comparison["added_edges"]) or "none"
    removed = ", ".join(f"{a}-{b}" for a, b in comparison["removed_edges"]) or "none"

    lines = [
        "# Q1.2(c) Discussion",
        "",
        "Top router pairs by IP/tuple overlap:",
    ]
    for _, row in top_pairs.iterrows():
        lines.append(
            f"- {row['router_a']}-{row['router_b']}: shared_ip_count={int(row['shared_ip_count'])}, "
            f"ip_jaccard={row['ip_jaccard']}, tuple_jaccard={row['tuple_jaccard']}"
        )

    lines.extend(
        [
            "",
            f"Edges retained from the feature-based MST: {retained}.",
            f"Edges added by the IP-based refinement: {added}.",
            f"Edges removed by the IP-based refinement: {removed}.",
            "",
            "If a feature-based edge is retained after adding IP and tuple overlap evidence, the two analyses are reinforcing each other.",
            "If an edge is removed or replaced, the IP-based analysis is contradicting the purely feature-based result and suggesting a different communication path pattern.",
        ]
    )
    return "\n".join(lines) + "\n"


def draw_graph(ax: plt.Axes, graph: nx.Graph, title: str) -> None:
    pos = nx.spring_layout(graph, seed=CONFIG.random_seed, weight="weight")
    weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()])
    widths = 1.5 + 3.5 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-9) if weights.size else 2.0
    nx.draw_networkx_nodes(graph, pos, node_color="#8ecae6", edgecolors="#1f1f1f", node_size=900, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(graph, pos, width=widths, edge_color=weights if weights.size else "#666666", edge_cmap=plt.cm.plasma if weights.size else None, ax=ax)
    ax.set_title(title)
    ax.axis("off")


def save_overlap_heatmap(
    matrix: pd.DataFrame,
    figure_path: Path,
    title: str,
    color_map: str,
    colorbar_label: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=color_map,
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.35,
        linecolor="white",
        annot_kws={"size": 9},
        cbar_kws={"label": colorbar_label, "shrink": 0.78},
        ax=ax,
    )
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("Router")
    ax.set_ylabel("Router")
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def save_refined_graph_figure(refined_graph: nx.Graph, figure_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    draw_graph(ax, refined_graph, "Refined MST (Feature + IP + Tuple Evidence)")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def make_figure(ip_matrix: pd.DataFrame, tuple_matrix: pd.DataFrame, refined_graph: nx.Graph, figure_dir: Path) -> dict[str, Path]:
    sns.set_theme(style="white", context="notebook")
    return {
        "ip_heatmap": save_overlap_heatmap(
            ip_matrix,
            figure_dir / "q1_2c_ip_jaccard_heatmap.png",
            "Combined IP-Set Overlap",
            "Blues",
            "IP Jaccard",
        ),
        "tuple_heatmap": save_overlap_heatmap(
            tuple_matrix,
            figure_dir / "q1_2c_tuple_jaccard_heatmap.png",
            "(src_port, dst_port, protocol) Tuple Overlap",
            "Greens",
            "Tuple Jaccard",
        ),
        "refined_graph": save_refined_graph_figure(
            refined_graph,
            figure_dir / "q1_2c_refined_mst.png",
        ),
    }


def main() -> None:
    args = parse_args()

    router_files = [
        path for path in sorted(args.data_dir.glob("*.csv")) + sorted(args.data_dir.glob("*.parquet"))
        if path.is_file()
    ]
    if not router_files:
        raise FileNotFoundError(f"No router files were found under {args.data_dir}.")

    grouped = group_router_files(router_files)
    ordered_items = sorted(grouped.items(), key=lambda item: int(item[0][1:]))
    accumulators = [summarize_router_group(router, paths, args.chunksize) for router, paths in ordered_items]

    ip_matrix, tuple_matrix, pairwise = build_pairwise_tables(accumulators)
    feature_similarity = load_similarity(args.similarity_csv)
    base_mst = build_maximum_spanning_tree(feature_similarity)

    refined_similarity = refine_similarity(feature_similarity, ip_matrix, tuple_matrix)
    refined_mst = build_maximum_spanning_tree(refined_similarity)
    refined_mst.graph["method"] = "Refined Maximum Spanning Tree"

    comparison = compare_graphs(base_mst, refined_mst)

    table_dir = ensure_directory(args.table_dir)
    figure_dir = ensure_directory(args.figure_dir)

    ip_matrix_path = table_dir / "q1_2c_ip_jaccard_matrix.csv"
    tuple_matrix_path = table_dir / "q1_2c_tuple_jaccard_matrix.csv"
    pairwise_path = table_dir / "q1_2c_pairwise_overlap_summary.csv"
    refined_edges_path = table_dir / "q1_2c_refined_graph_edges.csv"
    discussion_path = table_dir / "q1_2c_discussion.md"

    ip_matrix.to_csv(ip_matrix_path, index=True)
    tuple_matrix.to_csv(tuple_matrix_path, index=True)
    pairwise.to_csv(pairwise_path, index=False)
    edge_table(refined_mst).to_csv(refined_edges_path, index=False)
    discussion_path.write_text(build_discussion(pairwise, comparison), encoding="utf-8")
    figure_paths = make_figure(ip_matrix, tuple_matrix, refined_mst, figure_dir)

    print(f"Wrote IP Jaccard matrix to {ip_matrix_path}")
    print(f"Wrote tuple Jaccard matrix to {tuple_matrix_path}")
    print(f"Wrote pairwise overlap summary to {pairwise_path}")
    print(f"Wrote refined graph edges to {refined_edges_path}")
    print(f"Wrote discussion notes to {discussion_path}")
    for figure_name, figure_path in figure_paths.items():
        print(f"Wrote {figure_name} figure to {figure_path}")


if __name__ == "__main__":
    main()
