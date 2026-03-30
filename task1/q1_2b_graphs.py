"""Question 1.2(b): adjacency graph construction from router similarity."""

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
import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.paths import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Directory for generated Q1.2(b) tables.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "figures",
        help="Directory for generated Q1.2(b) figures.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=2,
        help="Number of nearest neighbors per router for the kNN graph.",
    )
    return parser.parse_args()


def load_similarity(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    ordered = sorted(frame.index, key=lambda item: int(item[1:]))
    return frame.loc[ordered, ordered]


def build_knn_graph(similarity: pd.DataFrame, k: int) -> nx.Graph:
    graph = nx.Graph(method=f"{k}-NN")
    routers = list(similarity.index)
    graph.add_nodes_from(routers)

    for router in routers:
        neighbors = (
            similarity.loc[router]
            .drop(index=router)
            .sort_values(ascending=False)
            .head(k)
        )
        for neighbor, weight in neighbors.items():
            graph.add_edge(router, neighbor, weight=float(weight))
    return graph


def build_maximum_spanning_tree(similarity: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph(method="Maximum Spanning Tree")
    routers = list(similarity.index)
    graph.add_nodes_from(routers)

    for i, source in enumerate(routers):
        for target in routers[i + 1 :]:
            graph.add_edge(source, target, weight=float(similarity.loc[source, target]))

    return nx.maximum_spanning_tree(graph, weight="weight")


def edge_table(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for source, target, data in sorted(graph.edges(data=True)):
        rows.append(
            {
                "source": source,
                "target": target,
                "weight": round(float(data["weight"]), 6),
            }
        )
    return pd.DataFrame(rows)


def graph_summary(graph: nx.Graph) -> dict[str, object]:
    degrees = dict(graph.degree())
    connected = nx.is_connected(graph) if graph.number_of_nodes() > 0 else False
    return {
        "method": graph.graph.get("method", "unknown"),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "connected": connected,
        "average_degree": round(float(np.mean(list(degrees.values()))), 3),
        "density": round(float(nx.density(graph)), 6),
    }


def build_discussion(knn_graph: nx.Graph, mst_graph: nx.Graph) -> str:
    knn_summary = graph_summary(knn_graph)
    mst_summary = graph_summary(mst_graph)

    lines = [
        "# Q1.2(b) Discussion",
        "",
        (
            f"The {knn_summary['method']} graph has {knn_summary['edges']} edges, "
            f"average degree {knn_summary['average_degree']}, and connected={knn_summary['connected']}."
        ),
        (
            f"The {mst_summary['method']} graph has {mst_summary['edges']} edges, "
            f"average degree {mst_summary['average_degree']}, and connected={mst_summary['connected']}."
        ),
        (
            "The kNN graph preserves each router's strongest local affinities and can reveal dense neighborhoods, "
            "but it may introduce shortcut edges that are less plausible as direct physical links."
        ),
        (
            "The maximum spanning tree is sparser and guarantees a connected backbone with exactly N-1 edges, "
            "which is often closer to a physically plausible first-pass topology reconstruction."
        ),
        (
            "For likely physical connectivity, the maximum spanning tree is the better baseline here because it avoids "
            "over-connecting the graph while still retaining the strongest similarity-supported structure."
        ),
    ]
    return "\n\n".join(lines) + "\n"


def draw_graph(ax: plt.Axes, graph: nx.Graph, title: str) -> None:
    pos = nx.spring_layout(graph, seed=CONFIG.random_seed, weight="weight")
    weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()])
    if weights.size:
        widths = 1.5 + 3.5 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
        edge_colors = weights
    else:
        widths = 2.0
        edge_colors = "#666666"

    nx.draw_networkx_nodes(graph, pos, node_color="#f2c14e", edgecolors="#1f1f1f", node_size=950, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=11, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos,
        width=widths,
        edge_color=edge_colors,
        edge_cmap=plt.cm.inferno if weights.size else None,
        ax=ax,
    )

    edge_labels = {
        (u, v): f"{data['weight']:.2f}"
        for u, v, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, rotate=False, ax=ax)
    ax.set_title(title)
    ax.axis("off")


def make_figure(knn_graph: nx.Graph, mst_graph: nx.Graph, figure_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Task 1.2(b): Router Adjacency Graph Constructions", fontsize=20, y=0.98)
    draw_graph(axes[0], knn_graph, knn_graph.graph["method"])
    draw_graph(axes[1], mst_graph, mst_graph.graph["method"])
    fig.tight_layout()

    output_path = figure_dir / "q1_2b_adjacency_graphs.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    similarity = load_similarity(args.similarity_csv)

    knn_graph = build_knn_graph(similarity, args.knn_k)
    mst_graph = build_maximum_spanning_tree(similarity)

    table_dir = ensure_directory(args.table_dir)
    figure_dir = ensure_directory(args.figure_dir)

    knn_edges_path = table_dir / "q1_2b_knn_edges.csv"
    mst_edges_path = table_dir / "q1_2b_mst_edges.csv"
    summary_path = table_dir / "q1_2b_graph_summary.csv"
    discussion_path = table_dir / "q1_2b_discussion.md"

    edge_table(knn_graph).to_csv(knn_edges_path, index=False)
    edge_table(mst_graph).to_csv(mst_edges_path, index=False)
    pd.DataFrame([graph_summary(knn_graph), graph_summary(mst_graph)]).to_csv(summary_path, index=False)
    discussion_path.write_text(build_discussion(knn_graph, mst_graph), encoding="utf-8")
    figure_path = make_figure(knn_graph, mst_graph, figure_dir)

    print(f"Wrote kNN edges to {knn_edges_path}")
    print(f"Wrote MST edges to {mst_edges_path}")
    print(f"Wrote graph summary to {summary_path}")
    print(f"Wrote discussion notes to {discussion_path}")
    print(f"Wrote graph figure to {figure_path}")


if __name__ == "__main__":
    main()
