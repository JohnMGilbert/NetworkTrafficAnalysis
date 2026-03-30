"""Question 1.3(b): graph-theoretic analysis of the reconstructed topology."""

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
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.paths import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology-csv",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "tables" / "q1_3a_inferred_topology_edges.csv",
        help="Path to the Q1.3(a) inferred topology edge table.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "tables",
        help="Directory for generated Q1.3(b) tables.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "figures",
        help="Directory for generated Q1.3(b) figures.",
    )
    return parser.parse_args()


def load_graph(path: Path) -> nx.Graph:
    edge_frame = pd.read_csv(path)
    graph = nx.Graph(method="Q1.3(a) inferred topology")
    routers = sorted(
        set(edge_frame["source"]).union(edge_frame["target"]),
        key=lambda item: int(item[1:]),
    )
    graph.add_nodes_from(routers)

    for row in edge_frame.itertuples(index=False):
        graph.add_edge(
            row.source,
            row.target,
            weight=float(row.composite_score),
            distance=float(row.latency_proxy),
            traffic_volume_proxy=float(row.traffic_volume_proxy),
            bandwidth_proxy=float(row.bandwidth_proxy),
        )
    return graph


def average_path_length_safe(graph: nx.Graph, weight: str | None) -> float:
    if graph.number_of_nodes() <= 1:
        return 0.0
    if nx.is_connected(graph):
        return float(nx.average_shortest_path_length(graph, weight=weight))
    largest_component = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_component).copy()
    return float(nx.average_shortest_path_length(subgraph, weight=weight))


def diameter_safe(graph: nx.Graph, weight: str | None) -> float:
    if graph.number_of_nodes() <= 1:
        return 0.0
    if nx.is_connected(graph):
        return float(nx.diameter(graph, weight=weight))
    largest_component = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_component).copy()
    return float(nx.diameter(subgraph, weight=weight))


def build_router_metrics(graph: nx.Graph) -> pd.DataFrame:
    unweighted_degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(graph, weight="distance", normalized=True)

    rows = []
    for router in sorted(graph.nodes(), key=lambda item: int(item[1:])):
        rows.append(
            {
                "router": router,
                "degree": int(unweighted_degree[router]),
                "weighted_degree": round(float(weighted_degree[router]), 6),
                "betweenness_centrality": round(float(betweenness[router]), 6),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["betweenness_centrality", "degree", "router"],
        ascending=[False, False, True],
    )


def build_graph_summary(graph: nx.Graph, router_metrics: pd.DataFrame) -> pd.DataFrame:
    degree_counts = (
        router_metrics["degree"]
        .value_counts()
        .sort_index()
        .rename_axis("degree")
        .reset_index(name="router_count")
    )
    degree_distribution = ", ".join(
        f"{int(row.degree)}:{int(row.router_count)}" for row in degree_counts.itertuples(index=False)
    )
    return pd.DataFrame(
        [
            {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "connected": nx.is_connected(graph),
                "density": round(float(nx.density(graph)), 6),
                "degree_distribution": degree_distribution,
                "diameter_hops": int(diameter_safe(graph, weight=None)),
                "average_path_length_hops": round(average_path_length_safe(graph, weight=None), 6),
                "weighted_diameter_latency_proxy": round(diameter_safe(graph, weight="distance"), 6),
                "weighted_average_path_length_latency_proxy": round(
                    average_path_length_safe(graph, weight="distance"),
                    6,
                ),
                "highest_betweenness_router": router_metrics.iloc[0]["router"],
                "highest_betweenness_value": float(router_metrics.iloc[0]["betweenness_centrality"]),
            }
        ]
    )


def build_removal_impact(graph: nx.Graph, critical_router: str) -> pd.DataFrame:
    reduced = graph.copy()
    reduced.remove_node(critical_router)

    component_sizes = sorted((len(component) for component in nx.connected_components(reduced)), reverse=True)
    largest_component_size = component_sizes[0] if component_sizes else 0
    return pd.DataFrame(
        [
            {
                "removed_router": critical_router,
                "remaining_nodes": reduced.number_of_nodes(),
                "remaining_edges": reduced.number_of_edges(),
                "connected_after_removal": nx.is_connected(reduced) if reduced.number_of_nodes() > 0 else False,
                "component_count_after_removal": nx.number_connected_components(reduced) if reduced.number_of_nodes() > 0 else 0,
                "largest_component_size_after_removal": largest_component_size,
                "diameter_hops_after_removal": round(diameter_safe(reduced, weight=None), 6),
                "average_path_length_hops_after_removal": round(
                    average_path_length_safe(reduced, weight=None),
                    6,
                ),
                "weighted_diameter_latency_proxy_after_removal": round(
                    diameter_safe(reduced, weight="distance"),
                    6,
                ),
                "weighted_average_path_length_latency_proxy_after_removal": round(
                    average_path_length_safe(reduced, weight="distance"),
                    6,
                ),
            }
        ]
    )


def build_discussion(
    graph: nx.Graph,
    router_metrics: pd.DataFrame,
    graph_summary: pd.DataFrame,
    removal_impact: pd.DataFrame,
) -> str:
    summary = graph_summary.iloc[0]
    impact = removal_impact.iloc[0]
    critical_router = str(summary["highest_betweenness_router"])
    top_central = router_metrics.head(3)

    lines = [
        "# Q1.3(b) Discussion",
        "",
        (
            f"The reconstructed topology has degree distribution {summary['degree_distribution']}, "
            f"diameter {int(summary['diameter_hops'])} hops, and average path length "
            f"{summary['average_path_length_hops']:.3f} hops."
        ),
        (
            f"Using latency proxy as the shortest-path weight yields weighted diameter "
            f"{summary['weighted_diameter_latency_proxy']:.3f} and weighted average path length "
            f"{summary['weighted_average_path_length_latency_proxy']:.3f}."
        ),
        "",
        "Routers with the highest betweenness centrality are:",
    ]

    for row in top_central.itertuples(index=False):
        lines.append(
            f"- {row.router}: betweenness={row.betweenness_centrality:.3f}, "
            f"degree={row.degree}, weighted_degree={row.weighted_degree:.3f}"
        )

    lines.extend(
        [
            "",
            (
                f"{critical_router} is the most critical router because it sits on the largest share of "
                "shortest low-latency paths, so failures there disrupt inter-router transit more than failures at edge-like nodes."
            ),
            (
                f"Removing {critical_router} leaves connected={bool(impact['connected_after_removal'])}, "
                f"with {int(impact['component_count_after_removal'])} connected components and largest component size "
                f"{int(impact['largest_component_size_after_removal'])}."
            ),
            (
                "If the graph fragments after removal, that router behaves like a bridge or backbone relay; "
                "if the graph stays connected but path lengths increase, it acts more like a high-throughput shortcut."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def make_figure(
    router_metrics: pd.DataFrame,
    graph_summary: pd.DataFrame,
    removal_impact: pd.DataFrame,
    figure_dir: Path,
) -> Path:
    summary = graph_summary.iloc[0]
    impact = removal_impact.iloc[0]

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    degree_view = router_metrics.sort_values(["degree", "router"], ascending=[False, True])
    sns.barplot(
        data=degree_view,
        x="router",
        y="degree",
        hue="router",
        palette="crest",
        dodge=False,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Router Degree")
    axes[0].set_xlabel("Router")
    axes[0].set_ylabel("Degree")

    centrality_view = router_metrics.sort_values(
        ["betweenness_centrality", "router"],
        ascending=[False, True],
    )
    sns.barplot(
        data=centrality_view,
        x="router",
        y="betweenness_centrality",
        hue="router",
        palette="flare",
        dodge=False,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Betweenness Centrality")
    axes[1].set_xlabel("Router")
    axes[1].set_ylabel("Centrality")

    impact_frame = pd.DataFrame(
        [
            {"metric": "Diameter (hops)", "baseline": summary["diameter_hops"], "after_removal": impact["diameter_hops_after_removal"]},
            {"metric": "Avg path length (hops)", "baseline": summary["average_path_length_hops"], "after_removal": impact["average_path_length_hops_after_removal"]},
            {"metric": "Components", "baseline": 1, "after_removal": impact["component_count_after_removal"]},
        ]
    )
    impact_long = impact_frame.melt(id_vars="metric", var_name="state", value_name="value")
    sns.barplot(data=impact_long, x="metric", y="value", hue="state", palette="viridis", ax=axes[2])
    axes[2].set_title(f"Impact of Removing {impact['removed_router']}")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Value")
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].legend(title="")

    fig.tight_layout()

    output_path = figure_dir / "q1_3b_graph_properties.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    graph = load_graph(args.topology_csv)
    router_metrics = build_router_metrics(graph)
    graph_summary = build_graph_summary(graph, router_metrics)
    removal_impact = build_removal_impact(graph, str(router_metrics.iloc[0]["router"]))

    table_dir = ensure_directory(args.table_dir)
    figure_dir = ensure_directory(args.figure_dir)

    router_metrics_path = table_dir / "q1_3b_router_metrics.csv"
    summary_path = table_dir / "q1_3b_graph_summary.csv"
    removal_path = table_dir / "q1_3b_removal_impact.csv"
    discussion_path = table_dir / "q1_3b_discussion.md"

    router_metrics.to_csv(router_metrics_path, index=False)
    graph_summary.to_csv(summary_path, index=False)
    removal_impact.to_csv(removal_path, index=False)
    discussion_path.write_text(
        build_discussion(graph, router_metrics, graph_summary, removal_impact),
        encoding="utf-8",
    )
    figure_path = make_figure(router_metrics, graph_summary, removal_impact, figure_dir)

    print(f"Wrote router metrics to {router_metrics_path}")
    print(f"Wrote graph summary to {summary_path}")
    print(f"Wrote removal impact to {removal_path}")
    print(f"Wrote discussion notes to {discussion_path}")
    print(f"Wrote graph figure to {figure_path}")


if __name__ == "__main__":
    main()
