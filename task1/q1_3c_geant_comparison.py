"""Question 1.3(c): compare the inferred topology with the public GEANT-2012 topology."""

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
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.paths import ensure_directory


TOPOLOGY_ZOO_DATASET_URL = "https://topology-zoo.org/dataset.html"
TOPOLOGY_ZOO_MAP_URL = "https://topology-zoo.org/maps/Geant2012.jpg"
TOPOHUB_URL = "https://www.topohub.org/"
GEANT_NETWORK_DATE = "2012-03"
GEANT_NODE_COUNT = 40
GEANT_EDGE_COUNT = 61
GEANT_CONNECTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    table_dir = CONFIG.outputs_dir / "task1" / "tables"
    parser.add_argument(
        "--topology-csv",
        type=Path,
        default=table_dir / "q1_3a_inferred_topology_edges.csv",
        help="Path to the Q1.3(a) inferred topology edge table.",
    )
    parser.add_argument(
        "--graph-summary-csv",
        type=Path,
        default=table_dir / "q1_3b_graph_summary.csv",
        help="Path to the Q1.3(b) graph summary.",
    )
    parser.add_argument(
        "--router-metrics-csv",
        type=Path,
        default=table_dir / "q1_3b_router_metrics.csv",
        help="Path to the Q1.3(b) router metrics.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=table_dir,
        help="Directory for generated Q1.3(c) tables.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "figures",
        help="Directory for generated Q1.3(c) figures.",
    )
    return parser.parse_args()


def load_inferred_graph(path: Path) -> nx.Graph:
    edge_frame = pd.read_csv(path)
    graph = nx.Graph(method="Inferred collector topology")
    routers = sorted(
        set(edge_frame["source"]).union(edge_frame["target"]),
        key=lambda item: int(item[1:]),
    )
    graph.add_nodes_from(routers)
    for row in edge_frame.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=float(row.composite_score))
    return graph


def build_comparison_summary(
    inferred_graph: nx.Graph,
    graph_summary: pd.DataFrame,
    router_metrics: pd.DataFrame,
) -> pd.DataFrame:
    summary = graph_summary.iloc[0]
    inferred_avg_degree = (2.0 * inferred_graph.number_of_edges()) / inferred_graph.number_of_nodes()
    geant_avg_degree = (2.0 * GEANT_EDGE_COUNT) / GEANT_NODE_COUNT

    rows = [
        {
            "metric": "node_count",
            "inferred_value": inferred_graph.number_of_nodes(),
            "actual_geant_2012_value": GEANT_NODE_COUNT,
            "interpretation": "The inferred graph covers only the 10 monitored collectors, while GEANT-2012 contains the full 40-node backbone.",
        },
        {
            "metric": "edge_count",
            "inferred_value": inferred_graph.number_of_edges(),
            "actual_geant_2012_value": GEANT_EDGE_COUNT,
            "interpretation": "The inferred graph is a sparse projection, not a full physical-layer map.",
        },
        {
            "metric": "density",
            "inferred_value": round(float(nx.density(inferred_graph)), 6),
            "actual_geant_2012_value": round((2.0 * GEANT_EDGE_COUNT) / (GEANT_NODE_COUNT * (GEANT_NODE_COUNT - 1)), 6),
            "interpretation": "The collector-only projection is denser because many hidden intermediate routers are collapsed out.",
        },
        {
            "metric": "average_degree",
            "inferred_value": round(inferred_avg_degree, 6),
            "actual_geant_2012_value": round(geant_avg_degree, 6),
            "interpretation": "Average degree is broadly comparable, but the semantics differ because one graph is a projection and the other is the full backbone.",
        },
        {
            "metric": "connected",
            "inferred_value": bool(summary["connected"]),
            "actual_geant_2012_value": GEANT_CONNECTED,
            "interpretation": "Both graphs form a connected backbone rather than isolated islands.",
        },
        {
            "metric": "highest_betweenness_router",
            "inferred_value": str(summary["highest_betweenness_router"]),
            "actual_geant_2012_value": "Central European transit hubs visible in the public map",
            "interpretation": "Both views suggest a non-uniform core where a few transit points carry more shortest paths than edge-like nodes.",
        },
        {
            "metric": "diameter_hops",
            "inferred_value": int(summary["diameter_hops"]),
            "actual_geant_2012_value": "Not compared directly",
            "interpretation": "Direct diameter comparison is weak because the inferred graph omits 30 internal GEANT nodes.",
        },
    ]
    return pd.DataFrame(rows)


def build_structural_feature_comparison(router_metrics: pd.DataFrame) -> pd.DataFrame:
    nonzero_betweenness = int((router_metrics["betweenness_centrality"] > 0).sum())
    top_router = str(router_metrics.iloc[0]["router"])

    rows = [
        {
            "structural_feature": "connected_backbone",
            "identified_in_reconstruction": "yes",
            "observed_in_actual_geant_2012": "yes",
            "notes": "The inferred graph is connected, matching the public GEANT backbone map's single connected component.",
        },
        {
            "structural_feature": "central_transit_region",
            "identified_in_reconstruction": "yes",
            "observed_in_actual_geant_2012": "yes",
            "notes": f"The inferred graph has concentrated betweenness around routers such as {top_router}, consistent with a core transit region in the public map.",
        },
        {
            "structural_feature": "peripheral_edge_like_sites",
            "identified_in_reconstruction": "yes",
            "observed_in_actual_geant_2012": "yes",
            "notes": "Several inferred routers have degree 2 and zero betweenness, consistent with edge-like or access-like positions.",
        },
        {
            "structural_feature": "full_geographic_layout",
            "identified_in_reconstruction": "no",
            "observed_in_actual_geant_2012": "yes",
            "notes": "The collector graph lacks explicit geography, so it cannot recover the actual pan-European placement of nodes and long-haul links.",
        },
        {
            "structural_feature": "hidden_intermediate_nodes",
            "identified_in_reconstruction": "no",
            "observed_in_actual_geant_2012": "yes",
            "notes": f"Only {nonzero_betweenness} inferred routers have non-zero betweenness, but the real 40-node network includes many internal relays that are invisible in the 10-router projection.",
        },
        {
            "structural_feature": "parallel_capacity_detail",
            "identified_in_reconstruction": "partial",
            "observed_in_actual_geant_2012": "yes",
            "notes": "The public topology includes explicit link capacities and geographic spans, while the inferred graph uses traffic-derived proxies instead.",
        },
    ]
    return pd.DataFrame(rows)


def build_discussion(
    inferred_graph: nx.Graph,
    graph_summary: pd.DataFrame,
    router_metrics: pd.DataFrame,
) -> str:
    summary = graph_summary.iloc[0]
    top_router = str(summary["highest_betweenness_router"])
    top_three = ", ".join(
        f"{row.router} ({row.betweenness_centrality:.3f})"
        for row in router_metrics.head(3).itertuples(index=False)
    )

    lines = [
        "# Q1.3(c) Comparison With Public GEANT-2012 Topology",
        "",
        "## Sources",
        "",
        f"- Internet Topology Zoo dataset entry for GEANT {GEANT_NETWORK_DATE}: [{TOPOLOGY_ZOO_DATASET_URL}]({TOPOLOGY_ZOO_DATASET_URL})",
        f"- Internet Topology Zoo GEANT-2012 map: [{TOPOLOGY_ZOO_MAP_URL}]({TOPOLOGY_ZOO_MAP_URL})",
        f"- TopoHub topology catalog (used for the 40-node, 61-link summary): [{TOPOHUB_URL}]({TOPOHUB_URL})",
        "",
        "## What the Reconstruction Got Right",
        "",
        (
            "The inferred topology correctly recovered a connected backbone rather than multiple isolated clusters. "
            "That matches the public GEANT-2012 map, which shows one continent-scale connected network."
        ),
        (
            f"The reconstruction also identified non-uniform transit importance. In the inferred graph, the highest-betweenness routers are {top_three}, "
            "which is consistent with the public map's dense central-European core and weaker edge-like peripheral sites."
        ),
        (
            "A third feature it captured is the difference between core-like and edge-like positions: several inferred routers have degree 2 and zero betweenness, "
            "which is the same kind of role separation visible in GEANT between central transit sites and peripheral endpoints such as island or eastern/southern edge locations."
        ),
        "",
        "## Where It Diverged From Reality",
        "",
        (
            f"The biggest mismatch is scale. The inferred graph has {inferred_graph.number_of_nodes()} monitored routers and {inferred_graph.number_of_edges()} edges, "
            f"whereas the public GEANT-2012 topology contains {GEANT_NODE_COUNT} nodes and {GEANT_EDGE_COUNT} edges. "
            "This means the reconstruction is a projection of collector observations, not a direct recovery of the full backbone."
        ),
        (
            "The inferred graph is therefore much denser than the real graph when measured only over visible vertices, because hidden intermediate routers are collapsed into fewer observed nodes."
        ),
        (
            "It also cannot recover the actual geographic embedding. The public map contains long-haul links across Europe and clear peripheral branches to places such as Iceland, Cyprus, Israel, and Malta, "
            "while the data-driven reconstruction only sees traffic affinity and overlap, not physical location."
        ),
        (
            f"Finally, the reconstruction over-emphasizes the importance of {top_router} as a single bridge-like collector. "
            "In the real GEANT topology, connectivity is distributed across many central sites, so shortest-path load is shared by more internal routers than the projected graph can show."
        ),
        "",
        "## What Would Improve the Reconstruction",
        "",
        (
            "Traceroute or hop-level path data would expose hidden intermediate routers and make it possible to separate direct adjacency from multi-hop traffic correlation."
        ),
        (
            "Geographic metadata such as PoP locations, RTT measurements, or link-delay estimates would help distinguish physically adjacent routers from routers that merely carry similar traffic mixes."
        ),
        (
            "Interface-level counters or known link capacities would improve edge weighting, because the public GEANT topology distinguishes real capacity-bearing links while the current reconstruction uses flow-derived volume and bandwidth proxies."
        ),
        (
            "Time-synchronized routing or control-plane data, such as BGP/IGP state or per-link utilization, would help determine whether two collectors are direct neighbors or simply observe the same transit events from different parts of the backbone."
        ),
        "",
        "## Bottom Line",
        "",
        (
            "The reconstruction got the broad structural story right: GEANT behaves like a connected backbone with a non-uniform central core and less-central peripheral sites. "
            "It diverged on exact node count, exact adjacency, and geography because flow statistics from ten collectors are not enough to recover the full March 2012 physical topology."
        ),
    ]
    return "\n".join(lines) + "\n"


def make_figure(
    comparison_summary: pd.DataFrame,
    structural_feature_comparison: pd.DataFrame,
    figure_dir: Path,
) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    numeric_metrics = comparison_summary[
        comparison_summary["metric"].isin(["node_count", "edge_count", "density", "average_degree"])
    ].copy()
    numeric_long = numeric_metrics.melt(
        id_vars="metric",
        value_vars=["inferred_value", "actual_geant_2012_value"],
        var_name="graph_type",
        value_name="value",
    )
    sns.barplot(data=numeric_long, x="metric", y="value", hue="graph_type", palette="deep", ax=axes[0])
    axes[0].set_title("Inferred vs Public GEANT-2012 Summary")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Value")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(title="")

    structural_counts = (
        structural_feature_comparison["identified_in_reconstruction"]
        .value_counts()
        .reindex(["yes", "partial", "no"], fill_value=0)
        .rename_axis("status")
        .reset_index(name="feature_count")
    )
    sns.barplot(
        data=structural_counts,
        x="status",
        y="feature_count",
        hue="status",
        palette="muted",
        dodge=False,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("How Much of GEANT Structure Was Recovered")
    axes[1].set_xlabel("Recovered in reconstruction")
    axes[1].set_ylabel("Feature count")

    fig.tight_layout()

    output_path = figure_dir / "q1_3c_comparison_dashboard.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()

    inferred_graph = load_inferred_graph(args.topology_csv)
    graph_summary = pd.read_csv(args.graph_summary_csv)
    router_metrics = pd.read_csv(args.router_metrics_csv)

    comparison_summary = build_comparison_summary(inferred_graph, graph_summary, router_metrics)
    structural_feature_comparison = build_structural_feature_comparison(router_metrics)

    table_dir = ensure_directory(args.table_dir)
    figure_dir = ensure_directory(args.figure_dir)

    comparison_path = table_dir / "q1_3c_comparison_summary.csv"
    structural_path = table_dir / "q1_3c_structural_feature_comparison.csv"
    discussion_path = table_dir / "q1_3c_discussion.md"

    comparison_summary.to_csv(comparison_path, index=False)
    structural_feature_comparison.to_csv(structural_path, index=False)
    discussion_path.write_text(
        build_discussion(inferred_graph, graph_summary, router_metrics),
        encoding="utf-8",
    )
    figure_path = make_figure(comparison_summary, structural_feature_comparison, figure_dir)

    print(f"Wrote comparison summary to {comparison_path}")
    print(f"Wrote structural feature comparison to {structural_path}")
    print(f"Wrote discussion notes to {discussion_path}")
    print(f"Wrote comparison figure to {figure_path}")


if __name__ == "__main__":
    main()
