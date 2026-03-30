"""Question 1.3(a): data-driven inferred router topology."""

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


EDGE_WEIGHTS = {
    "feature_similarity": 0.30,
    "ip_jaccard": 0.20,
    "tuple_jaccard": 0.15,
    "neighbor_consistency": 0.20,
    "volume_compatibility": 0.15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    table_dir = CONFIG.outputs_dir / "task1" / "tables"
    parser.add_argument(
        "--similarity-csv",
        type=Path,
        default=table_dir / "q1_2a_router_similarity_matrix.csv",
        help="Path to the Q1.2(a) feature similarity matrix.",
    )
    parser.add_argument(
        "--ip-jaccard-csv",
        type=Path,
        default=table_dir / "q1_2c_ip_jaccard_matrix.csv",
        help="Path to the Q1.2(c) IP Jaccard matrix.",
    )
    parser.add_argument(
        "--tuple-jaccard-csv",
        type=Path,
        default=table_dir / "q1_2c_tuple_jaccard_matrix.csv",
        help="Path to the Q1.2(c) tuple Jaccard matrix.",
    )
    parser.add_argument(
        "--feature-profiles-csv",
        type=Path,
        default=table_dir / "q1_2a_router_feature_profiles.csv",
        help="Path to the Q1.2(a) router feature profiles.",
    )
    parser.add_argument(
        "--volume-summary-csv",
        type=Path,
        default=table_dir / "q1_1b_volume_summary.csv",
        help="Path to the Q1.1(b) volume summary table.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=table_dir,
        help="Directory for generated Q1.3(a) tables.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "figures",
        help="Directory for generated Q1.3(a) figures.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Local candidate edge count retained per router before connectivity repair.",
    )
    parser.add_argument(
        "--score-quantile",
        type=float,
        default=0.75,
        help="Global composite-score quantile used for strong-edge retention.",
    )
    return parser.parse_args()


def load_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    ordered = sorted(frame.index, key=lambda item: int(item[1:]))
    return frame.loc[ordered, ordered].astype(float)


def normalize_zero_one(series: pd.Series) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())
    if np.isclose(minimum, maximum):
        return pd.Series(1.0, index=series.index)
    return (series - minimum) / (maximum - minimum)


def normalize_similarity(similarity: pd.DataFrame) -> pd.DataFrame:
    normalized = (similarity + 1.0) / 2.0
    for router in normalized.index:
        normalized.loc[router, router] = 1.0
    return normalized


def compute_neighbor_consistency(similarity: pd.DataFrame) -> pd.DataFrame:
    routers = list(similarity.index)
    matrix = pd.DataFrame(index=routers, columns=routers, dtype=float)
    for left in routers:
        left_vec = similarity.loc[left].drop(index=left).to_numpy(dtype=np.float64)
        left_norm = np.linalg.norm(left_vec)
        for right in routers:
            if left == right:
                matrix.loc[left, right] = 1.0
                continue
            right_vec = similarity.loc[right].drop(index=right).to_numpy(dtype=np.float64)
            right_norm = np.linalg.norm(right_vec)
            if left_norm == 0.0 or right_norm == 0.0:
                score = 0.0
            else:
                score = float(np.dot(left_vec, right_vec) / (left_norm * right_norm))
            matrix.loc[left, right] = (score + 1.0) / 2.0
    return matrix


def compute_volume_compatibility(volume_summary: pd.DataFrame) -> pd.DataFrame:
    volume_summary = volume_summary.copy()
    volume_summary["log_volume_mean"] = np.log1p(volume_summary["volume_mean"].astype(float))
    scaled = normalize_zero_one(volume_summary.set_index("router")["log_volume_mean"])
    volume_summary["scaled"] = volume_summary["router"].map(scaled)

    routers = volume_summary["router"].tolist()
    values = volume_summary.set_index("router")["scaled"].to_dict()
    matrix = pd.DataFrame(index=routers, columns=routers, dtype=float)
    for left in routers:
        for right in routers:
            matrix.loc[left, right] = 1.0 - abs(values[left] - values[right])
    return matrix


def build_pair_score_table(
    similarity: pd.DataFrame,
    ip_jaccard: pd.DataFrame,
    tuple_jaccard: pd.DataFrame,
    neighbor_consistency: pd.DataFrame,
    volume_compatibility: pd.DataFrame,
    feature_profiles: pd.DataFrame,
    volume_summary: pd.DataFrame,
) -> pd.DataFrame:
    routers = list(similarity.index)
    profiles = feature_profiles.set_index("router")
    volume_stats = volume_summary.set_index("router")
    rows = []

    for i, left in enumerate(routers):
        for right in routers[i + 1 :]:
            components = {
                "feature_similarity": float(similarity.loc[left, right]),
                "ip_jaccard": float(ip_jaccard.loc[left, right]),
                "tuple_jaccard": float(tuple_jaccard.loc[left, right]),
                "neighbor_consistency": float(neighbor_consistency.loc[left, right]),
                "volume_compatibility": float(volume_compatibility.loc[left, right]),
            }
            composite = sum(EDGE_WEIGHTS[name] * value for name, value in components.items())
            rows.append(
                {
                    "router_a": left,
                    "router_b": right,
                    **{name: round(value, 6) for name, value in components.items()},
                    "composite_score": round(composite, 6),
                    "traffic_volume_proxy": round(
                        float(
                            (
                                volume_stats.loc[left, "volume_mean"]
                                + volume_stats.loc[right, "volume_mean"]
                            )
                            / 2.0
                        ),
                        3,
                    ),
                    "bandwidth_proxy": round(
                        float(
                            (
                                profiles.loc[left, "flow_byts_s"]
                                + profiles.loc[right, "flow_byts_s"]
                            )
                            / 2.0
                        ),
                        3,
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)


def build_complete_graph(pair_scores: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    routers = sorted(
        set(pair_scores["router_a"]).union(pair_scores["router_b"]),
        key=lambda item: int(item[1:]),
    )
    graph.add_nodes_from(routers)
    for row in pair_scores.itertuples(index=False):
        graph.add_edge(
            row.router_a,
            row.router_b,
            weight=float(row.composite_score),
            feature_similarity=float(row.feature_similarity),
            ip_jaccard=float(row.ip_jaccard),
            tuple_jaccard=float(row.tuple_jaccard),
            neighbor_consistency=float(row.neighbor_consistency),
            volume_compatibility=float(row.volume_compatibility),
            traffic_volume_proxy=float(row.traffic_volume_proxy),
            bandwidth_proxy=float(row.bandwidth_proxy),
        )
    return graph


def select_seed_edges(pair_scores: pd.DataFrame, routers: list[str], top_k: int, score_quantile: float) -> set[tuple[str, str]]:
    selected: set[tuple[str, str]] = set()
    threshold = float(pair_scores["composite_score"].quantile(score_quantile))

    for router in routers:
        candidates = pair_scores[
            (pair_scores["router_a"] == router) | (pair_scores["router_b"] == router)
        ].nlargest(top_k, "composite_score")
        for row in candidates.itertuples(index=False):
            selected.add(tuple(sorted((row.router_a, row.router_b))))

    for row in pair_scores.itertuples(index=False):
        if float(row.composite_score) >= threshold:
            selected.add(tuple(sorted((row.router_a, row.router_b))))

    return selected


def build_inferred_graph(
    complete_graph: nx.Graph,
    pair_scores: pd.DataFrame,
    top_k: int,
    score_quantile: float,
) -> nx.Graph:
    routers = sorted(complete_graph.nodes(), key=lambda item: int(item[1:]))
    inferred = nx.Graph(method="Multi-signal sparse inferred topology")
    inferred.add_nodes_from(routers)

    for source, target in select_seed_edges(pair_scores, routers, top_k, score_quantile):
        inferred.add_edge(source, target, **complete_graph[source][target])

    if not nx.is_connected(inferred):
        mst = nx.maximum_spanning_tree(complete_graph, weight="weight")
        for source, target in mst.edges():
            if inferred.has_edge(source, target):
                continue
            inferred.add_edge(source, target, **complete_graph[source][target])
            if nx.is_connected(inferred):
                break

    return inferred


def edge_table(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for source, target, data in sorted(graph.edges(data=True)):
        rows.append(
            {
                "source": source,
                "target": target,
                "composite_score": round(float(data["weight"]), 6),
                "feature_similarity": round(float(data["feature_similarity"]), 6),
                "ip_jaccard": round(float(data["ip_jaccard"]), 6),
                "tuple_jaccard": round(float(data["tuple_jaccard"]), 6),
                "neighbor_consistency": round(float(data["neighbor_consistency"]), 6),
                "volume_compatibility": round(float(data["volume_compatibility"]), 6),
                "traffic_volume_proxy": round(float(data["traffic_volume_proxy"]), 3),
                "bandwidth_proxy": round(float(data["bandwidth_proxy"]), 3),
                "latency_proxy": round(1.0 - float(data["weight"]), 6),
            }
        )
    return pd.DataFrame(rows).sort_values("composite_score", ascending=False)


def build_discussion(graph: nx.Graph, pair_scores: pd.DataFrame) -> str:
    top_global = pair_scores.head(8)
    inferred_edges = edge_table(graph)
    lines = [
        "# Q1.3(a) Discussion",
        "",
        "This topology is inferred from data rather than imposed from known ground truth.",
        "Each router pair receives a composite edge score from normalized feature similarity, IP overlap, tuple overlap, shared-neighbor consistency, and volume compatibility.",
        "The final graph keeps the strongest local edges per router, retains globally strong edges, and then adds only the extra high-scoring links needed to ensure connectivity.",
        "",
        (
            f"The inferred graph has {graph.number_of_nodes()} routers, {graph.number_of_edges()} edges, "
            f"density={nx.density(graph):.3f}, and connected={nx.is_connected(graph)}."
        ),
        "",
        "Highest-scoring router pairs overall:",
    ]

    for row in top_global.itertuples(index=False):
        lines.append(
            f"- {row.router_a}-{row.router_b}: composite={row.composite_score:.3f}, "
            f"feature={row.feature_similarity:.3f}, ip={row.ip_jaccard:.3f}, "
            f"tuple={row.tuple_jaccard:.3f}, neighbor={row.neighbor_consistency:.3f}, "
            f"volume={row.volume_compatibility:.3f}"
        )

    lines.extend(["", "Edges kept in the inferred topology:"])
    for row in inferred_edges.itertuples(index=False):
        lines.append(
            f"- {row.source}-{row.target}: composite={row.composite_score:.3f}, "
            f"feature={row.feature_similarity:.3f}, ip={row.ip_jaccard:.3f}, "
            f"tuple={row.tuple_jaccard:.3f}, neighbor={row.neighbor_consistency:.3f}, "
            f"volume={row.volume_compatibility:.3f}"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- High-confidence edges are those supported by several signals at once, not just one similarity measure.",
            "- Missing edges should be read as lower-confidence under the available traffic evidence, not proof that no physical link exists.",
        ]
    )
    return "\n".join(lines) + "\n"


def draw_graph(ax: plt.Axes, graph: nx.Graph, title: str) -> None:
    pos = nx.spring_layout(graph, seed=CONFIG.random_seed, weight="weight")
    weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()])
    widths = 2.0 if weights.size == 0 else 2.0 + 4.0 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    nx.draw_networkx_nodes(graph, pos, node_color="#d9ed92", edgecolors="#1f1f1f", node_size=1000, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=11, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos,
        width=widths,
        edge_color=weights if weights.size else "#666666",
        edge_cmap=plt.cm.cividis if weights.size else None,
        ax=ax,
    )
    ax.set_title(title)
    ax.axis("off")


def make_figure(pair_scores: pd.DataFrame, inferred_graph: nx.Graph, figure_dir: Path) -> Path:
    routers = sorted(
        set(pair_scores["router_a"]).union(pair_scores["router_b"]),
        key=lambda item: int(item[1:]),
    )
    matrix = pd.DataFrame(index=routers, columns=routers, dtype=float)
    for router in routers:
        matrix.loc[router, router] = 1.0
    for row in pair_scores.itertuples(index=False):
        matrix.loc[row.router_a, row.router_b] = float(row.composite_score)
        matrix.loc[row.router_b, row.router_a] = float(row.composite_score)

    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Composite Edge Score"},
        ax=axes[0],
    )
    axes[0].set_title("Pairwise Composite Edge Scores")
    axes[0].set_xlabel("Router")
    axes[0].set_ylabel("Router")

    draw_graph(axes[1], inferred_graph, "Inferred Sparse Connected Topology")
    fig.tight_layout()

    output_path = figure_dir / "q1_3a_inferred_topology.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()

    similarity = normalize_similarity(load_matrix(args.similarity_csv))
    ip_jaccard = load_matrix(args.ip_jaccard_csv)
    tuple_jaccard = load_matrix(args.tuple_jaccard_csv)
    feature_profiles = pd.read_csv(args.feature_profiles_csv)
    volume_summary = pd.read_csv(args.volume_summary_csv)

    neighbor_consistency = compute_neighbor_consistency(similarity)
    volume_compatibility = compute_volume_compatibility(volume_summary)

    pair_scores = build_pair_score_table(
        similarity=similarity,
        ip_jaccard=ip_jaccard,
        tuple_jaccard=tuple_jaccard,
        neighbor_consistency=neighbor_consistency,
        volume_compatibility=volume_compatibility,
        feature_profiles=feature_profiles,
        volume_summary=volume_summary,
    )

    complete_graph = build_complete_graph(pair_scores)
    inferred_graph = build_inferred_graph(
        complete_graph=complete_graph,
        pair_scores=pair_scores,
        top_k=args.top_k,
        score_quantile=args.score_quantile,
    )

    table_dir = ensure_directory(args.table_dir)
    figure_dir = ensure_directory(args.figure_dir)

    pair_scores_path = table_dir / "q1_3a_pair_scores.csv"
    edges_path = table_dir / "q1_3a_inferred_topology_edges.csv"
    discussion_path = table_dir / "q1_3a_discussion.md"

    pair_scores.to_csv(pair_scores_path, index=False)
    edge_table(inferred_graph).to_csv(edges_path, index=False)
    discussion_path.write_text(build_discussion(inferred_graph, pair_scores), encoding="utf-8")
    figure_path = make_figure(pair_scores, inferred_graph, figure_dir)

    print(f"Wrote pairwise edge scores to {pair_scores_path}")
    print(f"Wrote inferred topology edges to {edges_path}")
    print(f"Wrote discussion notes to {discussion_path}")
    print(f"Wrote inferred topology figure to {figure_path}")


if __name__ == "__main__":
    main()
