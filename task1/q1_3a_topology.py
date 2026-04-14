"""Question 1.3(a): backbone-oriented router topology reconstruction."""

from __future__ import annotations

import argparse
import os
import re
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
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.paths import ensure_directory


EDGE_WEIGHTS = {
    "feature_similarity": 0.14,
    "ip_jaccard": 0.22,
    "tuple_jaccard": 0.18,
    "port_overlap": 0.08,
    "neighbor_consistency": 0.10,
    "volume_compatibility": 0.07,
    "duration_compatibility": 0.06,
    "packet_balance_compatibility": 0.04,
    "backbone_role_plausibility": 0.04,
    "refined_mst_support": 0.07,
}

DIRECT_EVIDENCE_WEIGHTS = {
    "ip_jaccard": 0.50,
    "tuple_jaccard": 0.35,
    "port_overlap": 0.15,
}

WEAK_DIRECT_PENALTY = 0.82
MAX_ROUTER_DEGREE = 4


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
        "--router-summary-csv",
        type=Path,
        default=table_dir / "q1_1a_router_summary.csv",
        help="Path to the Q1.1(a) router summary table.",
    )
    parser.add_argument(
        "--volume-summary-csv",
        type=Path,
        default=table_dir / "q1_1b_volume_summary.csv",
        help="Path to the Q1.1(b) volume summary table.",
    )
    parser.add_argument(
        "--refined-mst-csv",
        type=Path,
        default=table_dir / "q1_2c_refined_graph_edges.csv",
        help="Optional Q1.2(c) refined MST edge table used as a prior.",
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
        "--triangle-redundancy-budget",
        type=int,
        default=1,
        help="Maximum number of local triangle-closing redundancy edges to add after the backbone MST.",
    )
    parser.add_argument(
        "--shortcut-budget",
        type=int,
        default=1,
        help="Maximum number of long-path shortcut edges to add after the backbone MST.",
    )
    parser.add_argument("--top-k", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--score-quantile", type=float, default=0.75, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    ordered = sorted(frame.index, key=lambda item: int(item[1:]))
    return frame.loc[ordered, ordered].astype(float)


def load_refined_edge_set(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()

    frame = pd.read_csv(path)
    required = {"source", "target"}
    if not required.issubset(frame.columns):
        return set()

    return {
        tuple(sorted((str(row.source), str(row.target))))
        for row in frame.itertuples(index=False)
    }


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


def matrix_from_values(values: pd.Series) -> pd.DataFrame:
    scaled = normalize_zero_one(values.astype(float))
    routers = scaled.index.tolist()
    matrix = pd.DataFrame(index=routers, columns=routers, dtype=float)
    for left in routers:
        for right in routers:
            matrix.loc[left, right] = 1.0 - abs(float(scaled[left]) - float(scaled[right]))
    return matrix


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


def parse_top_port_set(text: str) -> set[str]:
    ports: set[str] = set()
    for part in str(text).split(";"):
        match = re.match(r"\s*([^\s]+)\s*\(\d+\)", part.strip())
        if match:
            ports.add(match.group(1))
    return ports


def compute_port_overlap(router_summary: pd.DataFrame) -> pd.DataFrame:
    ports = {
        row.router: parse_top_port_set(row.top_5_destination_ports)
        for row in router_summary.itertuples(index=False)
    }
    routers = sorted(ports, key=lambda item: int(item[1:]))
    matrix = pd.DataFrame(index=routers, columns=routers, dtype=float)
    for left in routers:
        for right in routers:
            union = ports[left] | ports[right]
            if not union:
                matrix.loc[left, right] = 0.0
            else:
                matrix.loc[left, right] = len(ports[left] & ports[right]) / len(union)
    return matrix


def compute_duration_compatibility(router_summary: pd.DataFrame) -> pd.DataFrame:
    summary = router_summary.copy()
    summary["log_duration_mean"] = np.log1p(pd.to_numeric(summary["flow_duration_mean"], errors="coerce").fillna(0.0))
    summary["log_duration_median"] = np.log1p(pd.to_numeric(summary["flow_duration_median"], errors="coerce").fillna(0.0))
    mean_matrix = matrix_from_values(summary.set_index("router")["log_duration_mean"])
    median_matrix = matrix_from_values(summary.set_index("router")["log_duration_median"])
    return 0.5 * mean_matrix + 0.5 * median_matrix


def compute_packet_balance_compatibility(router_summary: pd.DataFrame) -> pd.DataFrame:
    ratios = pd.to_numeric(
        router_summary["forward_to_backward_packet_ratio"].replace({"NA": np.nan, "inf": np.nan}),
        errors="coerce",
    )
    ratios = ratios.fillna(ratios.median() if ratios.notna().any() else 1.0)
    log_ratio = np.log(np.clip(ratios.astype(float), 1e-6, None))
    return matrix_from_values(pd.Series(log_ratio.to_numpy(), index=router_summary["router"]))


def compute_transit_scores(
    router_summary: pd.DataFrame,
    volume_summary: pd.DataFrame,
) -> pd.Series:
    merged = router_summary[["router", "flow_records", "forward_to_backward_packet_ratio"]].merge(
        volume_summary[["router", "volume_mean", "volume_p99"]],
        on="router",
        how="left",
    )
    merged["flow_records"] = pd.to_numeric(merged["flow_records"], errors="coerce").fillna(0.0)
    merged["volume_mean"] = pd.to_numeric(merged["volume_mean"], errors="coerce").fillna(0.0)
    merged["volume_p99"] = pd.to_numeric(merged["volume_p99"], errors="coerce").fillna(0.0)
    ratios = pd.to_numeric(
        merged["forward_to_backward_packet_ratio"].replace({"NA": np.nan, "inf": np.nan}),
        errors="coerce",
    )
    ratios = ratios.fillna(ratios.median() if ratios.notna().any() else 1.0)

    merged["log_flow_records"] = np.log1p(merged["flow_records"])
    merged["log_volume_mean"] = np.log1p(merged["volume_mean"])
    merged["log_volume_p99"] = np.log1p(merged["volume_p99"])
    merged["balance_score"] = 1.0 / (1.0 + np.abs(np.log(np.clip(ratios.astype(float), 1e-6, None))))

    flow_scaled = normalize_zero_one(merged.set_index("router")["log_flow_records"])
    volume_scaled = normalize_zero_one(merged.set_index("router")["log_volume_mean"])
    tail_scaled = normalize_zero_one(merged.set_index("router")["log_volume_p99"])
    balance_scaled = normalize_zero_one(merged.set_index("router")["balance_score"])

    return (
        0.40 * flow_scaled
        + 0.25 * balance_scaled
        + 0.20 * volume_scaled
        + 0.15 * tail_scaled
    ).sort_index(key=lambda index: index.str[1:].astype(int))


def compute_backbone_role_plausibility(transit_scores: pd.Series) -> pd.DataFrame:
    routers = transit_scores.index.tolist()
    matrix = pd.DataFrame(index=routers, columns=routers, dtype=float)
    for left in routers:
        for right in routers:
            left_score = float(transit_scores[left])
            right_score = float(transit_scores[right])
            matrix.loc[left, right] = 1.0 - (1.0 - left_score) * (1.0 - right_score)
    return matrix


def compute_refined_mst_support(
    routers: list[str],
    refined_edges: set[tuple[str, str]],
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=routers, columns=routers, dtype=float)
    for left in routers:
        for right in routers:
            if left == right:
                matrix.loc[left, right] = 1.0
            else:
                matrix.loc[left, right] = float(tuple(sorted((left, right))) in refined_edges)
    return matrix


def build_pair_score_table(
    similarity: pd.DataFrame,
    ip_jaccard: pd.DataFrame,
    tuple_jaccard: pd.DataFrame,
    port_overlap: pd.DataFrame,
    neighbor_consistency: pd.DataFrame,
    volume_compatibility: pd.DataFrame,
    duration_compatibility: pd.DataFrame,
    packet_balance_compatibility: pd.DataFrame,
    backbone_role_plausibility: pd.DataFrame,
    refined_mst_support: pd.DataFrame,
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
                "port_overlap": float(port_overlap.loc[left, right]),
                "neighbor_consistency": float(neighbor_consistency.loc[left, right]),
                "volume_compatibility": float(volume_compatibility.loc[left, right]),
                "duration_compatibility": float(duration_compatibility.loc[left, right]),
                "packet_balance_compatibility": float(packet_balance_compatibility.loc[left, right]),
                "backbone_role_plausibility": float(backbone_role_plausibility.loc[left, right]),
                "refined_mst_support": float(refined_mst_support.loc[left, right]),
            }
            direct_evidence = sum(
                DIRECT_EVIDENCE_WEIGHTS[name] * components[name]
                for name in DIRECT_EVIDENCE_WEIGHTS
            )
            base_score = sum(EDGE_WEIGHTS[name] * value for name, value in components.items())
            rows.append(
                {
                    "router_a": left,
                    "router_b": right,
                    **components,
                    "direct_evidence": float(direct_evidence),
                    "base_score": float(base_score),
                    "traffic_volume_proxy": float(
                        (
                            volume_stats.loc[left, "volume_mean"]
                            + volume_stats.loc[right, "volume_mean"]
                        )
                        / 2.0
                    ),
                    "bandwidth_proxy": float(
                        (
                            profiles.loc[left, "flow_byts_s"]
                            + profiles.loc[right, "flow_byts_s"]
                        )
                        / 2.0
                    ),
                }
            )

    pair_scores = pd.DataFrame(rows)

    ip_threshold = max(float(pair_scores["ip_jaccard"].quantile(0.75)), 0.45)
    tuple_threshold = max(float(pair_scores["tuple_jaccard"].quantile(0.75)), 0.10)
    port_threshold = max(float(pair_scores["port_overlap"].quantile(0.75)), 1.0 / 9.0)

    pair_scores["direct_support_count"] = (
        (pair_scores["ip_jaccard"] >= ip_threshold).astype(int)
        + (pair_scores["tuple_jaccard"] >= tuple_threshold).astype(int)
        + (pair_scores["port_overlap"] >= port_threshold).astype(int)
    )

    weak_direct_mask = (
        (pair_scores["direct_support_count"] == 0)
        & (pair_scores["refined_mst_support"] < 0.5)
        & (pair_scores["ip_jaccard"] < float(pair_scores["ip_jaccard"].median()))
        & (pair_scores["tuple_jaccard"] < float(pair_scores["tuple_jaccard"].median()))
    )
    pair_scores["weak_direct_penalty"] = np.where(weak_direct_mask, WEAK_DIRECT_PENALTY, 1.0)
    pair_scores["composite_score"] = pair_scores["base_score"] * pair_scores["weak_direct_penalty"]
    pair_scores["latency_proxy"] = 1.0 - (
        0.55 * pair_scores["direct_evidence"] + 0.45 * pair_scores["composite_score"]
    )

    rounded = pair_scores.copy()
    for column in [
        "feature_similarity",
        "ip_jaccard",
        "tuple_jaccard",
        "port_overlap",
        "neighbor_consistency",
        "volume_compatibility",
        "duration_compatibility",
        "packet_balance_compatibility",
        "backbone_role_plausibility",
        "refined_mst_support",
        "direct_evidence",
        "base_score",
        "weak_direct_penalty",
        "composite_score",
        "latency_proxy",
    ]:
        rounded[column] = rounded[column].astype(float).round(6)
    rounded["traffic_volume_proxy"] = rounded["traffic_volume_proxy"].astype(float).round(3)
    rounded["bandwidth_proxy"] = rounded["bandwidth_proxy"].astype(float).round(3)
    rounded["direct_support_count"] = rounded["direct_support_count"].astype(int)
    return rounded.sort_values("composite_score", ascending=False).reset_index(drop=True)


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
            distance=float(row.latency_proxy),
            feature_similarity=float(row.feature_similarity),
            ip_jaccard=float(row.ip_jaccard),
            tuple_jaccard=float(row.tuple_jaccard),
            port_overlap=float(row.port_overlap),
            neighbor_consistency=float(row.neighbor_consistency),
            volume_compatibility=float(row.volume_compatibility),
            duration_compatibility=float(row.duration_compatibility),
            packet_balance_compatibility=float(row.packet_balance_compatibility),
            backbone_role_plausibility=float(row.backbone_role_plausibility),
            refined_mst_support=float(row.refined_mst_support),
            direct_evidence=float(row.direct_evidence),
            base_score=float(row.base_score),
            weak_direct_penalty=float(row.weak_direct_penalty),
            traffic_volume_proxy=float(row.traffic_volume_proxy),
            bandwidth_proxy=float(row.bandwidth_proxy),
        )
    return graph


def add_edge_with_tier(graph: nx.Graph, complete_graph: nx.Graph, source: str, target: str, tier: str) -> None:
    data = dict(complete_graph[source][target])
    data["selection_tier"] = tier
    graph.add_edge(source, target, **data)


def valid_degree_addition(graph: nx.Graph, source: str, target: str) -> bool:
    return graph.degree(source) < MAX_ROUTER_DEGREE and graph.degree(target) < MAX_ROUTER_DEGREE


def triangle_candidates(graph: nx.Graph, pair_scores: pd.DataFrame) -> list[tuple[float, str, str]]:
    score_threshold = float(pair_scores["composite_score"].quantile(0.80))
    candidates: list[tuple[float, str, str]] = []
    for row in pair_scores.itertuples(index=False):
        source = str(row.router_a)
        target = str(row.router_b)
        if graph.has_edge(source, target):
            continue
        if row.composite_score < score_threshold or int(row.direct_support_count) < 2:
            continue
        if not valid_degree_addition(graph, source, target):
            continue

        common_neighbors = set(graph.neighbors(source)) & set(graph.neighbors(target))
        if not common_neighbors:
            continue

        local_support = max(
            min(
                float(graph[source][neighbor]["weight"]),
                float(graph[target][neighbor]["weight"]),
            )
            for neighbor in common_neighbors
        )
        candidate_score = 0.70 * float(row.composite_score) + 0.30 * local_support
        candidates.append((candidate_score, source, target))

    return sorted(candidates, reverse=True)


def shortcut_candidates(
    graph: nx.Graph,
    pair_scores: pd.DataFrame,
    transit_scores: pd.Series,
) -> list[tuple[float, str, str]]:
    score_threshold = float(pair_scores["composite_score"].quantile(0.75))
    direct_threshold = float(pair_scores["direct_evidence"].quantile(0.70))
    transit_threshold = float(transit_scores.median())
    candidates: list[tuple[float, str, str]] = []

    for row in pair_scores.itertuples(index=False):
        source = str(row.router_a)
        target = str(row.router_b)
        if graph.has_edge(source, target):
            continue
        if row.composite_score < score_threshold or row.direct_evidence < direct_threshold:
            continue
        if max(float(transit_scores[source]), float(transit_scores[target])) < transit_threshold:
            continue
        if not valid_degree_addition(graph, source, target):
            continue

        path_hops = nx.shortest_path_length(graph, source, target)
        if path_hops < 3:
            continue

        current_distance = nx.shortest_path_length(graph, source, target, weight="distance")
        improvement = current_distance - float(row.latency_proxy)
        candidate_score = improvement + 0.35 * float(row.composite_score)
        candidates.append((candidate_score, source, target))

    return sorted(candidates, reverse=True)


def build_inferred_graph(
    complete_graph: nx.Graph,
    pair_scores: pd.DataFrame,
    transit_scores: pd.Series,
    triangle_budget: int,
    shortcut_budget: int,
) -> nx.Graph:
    routers = sorted(complete_graph.nodes(), key=lambda item: int(item[1:]))
    inferred = nx.Graph(method="Backbone-oriented inferred topology")
    inferred.add_nodes_from(routers)

    backbone = nx.maximum_spanning_tree(complete_graph, weight="weight")
    for source, target in backbone.edges():
        add_edge_with_tier(inferred, complete_graph, source, target, "backbone_mst")

    for _ in range(max(triangle_budget, 0)):
        candidates = triangle_candidates(inferred, pair_scores)
        if not candidates:
            break
        _, source, target = candidates[0]
        add_edge_with_tier(inferred, complete_graph, source, target, "redundancy_triangle")

    for _ in range(max(shortcut_budget, 0)):
        candidates = shortcut_candidates(inferred, pair_scores, transit_scores)
        if not candidates:
            break
        _, source, target = candidates[0]
        add_edge_with_tier(inferred, complete_graph, source, target, "backbone_shortcut")

    return inferred


def edge_table(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for source, target, data in sorted(graph.edges(data=True)):
        rows.append(
            {
                "source": source,
                "target": target,
                "selection_tier": data.get("selection_tier", "selected"),
                "composite_score": round(float(data["weight"]), 6),
                "direct_evidence": round(float(data["direct_evidence"]), 6),
                "feature_similarity": round(float(data["feature_similarity"]), 6),
                "ip_jaccard": round(float(data["ip_jaccard"]), 6),
                "tuple_jaccard": round(float(data["tuple_jaccard"]), 6),
                "port_overlap": round(float(data["port_overlap"]), 6),
                "neighbor_consistency": round(float(data["neighbor_consistency"]), 6),
                "volume_compatibility": round(float(data["volume_compatibility"]), 6),
                "duration_compatibility": round(float(data["duration_compatibility"]), 6),
                "packet_balance_compatibility": round(float(data["packet_balance_compatibility"]), 6),
                "backbone_role_plausibility": round(float(data["backbone_role_plausibility"]), 6),
                "refined_mst_support": round(float(data["refined_mst_support"]), 6),
                "weak_direct_penalty": round(float(data["weak_direct_penalty"]), 6),
                "traffic_volume_proxy": round(float(data["traffic_volume_proxy"]), 3),
                "bandwidth_proxy": round(float(data["bandwidth_proxy"]), 3),
                "latency_proxy": round(float(data["distance"]), 6),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selection_tier", "composite_score"],
        ascending=[True, False],
    )


def format_tier_name(tier: str) -> str:
    labels = {
        "backbone_mst": "backbone edge",
        "redundancy_triangle": "local redundancy edge",
        "backbone_shortcut": "backbone shortcut",
    }
    return labels.get(tier, tier.replace("_", " "))


def strongest_supports(data: pd.Series | dict[str, object], top_n: int = 3) -> str:
    support_map = {
        "feature": float(data["feature_similarity"]),
        "ip": float(data["ip_jaccard"]),
        "tuple": float(data["tuple_jaccard"]),
        "ports": float(data["port_overlap"]),
        "neighbor": float(data["neighbor_consistency"]),
        "role": float(data["backbone_role_plausibility"]),
    }
    ordered = sorted(support_map.items(), key=lambda item: item[1], reverse=True)[:top_n]
    summary = ", ".join(f"{name}={value:.2f}" for name, value in ordered)
    if float(data["refined_mst_support"]) > 0.5:
        summary += "; also backed by the Q1.2(c) refined MST"
    return summary


def omitted_edge_reason(row: pd.Series, graph: nx.Graph) -> str:
    path = nx.shortest_path(graph, row["router_a"], row["router_b"])
    path_text = " -> ".join(path)
    reasons = []
    if int(row["direct_support_count"]) == 0:
        reasons.append("shared-IP/tuple/port evidence is weak")
    if float(row["feature_similarity"]) >= 0.60 and int(row["direct_support_count"]) == 0:
        reasons.append("the pair looks more like same-role similarity than a direct link")
    if float(row["refined_mst_support"]) < 0.5:
        reasons.append("Q1.2(c) did not keep the edge in the refined backbone")
    reasons.append(f"the selected graph already connects them through {path_text}")
    return "; ".join(dict.fromkeys(reasons))


def build_discussion(
    graph: nx.Graph,
    pair_scores: pd.DataFrame,
    transit_scores: pd.Series,
) -> str:
    inferred_edges = edge_table(graph)
    selected_edges = {
        tuple(sorted((str(row.source), str(row.target))))
        for row in inferred_edges.itertuples(index=False)
    }
    omitted = pair_scores[
        ~pair_scores.apply(
            lambda row: tuple(sorted((str(row["router_a"]), str(row["router_b"])))) in selected_edges,
            axis=1,
        )
    ].copy()

    tier_counts = inferred_edges["selection_tier"].value_counts().to_dict()
    top_transit = transit_scores.sort_values(ascending=False).head(4)

    lines = [
        "# Q1.3(a) Discussion",
        "",
        "## Reconstruction Assumptions",
        "",
        "- The ten collectors are treated as a sparse connected projection of a larger ISP backbone, so the reconstruction starts from a maximum-spanning backbone tree rather than a dense similarity graph.",
        "- Shared IPs, shared `(src_port, dst_port, protocol)` tuples, and shared dominant destination ports are treated as stronger evidence of direct adjacency than global feature similarity alone.",
        "- If two routers look globally similar but share weak direct-overlap evidence, the edge is penalized because it is more likely to reflect similar roles or regions than a physical link.",
        "- Routers with high flow counts, high traffic volumes, and relatively balanced forward/backward packet ratios are treated as more transit-like; redundancy edges are therefore biased toward those routers.",
        "- Only a small amount of redundancy is added after the backbone tree, because ISP backbones are resilient but still sparse.",
        "",
        "## Topology Summary",
        "",
        (
            f"The reconstructed graph has {graph.number_of_nodes()} routers, {graph.number_of_edges()} edges, "
            f"density={nx.density(graph):.3f}, and connected={nx.is_connected(graph)}."
        ),
        (
            "Selected tiers: "
            f"{tier_counts.get('backbone_mst', 0)} backbone edges, "
            f"{tier_counts.get('redundancy_triangle', 0)} triangle-closing redundancy edges, and "
            f"{tier_counts.get('backbone_shortcut', 0)} longer-path shortcut edges."
        ),
        (
            "Most transit-like routers under the Q1.1-derived heuristic are "
            + ", ".join(f"{router} ({score:.2f})" for router, score in top_transit.items())
            + "."
        ),
        "",
        "## Why Each Selected Edge Was Kept",
        "",
    ]

    for row in inferred_edges.itertuples(index=False):
        lines.append(
            f"- {row.source}-{row.target} ({format_tier_name(row.selection_tier)}): "
            f"composite={row.composite_score:.3f}, direct_evidence={row.direct_evidence:.3f}, "
            f"latency_proxy={row.latency_proxy:.3f}. Strongest supports: {strongest_supports(row._asdict())}."
        )

    lines.extend(
        [
            "",
            "## Why Strong Alternative Edges Were Left Out",
            "",
        ]
    )

    if omitted.empty:
        lines.append("- No omitted edges remained after selection.")
    else:
        omitted = omitted.sort_values("composite_score", ascending=False).head(8)
        for _, row in omitted.iterrows():
            lines.append(
                f"- {row['router_a']}-{row['router_b']}: composite={row['composite_score']:.3f}, "
                f"direct_evidence={row['direct_evidence']:.3f}. "
                f"Excluded because {omitted_edge_reason(row, graph)}."
            )

    lines.extend(
        [
            "",
            "Missing edges should therefore be read as lower-confidence under the available traffic evidence, not as proof that no physical GEANT link existed between those collectors.",
        ]
    )
    return "\n".join(lines) + "\n"


def draw_graph(
    ax: plt.Axes,
    graph: nx.Graph,
    transit_scores: pd.Series,
    title: str,
) -> None:
    pos = nx.spring_layout(graph, seed=CONFIG.random_seed, weight="weight", k=1.05)
    nodes = list(sorted(graph.nodes(), key=lambda item: int(item[1:])))
    node_colors = np.array([float(transit_scores[node]) for node in nodes])
    node_artist = nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=nodes,
        node_color=node_colors,
        cmap=plt.cm.YlOrRd,
        vmin=0.0,
        vmax=1.0,
        edgecolors="#1f1f1f",
        node_size=1050,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=11, font_weight="bold", ax=ax)

    style_map = {
        "backbone_mst": ("#355070", "solid"),
        "redundancy_triangle": ("#6d597a", "dashed"),
        "backbone_shortcut": ("#bc4749", "dashdot"),
    }
    for tier, (color, style) in style_map.items():
        tier_edges = [(u, v) for u, v, data in graph.edges(data=True) if data.get("selection_tier") == tier]
        if not tier_edges:
            continue
        tier_widths = [2.0 + 4.0 * float(graph[u][v]["weight"]) for u, v in tier_edges]
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=tier_edges,
            width=tier_widths,
            edge_color=color,
            style=style,
            ax=ax,
        )

    edge_labels = {
        (u, v): f"{data['weight']:.2f}"
        for u, v, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, rotate=False, ax=ax)

    legend_handles = [
        Line2D([0], [0], color=color, lw=2.5, linestyle=style, label=format_tier_name(tier))
        for tier, (color, style) in style_map.items()
        if any(data.get("selection_tier") == tier for _, _, data in graph.edges(data=True))
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", frameon=True)

    plt.colorbar(node_artist, ax=ax, fraction=0.046, pad=0.04, label="Transit-likelihood proxy")
    ax.set_title(title)
    ax.axis("off")


def make_figure(
    pair_scores: pd.DataFrame,
    inferred_graph: nx.Graph,
    figure_dir: Path,
    transit_scores: pd.Series,
) -> Path:
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
        cbar_kws={"label": "Composite Adjacency Score"},
        ax=axes[0],
    )
    axes[0].set_title("Pairwise Backbone-Oriented Adjacency Scores")
    axes[0].set_xlabel("Router")
    axes[0].set_ylabel("Router")

    draw_graph(axes[1], inferred_graph, transit_scores, "Reconstructed Sparse Backbone Topology")
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
    router_summary = pd.read_csv(args.router_summary_csv)
    volume_summary = pd.read_csv(args.volume_summary_csv)

    neighbor_consistency = compute_neighbor_consistency(similarity)
    volume_compatibility = compute_volume_compatibility(volume_summary)
    port_overlap = compute_port_overlap(router_summary)
    duration_compatibility = compute_duration_compatibility(router_summary)
    packet_balance_compatibility = compute_packet_balance_compatibility(router_summary)
    transit_scores = compute_transit_scores(router_summary, volume_summary)
    backbone_role_plausibility = compute_backbone_role_plausibility(transit_scores)
    refined_edges = load_refined_edge_set(args.refined_mst_csv)
    refined_mst_support = compute_refined_mst_support(list(similarity.index), refined_edges)

    pair_scores = build_pair_score_table(
        similarity=similarity,
        ip_jaccard=ip_jaccard,
        tuple_jaccard=tuple_jaccard,
        port_overlap=port_overlap,
        neighbor_consistency=neighbor_consistency,
        volume_compatibility=volume_compatibility,
        duration_compatibility=duration_compatibility,
        packet_balance_compatibility=packet_balance_compatibility,
        backbone_role_plausibility=backbone_role_plausibility,
        refined_mst_support=refined_mst_support,
        feature_profiles=feature_profiles,
        volume_summary=volume_summary,
    )

    complete_graph = build_complete_graph(pair_scores)
    inferred_graph = build_inferred_graph(
        complete_graph=complete_graph,
        pair_scores=pair_scores,
        transit_scores=transit_scores,
        triangle_budget=args.triangle_redundancy_budget,
        shortcut_budget=args.shortcut_budget,
    )

    table_dir = ensure_directory(args.table_dir)
    figure_dir = ensure_directory(args.figure_dir)

    pair_scores_path = table_dir / "q1_3a_pair_scores.csv"
    edges_path = table_dir / "q1_3a_inferred_topology_edges.csv"
    discussion_path = table_dir / "q1_3a_discussion.md"

    pair_scores.to_csv(pair_scores_path, index=False)
    edge_table(inferred_graph).to_csv(edges_path, index=False)
    discussion_path.write_text(
        build_discussion(inferred_graph, pair_scores, transit_scores),
        encoding="utf-8",
    )
    figure_path = make_figure(pair_scores, inferred_graph, figure_dir, transit_scores)

    print(f"Wrote pairwise edge scores to {pair_scores_path}")
    print(f"Wrote inferred topology edges to {edges_path}")
    print(f"Wrote discussion notes to {discussion_path}")
    print(f"Wrote inferred topology figure to {figure_path}")


if __name__ == "__main__":
    main()
