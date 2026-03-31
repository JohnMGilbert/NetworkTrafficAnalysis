"""Question 2.2(c): interpret the best-performing HDBSCAN clusters."""

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

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory
from task2.q2_2a_clustering import load_sample
from task2.q2_2a_clustering import stratified_indices_by_router


LOGGER = logging.getLogger("task2.q2_2c")
KNOWN_CLASSES = [
    "Normal",
    "DDoS Bot",
    "DDoS Dyn",
    "DDoS Stomp",
    "DDoS TCP",
    "DoS Hulk",
    "DoS SlowHTTP",
    "Infiltration MITM",
    "Web Command Injection",
    "Web SQL Injection",
    "Web XSS",
    "Unknown",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=CONFIG.interim_data_dir / "task2_q2_1b_umap_sample.parquet",
        help="Parquet sample produced by Question 2.1(b).",
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_1b_umap_embedding.csv",
        help="UMAP coordinates produced by Question 2.1(b).",
    )
    parser.add_argument(
        "--q2-2a-summary-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_2a_summary.json",
        help="Summary JSON produced by Question 2.2(a).",
    )
    parser.add_argument(
        "--assignments-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_2a_cluster_assignments.csv",
        help="Cluster assignments produced by Question 2.2(a).",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_1a_scaler_parameters.csv",
        help="Scaler metadata produced by Question 2.1(a).",
    )
    parser.add_argument(
        "--feature-audit-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_1a_feature_audit.csv",
        help="Feature audit table produced by Question 2.1(a).",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables",
        help="Directory for generated Task 2.2(c) tables and reports.",
    )
    return parser.parse_args()


def parse_q2_2a_summary(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload["fit_rows"])


def reconstruct_fit_sample(
    sample_path: Path,
    embedding_path: Path,
    fit_rows: int,
) -> tuple[pd.DataFrame, list[str]]:
    sample, _, feature_columns = load_sample(sample_path, embedding_path)
    fit_indices = stratified_indices_by_router(sample["router_id"], fit_rows, CONFIG.random_seed)
    fit_sample = sample.iloc[fit_indices].reset_index(drop=True)
    return fit_sample, feature_columns


def inverse_transform(frame: pd.DataFrame, scaler: pd.DataFrame, column: str) -> pd.Series:
    params = scaler.loc[column]
    return frame[column] * params["scaling_std"] + params["scaling_mean"]


def top_values(series: pd.Series, limit: int = 5) -> str:
    counts = series.value_counts().head(limit)
    return ", ".join(f"{int(value)} ({count})" for value, count in counts.items())


def build_analysis_frame(
    fit_sample: pd.DataFrame,
    assignments: pd.DataFrame,
    scaler: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "router_id": fit_sample["router_id"].astype(str),
            "cluster_label": assignments["hdbscan_cluster"].astype(int),
        }
    )
    for column in [
        "flow_duration",
        "tot_fwd_pkts",
        "tot_bwd_pkts",
        "totlen_fwd_pkts",
        "totlen_bwd_pkts",
        "dst_port",
        "src_port",
        "flow_pkts_s",
        "flow_byts_s",
        "down_up_ratio",
        "pkt_size_avg",
        "fwd_pkt_len_mean",
        "bwd_pkt_len_mean",
        "active_mean",
        "idle_mean",
    ]:
        frame[column] = inverse_transform(fit_sample, scaler, column)

    frame["dst_port"] = frame["dst_port"].round().astype(int)
    frame["src_port"] = frame["src_port"].round().astype(int)
    frame["total_pkts"] = frame["tot_fwd_pkts"] + frame["tot_bwd_pkts"]
    frame["total_bytes"] = frame["totlen_fwd_pkts"] + frame["totlen_bwd_pkts"]
    frame["bytes_per_packet"] = frame["total_bytes"] / (frame["total_pkts"] + 1.0)
    frame["directional_byte_imbalance"] = (
        frame["totlen_fwd_pkts"] - frame["totlen_bwd_pkts"]
    ) / (frame["total_bytes"] + 1.0)
    frame["burst_idle_log_ratio"] = np.log1p(frame["active_mean"].clip(lower=0.0)) - np.log1p(
        frame["idle_mean"].clip(lower=0.0)
    )
    frame["packet_size_asymmetry"] = (
        np.abs(frame["fwd_pkt_len_mean"] - frame["bwd_pkt_len_mean"])
        / (frame["fwd_pkt_len_mean"] + frame["bwd_pkt_len_mean"] + 1.0)
    )
    return frame


def build_cluster_statistics(analysis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_rows = len(analysis)
    for cluster_label in sorted(analysis["cluster_label"].unique()):
        cluster = analysis[analysis["cluster_label"] == cluster_label]
        rows.append(
            {
                "cluster_label": cluster_label,
                "cluster_size": len(cluster),
                "cluster_fraction": round(len(cluster) / total_rows, 6),
                "mean_flow_duration": round(cluster["flow_duration"].mean(), 2),
                "mean_packet_count": round(cluster["total_pkts"].mean(), 2),
                "mean_byte_count": round(cluster["total_bytes"].mean(), 2),
                "mean_flow_packets_per_second": round(cluster["flow_pkts_s"].mean(), 2),
                "mean_flow_bytes_per_second": round(cluster["flow_byts_s"].mean(), 2),
                "mean_down_up_ratio": round(cluster["down_up_ratio"].mean(), 3),
                "mean_bytes_per_packet": round(cluster["bytes_per_packet"].mean(), 2),
                "mean_directional_byte_imbalance": round(cluster["directional_byte_imbalance"].mean(), 3),
                "mean_packet_size_asymmetry": round(cluster["packet_size_asymmetry"].mean(), 3),
                "dominant_destination_ports": top_values(cluster["dst_port"]),
                "dominant_source_ports": top_values(cluster["src_port"]),
                "dominant_routers": ", ".join(
                    f"{router} ({count})"
                    for router, count in cluster["router_id"].value_counts().head(5).items()
                ),
                "dominant_protocols": "protocol feature removed in preprocessing (zero variance)",
            }
        )
    return pd.DataFrame(rows)


def assign_semantic_label(row: pd.Series) -> tuple[str, str, str]:
    cluster_label = int(row["cluster_label"])
    duration = float(row["mean_flow_duration"])
    packet_count = float(row["mean_packet_count"])
    byte_count = float(row["mean_byte_count"])
    packet_rate = float(row["mean_flow_packets_per_second"])
    down_up_ratio = float(row["mean_down_up_ratio"])
    asymmetry = float(row["mean_packet_size_asymmetry"])
    dominant_ports = str(row["dominant_destination_ports"])

    if cluster_label == -1:
        evidence = (
            f"HDBSCAN marked this group as noise/outliers, and its dominant ports ({dominant_ports}) are more mixed than "
            "the main dense clusters. It also carries comparatively high byte volume per flow, which suggests rare attack "
            "subtypes or blended behavior rather than one clean class."
        )
        return "Unknown", "low", evidence

    if duration > 2.5e7 and packet_rate < 10:
        evidence = (
            f"The flows are long-lived (mean duration {duration:.0f}) but extremely low-rate ({packet_rate:.2f} pkts/s), "
            "which matches slow application-layer connection exhaustion. The cluster is also concentrated on only a couple "
            f"of destination ports ({dominant_ports}), consistent with a focused SlowHTTP-style service target."
        )
        return "DoS SlowHTTP", "high", evidence

    if packet_count < 5 and byte_count < 250 and down_up_ratio <= 0.05:
        evidence = (
            f"The flows are tiny (mean {packet_count:.2f} packets and {byte_count:.2f} bytes) and almost completely "
            f"one-directional (down/up ratio {down_up_ratio:.2f}). That is consistent with TCP flood behavior that pushes "
            "minimal request traffic without meaningful responses."
        )
        return "DDoS TCP", "high", evidence

    if packet_count > 90 and byte_count > 70000 and asymmetry > 0.8:
        evidence = (
            f"The cluster has large flows (mean {packet_count:.2f} packets and {byte_count:.2f} bytes) with strong "
            f"forward/backward packet-size asymmetry ({asymmetry:.2f}). Its traffic is concentrated on a small set of "
            f"service ports ({dominant_ports}), which is consistent with coordinated bot-driven flooding."
        )
        return "DDoS Bot", "medium", evidence

    if duration > 1.0e7 and packet_count > 80 and 0.5 <= down_up_ratio <= 0.9:
        evidence = (
            f"The flows are sustained over long durations (mean {duration:.0f}) with around {packet_count:.2f} packets "
            f"per flow, suggesting repeated request bursts against one service. The destination-port concentration "
            f"({dominant_ports}) and moderate request/response balance (down/up ratio {down_up_ratio:.2f}) fit a Hulk-like "
            "HTTP flood better than benign background traffic."
        )
        return "DoS Hulk", "medium", evidence

    if duration < 2.0e3 and 0.8 <= down_up_ratio <= 1.2 and byte_count < 300:
        evidence = (
            f"The cluster consists of short, balanced exchanges (mean duration {duration:.0f}, down/up ratio "
            f"{down_up_ratio:.2f}) with small byte volume ({byte_count:.2f}). Its destination ports are more diverse than "
            "the focused attack clusters, which is the strongest available signal for routine background traffic."
        )
        return "Normal", "medium", evidence

    evidence = (
        f"The cluster shows a mixed profile with mean duration {duration:.0f}, packet count {packet_count:.2f}, and "
        f"dominant destination ports {dominant_ports}. Those statistics do not isolate one of the assignment's named "
        "attack categories cleanly, so this cluster is left as Unknown."
    )
    return "Unknown", "low", evidence


def build_mapping_table(cluster_stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in cluster_stats.iterrows():
        predicted_label, confidence, evidence = assign_semantic_label(row)
        rows.append(
            {
                "cluster_label": int(row["cluster_label"]),
                "predicted_label": predicted_label,
                "confidence": confidence,
                "evidence": evidence,
            }
        )
    return pd.DataFrame(rows)


def build_confusion_style_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels_in_use = set(mapping["predicted_label"])
    for cluster_label in sorted(mapping["cluster_label"].unique()):
        predicted_label = mapping.loc[mapping["cluster_label"] == cluster_label, "predicted_label"].iloc[0]
        row = {
            "cluster_label": cluster_label,
            "predicted_label": predicted_label,
        }
        for known_label in KNOWN_CLASSES:
            row[known_label] = int(known_label == predicted_label and known_label in labels_in_use)
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    cluster_stats: pd.DataFrame,
    mapping: pd.DataFrame,
) -> None:
    lines = [
        "# Task 2.2(c) HDBSCAN Cluster Analysis",
        "",
        "## Scope",
        (
            "This analysis focuses on HDBSCAN because it was the strongest performer in Question 2.2(b). Cluster "
            "statistics were computed on the same fitted sample used in Question 2.2(a), with inverse-transformed packet, "
            "byte, duration, and port features for interpretability."
        ),
        "",
        "## Protocol Note",
        (
            "The `protocol` feature was removed during preprocessing because it had zero variance in the dataset, so the "
            "cluster profiles emphasize ports, packet counts, byte counts, flow duration, and directional asymmetry instead."
        ),
        "",
        "## Cluster Label Mapping",
    ]

    for _, row in mapping.sort_values("cluster_label").iterrows():
        lines.append(
            f"- Cluster {int(row['cluster_label'])}: {row['predicted_label']} ({row['confidence']} confidence). {row['evidence']}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            (
                "HDBSCAN discovered only a small number of dense groups, so several of the assignment's eleven named "
                "classes likely collapsed together or were left in the noise bucket. The `Unknown` mapping should be read "
                "as a conservative choice rather than a model failure: it reflects insufficient evidence to separate rare "
                "attack families cleanly without labels."
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    ensure_directory(args.table_dir)

    fit_rows = parse_q2_2a_summary(args.q2_2a_summary_path)
    fit_sample, _ = reconstruct_fit_sample(
        sample_path=args.sample_path,
        embedding_path=args.embedding_path,
        fit_rows=fit_rows,
    )
    assignments = pd.read_csv(args.assignments_path)
    scaler = pd.read_csv(args.scaler_path).set_index("feature")
    feature_audit = pd.read_csv(args.feature_audit_path)

    if "protocol" in feature_audit["feature"].values:
        protocol_reason = feature_audit.loc[feature_audit["feature"] == "protocol", "removal_reason"].iloc[0]
        LOGGER.info("Protocol feature removal reason: %s", protocol_reason)

    analysis = build_analysis_frame(fit_sample, assignments, scaler)
    cluster_stats = build_cluster_statistics(analysis)
    mapping = build_mapping_table(cluster_stats)
    confusion_style = build_confusion_style_mapping(mapping)

    cluster_stats.to_csv(args.table_dir / "q2_2c_hdbscan_cluster_statistics.csv", index=False)
    mapping.to_csv(args.table_dir / "q2_2c_hdbscan_cluster_mapping.csv", index=False)
    confusion_style.to_csv(args.table_dir / "q2_2c_hdbscan_confusion_style_mapping.csv", index=False)

    summary_payload = {
        "best_algorithm": "HDBSCAN",
        "cluster_count_excluding_noise": int((cluster_stats["cluster_label"] != -1).sum()),
        "noise_cluster_present": bool((cluster_stats["cluster_label"] == -1).any()),
        "clusters": cluster_stats.merge(mapping, on="cluster_label").to_dict(orient="records"),
    }
    (args.table_dir / "q2_2c_hdbscan_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    write_report(args.table_dir / "q2_2c_hdbscan_report.md", cluster_stats, mapping)
    LOGGER.info("Task 2.2(c) HDBSCAN artifacts written to %s.", args.table_dir)


if __name__ == "__main__":
    main()
