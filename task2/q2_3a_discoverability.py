"""Question 2.3(a): infer per-class discoverability from unsupervised results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.paths import ensure_directory


CLASSES = [
    "DDoS TCP",
    "DoS SlowHTTP",
    "DDoS Bot",
    "DoS Hulk",
    "Normal",
    "DDoS Dyn",
    "DDoS Stomp",
    "Web XSS",
    "Infiltration MITM",
    "Web SQL Injection",
    "Web Command Injection",
]

CONFIDENCE_BONUS = {"high": 3.0, "medium": 2.0, "low": 0.5, "none": 0.0}

CLASS_PRIORS = {
    "DDoS TCP": {
        "prevalence": "very high",
        "separability": "very high",
        "mapped_cluster": "DDoS TCP",
        "hardness": "Minimal packets, strong one-way asymmetry, and overwhelming support make this class easy to isolate.",
        "helpful_signal": "Already well separated by packet count, packet size, and directionality.",
    },
    "DoS SlowHTTP": {
        "prevalence": "high",
        "separability": "very high",
        "mapped_cluster": "DoS SlowHTTP",
        "hardness": "Long-lived, very low-rate flows form an unusually clean slow-attack signature.",
        "helpful_signal": "Connection duration and packets-per-second already separate it well.",
    },
    "DDoS Bot": {
        "prevalence": "high",
        "separability": "high",
        "mapped_cluster": "DDoS Bot",
        "hardness": "Large, asymmetric bot-driven flood flows create a recognizable dense region.",
        "helpful_signal": "Service-port concentration and payload asymmetry are the key cues.",
    },
    "DoS Hulk": {
        "prevalence": "high",
        "separability": "high",
        "mapped_cluster": "DoS Hulk",
        "hardness": "Burst-heavy HTTP flooding is distinct, but it partially overlaps with other volumetric attacks.",
        "helpful_signal": "Burst timing and request-heavy packet patterns help.",
    },
    "Normal": {
        "prevalence": "medium",
        "separability": "medium",
        "mapped_cluster": "Normal",
        "hardness": "Normal traffic is broad and heterogeneous, so it forms a usable but not perfectly tight cluster.",
        "helpful_signal": "Port diversity and balanced bidirectional exchanges help most.",
    },
    "DDoS Dyn": {
        "prevalence": "medium",
        "separability": "medium-low",
        "mapped_cluster": None,
        "hardness": "Likely overlaps with other DDoS variants because the current feature set mostly captures flood intensity, not variant-specific control behavior.",
        "helpful_signal": "Temporal burst shape, TCP flag sequences, and target-host concentration would help.",
    },
    "DDoS Stomp": {
        "prevalence": "medium",
        "separability": "medium-low",
        "mapped_cluster": None,
        "hardness": "Likely collapses into the broader DDoS region because its packet-rate profile resembles other flooding attacks.",
        "helpful_signal": "Handshake-state ratios, retransmission patterns, and SYN/ACK timing would help.",
    },
    "Web XSS": {
        "prevalence": "low",
        "separability": "low",
        "mapped_cluster": None,
        "hardness": "Application-layer abuse can look close to ordinary web sessions in flow-only statistics.",
        "helpful_signal": "HTTP method, URI length, response code, and header-level metadata would help.",
    },
    "Infiltration MITM": {
        "prevalence": "very low",
        "separability": "very low",
        "mapped_cluster": None,
        "hardness": "MITM activity often hides inside otherwise plausible client-server exchanges and may be sparse enough to be treated as noise.",
        "helpful_signal": "Host context, TLS fingerprints, ARP/DNS anomalies, and sequence-level timing would help.",
    },
    "Web SQL Injection": {
        "prevalence": "very low",
        "separability": "very low",
        "mapped_cluster": None,
        "hardness": "SQL injection attempts are rare and can be nearly indistinguishable from legitimate HTTP flows at the aggregate flow level.",
        "helpful_signal": "Request payload features, URI/query tokenization, and server response semantics would help.",
    },
    "Web Command Injection": {
        "prevalence": "extreme low",
        "separability": "very low",
        "mapped_cluster": None,
        "hardness": "The assignment explicitly states this class has only 330 training instances, so it is too rare to form a stable dense cluster and is likely absorbed into noise or benign-looking web traffic.",
        "helpful_signal": "HTTP payload inspection, shell-metacharacter features, and endpoint telemetry would help.",
    },
}

SEPARABILITY_SCORE = {
    "very high": 4.0,
    "high": 3.0,
    "medium": 2.0,
    "medium-low": 1.25,
    "low": 0.75,
    "very low": 0.25,
}

PREVALENCE_SCORE = {
    "very high": 2.5,
    "high": 2.0,
    "medium": 1.2,
    "medium-low": 0.8,
    "low": 0.3,
    "very low": -0.3,
    "extreme low": -0.8,
}

HARDEST_CLASS_DETAILS = {
    "Web Command Injection": (
        "is hardest because it is extremely rare and application-layer in nature, so flow statistics alone do not "
        "give it a stable cluster. Extra help would come from HTTP payload inspection, shell-metacharacter counts, "
        "and endpoint telemetry."
    ),
    "Web SQL Injection": (
        "is near the bottom because it is also rare and looks too similar to ordinary web sessions at the flow level. "
        "URI/query-token features and response-code patterns would help."
    ),
    "Infiltration MITM": (
        "remains difficult because MITM behavior can preserve plausible client-server volumes while only changing subtle "
        "session context. TLS fingerprints, DNS/ARP anomalies, and host-level context would help."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q2-2c-summary-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_2c_hdbscan_summary.json",
        help="Summary JSON produced by Question 2.2(c).",
    )
    parser.add_argument(
        "--q2-2b-summary-path",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables" / "q2_2b_summary.json",
        help="Summary JSON produced by Question 2.2(b).",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task2" / "tables",
        help="Directory for generated Task 2.3(a) outputs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mapped_cluster_lookup(summary: dict) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for cluster in summary["clusters"]:
        predicted_label = cluster["predicted_label"]
        if predicted_label != "Unknown":
            lookup[predicted_label] = cluster
    return lookup


def build_discoverability_table(q2_2c_summary: dict) -> pd.DataFrame:
    mapped = mapped_cluster_lookup(q2_2c_summary)
    rows: list[dict[str, object]] = []
    for class_name in CLASSES:
        prior = CLASS_PRIORS[class_name]
        cluster = mapped.get(class_name)
        cluster_fraction = float(cluster["cluster_fraction"]) if cluster else 0.0
        confidence = str(cluster["confidence"]) if cluster else "none"
        score = (
            SEPARABILITY_SCORE[prior["separability"]]
            + PREVALENCE_SCORE[prior["prevalence"]]
            + CONFIDENCE_BONUS[confidence]
            + (2.0 * cluster_fraction)
        )
        rows.append(
            {
                "class_name": class_name,
                "discoverability_score": round(score, 3),
                "discoverability_rank": 0,
                "mapped_to_hdbscan_cluster": cluster is not None,
                "mapped_cluster_label": int(cluster["cluster_label"]) if cluster else None,
                "cluster_confidence": confidence,
                "cluster_fraction": round(cluster_fraction, 4),
                "prevalence_assumption": prior["prevalence"],
                "separability_assumption": prior["separability"],
                "why_ranked_here": prior["hardness"],
                "additional_features_or_data": prior["helpful_signal"],
            }
        )

    table = pd.DataFrame(rows).sort_values(
        by=["discoverability_score", "cluster_fraction", "class_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    table["discoverability_rank"] = table.index + 1
    return table


def render_report(table: pd.DataFrame, q2_2b_summary: dict) -> str:
    hardest = table.nlargest(3, "discoverability_rank")["class_name"].tolist()
    lines = [
        "# Task 2.3(a) Discoverability Analysis",
        "",
        "## Method",
        "This ranking is an evidence-based inference, not a ground-truth evaluation. The Task 2 dataset is unlabeled during discovery, so the ordering combines:",
        f"- the best unsupervised model selected in Question 2.2(b): `{q2_2b_summary['best_algorithm']}`;",
        "- whether Question 2.2(c) produced a dedicated HDBSCAN cluster for a class and how confident that mapping was;",
        "- the structural distinctiveness of each class in the existing flow features;",
        "- the assignment's stated imbalance facts, especially that `Web Command Injection` has only 330 instances while `DDoS TCP` has over 1.8 million.",
        "",
        "## Discoverability Ranking",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"{row.discoverability_rank}. **{row.class_name}** "
            f"(score {row.discoverability_score:.3f}, cluster confidence: {row.cluster_confidence}) - "
            f"{row.why_ranked_here}"
        )

    lines.extend(
        [
            "",
            "## Hardest Classes",
            *(f"- **{class_name}** {HARDEST_CLASS_DETAILS[class_name]}" for class_name in hardest),
            "",
            "## Interpretation",
            "- The easiest classes are the ones with extreme, low-entropy flow signatures: tiny one-way floods (`DDoS TCP`) or very long low-rate sessions (`DoS SlowHTTP`).",
            "- Mid-tier classes are still discoverable, but only as coarse behavior families. `DDoS Dyn` and `DDoS Stomp` likely blend into broader DDoS structure instead of forming their own dense clusters.",
            "- The web attacks and `Infiltration MITM` are hardest because they are both rare and semantically defined by payload or endpoint context that the current CICFlowMeter-style flow features do not capture.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_directory(args.table_dir)

    q2_2c_summary = load_json(args.q2_2c_summary_path)
    q2_2b_summary = load_json(args.q2_2b_summary_path)

    table = build_discoverability_table(q2_2c_summary)
    report = render_report(table, q2_2b_summary)
    table["mapped_cluster_label"] = table["mapped_cluster_label"].astype(object)
    table.loc[table["mapped_to_hdbscan_cluster"] == False, "mapped_cluster_label"] = None
    json_ready_table: list[dict[str, object]] = []
    for row in table.itertuples(index=False, name="DiscoverabilityRow"):
        cleaned = {}
        for key in table.columns:
            value = getattr(row, key)
            cleaned[key] = None if pd.isna(value) else value
        json_ready_table.append(cleaned)

    table_path = args.table_dir / "q2_3a_discoverability_ranking.csv"
    report_path = args.table_dir / "q2_3a_report.md"
    summary_path = args.table_dir / "q2_3a_summary.json"

    table.to_csv(table_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "best_algorithm": q2_2b_summary["best_algorithm"],
                "ranking_method": (
                    "Inference from HDBSCAN cluster mapping confidence, class-imbalance facts stated in the assignment, "
                    "and qualitative separability of the existing flow features."
                ),
                "hardest_classes": table.nlargest(3, "discoverability_rank")["class_name"].tolist(),
                "ranking": json_ready_table,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
