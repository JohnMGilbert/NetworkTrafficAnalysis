"""Question 1.1(a): router-level summary statistics for Task 1."""

from __future__ import annotations

import argparse
import math
import sys
from array import array
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.common.io import list_router_files, normalize_columns
from src.common.logging_utils import configure_logging
from src.common.paths import ensure_directory


FLOW_DURATION_CANDIDATES = (
    "flow_duration",
    "flow_duration_ms",
    "duration",
)
DEST_PORT_CANDIDATES = (
    "dst_port",
    "destination_port",
)
FWD_PACKET_CANDIDATES = (
    "tot_fwd_pkts",
    "total_fwd_packets",
    "fwd_pkt_count",
)
BWD_PACKET_CANDIDATES = (
    "tot_bwd_pkts",
    "total_bwd_packets",
    "bwd_pkt_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=CONFIG.raw_data_dir,
        help="Directory containing per-router CSV or parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONFIG.outputs_dir / "task1" / "tables",
        help="Directory for generated Q1.1(a) artifacts.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="CSV chunk size used to keep memory bounded.",
    )
    return parser.parse_args()


def find_first_available(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def infer_router_name(path: Path) -> str:
    stem = path.stem
    if "-" not in stem:
        return stem
    router_suffix = stem.rsplit("-", 1)[-1]
    if router_suffix.isdigit():
        return f"D{int(router_suffix)}"
    return stem


@dataclass
class RunningStats:
    count: int = 0
    mean_value: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: pd.Series) -> None:
        clean = pd.to_numeric(values, errors="coerce").dropna()
        if clean.empty:
            return
        chunk_count = int(clean.shape[0])
        chunk_mean = float(clean.mean())
        chunk_var = float(clean.var(ddof=0))
        chunk_m2 = chunk_var * chunk_count

        if self.count == 0:
            self.count = chunk_count
            self.mean_value = chunk_mean
            self.m2 = chunk_m2
        else:
            combined_count = self.count + chunk_count
            delta = chunk_mean - self.mean_value
            self.m2 = (
                self.m2
                + chunk_m2
                + (delta * delta) * self.count * chunk_count / combined_count
            )
            self.mean_value += delta * chunk_count / combined_count
            self.count = combined_count

        self.minimum = min(self.minimum, float(clean.min()))
        self.maximum = max(self.maximum, float(clean.max()))

    @property
    def mean(self) -> float | None:
        if self.count == 0:
            return None
        return self.mean_value

    @property
    def std(self) -> float | None:
        if self.count == 0:
            return None
        variance = max(self.m2 / self.count, 0.0)
        return math.sqrt(variance)


@dataclass
class RouterAccumulator:
    router_name: str
    flow_count: int = 0
    duration_stats: RunningStats = field(default_factory=RunningStats)
    duration_values: array = field(default_factory=lambda: array("d"))
    dest_ports: Counter[str] = field(default_factory=Counter)
    fwd_packet_total: float = 0.0
    bwd_packet_total: float = 0.0

    def update(
        self,
        frame: pd.DataFrame,
        duration_col: str | None,
        dest_port_col: str | None,
        fwd_col: str | None,
        bwd_col: str | None,
    ) -> None:
        self.flow_count += int(frame.shape[0])

        if duration_col:
            durations = pd.to_numeric(frame[duration_col], errors="coerce").dropna()
            self.duration_stats.update(durations)
            self.duration_values.extend(durations.astype(float).to_numpy())

        if dest_port_col:
            ports = frame[dest_port_col].dropna().astype(str)
            self.dest_ports.update(ports.tolist())

        if fwd_col:
            self.fwd_packet_total += float(pd.to_numeric(frame[fwd_col], errors="coerce").fillna(0).sum())
        if bwd_col:
            self.bwd_packet_total += float(pd.to_numeric(frame[bwd_col], errors="coerce").fillna(0).sum())

    def finalize_row(self) -> dict[str, object]:
        median = None
        if self.duration_values:
            median = float(pd.Series(self.duration_values, copy=False).median())

        if self.bwd_packet_total == 0:
            packet_ratio = None if self.fwd_packet_total == 0 else math.inf
        else:
            packet_ratio = self.fwd_packet_total / self.bwd_packet_total

        top_ports = "; ".join(
            f"{port} ({count})"
            for port, count in self.dest_ports.most_common(5)
        )

        return {
            "router": self.router_name,
            "flow_records": self.flow_count,
            "flow_duration_min": normalize_number(self.duration_stats.minimum),
            "flow_duration_max": normalize_number(self.duration_stats.maximum),
            "flow_duration_mean": normalize_number(self.duration_stats.mean),
            "flow_duration_median": normalize_number(median),
            "flow_duration_std": normalize_number(self.duration_stats.std),
            "top_5_destination_ports": top_ports,
            "forward_packets_total": normalize_number(self.fwd_packet_total),
            "backward_packets_total": normalize_number(self.bwd_packet_total),
            "forward_to_backward_packet_ratio": format_ratio(packet_ratio),
        }


def normalize_number(value: float | None) -> float | None:
    if value is None or value in (math.inf, -math.inf):
        return None
    return round(float(value), 6)


def format_ratio(value: float | None) -> str:
    if value is None:
        return "NA"
    if math.isinf(value):
        return "inf"
    return f"{value:.6f}"


def iter_router_frames(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    if path.suffix.lower() == ".parquet":
        yield normalize_columns(pd.read_parquet(path))
        return

    chunk_iter = pd.read_csv(path, chunksize=chunksize, low_memory=False)
    for chunk in chunk_iter:
        yield normalize_columns(chunk)


def summarize_router(path: Path, chunksize: int) -> dict[str, object]:
    accumulator = RouterAccumulator(router_name=infer_router_name(path))
    duration_col = None
    dest_port_col = None
    fwd_col = None
    bwd_col = None

    for frame in iter_router_frames(path, chunksize):
        if duration_col is None:
            duration_col = find_first_available(frame.columns, FLOW_DURATION_CANDIDATES)
            dest_port_col = find_first_available(frame.columns, DEST_PORT_CANDIDATES)
            fwd_col = find_first_available(frame.columns, FWD_PACKET_CANDIDATES)
            bwd_col = find_first_available(frame.columns, BWD_PACKET_CANDIDATES)
        accumulator.update(frame, duration_col, dest_port_col, fwd_col, bwd_col)

    return accumulator.finalize_row()


def summarize_router_group(paths: list[Path], chunksize: int) -> dict[str, object]:
    accumulator = RouterAccumulator(router_name=infer_router_name(paths[0]))
    duration_col = None
    dest_port_col = None
    fwd_col = None
    bwd_col = None

    for path in sorted(paths):
        for frame in iter_router_frames(path, chunksize):
            if duration_col is None:
                duration_col = find_first_available(frame.columns, FLOW_DURATION_CANDIDATES)
                dest_port_col = find_first_available(frame.columns, DEST_PORT_CANDIDATES)
                fwd_col = find_first_available(frame.columns, FWD_PACKET_CANDIDATES)
                bwd_col = find_first_available(frame.columns, BWD_PACKET_CANDIDATES)
            accumulator.update(frame, duration_col, dest_port_col, fwd_col, bwd_col)

    return accumulator.finalize_row()


def group_router_files(router_files: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in router_files:
        router_name = infer_router_name(path)
        grouped.setdefault(router_name, []).append(path)
    return grouped


def build_discussion(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "# Q1.1(a) Discussion\n\nNo router data was available.\n"

    flow_leader = summary.loc[summary["flow_records"].idxmax()]
    duration_leader = summary.loc[summary["flow_duration_mean"].fillna(-1).idxmax()]

    ratio_values = pd.to_numeric(
        summary["forward_to_backward_packet_ratio"].replace({"inf": None, "NA": None}),
        errors="coerce",
    )
    imbalance_idx = ratio_values.sub(1.0).abs().idxmax() if ratio_values.notna().any() else None

    lines = [
        "# Q1.1(a) Discussion",
        "",
        (
            f"{flow_leader['router']} carries the largest observed flow volume "
            f"({int(flow_leader['flow_records']):,} records), which suggests it may sit on a busier "
            "portion of the network or aggregate traffic from multiple neighboring segments."
        ),
        (
            f"{duration_leader['router']} shows the highest mean flow duration "
            f"({duration_leader['flow_duration_mean']}) among the sampled routers, pointing to longer-lived "
            "sessions or slower application patterns passing through that vantage point."
        ),
    ]

    if imbalance_idx is not None:
        imbalance_row = summary.loc[imbalance_idx]
        lines.append(
            (
                f"{imbalance_row['router']} has the most asymmetric forward/backward packet ratio "
                f"({imbalance_row['forward_to_backward_packet_ratio']}), which may indicate a directional role "
                "such as heavier request fan-out, response aggregation, or visibility into only one side of some flows."
            )
        )

    lines.append(
        "Destination-port mixes should be compared alongside these aggregates: repeated concentration on a small set "
        "of ports often indicates specialized application traffic, while broader port diversity can indicate transit "
        "or mixed-service behavior."
    )
    return "\n\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    configure_logging()

    router_files = list_router_files(args.data_dir)
    if not router_files:
        raise FileNotFoundError(
            f"No router CSV/parquet files were found under {args.data_dir}. "
            "Place the FLNET2023 router files in data/raw/ first."
        )

    output_dir = ensure_directory(args.output_dir)
    router_groups = group_router_files(router_files)
    rows = [
        summarize_router_group(paths, args.chunksize)
        for _, paths in sorted(router_groups.items(), key=lambda item: int(item[0][1:]))
    ]
    summary = pd.DataFrame(rows).sort_values("router", key=lambda s: s.str[1:].astype(int)).reset_index(drop=True)

    csv_path = output_dir / "q1_1a_router_summary.csv"
    md_path = output_dir / "q1_1a_discussion.md"

    summary.to_csv(csv_path, index=False)
    md_path.write_text(build_discussion(summary), encoding="utf-8")

    print(f"Wrote summary table to {csv_path}")
    print(f"Wrote discussion notes to {md_path}")


if __name__ == "__main__":
    main()
