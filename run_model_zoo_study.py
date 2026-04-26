#!/usr/bin/env python3
"""Run LeNet/VGG/ResNet trace-driven NoC studies with visualizations.

Outputs:
- traces per model
- per-model BookSim run outputs
- combined CSV
- plots (PNG)
- markdown insights report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from hardware_config import (
    FIXED_HW_8X8_MESH,
    FIXED_INJECTION_RATE,
    FIXED_PACKET_SIZE_FLITS,
    FIXED_SEED,
)

NODES = FIXED_HW_8X8_MESH.num_nodes
GRID = FIXED_HW_8X8_MESH.k


@dataclass
class Phase:
    name: str
    cycles: int
    kind: str
    strength: float


def coord(node: int) -> Tuple[int, int]:
    return (node // GRID, node % GRID)


def node_id(r: int, c: int) -> int:
    return r * GRID + c


def manhattan(a: int, b: int) -> int:
    ar, ac = coord(a)
    br, bc = coord(b)
    return abs(ar - br) + abs(ac - bc)


def add_edge(edges: Dict[Tuple[int, int], float], src: int, dst: int, w: float) -> None:
    if w <= 0:
        return
    edges[(src, dst)] = edges.get((src, dst), 0.0) + w


def build_local_conv_phase(strength: float) -> Dict[Tuple[int, int], float]:
    edges: Dict[Tuple[int, int], float] = {}
    for s in range(NODES):
        # Local stencil communication; stronger to closer neighbors.
        for d in range(NODES):
            if s == d:
                continue
            dist = manhattan(s, d)
            if dist == 1:
                add_edge(edges, s, d, 3.0 * strength)
            elif dist == 2:
                add_edge(edges, s, d, 1.5 * strength)
        # Keep tiny self-edge so every source is covered in strict mode.
        add_edge(edges, s, s, 0.05 * strength)
    return edges


def build_pool_phase(strength: float) -> Dict[Tuple[int, int], float]:
    edges: Dict[Tuple[int, int], float] = {}
    for r in range(0, GRID, 2):
        for c in range(0, GRID, 2):
            collector = node_id(r, c)
            members = [
                node_id(r, c),
                node_id(r, c + 1),
                node_id(r + 1, c),
                node_id(r + 1, c + 1),
            ]
            for s in members:
                if s != collector:
                    add_edge(edges, s, collector, 2.0 * strength)
                add_edge(edges, s, s, 0.05 * strength)
    return edges


def build_fc_phase(strength: float) -> Dict[Tuple[int, int], float]:
    edges: Dict[Tuple[int, int], float] = {}
    for s in range(NODES):
        for d in range(NODES):
            if s == d:
                continue
            add_edge(edges, s, d, strength)
        add_edge(edges, s, s, 0.05 * strength)
    return edges


def build_residual_merge_phase(strength: float) -> Dict[Tuple[int, int], float]:
    edges: Dict[Tuple[int, int], float] = {}
    # 8 merge points across rows; each source contributes to one merge point.
    merge_nodes = [node_id(r, 4) for r in range(GRID)]
    for s in range(NODES):
        r, _ = coord(s)
        m = merge_nodes[r]
        add_edge(edges, s, m, 2.5 * strength)
        # small side traffic on skip path
        add_edge(edges, s, (s + 1) % NODES, 0.7 * strength)
        add_edge(edges, s, s, 0.05 * strength)
    return edges


def phase_edges(kind: str, strength: float) -> Dict[Tuple[int, int], float]:
    if kind == "conv":
        return build_local_conv_phase(strength)
    if kind == "pool":
        return build_pool_phase(strength)
    if kind == "fc":
        return build_fc_phase(strength)
    if kind == "res_merge":
        return build_residual_merge_phase(strength)
    raise ValueError(f"Unknown phase kind: {kind}")


def model_specs_template() -> Dict[str, List[Phase]]:
    return {
        "lenet": [
            Phase("conv1", 2600, "conv", 1.0),
            Phase("pool1", 1400, "pool", 0.8),
            Phase("conv2", 2400, "conv", 1.2),
            Phase("pool2", 1400, "pool", 0.7),
            Phase("fc1", 2000, "fc", 0.35),
            Phase("fc2", 1600, "fc", 0.30),
        ],
        "vgg": [
            Phase("conv1_1", 2800, "conv", 1.1),
            Phase("conv1_2", 2800, "conv", 1.1),
            Phase("pool1", 1400, "pool", 0.8),
            Phase("conv2_1", 3000, "conv", 1.2),
            Phase("conv2_2", 3000, "conv", 1.2),
            Phase("pool2", 1400, "pool", 0.8),
            Phase("conv3_1", 3200, "conv", 1.3),
            Phase("conv3_2", 3200, "conv", 1.3),
            Phase("pool3", 1500, "pool", 0.9),
            Phase("fc1", 2200, "fc", 0.45),
            Phase("fc2", 2000, "fc", 0.40),
        ],
        "resnet": [
            Phase("stem_conv", 2800, "conv", 1.1),
            Phase("pool", 1400, "pool", 0.8),
            Phase("block1_conv1", 2600, "conv", 1.0),
            Phase("block1_conv2", 2600, "conv", 1.0),
            Phase("block1_merge", 1800, "res_merge", 1.0),
            Phase("block2_conv1", 2800, "conv", 1.1),
            Phase("block2_conv2", 2800, "conv", 1.1),
            Phase("block2_merge", 1800, "res_merge", 1.1),
            Phase("fc", 2200, "fc", 0.35),
        ],
    }


def model_specs_pytorch() -> Dict[str, List[Phase]]:
    from pytorch_trace_generator import generate_pytorch_phases

    specs: Dict[str, List[Phase]] = {}
    for model in ["lenet", "vgg", "resnet"]:
        raw_phases = generate_pytorch_phases(model)
        phases: List[Phase] = []
        for p in raw_phases:
            phases.append(
                Phase(
                    name=str(p["name"]),
                    cycles=int(p["cycles"]),
                    kind=str(p["kind"]),
                    strength=float(p["strength"]),
                )
            )
        specs[model] = phases
    return specs


def write_trace_for_model(model: str, phases: List[Phase], trace_path: Path, manifest_path: Path) -> List[int]:
    phase_cycles = [p.cycles for p in phases]
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    with trace_path.open("w", encoding="utf-8") as f:
        f.write("# phase src dst weight\n")
        for i, p in enumerate(phases):
            edges = phase_edges(p.kind, p.strength)
            for (src, dst), w in sorted(edges.items()):
                f.write(f"{i} {src} {dst} {w:.8f}\n")

    manifest = {
        "model": model,
        "num_nodes": NODES,
        "phase_cycles": phase_cycles,
        "phases": [p.__dict__ for p in phases],
        "trace_file": str(trace_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return phase_cycles


def write_trace_for_explicit_phases(
    model: str,
    explicit: Dict[str, object],
    trace_path: Path,
    manifest_path: Path,
) -> List[int]:
    phases_raw = explicit.get("phases", [])
    if not isinstance(phases_raw, list) or not phases_raw:
        raise ValueError(f"No explicit phases generated for model: {model}")

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    phase_cycles: List[int] = []

    with trace_path.open("w", encoding="utf-8") as f:
        f.write("# phase src dst weight\n")
        for i, p in enumerate(phases_raw):
            if not isinstance(p, dict):
                continue
            cyc = int(p.get("cycles", 1000))
            phase_cycles.append(cyc)
            edges = p.get("edges", [])
            if not isinstance(edges, list):
                edges = []
            for e in edges:
                if not isinstance(e, dict):
                    continue
                src = int(e.get("src", -1))
                dst = int(e.get("dst", -1))
                w = float(e.get("weight", 0.0))
                if src < 0 or dst < 0 or w <= 0.0:
                    continue
                f.write(f"{i} {src} {dst} {w:.8f}\n")

    manifest = {
        "model": model,
        "num_nodes": NODES,
        "mapping_mode": explicit.get("mapping_mode", "strict_tiled"),
        "memory_nodes": explicit.get("memory_nodes", [explicit.get("memory_node", NODES - 1)]),
        "memory_node": explicit.get("memory_node", NODES - 1),
        "max_sim_tiles_per_layer": explicit.get("max_sim_tiles_per_layer", None),
        "phase_cycles": phase_cycles,
        "layers": explicit.get("layers", []),
        "phases": [
            {
                "name": p.get("name", f"phase_{idx}"),
                "kind": p.get("kind", "unknown"),
                "cycles": int(p.get("cycles", 1000)),
                "edge_count": len(p.get("edges", [])) if isinstance(p.get("edges", []), list) else 0,
                "actual_tiles_covered": p.get("actual_tiles_covered", 1),
            }
            for idx, p in enumerate(phases_raw)
            if isinstance(p, dict)
        ],
        "trace_file": str(trace_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return phase_cycles


def run_trace_study(
    booksim_bin: Path,
    runner_path: Path,
    trace_file: Path,
    phase_cycles: List[int],
    rates: List[float],
    topologies: List[str],
    seeds: List[int],
    packet_sizes: List[int],
    vc_buf_sizes: List[int] | None,
    routing_functions: List[str] | None,
    outdir: Path,
) -> None:
    cmd = [
        "python3",
        str(runner_path),
        "--booksim-bin",
        str(booksim_bin),
        "--trace-file",
        str(trace_file),
        "--phase-cycles",
        ",".join(str(x) for x in phase_cycles),
        "--rates",
        ",".join(f"{r:.4f}" for r in rates),
        "--topologies",
        ",".join(topologies),
        "--packet-sizes",
        ",".join(str(x) for x in packet_sizes),
        "--injection-process",
        "trace",
        "--trace-injection-use-source-weights",
        "1",
        "--seeds",
        ",".join(str(s) for s in seeds),
        "--fallback-uniform",
        "0",
        "--outdir",
        str(outdir),
    ]
    if vc_buf_sizes:
        cmd.extend(["--vc-buf-sizes", ",".join(str(x) for x in vc_buf_sizes)])
    if routing_functions:
        cmd.extend(["--routing-functions", ",".join(routing_functions)])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    (outdir / "runner_stdout.log").write_text(proc.stdout + "\n\n[stderr]\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            "run_trace_noc_study.py failed for model run directory "
            f"{outdir}. See {(outdir / 'runner_stdout.log').resolve()}"
        )



def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_seed_rows(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    groups: Dict[Tuple[str, float, float, str, str], List[Dict[str, float]]] = defaultdict(list)
    for r in rows:
        groups[
            (
                str(r["topology"]),
                float(r["injection_rate"]),
                float(r.get("packet_size", 4.0)),
                str(r.get("vc_buf_size", "default")),
                str(r.get("routing_function", "default")),
            )
        ].append(r)

    out: List[Dict[str, float]] = []
    for (topo, inj, packet_size, vc_buf_size, routing_function), g in sorted(
        groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][4])
    ):
        lat = [float(x["packet_latency_avg"]) for x in g]
        thr = [float(x["accepted_flit_rate_avg"]) for x in g]
        hops = [float(x["hops_avg"]) for x in g]
        out.append(
            {
                "topology": topo,
                "injection_rate": inj,
                "packet_size": packet_size,
                "vc_buf_size": vc_buf_size,
                "routing_function": routing_function,
                "packet_latency_avg": sum(lat) / len(lat),
                "accepted_flit_rate_avg": sum(thr) / len(thr),
                "hops_avg": sum(hops) / len(hops),
            }
        )
    return out


def to_float(x: str, default: float = math.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def build_rate_sweep(start: float, stop: float, step: float) -> List[float]:
    if step <= 0.0:
        raise ValueError("rate-step must be positive")
    if stop < start:
        raise ValueError("rate-stop must be >= rate-start")

    values: List[float] = []
    cur = start
    eps = step * 1e-6
    while cur <= stop + eps:
        values.append(round(cur, 8))
        cur += step
    if not values:
        raise ValueError("Generated rate sweep is empty")
    return values


def build_seed_list(seed_start: int, num_repeats: int) -> List[int]:
    if num_repeats <= 0:
        raise ValueError("num-repeats must be positive")
    return [seed_start + i for i in range(num_repeats)]


def saturation_estimate(rows: List[Dict[str, float]]) -> float:
    ordered = sorted(rows, key=lambda r: r["injection_rate"])
    if not ordered:
        return math.nan
    baseline = ordered[0]["packet_latency_avg"]
    for i in range(1, len(ordered)):
        prev = ordered[i - 1]
        cur = ordered[i]
        lat_jump = cur["packet_latency_avg"] / max(prev["packet_latency_avg"], 1e-9)
        thr_gain = cur["accepted_flit_rate_avg"] - prev["accepted_flit_rate_avg"]
        if cur["packet_latency_avg"] > 2.0 * baseline and (lat_jump > 1.2 or thr_gain < 0.002):
            return cur["injection_rate"]
    return ordered[-1]["injection_rate"]


def plot_model(model: str, rows: List[Dict[str, float]], outdir: Path) -> None:
    metrics = [
        ("packet_latency_avg", "Packet Latency"),
        ("accepted_flit_rate_avg", "Accepted Flit Rate"),
        ("hops_avg", "Average Hops"),
    ]

    for key, ylabel in metrics:
        unique_rates = sorted(set(float(r["injection_rate"]) for r in rows))
        plt.figure(figsize=(12.5, 5.8))
        configs = sorted(
            set(
                (
                    str(r["topology"]),
                    float(r.get("packet_size", 4.0)),
                    str(r.get("vc_buf_size", "default")),
                    str(r.get("routing_function", "default")),
                )
                for r in rows
            )
        )
        if len(unique_rates) == 1:
            labels: List[str] = []
            vals: List[float] = []
            for topo, packet_size, vc_buf_size, routing_function in configs:
                pts = [
                    r
                    for r in rows
                    if r["topology"] == topo
                    and float(r.get("packet_size", 4.0)) == packet_size
                    and str(r.get("vc_buf_size", "default")) == vc_buf_size
                    and str(r.get("routing_function", "default")) == routing_function
                ]
                if not pts:
                    continue
                labels.append(f"{topo}\nps={int(packet_size)}")
                vals.append(float(pts[0][key]))
            plt.bar(labels, vals)
            plt.xlabel(f"Topology/Config @ injection_rate={unique_rates[0]:.4f}")
        else:
            for topo, packet_size, vc_buf_size, routing_function in configs:
                pts = sorted(
                    [
                        r
                        for r in rows
                        if r["topology"] == topo
                        and float(r.get("packet_size", 4.0)) == packet_size
                        and str(r.get("vc_buf_size", "default")) == vc_buf_size
                        and str(r.get("routing_function", "default")) == routing_function
                    ],
                    key=lambda x: x["injection_rate"],
                )
                xs = [p["injection_rate"] for p in pts]
                ys = [p[key] for p in pts]
                label = f"{topo} ps={int(packet_size)} vc={vc_buf_size} rf={routing_function}"
                plt.plot(xs, ys, marker="o", linewidth=1.5, label=label)
            plt.xlabel("Injection Rate")

        plt.title(f"{model.upper()} - {ylabel} vs Topology (fixed operating point)")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        # Keep the plotting area clear when sweeping many packet/VC configurations.
        if len(unique_rates) == 1:
            plt.tight_layout()
        else:
            plt.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=9,
                frameon=True,
                borderaxespad=0.8,
            )
            plt.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))
        plt.savefig(outdir / f"{model}_{key}.png", dpi=180, bbox_inches="tight")
        plt.close()


def plot_cross_model_best_latency(summary_rows: List[Dict[str, str]], topologies: List[str], outdir: Path) -> None:
    models = sorted(set(r["model"] for r in summary_rows))
    x = list(range(len(models)))

    plt.figure(figsize=(8.5, 5.0))
    for topo in topologies:
        vals = []
        for m in models:
            cands = [r for r in summary_rows if r["model"] == m and r["topology"] == topo]
            v = min(to_float(r["best_latency"]) for r in cands) if cands else math.nan
            vals.append(v)
        plt.plot(x, vals, marker="o", linewidth=2.0, label=topo)

    plt.xticks(x, [m.upper() for m in models])
    plt.ylabel("Best Observed Packet Latency")
    plt.title("Cross-Model Best Latency by Topology")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "cross_model_best_latency.png", dpi=180)
    plt.close()


def plot_cross_model_metric_grouped_bar(
    combined_rows_averaged: List[Dict[str, str]],
    topologies: List[str],
    outdir: Path,
    metric_key: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    known_order = ["lenet", "vgg", "resnet"]
    models_found = set(r["model"] for r in combined_rows_averaged)
    models = [m for m in known_order if m in models_found] + sorted(models_found - set(known_order))
    x = list(range(len(models)))
    group_width = 0.78
    topo_count = max(1, len(topologies))
    bar_width = group_width / float(topo_count)

    shade_palette = ["#8ecae6", "#ffb703", "#90be6d", "#adb5bd"]
    hatch_palette = ["///", "\\\\", "...", "xx"]
    color_for_topo = {
        topo: shade_palette[idx % len(shade_palette)] for idx, topo in enumerate(topologies)
    }
    hatch_for_topo = {
        topo: hatch_palette[idx % len(hatch_palette)] for idx, topo in enumerate(topologies)
    }

    plt.figure(figsize=(8.8, 5.2))
    for topo_idx, topo in enumerate(topologies):
        vals: List[float] = []
        for model in models:
            cands = [
                r
                for r in combined_rows_averaged
                if r["model"] == model and r["topology"] == topo
            ]
            if not cands:
                vals.append(math.nan)
                continue
            metric_vals = [to_float(r[metric_key]) for r in cands]
            vals.append(sum(metric_vals) / float(len(metric_vals)))

        bar_vals = [0.0 if math.isnan(v) else v for v in vals]
        offset = -group_width / 2.0 + (topo_idx + 0.5) * bar_width
        xpos = [xi + offset for xi in x]
        plt.bar(
            xpos,
            bar_vals,
            width=bar_width,
            label=topo,
            color=color_for_topo[topo],
            edgecolor="black",
            linewidth=0.9,
            hatch=hatch_for_topo[topo],
        )

    plt.xticks(x, [m.upper() for m in models])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.legend(title="Topology")
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=180)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Run model-zoo (LeNet/VGG/ResNet) BookSim study")
    ap.add_argument("--booksim-bin", default="../booksim2/src/booksim")
    ap.add_argument("--topologies", default="mesh,torus,flatfly")
    ap.add_argument(
        "--max-sim-tiles-per-layer",
        type=int,
        default=32,
        help="Cap simulated tile phases per layer while preserving temporal multiplexing scale.",
    )
    ap.add_argument(
        "--num-repeats",
        type=int,
        default=5,
        help="Number of stochastic repetitions per operating point (different seeds).",
    )
    ap.add_argument(
        "--seed-start",
        type=int,
        default=FIXED_SEED,
        help="Base seed used to generate consecutive seeds for repeated runs.",
    )
    ap.add_argument("--outdir", default="results/model_zoo_study")
    args = ap.parse_args()

    booksim_bin = Path(args.booksim_bin).resolve()
    runner = (Path(__file__).parent / "run_trace_noc_study.py").resolve()
    outdir = Path(args.outdir).resolve()
    trace_dir = outdir / "traces"
    plot_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not booksim_bin.exists():
        raise FileNotFoundError(
            f"BookSim binary not found: {booksim_bin}. "
            "If you are in dnn-noc/, try --booksim-bin ../booksim2/src/booksim"
        )
    if not runner.exists():
        raise FileNotFoundError(f"Trace runner not found: {runner}")

    topologies = [x.strip() for x in args.topologies.split(",") if x.strip()]
    rates = [FIXED_INJECTION_RATE]
    seeds = build_seed_list(args.seed_start, args.num_repeats)
    packet_sizes = [FIXED_PACKET_SIZE_FLITS]
    mapping_algorithm = "pytorch-strict"

    combined_rows: List[Dict[str, str]] = []
    combined_rows_averaged: List[Dict[str, str]] = []
    summary_rows: List[Dict[str, str]] = []

    from pytorch_strict_mapper import generate_strict_pytorch_trace

    strict_specs: Dict[str, Dict[str, object]] = {}
    models = ["lenet", "vgg", "resnet"]
    for model in models:
        strict_specs[model] = generate_strict_pytorch_trace(
            model_name=model,
            nodes=NODES,
            max_sim_tiles_per_layer=args.max_sim_tiles_per_layer,
            memory_nodes=FIXED_HW_8X8_MESH.memory_nodes,
        )

    for model in models:
        model_dir = outdir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        trace_file = trace_dir / f"{model}_trace.txt"
        manifest = trace_dir / f"{model}_trace_manifest.json"
        phase_cycles = write_trace_for_explicit_phases(model, strict_specs[model], trace_file, manifest)

        run_trace_study(
            booksim_bin=booksim_bin,
            runner_path=runner,
            trace_file=trace_file,
            phase_cycles=phase_cycles,
            rates=rates,
            topologies=topologies,
            seeds=seeds,
            packet_sizes=packet_sizes,
            vc_buf_sizes=None,
            routing_functions=None,
            outdir=model_dir,
        )

        runs_csv = model_dir / "runs.csv"
        if not runs_csv.exists():
            raise FileNotFoundError(
                f"Expected results not found: {runs_csv}. "
                f"Check {(model_dir / 'runner_stdout.log').resolve()}"
            )
        rows_raw = read_csv_rows(runs_csv)
        ok_rows: List[Dict[str, float]] = []

        for rr in rows_raw:
            if rr.get("metric_parse_ok", "False") != "True":
                continue
            row = {
                "model": model,
                "topology": rr["topology"],
                "injection_rate": to_float(rr["injection_rate"]),
                "seed": to_float(rr.get("seed", "nan")),
                "packet_size": to_float(rr.get("packet_size", "4")),
                "vc_buf_size": rr.get("vc_buf_size", "default"),
                "routing_function": rr.get("routing_function", "default"),
                "packet_latency_avg": to_float(rr["packet_latency_avg"]),
                "accepted_flit_rate_avg": to_float(rr["accepted_flit_rate_avg"]),
                "hops_avg": to_float(rr["hops_avg"]),
            }
            ok_rows.append(row)
            combined_rows.append({k: str(v) for k, v in row.items()})

        agg_rows = aggregate_seed_rows(ok_rows)

        for r in agg_rows:
            combined_rows_averaged.append(
                {
                    "model": model,
                    "topology": str(r["topology"]),
                    "injection_rate": f"{float(r['injection_rate']):.6f}",
                    "packet_size": str(int(float(r.get("packet_size", 4.0)))),
                    "vc_buf_size": str(r.get("vc_buf_size", "default")),
                    "routing_function": str(r.get("routing_function", "default")),
                    "packet_latency_avg": f"{float(r['packet_latency_avg']):.6f}",
                    "accepted_flit_rate_avg": f"{float(r['accepted_flit_rate_avg']):.6f}",
                    "hops_avg": f"{float(r['hops_avg']):.6f}",
                }
            )

        plot_model(model, agg_rows, plot_dir)

        for topo in topologies:
            trows = [r for r in agg_rows if r["topology"] == topo]
            if not trows:
                continue
            best_lat = min(r["packet_latency_avg"] for r in trows)
            best_thr = max(r["accepted_flit_rate_avg"] for r in trows)
            sat = saturation_estimate(trows)
            summary_rows.append(
                {
                    "model": model,
                    "topology": topo,
                    "best_latency": f"{best_lat:.6f}",
                    "best_accepted_flit_rate": f"{best_thr:.6f}",
                    "saturation_rate_est": f"{sat:.6f}",
                }
            )

    with (outdir / "combined_runs.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "model",
            "topology",
            "injection_rate",
            "seed",
            "packet_size",
            "vc_buf_size",
            "routing_function",
            "packet_latency_avg",
            "accepted_flit_rate_avg",
            "hops_avg",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(combined_rows)

    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "model",
            "topology",
            "best_latency",
            "best_accepted_flit_rate",
            "saturation_rate_est",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    with (outdir / "combined_runs_averaged.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "model",
            "topology",
            "injection_rate",
            "packet_size",
            "vc_buf_size",
            "routing_function",
            "packet_latency_avg",
            "accepted_flit_rate_avg",
            "hops_avg",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(combined_rows_averaged)

    plot_cross_model_best_latency(summary_rows, topologies, plot_dir)
    plot_cross_model_metric_grouped_bar(
        combined_rows_averaged=combined_rows_averaged,
        topologies=topologies,
        outdir=plot_dir,
        metric_key="packet_latency_avg",
        ylabel="Packet Latency",
        title="Cross-Model Average Latency by Topology (Grouped Bars)",
        filename="cross_model_packet_latency.png",
    )
    plot_cross_model_metric_grouped_bar(
        combined_rows_averaged=combined_rows_averaged,
        topologies=topologies,
        outdir=plot_dir,
        metric_key="accepted_flit_rate_avg",
        ylabel="Accepted Flit Rate",
        title="Cross-Model Average Accepted Flit Rate by Topology (Grouped Bars)",
        filename="cross_model_accepted_flit_rate.png",
    )
    plot_cross_model_metric_grouped_bar(
        combined_rows_averaged=combined_rows_averaged,
        topologies=topologies,
        outdir=plot_dir,
        metric_key="hops_avg",
        ylabel="Average Hops",
        title="Cross-Model Average Hops by Topology (Grouped Bars)",
        filename="cross_model_hops_avg.png",
    )

    # Insights report
    lines = [
        "# Model Zoo NoC Study",
        "",
        "Models: LeNet, VGG, ResNet",
        f"Topology sweep: {', '.join(topologies)}",
        f"Fixed node count: {FIXED_HW_8X8_MESH.num_nodes}",
        f"Fixed injection rate: {FIXED_INJECTION_RATE:.4f}",
        f"Fixed packet size: {FIXED_PACKET_SIZE_FLITS}",
        f"Seed start: {args.seed_start}",
        f"Repetitions per point: {args.num_repeats}",
        f"Injection rates: {', '.join(f'{r:.4f}' for r in rates)}",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        f"Packet sizes: {', '.join(str(x) for x in packet_sizes)}",
        f"Fixed mapping method: {mapping_algorithm}",
        f"Max simulated tiles per layer: {args.max_sim_tiles_per_layer}",
        "",
        "## Key Insights",
    ]

    for model in ["lenet", "resnet", "vgg"]:
        mrows = [r for r in summary_rows if r["model"] == model]
        if not mrows:
            continue
        best_lat_row = min(mrows, key=lambda r: float(r["best_latency"]))
        best_thr_row = max(mrows, key=lambda r: float(r["best_accepted_flit_rate"]))
        lines.append(
            f"- {model.upper()}: lowest latency on {best_lat_row['topology']} "
            f"({float(best_lat_row['best_latency']):.3f}), highest accepted flit rate on "
            f"{best_thr_row['topology']} ({float(best_thr_row['best_accepted_flit_rate']):.5f})."
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "- combined_runs.csv: all successful runs across models and topologies.",
            "- summary.csv: per-model per-topology summary metrics at fixed operating point.",
            "- plots/: latency, throughput, hops curves per model + cross-model comparison plot.",
            "",
            "## Notes",
            "- This study uses trace-driven destination and source injection processes.",
            "- Curves and summaries are computed from seed-averaged metrics at each operating point.",
            "- Residual behavior is represented explicitly via residual-merge traffic phases in the ResNet trace.",
            "- Topology is the only sweep axis in this study.",
            "- Injection rate, packet size, seed, and mapping method are fixed to realistic deployment assumptions.",
        ]
    )

    (outdir / "insights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "models": ["lenet", "vgg", "resnet"],
        "topologies": topologies,
        "trace_source": mapping_algorithm,
        "rates": rates,
        "seeds": seeds,
        "seed_start": args.seed_start,
        "num_repeats": args.num_repeats,
        "packet_sizes": packet_sizes,
        "vc_buf_sizes": ["topology_default"],
        "routing_functions": ["topology_default"],
        "max_sim_tiles_per_layer": args.max_sim_tiles_per_layer,
        "hardware_profile": {
            "name": FIXED_HW_8X8_MESH.name,
            "topology": FIXED_HW_8X8_MESH.topology,
            "k": FIXED_HW_8X8_MESH.k,
            "n": FIXED_HW_8X8_MESH.n,
            "xr": FIXED_HW_8X8_MESH.xr,
            "num_nodes": FIXED_HW_8X8_MESH.num_nodes,
            "routing_function": FIXED_HW_8X8_MESH.routing_function,
            "num_vcs": FIXED_HW_8X8_MESH.num_vcs,
            "vc_buf_size": FIXED_HW_8X8_MESH.vc_buf_size,
            "router_pipeline_stages": FIXED_HW_8X8_MESH.router_pipeline_stages,
            "bits_per_flit": FIXED_HW_8X8_MESH.bits_per_flit,
            "memory_nodes": FIXED_HW_8X8_MESH.memory_nodes,
        },
        "combined_runs_csv": str((outdir / "combined_runs.csv").resolve()),
        "combined_runs_averaged_csv": str((outdir / "combined_runs_averaged.csv").resolve()),
        "summary_csv": str((outdir / "summary.csv").resolve()),
        "insights_md": str((outdir / "insights.md").resolve()),
        "plots_dir": str(plot_dir.resolve()),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Completed model-zoo study at: {outdir.resolve()}")
    print(f"- {outdir / 'combined_runs.csv'}")
    print(f"- {outdir / 'combined_runs_averaged.csv'}")
    print(f"- {outdir / 'summary.csv'}")
    print(f"- {outdir / 'insights.md'}")
    print(f"- {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
