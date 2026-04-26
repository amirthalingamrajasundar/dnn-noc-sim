#!/usr/bin/env python3
"""Run BookSim NoC studies using trace-driven traffic.

This is the recommended runner for realistic, extendable DNN studies.
Provide a trace file generated from your DNN mapper/extractor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from hardware_config import (
    FIXED_HW_8X8_MESH,
    FIXED_INJECTION_RATE,
    FIXED_PACKET_SIZE_FLITS,
    FIXED_SEED,
)

OVERALL_SECTION = "====== Overall Traffic Statistics ======"


def topology_cfg_lines(
    name: str,
    routing_function_override: Optional[str] = None,
    vc_buf_size_override: Optional[int] = None,
) -> List[str]:
    common = [
        "subnets = 1;",
        "injection_process = bernoulli;",
        "sim_type = latency;",
        "warmup_periods = 2;",
        "sample_period = 2000;",
        "max_samples = 6;",
        "measure_stats = 1;",
    ]

    if name == FIXED_HW_8X8_MESH.topology:
        routing_function = (
            routing_function_override
            if routing_function_override
            else FIXED_HW_8X8_MESH.routing_function
        )
        vc_buf_size = (
            vc_buf_size_override
            if vc_buf_size_override is not None
            else FIXED_HW_8X8_MESH.vc_buf_size
        )
        return [
            f"topology = {FIXED_HW_8X8_MESH.topology};",
            f"k = {FIXED_HW_8X8_MESH.k};",
            f"n = {FIXED_HW_8X8_MESH.n};",
            f"xr = {FIXED_HW_8X8_MESH.xr};",
            f"routing_function = {routing_function};",
            f"num_vcs = {FIXED_HW_8X8_MESH.num_vcs};",
            f"vc_buf_size = {vc_buf_size};",
        ] + common

    if name == "torus":
        routing_function = routing_function_override if routing_function_override else "dim_order"
        vc_buf_size = vc_buf_size_override if vc_buf_size_override is not None else FIXED_HW_8X8_MESH.vc_buf_size
        return [
            "topology = torus;",
            f"k = {FIXED_HW_8X8_MESH.k};",
            f"n = {FIXED_HW_8X8_MESH.n};",
            f"xr = {FIXED_HW_8X8_MESH.xr};",
            f"routing_function = {routing_function};",
            f"num_vcs = {FIXED_HW_8X8_MESH.num_vcs};",
            f"vc_buf_size = {vc_buf_size};",
        ] + common

    if name == "flatfly":
        routing_function = routing_function_override if routing_function_override else "ran_min"
        vc_buf_size = vc_buf_size_override if vc_buf_size_override is not None else 4
        return [
            "topology = flatfly;",
            "k = 4;",
            "n = 2;",
            "c = 4;",
            "x = 4;",
            "y = 4;",
            "xr = 2;",
            "yr = 2;",
            f"routing_function = {routing_function};",
            "num_vcs = 8;",
            f"vc_buf_size = {vc_buf_size};",
        ] + common

    raise ValueError(f"Unsupported topology: {name}")


def parse_rates(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No injection rates provided")
    return vals


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


def parse_phase_cycles(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No phase cycles provided")
    if any(v <= 0 for v in vals):
        raise ValueError("All phase cycles must be positive")
    return vals


def parse_seeds(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No seeds provided")
    return vals


def parse_int_list(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("List must contain at least one integer value")
    return vals


def parse_str_list(raw: str) -> List[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("List must contain at least one value")
    return vals


def parse_metrics(output: str) -> Dict[str, Optional[float]]:
    idx = output.rfind(OVERALL_SECTION)
    text = output[idx:] if idx >= 0 else output

    def pick(pattern: str) -> Optional[float]:
        matches = re.findall(pattern, text)
        if not matches:
            return None
        return float(matches[-1])

    return {
        "packet_latency_avg": pick(r"Packet latency average\s*=\s*([0-9.]+)"),
        "accepted_flit_rate_avg": pick(r"Accepted flit rate average\s*=\s*([0-9.]+)"),
        "hops_avg": pick(r"Hops average\s*=\s*([0-9.]+)"),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def _safe_std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
    mu = _safe_mean(values)
    if mu is None:
        return None
    var = sum((x - mu) ** 2 for x in values) / float(len(values) - 1)
    return math.sqrt(var)


def aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        key = (
            r["topology"],
            r["injection_rate"],
            r["packet_size"],
            r["vc_buf_size"],
            r["routing_function"],
            r["trace_file"],
            r["phase_cycles"],
        )
        groups[key].append(r)

    out: List[Dict[str, object]] = []
    for key in sorted(groups.keys(), key=lambda k: (str(k[0]), float(k[1]))):
        g = groups[key]
        lat = [float(x["packet_latency_avg"]) for x in g if x.get("packet_latency_avg") is not None]
        thr = [float(x["accepted_flit_rate_avg"]) for x in g if x.get("accepted_flit_rate_avg") is not None]
        hops = [float(x["hops_avg"]) for x in g if x.get("hops_avg") is not None]
        ok_count = sum(1 for x in g if x.get("metric_parse_ok"))
        exit_zero_count = sum(1 for x in g if int(x.get("exit_code", -1)) == 0)

        out.append(
            {
                "topology": key[0],
                "injection_rate": key[1],
                "packet_size": key[2],
                "vc_buf_size": key[3],
                "routing_function": key[4],
                "trace_file": key[5],
                "phase_cycles": key[6],
                "num_runs": len(g),
                "num_metric_ok": ok_count,
                "num_exit_code_zero": exit_zero_count,
                "packet_latency_avg_mean": _safe_mean(lat),
                "packet_latency_avg_std": _safe_std(lat),
                "accepted_flit_rate_avg_mean": _safe_mean(thr),
                "accepted_flit_rate_avg_std": _safe_std(thr),
                "hops_avg_mean": _safe_mean(hops),
                "hops_avg_std": _safe_std(hops),
            }
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Trace-driven multi-topology BookSim study")
    p.add_argument("--booksim-bin", default="../booksim2/src/booksim")
    p.add_argument("--trace-file", required=True)
    p.add_argument("--phase-cycles", required=True, help="Comma-separated phase cycles")
    p.add_argument(
        "--rates",
        default="",
        help="Optional comma-separated rates. Topology-only mode enforces one fixed realistic value.",
    )
    p.add_argument("--rate-start", type=float, default=0.005)
    p.add_argument("--rate-stop", type=float, default=0.020)
    p.add_argument("--rate-step", type=float, default=0.005)
    p.add_argument("--topologies", default="mesh,torus,flatfly")
    p.add_argument("--packet-size", type=int, default=FIXED_PACKET_SIZE_FLITS)
    p.add_argument(
        "--packet-sizes",
        default="",
        help="Optional comma-separated packet sizes sweep. Topology-only mode enforces one fixed value.",
    )
    p.add_argument(
        "--vc-buf-sizes",
        default="",
        help="Optional comma-separated VC buffer sizes sweep. Topology-only mode disables this sweep.",
    )
    p.add_argument(
        "--routing-functions",
        default="",
        help="Optional comma-separated routing functions sweep. Topology-only mode disables this sweep.",
    )
    p.add_argument(
        "--topology-only-sweep",
        type=int,
        default=1,
        choices=[0, 1],
        help="When 1, topology is the only sweep axis; all other parameters are fixed.",
    )
    p.add_argument("--loop", type=int, default=1, choices=[0, 1])
    p.add_argument("--fallback-uniform", type=int, default=0, choices=[0, 1])
    p.add_argument("--injection-process", default="trace", choices=["trace", "bernoulli", "on_off"])
    p.add_argument("--trace-injection-use-source-weights", type=int, default=1, choices=[0, 1])
    p.add_argument("--seeds", default=str(FIXED_SEED), help="Comma-separated random seeds")
    p.add_argument("--outdir", default="results/trace_study")
    args = p.parse_args()

    booksim = Path(args.booksim_bin).resolve()
    trace_file = Path(args.trace_file).resolve()
    if not booksim.exists():
        raise FileNotFoundError(f"BookSim binary not found: {booksim}")
    if not trace_file.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_file}")

    phase_cycles = parse_phase_cycles(args.phase_cycles)
    rates = parse_rates(args.rates) if args.rates.strip() else build_rate_sweep(args.rate_start, args.rate_stop, args.rate_step)
    seeds = parse_seeds(args.seeds)
    topologies = [x.strip() for x in args.topologies.split(",") if x.strip()]
    packet_sizes = parse_int_list(args.packet_sizes) if args.packet_sizes.strip() else [args.packet_size]
    vc_buf_sizes: List[Optional[int]] = (
        parse_int_list(args.vc_buf_sizes) if args.vc_buf_sizes.strip() else [None]
    )
    routing_functions: List[Optional[str]] = [None]
    if args.routing_functions.strip():
        routing_functions = [
            None if x.lower() == "default" else x for x in parse_str_list(args.routing_functions)
        ]

    if args.topology_only_sweep == 1:
        rates = [FIXED_INJECTION_RATE]
        packet_sizes = [FIXED_PACKET_SIZE_FLITS]
        args.injection_process = "trace"

        if args.vc_buf_sizes.strip() or args.routing_functions.strip():
            raise ValueError(
                "Topology-only mode disables sweeps for vc_buf_size/routing_function."
            )

        if args.packet_sizes.strip():
            explicit_packet_sizes = parse_int_list(args.packet_sizes)
            if any(ps != FIXED_PACKET_SIZE_FLITS for ps in explicit_packet_sizes):
                raise ValueError(
                    "Topology-only mode requires packet_size="
                    f"{FIXED_PACKET_SIZE_FLITS}."
                )

        vc_buf_sizes = [None]
        routing_functions = [None]

    outdir = Path(args.outdir)
    cfg_dir = outdir / "configs"
    log_dir = outdir / "logs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    phase_cycles_cfg = "{" + ",".join(str(x) for x in phase_cycles) + "}"

    for topo in topologies:
        for routing_function in routing_functions:
            for vc_buf_size in vc_buf_sizes:
                base = topology_cfg_lines(
                    topo,
                    routing_function_override=routing_function,
                    vc_buf_size_override=vc_buf_size,
                )
                for packet_size in packet_sizes:
                    for rate in rates:
                        for seed in seeds:
                            rf_tag = routing_function if routing_function else "default"
                            vb_tag = str(vc_buf_size) if vc_buf_size is not None else "default"
                            tag = (
                                f"{topo}_rf{rf_tag}_vb{vb_tag}_ps{packet_size}_r{rate:.3f}_s{seed}"
                                .replace(".", "p")
                                .replace("/", "_")
                            )
                            cfg_path = cfg_dir / f"{tag}.cfg"
                            log_path = log_dir / f"{tag}.log"

                            cfg = []
                            cfg.extend(base)
                            cfg.append("traffic = trace;")
                            cfg.append(f"trace_traffic_file = {trace_file};")
                            cfg.append(f"trace_phase_cycles = {phase_cycles_cfg};")
                            cfg.append(f"trace_traffic_loop = {args.loop};")
                            cfg.append(f"trace_fallback_uniform = {args.fallback_uniform};")
                            cfg.append(f"trace_injection_file = {trace_file};")
                            cfg.append(f"trace_injection_phase_cycles = {phase_cycles_cfg};")
                            cfg.append(f"trace_injection_loop = {args.loop};")
                            cfg.append(
                                f"trace_injection_use_source_weights = {args.trace_injection_use_source_weights};"
                            )
                            cfg.append(f"packet_size = {packet_size};")
                            cfg.append(f"injection_rate = {rate:.4f};")
                            cfg.append(f"injection_process = {args.injection_process};")
                            cfg.append(f"seed = {seed};")
                            cfg.append("")
                            cfg_path.write_text("\n".join(cfg), encoding="utf-8")

                            proc = subprocess.run(
                                [str(booksim), str(cfg_path)], capture_output=True, text=True, check=False
                            )
                            log_path.write_text(
                                proc.stdout + "\n\n[stderr]\n" + proc.stderr,
                                encoding="utf-8",
                            )

                            metrics = parse_metrics(proc.stdout)
                            ok = all(metrics[k] is not None for k in metrics)

                            rows.append(
                                {
                                    "topology": topo,
                                    "injection_rate": rate,
                                    "seed": seed,
                                    "packet_size": packet_size,
                                    "vc_buf_size": vc_buf_size if vc_buf_size is not None else "default",
                                    "routing_function": routing_function if routing_function else "default",
                                    "trace_file": str(trace_file),
                                    "phase_cycles": ",".join(str(x) for x in phase_cycles),
                                    "metric_parse_ok": ok,
                                    "exit_code": proc.returncode,
                                    **metrics,
                                    "cfg_path": str(cfg_path),
                                    "log_path": str(log_path),
                                }
                            )

    write_csv(outdir / "runs.csv", rows)
    agg_rows = aggregate_rows(rows)
    write_csv(outdir / "runs_aggregated.csv", agg_rows)

    summary = {
        "total_runs": len(rows),
        "successful_runs": sum(1 for r in rows if r["metric_parse_ok"]),
        "seeds": seeds,
        "packet_sizes": packet_sizes,
        "vc_buf_sizes": [v if v is not None else "default" for v in vc_buf_sizes],
        "routing_functions": [r if r else "default" for r in routing_functions],
        "trace_file": str(trace_file),
        "phase_cycles": phase_cycles,
        "rates": rates,
        "topologies": topologies,
        "topology_only_sweep": args.topology_only_sweep,
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
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Completed runs: {summary['total_runs']}")
    print(f"Successful metric parses: {summary['successful_runs']}")
    print(f"Output directory: {outdir.resolve()}")
    print(f"- {outdir / 'runs.csv'}")
    print(f"- {outdir / 'runs_aggregated.csv'}")
    print(f"- {outdir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
