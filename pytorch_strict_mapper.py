#!/usr/bin/env python3
"""Strict PyTorch-to-NoC trace mapper with explicit tiled PE scheduling.

This mapper captures real layer execution order and tensor shapes, then builds
trace phases using deterministic PE assignment. If workload exceeds available
PEs, it applies temporal multiplexing (multiple tile passes).

Improvements:
1. Spatial locality: Maps tasks to PEs using 2D geometric coordinates from feature maps
2. Multiple memory nodes: Distributes weight and activation fetches across memory controllers
3. Flit translation: Converts floating-point weights to discrete flits with injection rates
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class LayerInfo:
    name: str
    kind: str
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    weight_elems: int
    deps_per_output: int


def _shape_elems(shape: Tuple[int, ...]) -> int:
    n = 1
    for x in shape:
        n *= int(x)
    return n


def _tensor_shape(obj: Any) -> Tuple[int, ...]:
    if hasattr(obj, "shape"):
        return tuple(int(x) for x in obj.shape)
    if isinstance(obj, (list, tuple)) and obj:
        for item in obj:
            if hasattr(item, "shape"):
                return tuple(int(x) for x in item.shape)
    return tuple()


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _manhattan_distance(node_a: int, node_b: int, mesh_width: int) -> int:
    """Compute Manhattan distance between two nodes in a 2D mesh."""
    x_a, y_a = node_a % mesh_width, node_a // mesh_width
    x_b, y_b = node_b % mesh_width, node_b // mesh_width
    return abs(x_a - x_b) + abs(y_a - y_b)


def _add_edge(edges: Dict[Tuple[int, int], float], src: int, dst: int, w: float) -> None:
    if w <= 0.0:
        return
    edges[(src, dst)] = edges.get((src, dst), 0.0) + w


def _infer_mesh_dimensions(nodes: int) -> Tuple[int, int]:
    """Infer 2D mesh dimensions from node count. Tries to find square-like mesh."""
    size = int(math.sqrt(nodes))
    while size > 0:
        if nodes % size == 0:
            other = nodes // size
            return (size, other)
        size -= 1
    return (1, nodes)


def _get_default_memory_nodes(nodes: int) -> List[int]:
    """Generate default memory node positions (e.g., four corners for 8x8 mesh)."""
    mesh_w, mesh_h = _infer_mesh_dimensions(nodes)
    corners = [0, mesh_w - 1, (mesh_h - 1) * mesh_w, nodes - 1]
    return [c for c in corners if c < nodes]


def _weight_to_flits(
    weight_value: float,
    bits_per_item: int = 32,
    bytes_per_flit: int = 8,
) -> float:
    """Convert floating-point weight to number of flits.
    
    Args:
        weight_value: Floating-point weight (typically bytes or total data size).
        bits_per_item: Number of bits per work item (default 32 for FP32).
        bytes_per_flit: Flit size in bytes (default 8 for 64-bit flits).
    
    Returns:
        Number of flits as a float.
    """
    if weight_value <= 0.0:
        return 0.0
    bits_total = weight_value * bits_per_item
    bits_per_flit = bytes_per_flit * 8
    flits = bits_total / bits_per_flit
    return max(1.0, flits)


def _build_lenet5() -> Any:
    import torch.nn as nn

    class LeNet5(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(6, 16, kernel_size=5, stride=1),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 6 * 6, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10),
            )

        def forward(self, x: Any) -> Any:
            x = self.features(x)
            return self.classifier(x)

    return LeNet5()


def _build_vgg11() -> Any:
    from torchvision.models import vgg11

    return vgg11(weights=None, num_classes=10)


def _build_resnet18_cifar() -> Any:
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _model_and_input(model_name: str) -> Tuple[Any, Any]:
    import torch

    if model_name == "lenet":
        return _build_lenet5(), torch.randn(1, 1, 32, 32)
    if model_name == "vgg":
        return _build_vgg11(), torch.randn(1, 3, 32, 32)
    if model_name == "resnet":
        return _build_resnet18_cifar(), torch.randn(1, 3, 32, 32)
    raise ValueError(f"Unsupported model: {model_name}")


def _collect_layer_info(model_name: str) -> List[LayerInfo]:
    import torch
    import torch.nn as nn

    model, sample_input = _model_and_input(model_name)
    model.eval()

    infos: List[LayerInfo] = []
    hooks: List[Any] = []

    try:
        from torchvision.models.resnet import BasicBlock, Bottleneck

        residual_types = (BasicBlock, Bottleneck)
    except Exception:
        residual_types = tuple()

    def conv_hook(name: str, module: Any) -> Any:
        def fn(_module: Any, inp: Any, out: Any) -> None:
            inp_shape = _tensor_shape(inp)
            out_shape = _tensor_shape(out)
            cin = int(getattr(module, "in_channels", 1))
            k = getattr(module, "kernel_size", (1, 1))
            kh = int(k[0]) if isinstance(k, tuple) else int(k)
            kw = int(k[1]) if isinstance(k, tuple) else int(k)
            deps = max(1, cin * kh * kw)
            w = getattr(module, "weight", None)
            infos.append(
                LayerInfo(
                    name=name,
                    kind="conv",
                    in_shape=inp_shape,
                    out_shape=out_shape,
                    weight_elems=int(w.numel()) if w is not None else 0,
                    deps_per_output=deps,
                )
            )

        return fn

    def pool_hook(name: str, module: Any) -> Any:
        def fn(_module: Any, inp: Any, out: Any) -> None:
            inp_shape = _tensor_shape(inp)
            out_shape = _tensor_shape(out)
            k = getattr(module, "kernel_size", 2)
            if isinstance(k, tuple):
                karea = int(k[0]) * int(k[1])
            else:
                karea = int(k) * int(k)
            infos.append(
                LayerInfo(
                    name=name,
                    kind="pool",
                    in_shape=inp_shape,
                    out_shape=out_shape,
                    weight_elems=0,
                    deps_per_output=max(1, karea),
                )
            )

        return fn

    def fc_hook(name: str, module: Any) -> Any:
        def fn(_module: Any, inp: Any, out: Any) -> None:
            inp_shape = _tensor_shape(inp)
            out_shape = _tensor_shape(out)
            in_f = int(getattr(module, "in_features", 1))
            w = getattr(module, "weight", None)
            infos.append(
                LayerInfo(
                    name=name,
                    kind="fc",
                    in_shape=inp_shape,
                    out_shape=out_shape,
                    weight_elems=int(w.numel()) if w is not None else 0,
                    deps_per_output=max(1, in_f),
                )
            )

        return fn

    def merge_hook(name: str) -> Any:
        def fn(_module: Any, inp: Any, out: Any) -> None:
            infos.append(
                LayerInfo(
                    name=f"{name}_merge",
                    kind="res_merge",
                    in_shape=_tensor_shape(inp),
                    out_shape=_tensor_shape(out),
                    weight_elems=0,
                    deps_per_output=2,
                )
            )

        return fn

    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook(name, module)))
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            hooks.append(module.register_forward_hook(pool_hook(name, module)))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(fc_hook(name, module)))
        elif residual_types and isinstance(module, residual_types):
            hooks.append(module.register_forward_hook(merge_hook(name)))

    with torch.no_grad():
        _ = model(sample_input)

    for h in hooks:
        h.remove()

    if not infos:
        raise RuntimeError(f"No layers captured from model: {model_name}")

    return infos


def _work_items_for_layer(info: LayerInfo) -> int:
    # For conv/pool/residual: output tensor element count is the produced work set.
    # For FC: output tensor element count equals neuron outputs for current batch.
    return max(1, _shape_elems(info.out_shape))


def _ops_per_item(info: LayerInfo) -> int:
    if info.kind == "res_merge":
        return 2
    return max(1, info.deps_per_output)


def generate_strict_pytorch_trace(
    model_name: str,
    nodes: int,
    max_sim_tiles_per_layer: int = 32,
    memory_nodes: List[int] | None = None,
    bits_per_item: int = 32,
    bytes_per_flit: int = 8,
) -> Dict[str, Any]:
    """Build explicit tiled phases and edges from PyTorch layer execution.
    
    Args:
        model_name: Name of the model to trace (lenet, vgg, resnet).
        nodes: Total number of PE nodes in the NoC.
        max_sim_tiles_per_layer: Maximum simulated tiles per layer for temporal multiplexing.
        memory_nodes: List of memory controller node IDs. If None, uses default corners.
        bits_per_item: Bits per work item for flit conversion (default 32 for FP32).
        bytes_per_flit: Bytes per flit for NoC (default 8 for 64-bit flits).
    """
    if nodes <= 0:
        raise ValueError("nodes must be positive")

    infos = _collect_layer_info(model_name)
    
    # Setup memory nodes
    if memory_nodes is None:
        memory_nodes = _get_default_memory_nodes(nodes)
    for mem_id in memory_nodes:
        if mem_id < 0 or mem_id >= nodes:
            raise ValueError(f"memory_node {mem_id} must be in [0, {nodes})")
    
    mesh_w, mesh_h = _infer_mesh_dimensions(nodes)

    phases: List[Dict[str, Any]] = []
    per_layer: List[Dict[str, Any]] = []
    prev_work_items = 0
    prev_out_shape: Tuple[int, ...] = tuple()

    for layer_idx, info in enumerate(infos):
        work_items = _work_items_for_layer(info)
        ops_per_item = _ops_per_item(info)
        actual_tiles = int(math.ceil(work_items / float(nodes)))
        sim_tiles = min(max(1, actual_tiles), max_sim_tiles_per_layer)
        tiles_per_phase = int(math.ceil(actual_tiles / float(sim_tiles)))

        per_layer.append(
            {
                "name": info.name,
                "kind": info.kind,
                "work_items": work_items,
                "ops_per_item": ops_per_item,
                "actual_tiles": actual_tiles,
                "sim_tiles": sim_tiles,
                "tiles_per_phase": tiles_per_phase,
                "weight_elems": info.weight_elems,
            }
        )

        dep_samples = min(ops_per_item, 16)
        dep_scale = float(ops_per_item) / float(dep_samples)
        weight_per_task = float(info.weight_elems) / float(max(1, work_items))
        weight_flits = _weight_to_flits(weight_per_task, bits_per_item, bytes_per_flit)

        # Extract output spatial dimensions (for feature maps: assume [batch, channels, height, width])
        out_shape = info.out_shape
        if len(out_shape) >= 3:
            out_h, out_w = out_shape[-2], out_shape[-1]
        else:
            out_h, out_w = 1, 1

        for st in range(sim_tiles):
            start_tile = st * tiles_per_phase
            tasks_before = start_tile * nodes
            rem = max(0, work_items - tasks_before)
            active_pes = min(nodes, rem)
            if active_pes <= 0:
                continue

            edges: Dict[Tuple[int, int], float] = {}
            flit_counts: Dict[Tuple[int, int], float] = {}  # Track flit counts separately
            
            for local_idx in range(active_pes):
                task_id = tasks_before + local_idx
                
                # --- SPATIAL LOCALITY: Map task_id to 2D output coordinates ---
                out_idx = task_id % work_items if work_items > 0 else 0
                if out_h > 1 or out_w > 1:
                    out_y = (out_idx // out_w) % out_h
                    out_x = out_idx % out_w
                else:
                    out_y, out_x = 0, 0
                
                # Assign PE based on output feature map position
                dst = (out_x % mesh_w) + (out_y % mesh_h) * mesh_w
                dst = dst % nodes

                # --- DEPENDENCY MAPPING: Use input receptive field coordinates ---
                if prev_work_items > 0 and len(prev_out_shape) >= 3:
                    prev_h, prev_w = prev_out_shape[-2], prev_out_shape[-1]
                    # Sample input dependencies with spatial locality
                    for d in range(dep_samples):
                        # Map to input coordinates using kernel locality
                        kernel_offset = d % max(1, ops_per_item)
                        in_idx = (out_idx + kernel_offset) % prev_work_items
                        in_y = (in_idx // prev_w) % prev_h if prev_h > 0 else 0
                        in_x = in_idx % prev_w if prev_w > 0 else 0
                        
                        # Source PE based on input coordinate
                        src = (in_x % mesh_w) + (in_y % mesh_h) * mesh_w
                        src = src % nodes
                        _add_edge(edges, src, dst, dep_scale * tiles_per_phase)
                else:
                    # Initial layer: fetch from nearest memory node
                    mem_node = memory_nodes[local_idx % len(memory_nodes)]
                    _add_edge(edges, mem_node, dst, dep_scale * tiles_per_phase)

                # --- WEIGHT FETCHES: Distribute across multiple memory nodes ---
                if weight_per_task > 0.0:
                    # Round-robin or distance-based selection of memory node
                    mem_node = memory_nodes[local_idx % len(memory_nodes)]
                    _add_edge(edges, mem_node, dst, weight_flits * tiles_per_phase)
                    flit_counts[(mem_node, dst)] = flit_counts.get((mem_node, dst), 0.0) + weight_flits

                # Local computation (self-loop for pipelining/buffering)
                _add_edge(edges, dst, dst, 0.05 * tiles_per_phase)

            per_pe_ops = ops_per_item * tiles_per_phase
            cycles = _clamp_int(int(math.ceil(per_pe_ops / 128.0)), 80, 6000)
            
            # Calculate injection rate (flits per cycle) for weight traffic
            total_weight_flits = sum(fl for (_, _), fl in flit_counts.items())
            injection_rate = total_weight_flits / max(1, cycles) if cycles > 0 else 0.0

            phases.append(
                {
                    "name": f"{info.name}_tile{st}",
                    "kind": info.kind,
                    "cycles": cycles,
                    "edges": [
                        {
                            "src": int(src),
                            "dst": int(dst),
                            "weight": float(w),
                        }
                        for (src, dst), w in sorted(edges.items())
                    ],
                    "actual_tiles_covered": int(tiles_per_phase),
                    "total_weight_flits": float(total_weight_flits),
                    "injection_rate_flits_per_cycle": float(injection_rate),
                }
            )

        prev_work_items = work_items
        prev_out_shape = out_shape

    return {
        "model": model_name,
        "nodes": nodes,
        "mesh_dimensions": {"width": mesh_w, "height": mesh_h},
        "memory_nodes": memory_nodes,
        "mapping_mode": "strict_tiled_spatial",
        "max_sim_tiles_per_layer": max_sim_tiles_per_layer,
        "flit_parameters": {
            "bits_per_item": bits_per_item,
            "bytes_per_flit": bytes_per_flit,
        },
        "layers": per_layer,
        "phases": phases,
    }
