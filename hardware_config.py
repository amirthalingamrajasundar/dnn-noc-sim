#!/usr/bin/env python3
"""Fixed hardware profile for realistic DNN-NoC studies.

This module centralizes immutable hardware assumptions so experiments compare
mapping strategies and DNN scaling on the same silicon target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    topology: str
    k: int
    n: int
    xr: int
    num_nodes: int
    routing_function: str
    num_vcs: int
    vc_buf_size: int
    bits_per_flit: int
    router_pipeline_stages: int
    memory_nodes: List[int]


FIXED_HW_8X8_MESH = HardwareProfile(
    name="fixed_8x8_mesh_v1",
    topology="mesh",
    k=8,
    n=2,
    xr=1,
    num_nodes=64,
    routing_function="dim_order",
    num_vcs=4,
    vc_buf_size=8,
    bits_per_flit=64,
    router_pipeline_stages=4,
    memory_nodes=[0, 7, 56, 63],
)

# Fixed operating point used for topology-only sweeps.
# The injection rate is chosen as a nominal pre-saturation point for 64-node
# studies so topology differences are visible without global congestion collapse.
FIXED_INJECTION_RATE = 0.006
FIXED_PACKET_SIZE_FLITS = 4
FIXED_SEED = 1

