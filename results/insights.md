# Model Zoo NoC Study

Models: LeNet, VGG, ResNet
Topology sweep: mesh, torus, flatfly
Fixed node count: 64
Fixed injection rate: 0.0060
Fixed packet size: 4
Seed start: 1
Repetitions per point: 5
Injection rates: 0.0060
Seeds: 1, 2, 3, 4, 5
Packet sizes: 4
Fixed mapping method: pytorch-strict
Max simulated tiles per layer: 32

## Key Insights
- LENET: lowest latency on flatfly (29.034), highest accepted flit rate on torus (0.02423).
- RESNET: lowest latency on flatfly (16.938), highest accepted flit rate on torus (0.02415).
- VGG: lowest latency on flatfly (18.651), highest accepted flit rate on flatfly (0.02425).

## Output Files
- combined_runs.csv: all successful runs across models and topologies.
- summary.csv: per-model per-topology summary metrics at fixed operating point.
- plots/: latency, throughput, hops curves per model + cross-model comparison plot.

## Notes
- This study uses trace-driven destination and source injection processes.
- Curves and summaries are computed from seed-averaged metrics at each operating point.
- Residual behavior is represented explicitly via residual-merge traffic phases in the ResNet trace.
- Topology is the only sweep axis in this study.
- Injection rate, packet size, seed, and mapping method are fixed to realistic deployment assumptions.
