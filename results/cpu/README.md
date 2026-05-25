# Recorded results

`cpu/` — real measurements on a 24-core x86_64 host (the A100 box's CPU), f64,
via `python -m jaxbench.runner <task>`. Each `result_<task>.json` has the correctness
residuals and per-size latency/GFLOP/s sweep. GPU (A100) and TPU (GCP) baselines are
produced the same way with `--device gpu` / `--device tpu`.
