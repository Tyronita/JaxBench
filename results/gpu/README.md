# Recorded A100 results

Real measurements on an NVIDIA A100 80GB (sm_80) with jax[cuda12] 0.10.1, f32,
via `python -m jaxbench.runner <task>`. Each `result_<task>.json` has correctness
residuals (gated) and the per-size latency/GFLOP/s sweep. Reproduce on any A100:
`pip install -e ".[cuda]" && JAX_ENABLE_X64=1 python -m jaxbench.runner lu__gpu__f32`.
