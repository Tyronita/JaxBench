# ShinkaEvolve-style mutation results

Real end-to-end run of the JaxBench eval loop ShinkaEvolve drives — 5 candidate
kernels for `prng_threefry` on an A100, run inside `evolvebench.azurecr.io/jaxbench/build-cuda:0.2.0`
with the warm bazel disk_cache mounted (`/data/bazel-disk`).

Each candidate goes through: stage source → `bazel build //jaxlib/cuda:_prng`
(`--config=cuda12,cuda_libraries_from_stubs,build_cuda_with_clang`, sm_80) →
hot-swap `_prng.so` → correctness vs stock → latency sweep → speedup → restore.

Candidates are programmatic no-op comment variants (no LLM key in env) — semantically
identical to stock — so scores cluster within measurement noise of 1.0; the value of
this run is proving the entire loop works on real hardware, not finding a kernel win.
