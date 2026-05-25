# JaxBench

**A dataset of optimization challenges on JAX's own hot paths — KernelBench, but the
thing you optimize is JAX itself.**

Each challenge hands you the **current C++/CUDA source** of a JAX hot path as the
reference. You submit a **faster implementation**. JaxBench **rebuilds JAX** with your
change and scores the **speedup of the real library**, gated on **correctness against
stock JAX**. The mission: make the official `jax-ml/jax` measurably faster and land
high-impact upstream PRs.

> The challenge is not "how fast is JAX" — it's "**change a solver / a graph pass in
> C++, rebuild, keep it correct, make it faster.**" Correctness is a gate; speedup is
> the score.

## Two tiers of hot path

| tier | what you edit | rebuild | why it matters |
|---|---|--:|---|
| **`jaxlib_kernel`** | leaf C++/CUDA kernels — LU/QR/SVD/eigh/Cholesky solvers, Threefry PRNG, sparse, tridiagonal | ~4–23s (targeted `.so` + hot-swap) | the library-call path (`jnp.linalg.*`, `jax.random`, sparse) |
| **`xla_core`** | XLA compiler-graph logic — **scan/while lowering, algebraic simplifier, fusion, HLO IR** | ~27–134s (`--override_repository`) | the **computational** wins: faster `jax.lax.scan`, expensive ops, fusion → makes a whole class of programs faster |

Including XLA core is the point: optimizing `scan_expander`, `while_loop_simplifier`,
the `algebraic_simplifier`, and fusion is how you make JAX *more effective*, not just
how you speed up one library call. Their higher rebuild cost is exactly why the loop
runs **serverless + parallel + cross-device** (below).

Full dataset (15 challenges, 10 kernel + 5 XLA core): [`docs/CHALLENGES.md`](docs/CHALLENGES.md).

## Anatomy of a challenge

Each entry in `jaxbench/challenges.py` is `(file, quick build instructions, config,
speed-up test)`. Two examples:

**`prng_threefry`** (jaxlib_kernel — the cheapest, hottest first challenge)
- **file:** `jaxlib/gpu/prng_kernels.cu.cc` (edit inside `// EVOLVE-BLOCK`)
- **build:** `bazel build … --config=cuda_libraries_from_stubs --config=build_cuda_with_clang //jaxlib/cuda:_prng` (~4.8s)
- **apply:** hot-swap `jax_cuda12_plugin/_prng.so`
- **speed-up test:** `jax.random.uniform` over 2¹⁸…2²⁴; correctness = determinism + distribution

**`xla_scan_expander`** (xla_core — the "scan expensive operations" target)
- **file:** `xla/service/scan_expander.cc` (the scan→while lowering)
- **build:** `bazel build … --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel` (~60s)
- **apply:** rebuild + reinstall the plugin wheel (serverless: fresh container per candidate)
- **speed-up test:** `jax.lax.scan` cumulative recurrence over 1k…64k steps; correctness = output matches reference

## The loop (correctness-gated speedup)

```
candidate C++  ──▶  build (tier-aware)  ──▶  apply (.so hot-swap | wheel reinstall)
                                              │
   score = mean speedup vs stock  ◀──  speed-up test  ──▶  correctness vs stock JAX
   (0 if any test fails correctness)         (jax.lax.scan / jnp.linalg.* / …)
```

`python -m jaxbench.challenge_runner <challenge_id> --device gpu --build`

## The 15 hot paths — file, build, and what we measure

All builds share `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check`. Below is the per-path delta (flags + target), how the result is applied, the rebuild cost, the speed-up test, and the **values measured**.


### Tier 1 — jaxlib kernels (targeted `.so` + hot-swap) — 10 hot paths

| # | challenge | file · edit | `BASE` + … | rebuild | speed-up test | vals measured |
|--:|---|---|---|--:|---|---|
| 1 | `linalg_lu_getrf` | `jaxlib/gpu/solver_kernels_ffi.cc` · GetrfImpl / EVOLVE-BLOCK | `--config=cuda_libraries_from_stubs //jaxlib/cuda:_solver` → swap `_solver.so` | ~7.0s | `lu` 256…2048, gpu,cpu,tpu | correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock |
| 2 | `linalg_qr_geqrf` | `jaxlib/gpu/solver_kernels_ffi.cc` · GeqrfImpl / OrgqrImpl | `--config=cuda_libraries_from_stubs //jaxlib/cuda:_solver` → swap `_solver.so` | ~7.0s | `qr` 256…2048, gpu,cpu,tpu | correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock |
| 3 | `linalg_svd_gesvd` | `jaxlib/gpu/solver_kernels_ffi.cc` · GesvdImpl | `--config=cuda_libraries_from_stubs //jaxlib/cuda:_solver` → swap `_solver.so` | ~7.0s | `svd` 256…1024, gpu,cpu,tpu | correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock |
| 4 | `linalg_eigh_syevd` | `jaxlib/gpu/solver_kernels_ffi.cc` · SyevdImpl | `--config=cuda_libraries_from_stubs //jaxlib/cuda:_solver` → swap `_solver.so` | ~7.0s | `eigh` 256…1024, gpu,cpu,tpu | correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock |
| 5 | `linalg_cholesky_update` | `jaxlib/gpu/linalg_kernels.cu.cc` · drotg / CholeskyUpdateKernel (EVOLVE-BLOCK) | `--config=cuda_libraries_from_stubs --config=build_cuda_with_clang //jaxlib/cuda:_linalg` → swap `_linalg.so` | ~12.9s | `cholesky_update` 256…1024, gpu | correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock |
| 6 | `linalg_tridiagonal_solve` | `jaxlib/tridiagonal_solve_perturbed.h` · MaybePerturbPivot (EVOLVE-BLOCK) | `--config=cuda_libraries_from_stubs --config=build_cuda_with_clang //jaxlib/cuda:_linalg` → swap `_linalg.so` | ~17.0s | `tridiagonal_solve` 1024…16384, gpu,cpu | correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock |
| 7 | `linalg_householder` | `jaxlib/gpu/householder_kernels.cu.cc` · ProductOf...Reflectors...Kernel | `--config=cuda_libraries_from_stubs --config=build_cuda_with_clang //jaxlib/cuda:_solver` → swap `_solver.so` | ~5.5s | `householder_product` 256…1024, gpu | correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock |
| 8 | `prng_threefry` | `jaxlib/gpu/prng_kernels.cu.cc` · ThreeFry2x32Kernel (EVOLVE-BLOCK) | `--config=cuda_libraries_from_stubs --config=build_cuda_with_clang //jaxlib/cuda:_prng` → swap `_prng.so` | ~4.8s | `threefry_uniform` 262144…16777216, gpu | determinism, uniform_mean_var, latency_ms, speedup_vs_stock |
| 9 | `sparse_csr_matmul` | `jaxlib/gpu/sparse_kernels.cc` · CsrMatmul (EVOLVE-BLOCK) | `--config=cuda_libraries_from_stubs //jaxlib/cuda:_sparse` → swap `_sparse.so` | ~8.4s | `csr_matmul` 1024…16384, gpu,cpu | correctness_residual, latency_ms, speedup_vs_stock |
| 10 | `lapack_cpu_backend` | `jaxlib/cpu/lapack_kernels.cc` · TriMatrixEquationSolver / EVOLVE-BLOCK | `//jaxlib/cpu:_lapack` → swap `_lapack.so` | ~22.7s | `lu` 256…1024, cpu | correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock |

### Tier 2 — XLA core graph (`--override_repository`, rebuild+reinstall) — 5 hot paths

| # | challenge | file · edit | `BASE` + … | rebuild | speed-up test | vals measured |
|--:|---|---|---|--:|---|---|
| 1 | `xla_scan_expander` | `xla/service/scan_expander.cc` · ExpandInstruction / scan->while lowering | `--config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel` → reinstall wheel | ~60.0s | `scan_cumsum` 1024…65536, gpu,cpu,tpu | correctness_residual, compile_time_ms, exec_latency_ms, speedup_vs_stock |
| 2 | `xla_while_loop_simplifier` | `xla/service/while_loop_simplifier.cc` · while-loop simplification pass | `--config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel` → reinstall wheel | ~60.0s | `while_loop_iter` 1000…100000, gpu,cpu,tpu | correctness_residual, compile_time_ms, exec_latency_ms, speedup_vs_stock |
| 3 | `xla_algebraic_simplifier` | `xla/hlo/transforms/simplifiers/algebraic_simplifier.cc` · the simplifier rules | `--config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel` → reinstall wheel | ~43.0s | `algebra_graph` 512…2048, gpu,cpu,tpu | correctness_residual, compile_time_ms, exec_latency_ms, speedup_vs_stock |
| 4 | `xla_instruction_fusion` | `xla/service/instruction_fusion.cc` · fusion decisions | `--config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel` → reinstall wheel | ~60.0s | `elementwise_chain` 1048576…16777216, gpu,tpu | correctness_residual, compile_time_ms, exec_latency_ms, speedup_vs_stock |
| 5 | `xla_hlo_instruction` | `xla/hlo/ir/hlo_instruction.cc` · core IR routines | `--config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel` → reinstall wheel | ~27.0s | `compile_time_graph` 256…512, gpu,cpu | correctness_residual, compile_time_ms, exec_latency_ms, speedup_vs_stock |

### Validation per hot path — `(pytests, vals)`

Each challenge advertises the correctness pytests that gate it and the values measured:

| challenge | pytests | vals |
|---|---|---|
| `linalg_lu_getrf` | `tests/test_challenges.py::test_challenge_correct[linalg_lu_getrf]` · `tests/test_correctness.py::test_task_correct[lu*]` · `tests/test_integration.py` | (correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock) |
| `linalg_qr_geqrf` | `tests/test_challenges.py::test_challenge_correct[linalg_qr_geqrf]` · `tests/test_correctness.py::test_task_correct[qr*]` · `tests/test_integration.py` | (correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock) |
| `linalg_svd_gesvd` | `tests/test_challenges.py::test_challenge_correct[linalg_svd_gesvd]` · `tests/test_correctness.py::test_task_correct[svd*]` · `tests/test_integration.py` | (correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock) |
| `linalg_eigh_syevd` | `tests/test_challenges.py::test_challenge_correct[linalg_eigh_syevd]` · `tests/test_correctness.py::test_task_correct[eigh*]` · `tests/test_integration.py` | (correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock) |
| `linalg_cholesky_update` | `tests/test_challenges.py::test_challenge_correct[linalg_cholesky_update]` · `tests/test_correctness.py::test_task_correct[cholesky_update*]` · `tests/test_integration.py` | (correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock) |
| `linalg_tridiagonal_solve` | `tests/test_challenges.py::test_challenge_correct[linalg_tridiagonal_solve]` · `tests/test_correctness.py::test_task_correct[tridiagonal_solve*]` · `tests/test_integration.py` | (correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock) |
| `linalg_householder` | `tests/test_challenges.py::test_challenge_correct[linalg_householder]` · `tests/test_correctness.py::test_task_correct[householder_product*]` · `tests/test_integration.py` | (correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock) |
| `prng_threefry` | `tests/test_challenges.py::test_challenge_correct[prng_threefry]` · `tests/test_correctness.py::test_task_correct[threefry_uniform*]` · `tests/test_integration.py` | (determinism, uniform_mean_var, latency_ms, speedup_vs_stock) |
| `sparse_csr_matmul` | `tests/test_challenges.py::test_challenge_correct[sparse_csr_matmul]` · `tests/test_correctness.py::test_task_correct[csr_matmul*]` · `tests/test_integration.py` | (correctness_residual, latency_ms, speedup_vs_stock) |
| `lapack_cpu_backend` | `tests/test_challenges.py::test_challenge_correct[lapack_cpu_backend]` · `tests/test_correctness.py::test_task_correct[lu*]` · `tests/test_integration.py` | (correctness_residual, latency_ms, throughput_gflops, speedup_vs_stock) |
| `xla_scan_expander` | `tests/test_challenges.py::test_challenge_correct[xla_scan_expander]` · `tests/test_xla_core.py` | (correctness_residual, compile_time_ms, exec_latency_ms, speedup_vs_stock) |
| `xla_while_loop_simplifier` | `tests/test_challenges.py::test_challenge_correct[xla_while_loop_simplifier]` · `tests/test_xla_core.py` | (correctness_residual, compile_time_ms, exec_latency_ms, speedup_vs_stock) |
| `xla_algebraic_simplifier` | `tests/test_challenges.py::test_challenge_correct[xla_algebraic_simplifier]` · `tests/test_xla_core.py` | (correctness_residual, compile_time_ms, exec_latency_ms, speedup_vs_stock) |
| `xla_instruction_fusion` | `tests/test_challenges.py::test_challenge_correct[xla_instruction_fusion]` · `tests/test_xla_core.py` | (correctness_residual, compile_time_ms, exec_latency_ms, speedup_vs_stock) |
| `xla_hlo_instruction` | `tests/test_challenges.py::test_challenge_correct[xla_hlo_instruction]` · `tests/test_xla_core.py` | (correctness_residual, compile_time_ms, exec_latency_ms, speedup_vs_stock) |


## Serverless, cross-device

A candidate evaluation is a pure function, so it fans out one-invocation-per-candidate
across **GPU, TPU, and CPU**. Mount a warm bazel `--disk_cache` read-only so the cold
build is paid once globally; each candidate then does only the targeted rebuild
(seconds for kernels) or an XLA override build (parallelised). `docker/Dockerfile`
(`DEVICE=cpu|cuda|tpu`), `serverless/` (handler + Modal app + GCP-TPU path), and
`infra/` (Azure CPU VM + GCP TPU VM) make this one command per device.
Details + recommendation: [`serverless/README.md`](serverless/README.md).

### Prebuilt images per device (anonymised registry)

Every device + setup has a build-ready image so the loop runs anywhere. Registry refs
are **anonymised** — set `JAXBENCH_REGISTRY` to your own (e.g. an Azure Container
Registry) before building. `REGISTRY = ${JAXBENCH_REGISTRY:-your-registry.azurecr.io/jaxbench}`.

| device | eval image | build image (bazel+clang) | serverless runtime |
|---|---|---|---|
| **GPU** A100/H100 | `${REGISTRY}/runtime-cuda:0.2.0` | `${REGISTRY}/build-cuda:0.2.0` | Modal / RunPod / ACI-GPU / GKE |
| **CPU** many-core | `${REGISTRY}/runtime-cpu:0.2.0` | `${REGISTRY}/build-cpu:0.2.0` | Cloud Run / Azure Container Apps / Fargate |
| **TPU** | `${REGISTRY}/runtime-tpu:0.2.0` | `${REGISTRY}/build-tpu:0.2.0` | GCP Cloud Run TPU / GKE / TPU VM |

Build them all (eval + build variants, every device) in one command:
```bash
JAXBENCH_REGISTRY=<your-acr>.azurecr.io/jaxbench docker buildx bake -f docker/bake.hcl all
# or: bash docker/build_all.sh            # plain docker build; PUSH=1 to push
```
Full table + run examples: [`docker/REGISTRY.md`](docker/REGISTRY.md).

## Validated on real hardware

- **A100 80GB:** stock baselines recorded for 9 ops (`results/gpu/`); the full
  **mutate→build→hot-swap→run loop validated end-to-end** — a fork-built `_linalg.so`
  hot-swapped into stock `jax[cuda12]` ran `cholesky_update` correctly (residual 9.4e-8).
- **24-core CPU:** stock baselines recorded (`results/cpu/`); challenge runner verified
  for both tiers (kernel + XLA scan/while/simplifier workloads run + check on CPU).
- **TPU:** device class + workloads + commands wired; runs on **GCP** (Azure has no
  TPUs) via `infra/gcp_tpu_vm.sh`. No TPU numbers are claimed until that run executes.

Stock baselines are the references each challenge's speedup is scored against.

## Build-speed methodology (why the rebuild is fast enough to evolve)

Per single-kernel edit: naive `nvcc` wheel ~21s → **targeted `.so` + clang device +
hot-swap ~13s** (~4s host-only). Levers: content-addressed `--disk_cache`, one GPU
arch (`sm_80`), clang device compiler, build one extension `.so`, hot-swap it. XLA-tier
edits use `--override_repository` (the http_archive cache can't be edited in place).
Full detail + measured per-file costs: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Evolving with ShinkaEvolve (real dependency)

```bash
pip install -e ".[evolve]"               # pulls shinka-evolve from GitHub
export ANTHROPIC_API_KEY=...             # ShinkaEvolve's LLM
python shinka/run_evo.py --file jaxlib/gpu/prng_kernels.cu.cc --generations 20
```

ShinkaEvolve mutates the hot path's region; `shinka/evaluate.py` builds → applies →
correctness-gates → scores speedup. One island per hot path. See
[`shinka/README.md`](shinka/README.md) and [`docs/PR_PLAYBOOK.md`](docs/PR_PLAYBOOK.md).

## Quickstart

```bash
pip install -e . && pip install "jax[cpu]"     # or jax[cuda12] / jax[tpu]
python -m jaxbench.challenges                   # the challenge dataset
python -m jaxbench.challenge_runner xla_scan_expander --device cpu   # run one (no build)
```

## Layout

```
jaxbench/  challenges(dataset+validation tuples) · workloads(speed-up tests) · challenge_runner
           build(targeted .so + XLA override) · ops · reference · bench · metrics · sharding
tests/     test_challenges (per-challenge correctness gate) · test_xla_core · test_correctness · test_integration
docker/    Dockerfile (DEVICE=cpu|cuda|tpu) · bake.hcl · build_all.sh · REGISTRY.md (anonymised)
serverless/ runtime-agnostic handler · Modal app · cross-device guide
infra/     az (CPU VM) + gcloud (TPU VM) provisioning
shinka/    ShinkaEvolve evaluator + launcher (the evolutionary driver)
docs/      CHALLENGES · METHODOLOGY · METRICS · PR_PLAYBOOK
```

## License

Apache-2.0 (matches JAX), so improvements flow upstream.
