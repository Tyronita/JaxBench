# JaxBench challenges — the dataset

Each row is an **optimization challenge on a JAX hot path**. You get the current
C++/CUDA source as the reference; you submit a faster implementation; JaxBench
**rebuilds JAX** and scores the **speedup of the real library**, gated on correctness
against stock JAX. Two tiers:

- **jaxlib_kernel** — leaf kernels, cheap targeted `.so` rebuild + hot-swap.
- **xla_core** — compiler-graph logic (scan/while/simplifier/fusion/HLO): the
  *computational* wins; rebuilt via `--override_repository` (costlier, run serverless).


## Tier: `jaxlib_kernel` (10 challenges)

| id | hot path (file) | what to edit | speed-up test | devices | rebuild | why it matters |
|---|---|---|---|---|--:|---|
| `linalg_lu_getrf` | `jaxlib/gpu/solver_kernels_ffi.cc` | GetrfImpl / EVOLVE-BLOCK | `lu` (256…2048) | gpu,cpu,tpu | ~7.0s | LU backs jnp.linalg.solve/inv/det and every linear-solve-heavy program. |
| `linalg_qr_geqrf` | `jaxlib/gpu/solver_kernels_ffi.cc` | GeqrfImpl / OrgqrImpl | `qr` (256…2048) | gpu,cpu,tpu | ~7.0s | QR backs least-squares, orthogonalization, many iterative solvers. |
| `linalg_svd_gesvd` | `jaxlib/gpu/solver_kernels_ffi.cc` | GesvdImpl | `svd` (256…1024) | gpu,cpu,tpu | ~7.0s | SVD backs PCA, low-rank, pinv, spectral methods. |
| `linalg_eigh_syevd` | `jaxlib/gpu/solver_kernels_ffi.cc` | SyevdImpl | `eigh` (256…1024) | gpu,cpu,tpu | ~7.0s | eigh backs spectral methods, PCA, physics; a common bottleneck. |
| `linalg_cholesky_update` | `jaxlib/gpu/linalg_kernels.cu.cc` | drotg / CholeskyUpdateKernel (EVOLVE-BLOCK) | `cholesky_update` (256…1024) | gpu | ~12.9s | Online/streaming covariance, Kalman filters, GP updates. |
| `linalg_tridiagonal_solve` | `jaxlib/tridiagonal_solve_perturbed.h` | MaybePerturbPivot (EVOLVE-BLOCK) | `tridiagonal_solve` (1024…16384) | gpu,cpu | ~17.0s | PDE/ODE solvers, time series, banded systems; proven +41.7% in earlier work. |
| `linalg_householder` | `jaxlib/gpu/householder_kernels.cu.cc` | ProductOf...Reflectors...Kernel | `householder_product` (256…1024) | gpu | ~5.5s | QR application step; orthogonal transforms. |
| `prng_threefry` | `jaxlib/gpu/prng_kernels.cu.cc` | ThreeFry2x32Kernel (EVOLVE-BLOCK) | `threefry_uniform` (262144…16777216) | gpu | ~4.8s | JAX's RNG — threaded through every program (init, dropout, sampling). Cheapest kernel to rebuild (~4.8s) => ideal first challenge. |
| `sparse_csr_matmul` | `jaxlib/gpu/sparse_kernels.cc` | CsrMatmul (EVOLVE-BLOCK) | `csr_matmul` (1024…16384) | gpu,cpu | ~8.4s | GNNs, scientific computing, sparse linear algebra. |
| `lapack_cpu_backend` | `jaxlib/cpu/lapack_kernels.cc` | TriMatrixEquationSolver / EVOLVE-BLOCK | `lu` (256…1024) | cpu | ~22.7s | The CPU backend for all of jnp.linalg; most-maintained kernel upstream. NOTE: 22.7s monolith — split the hot routine into its own .cc before iterating. |

## Tier: `xla_core` (5 challenges)

| id | hot path (file) | what to edit | speed-up test | devices | rebuild | why it matters |
|---|---|---|---|---|--:|---|
| `xla_scan_expander` | `xla/service/scan_expander.cc` | ExpandInstruction / scan->while lowering | `scan_cumsum` (1024…65536) | gpu,cpu,tpu | ~60.0s | THE 'scan expensive operations' target. jax.lax.scan lowers through this; RNNs, ODE integrators, cumulative ops, associative scans. Optimizing the lowering makes a huge class of programs faster. |
| `xla_while_loop_simplifier` | `xla/service/while_loop_simplifier.cc` | while-loop simplification pass | `while_loop_iter` (1000…100000) | gpu,cpu,tpu | ~60.0s | jax.lax.while_loop / fori_loop / scan all lower to While; trip-count and invariant-hoisting optimizations speed up every loop-heavy program. |
| `xla_algebraic_simplifier` | `xla/hlo/transforms/simplifiers/algebraic_simplifier.cc` | the simplifier rules | `algebra_graph` (512…2048) | gpu,cpu,tpu | ~43.0s | The biggest single optimization pass (428KB). Better/cheaper algebraic rewrites speed up compilation AND the compiled graph for nearly all programs. |
| `xla_instruction_fusion` | `xla/service/instruction_fusion.cc` | fusion decisions | `elementwise_chain` (1048576…16777216) | gpu,tpu | ~60.0s | Fusion eliminates memory traffic for elementwise/reduction chains — the core lever for memory-bound DL graphs (activations, normalizations, softmax). |
| `xla_hlo_instruction` | `xla/hlo/ir/hlo_instruction.cc` | core IR routines | `compile_time_graph` (256…512) | gpu,cpu | ~27.0s | The IR every pass touches; micro-optimizations here speed up *compilation* of all programs. Highest fan-out (the .h is 134s/98 libs) — edit the .cc only. |

## Build instructions per challenge

### `linalg_lu_getrf` — LU decomposition (getrf)
- **file:** `jaxlib/gpu/solver_kernels_ffi.cc`  ·  **tier:** jaxlib_kernel  ·  **devices:** gpu, cpu, tpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs //jaxlib/cuda:_solver`
- **apply:** hot-swap `jax_cuda12_plugin/_solver.so`
- **speed-up test:** `lu` over sizes [256, 512, 1024, 2048], dtypes ['f32', 'f64', 'c64', 'c128']; correctness = `lu_reconstruct`

### `linalg_qr_geqrf` — QR decomposition (geqrf/orgqr)
- **file:** `jaxlib/gpu/solver_kernels_ffi.cc`  ·  **tier:** jaxlib_kernel  ·  **devices:** gpu, cpu, tpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs //jaxlib/cuda:_solver`
- **apply:** hot-swap `jax_cuda12_plugin/_solver.so`
- **speed-up test:** `qr` over sizes [256, 512, 1024, 2048], dtypes ['f32', 'f64', 'c64', 'c128']; correctness = `qr_reconstruct`

### `linalg_svd_gesvd` — SVD (gesvd/gesvdj)
- **file:** `jaxlib/gpu/solver_kernels_ffi.cc`  ·  **tier:** jaxlib_kernel  ·  **devices:** gpu, cpu, tpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs //jaxlib/cuda:_solver`
- **apply:** hot-swap `jax_cuda12_plugin/_solver.so`
- **speed-up test:** `svd` over sizes [256, 512, 1024], dtypes ['f32', 'f64']; correctness = `svd_reconstruct`

### `linalg_eigh_syevd` — Symmetric/Hermitian eig (syevd/heevd)
- **file:** `jaxlib/gpu/solver_kernels_ffi.cc`  ·  **tier:** jaxlib_kernel  ·  **devices:** gpu, cpu, tpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs //jaxlib/cuda:_solver`
- **apply:** hot-swap `jax_cuda12_plugin/_solver.so`
- **speed-up test:** `eigh` over sizes [256, 512, 1024], dtypes ['f32', 'f64', 'c64', 'c128']; correctness = `eigh_residual`

### `linalg_cholesky_update` — Rank-1 Cholesky update (drotg)
- **file:** `jaxlib/gpu/linalg_kernels.cu.cc`  ·  **tier:** jaxlib_kernel  ·  **devices:** gpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --config=build_cuda_with_clang //jaxlib/cuda:_linalg`
- **apply:** hot-swap `jax_cuda12_plugin/_linalg.so`
- **speed-up test:** `cholesky_update` over sizes [256, 512, 1024], dtypes ['f32', 'f64']; correctness = `cholesky_update_residual`

### `linalg_tridiagonal_solve` — Tridiagonal solve (perturbed pivots)
- **file:** `jaxlib/tridiagonal_solve_perturbed.h`  ·  **tier:** jaxlib_kernel  ·  **devices:** gpu, cpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --config=build_cuda_with_clang //jaxlib/cuda:_linalg`
- **apply:** hot-swap `jax_cuda12_plugin/_linalg.so`
- **speed-up test:** `tridiagonal_solve` over sizes [1024, 4096, 16384], dtypes ['f32', 'f64']; correctness = `tridiagonal_residual`

### `linalg_householder` — Householder reflector product
- **file:** `jaxlib/gpu/householder_kernels.cu.cc`  ·  **tier:** jaxlib_kernel  ·  **devices:** gpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --config=build_cuda_with_clang //jaxlib/cuda:_solver`
- **apply:** hot-swap `jax_cuda12_plugin/_solver.so`
- **speed-up test:** `householder_product` over sizes [256, 512, 1024], dtypes ['f32', 'f64', 'c64', 'c128']; correctness = `householder_orthonormal`

### `prng_threefry` — Threefry2x32 PRNG
- **file:** `jaxlib/gpu/prng_kernels.cu.cc`  ·  **tier:** jaxlib_kernel  ·  **devices:** gpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --config=build_cuda_with_clang //jaxlib/cuda:_prng`
- **apply:** hot-swap `jax_cuda12_plugin/_prng.so`
- **speed-up test:** `threefry_uniform` over sizes [262144, 4194304, 16777216], dtypes ['f32']; correctness = `prng_statistical`

### `sparse_csr_matmul` — CSR sparse × dense matmul
- **file:** `jaxlib/gpu/sparse_kernels.cc`  ·  **tier:** jaxlib_kernel  ·  **devices:** gpu, cpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs //jaxlib/cuda:_sparse`
- **apply:** hot-swap `jax_cuda12_plugin/_sparse.so`
- **speed-up test:** `csr_matmul` over sizes [1024, 4096, 16384], dtypes ['f32', 'f64']; correctness = `csr_vs_dense`

### `lapack_cpu_backend` — CPU LAPACK backend (lu/qr/svd/eigh)
- **file:** `jaxlib/cpu/lapack_kernels.cc`  ·  **tier:** jaxlib_kernel  ·  **devices:** cpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check //jaxlib/cpu:_lapack`
- **apply:** hot-swap `jaxlib/cpu/_lapack.so`
- **speed-up test:** `lu` over sizes [256, 512, 1024], dtypes ['f32', 'f64']; correctness = `lu_reconstruct`

### `xla_scan_expander` — Scan lowering (jax.lax.scan)
- **file:** `xla/service/scan_expander.cc`  ·  **tier:** xla_core  ·  **devices:** gpu, cpu, tpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel`
- **apply:** rebuild + reinstall the plugin wheel (serverless: fresh container)
- **speed-up test:** `scan_cumsum` over sizes [1024, 8192, 65536], dtypes ['f32', 'f64']; correctness = `scan_vs_reference`

### `xla_while_loop_simplifier` — While-loop simplification
- **file:** `xla/service/while_loop_simplifier.cc`  ·  **tier:** xla_core  ·  **devices:** gpu, cpu, tpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel`
- **apply:** rebuild + reinstall the plugin wheel (serverless: fresh container)
- **speed-up test:** `while_loop_iter` over sizes [1000, 10000, 100000], dtypes ['f32']; correctness = `while_vs_reference`

### `xla_algebraic_simplifier` — Algebraic simplification pass
- **file:** `xla/hlo/transforms/simplifiers/algebraic_simplifier.cc`  ·  **tier:** xla_core  ·  **devices:** gpu, cpu, tpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel`
- **apply:** rebuild + reinstall the plugin wheel (serverless: fresh container)
- **speed-up test:** `algebra_graph` over sizes [512, 1024, 2048], dtypes ['f32']; correctness = `graph_vs_reference`

### `xla_instruction_fusion` — Instruction fusion
- **file:** `xla/service/instruction_fusion.cc`  ·  **tier:** xla_core  ·  **devices:** gpu, tpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel`
- **apply:** rebuild + reinstall the plugin wheel (serverless: fresh container)
- **speed-up test:** `elementwise_chain` over sizes [1048576, 4194304, 16777216], dtypes ['f32']; correctness = `graph_vs_reference`

### `xla_hlo_instruction` — Core HLO IR (hlo_instruction)
- **file:** `xla/hlo/ir/hlo_instruction.cc`  ·  **tier:** xla_core  ·  **devices:** gpu, cpu
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel`
- **apply:** rebuild + reinstall the plugin wheel (serverless: fresh container)
- **speed-up test:** `compile_time_graph` over sizes [256, 512], dtypes ['f32']; correctness = `graph_vs_reference`

