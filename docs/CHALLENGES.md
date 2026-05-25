# JaxBench challenges — the dataset

Each row is an **optimization challenge on a JAX hot path**: you get the current C++/CUDA
source as reference, submit a faster version, and JaxBench rebuilds JAX and scores the
speedup of the real library, gated on correctness. `(pytests, vals)` is the validation tuple.


## Tier: `jaxlib_kernel` (10)

### `linalg_lu_getrf` — LU decomposition (getrf)
- **file / edit:** `jaxlib/gpu/solver_kernels_ffi.cc` · GetrfImpl / EVOLVE-BLOCK
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs //jaxlib/cuda:_solver`
- **apply:** hot-swap `jax_cuda12_plugin/_solver.so`
- **rebuild:** ~7.0s · **devices:** gpu, cpu, tpu
- **speed-up test:** `lu` sizes=[256, 512, 1024, 2048] dtypes=['f32', 'f64', 'c64', 'c128']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[linalg_lu_getrf]', 'tests/test_correctness.py::test_task_correct[lu*]', 'tests/test_integration.py'], vals=['correctness_residual', 'latency_ms', 'throughput_gflops', 'speedup_vs_stock'])
- **why:** LU backs jnp.linalg.solve/inv/det and every linear-solve-heavy program.

### `linalg_qr_geqrf` — QR decomposition (geqrf/orgqr)
- **file / edit:** `jaxlib/gpu/solver_kernels_ffi.cc` · GeqrfImpl / OrgqrImpl
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs //jaxlib/cuda:_solver`
- **apply:** hot-swap `jax_cuda12_plugin/_solver.so`
- **rebuild:** ~7.0s · **devices:** gpu, cpu, tpu
- **speed-up test:** `qr` sizes=[256, 512, 1024, 2048] dtypes=['f32', 'f64', 'c64', 'c128']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[linalg_qr_geqrf]', 'tests/test_correctness.py::test_task_correct[qr*]', 'tests/test_integration.py'], vals=['correctness_residual', 'latency_ms', 'throughput_gflops', 'speedup_vs_stock'])
- **why:** QR backs least-squares, orthogonalization, many iterative solvers.

### `linalg_svd_gesvd` — SVD (gesvd/gesvdj)
- **file / edit:** `jaxlib/gpu/solver_kernels_ffi.cc` · GesvdImpl
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs //jaxlib/cuda:_solver`
- **apply:** hot-swap `jax_cuda12_plugin/_solver.so`
- **rebuild:** ~7.0s · **devices:** gpu, cpu, tpu
- **speed-up test:** `svd` sizes=[256, 512, 1024] dtypes=['f32', 'f64']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[linalg_svd_gesvd]', 'tests/test_correctness.py::test_task_correct[svd*]', 'tests/test_integration.py'], vals=['correctness_residual', 'latency_ms', 'throughput_gflops', 'speedup_vs_stock'])
- **why:** SVD backs PCA, low-rank, pinv, spectral methods.

### `linalg_eigh_syevd` — Symmetric/Hermitian eig (syevd/heevd)
- **file / edit:** `jaxlib/gpu/solver_kernels_ffi.cc` · SyevdImpl
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs //jaxlib/cuda:_solver`
- **apply:** hot-swap `jax_cuda12_plugin/_solver.so`
- **rebuild:** ~7.0s · **devices:** gpu, cpu, tpu
- **speed-up test:** `eigh` sizes=[256, 512, 1024] dtypes=['f32', 'f64', 'c64', 'c128']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[linalg_eigh_syevd]', 'tests/test_correctness.py::test_task_correct[eigh*]', 'tests/test_integration.py'], vals=['correctness_residual', 'latency_ms', 'throughput_gflops', 'speedup_vs_stock'])
- **why:** eigh backs spectral methods, PCA, physics; a common bottleneck.

### `linalg_cholesky_update` — Rank-1 Cholesky update (drotg)
- **file / edit:** `jaxlib/gpu/linalg_kernels.cu.cc` · drotg / CholeskyUpdateKernel (EVOLVE-BLOCK)
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --config=build_cuda_with_clang //jaxlib/cuda:_linalg`
- **apply:** hot-swap `jax_cuda12_plugin/_linalg.so`
- **rebuild:** ~12.9s · **devices:** gpu
- **speed-up test:** `cholesky_update` sizes=[256, 512, 1024] dtypes=['f32', 'f64']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[linalg_cholesky_update]', 'tests/test_correctness.py::test_task_correct[cholesky_update*]', 'tests/test_integration.py'], vals=['correctness_residual', 'latency_ms', 'throughput_gflops', 'speedup_vs_stock'])
- **why:** Online/streaming covariance, Kalman filters, GP updates.

### `linalg_tridiagonal_solve` — Tridiagonal solve (perturbed pivots)
- **file / edit:** `jaxlib/tridiagonal_solve_perturbed.h` · MaybePerturbPivot (EVOLVE-BLOCK)
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --config=build_cuda_with_clang //jaxlib/cuda:_linalg`
- **apply:** hot-swap `jax_cuda12_plugin/_linalg.so`
- **rebuild:** ~17.0s · **devices:** gpu, cpu
- **speed-up test:** `tridiagonal_solve` sizes=[1024, 4096, 16384] dtypes=['f32', 'f64']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[linalg_tridiagonal_solve]', 'tests/test_correctness.py::test_task_correct[tridiagonal_solve*]', 'tests/test_integration.py'], vals=['correctness_residual', 'latency_ms', 'throughput_gflops', 'speedup_vs_stock'])
- **why:** PDE/ODE solvers, time series, banded systems; proven +41.7% in earlier work.

### `linalg_householder` — Householder reflector product
- **file / edit:** `jaxlib/gpu/householder_kernels.cu.cc` · ProductOf...Reflectors...Kernel
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --config=build_cuda_with_clang //jaxlib/cuda:_solver`
- **apply:** hot-swap `jax_cuda12_plugin/_solver.so`
- **rebuild:** ~5.5s · **devices:** gpu
- **speed-up test:** `householder_product` sizes=[256, 512, 1024] dtypes=['f32', 'f64', 'c64', 'c128']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[linalg_householder]', 'tests/test_correctness.py::test_task_correct[householder_product*]', 'tests/test_integration.py'], vals=['correctness_residual', 'latency_ms', 'throughput_gflops', 'speedup_vs_stock'])
- **why:** QR application step; orthogonal transforms.

### `prng_threefry` — Threefry2x32 PRNG
- **file / edit:** `jaxlib/gpu/prng_kernels.cu.cc` · ThreeFry2x32Kernel (EVOLVE-BLOCK)
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --config=build_cuda_with_clang //jaxlib/cuda:_prng`
- **apply:** hot-swap `jax_cuda12_plugin/_prng.so`
- **rebuild:** ~4.8s · **devices:** gpu
- **speed-up test:** `threefry_uniform` sizes=[262144, 4194304, 16777216] dtypes=['f32']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[prng_threefry]', 'tests/test_correctness.py::test_task_correct[threefry_uniform*]', 'tests/test_integration.py'], vals=['determinism', 'uniform_mean_var', 'latency_ms', 'speedup_vs_stock'])
- **why:** JAX's RNG — threaded through every program (init, dropout, sampling). Cheapest kernel to rebuild (~4.8s) => ideal first challenge.

### `sparse_csr_matmul` — CSR sparse × dense matmul
- **file / edit:** `jaxlib/gpu/sparse_kernels.cc` · CsrMatmul (EVOLVE-BLOCK)
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs //jaxlib/cuda:_sparse`
- **apply:** hot-swap `jax_cuda12_plugin/_sparse.so`
- **rebuild:** ~8.4s · **devices:** gpu, cpu
- **speed-up test:** `csr_matmul` sizes=[1024, 4096, 16384] dtypes=['f32', 'f64']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[sparse_csr_matmul]', 'tests/test_correctness.py::test_task_correct[csr_matmul*]', 'tests/test_integration.py'], vals=['correctness_residual', 'latency_ms', 'speedup_vs_stock'])
- **why:** GNNs, scientific computing, sparse linear algebra.

### `lapack_cpu_backend` — CPU LAPACK backend (lu/qr/svd/eigh)
- **file / edit:** `jaxlib/cpu/lapack_kernels.cc` · TriMatrixEquationSolver / EVOLVE-BLOCK
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check //jaxlib/cpu:_lapack`
- **apply:** hot-swap `jaxlib/cpu/_lapack.so`
- **rebuild:** ~22.7s · **devices:** cpu
- **speed-up test:** `lu` sizes=[256, 512, 1024] dtypes=['f32', 'f64']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[lapack_cpu_backend]', 'tests/test_correctness.py::test_task_correct[lu*]', 'tests/test_integration.py'], vals=['correctness_residual', 'latency_ms', 'throughput_gflops', 'speedup_vs_stock'])
- **why:** The CPU backend for all of jnp.linalg; most-maintained kernel upstream. NOTE: 22.7s monolith — split the hot routine into its own .cc before iterating.


## Tier: `xla_core` (5)

### `xla_scan_expander` — Scan lowering (jax.lax.scan)
- **file / edit:** `xla/service/scan_expander.cc` · ExpandInstruction / scan->while lowering
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel`
- **apply:** rebuild + reinstall plugin wheel
- **rebuild:** ~60.0s · **devices:** gpu, cpu, tpu
- **speed-up test:** `scan_cumsum` sizes=[1024, 8192, 65536] dtypes=['f32', 'f64']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[xla_scan_expander]', 'tests/test_xla_core.py'], vals=['correctness_residual', 'compile_time_ms', 'exec_latency_ms', 'speedup_vs_stock'])
- **why:** THE 'scan expensive operations' target. jax.lax.scan lowers through this; RNNs, ODE integrators, cumulative ops, associative scans. Optimizing the lowering makes a huge class of programs faster.

### `xla_while_loop_simplifier` — While-loop simplification
- **file / edit:** `xla/service/while_loop_simplifier.cc` · while-loop simplification pass
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel`
- **apply:** rebuild + reinstall plugin wheel
- **rebuild:** ~60.0s · **devices:** gpu, cpu, tpu
- **speed-up test:** `while_loop_iter` sizes=[1000, 10000, 100000] dtypes=['f32']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[xla_while_loop_simplifier]', 'tests/test_xla_core.py'], vals=['correctness_residual', 'compile_time_ms', 'exec_latency_ms', 'speedup_vs_stock'])
- **why:** jax.lax.while_loop / fori_loop / scan all lower to While; trip-count and invariant-hoisting optimizations speed up every loop-heavy program.

### `xla_algebraic_simplifier` — Algebraic simplification pass
- **file / edit:** `xla/hlo/transforms/simplifiers/algebraic_simplifier.cc` · the simplifier rules
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel`
- **apply:** rebuild + reinstall plugin wheel
- **rebuild:** ~43.0s · **devices:** gpu, cpu, tpu
- **speed-up test:** `algebra_graph` sizes=[512, 1024, 2048] dtypes=['f32']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[xla_algebraic_simplifier]', 'tests/test_xla_core.py'], vals=['correctness_residual', 'compile_time_ms', 'exec_latency_ms', 'speedup_vs_stock'])
- **why:** The biggest single optimization pass (428KB). Better/cheaper algebraic rewrites speed up compilation AND the compiled graph for nearly all programs.

### `xla_instruction_fusion` — Instruction fusion
- **file / edit:** `xla/service/instruction_fusion.cc` · fusion decisions
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel`
- **apply:** rebuild + reinstall plugin wheel
- **rebuild:** ~60.0s · **devices:** gpu, tpu
- **speed-up test:** `elementwise_chain` sizes=[1048576, 4194304, 16777216] dtypes=['f32']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[xla_instruction_fusion]', 'tests/test_xla_core.py'], vals=['correctness_residual', 'compile_time_ms', 'exec_latency_ms', 'speedup_vs_stock'])
- **why:** Fusion eliminates memory traffic for elementwise/reduction chains — the core lever for memory-bound DL graphs (activations, normalizations, softmax).

### `xla_hlo_instruction` — Core HLO IR (hlo_instruction)
- **file / edit:** `xla/hlo/ir/hlo_instruction.cc` · core IR routines
- **build:** `bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.12 --disk_cache=/data/bazel-disk --features=-layering_check --config=cuda_libraries_from_stubs --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel`
- **apply:** rebuild + reinstall plugin wheel
- **rebuild:** ~27.0s · **devices:** gpu, cpu
- **speed-up test:** `compile_time_graph` sizes=[256, 512] dtypes=['f32']
- **validation tuple:** (pytests=['tests/test_challenges.py::test_challenge_correct[xla_hlo_instruction]', 'tests/test_xla_core.py'], vals=['correctness_residual', 'compile_time_ms', 'exec_latency_ms', 'speedup_vs_stock'])
- **why:** The IR every pass touches; micro-optimizations here speed up *compilation* of all programs. Highest fan-out (the .h is 134s/98 libs) — edit the .cc only.

