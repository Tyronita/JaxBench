# JaxBench — the 100 tasks

Auto-generated from `jaxbench/registry.py`. Each task = one (op × device × dtype),
its backing jaxlib file (the ShinkaEvolve mutation surface), the targeted build,
the extension `.so` hot-swapped, and the N-sweep. Ordered by rebuild cost (cheapest
loop first).

**100 tasks** · families: prng 4, linalg 88, sparse 4, tridiagonal 4 · devices: gpu 61, cpu 39

| # | task id | op | device | dtype | file (mutation surface) | build target | .so | wheel |
|--:|---|---|---|---|---|---|---|---|
| 1 | `threefry_normal__gpu__f32` | threefry_normal | gpu | f32 | `jaxlib/gpu/prng_kernels.cu.cc` | `//jaxlib/cuda:_prng` | `_prng.so` | jax-cuda-plugin |
| 2 | `threefry_normal__gpu__f64` | threefry_normal | gpu | f64 | `jaxlib/gpu/prng_kernels.cu.cc` | `//jaxlib/cuda:_prng` | `_prng.so` | jax-cuda-plugin |
| 3 | `threefry_uniform__gpu__f32` | threefry_uniform | gpu | f32 | `jaxlib/gpu/prng_kernels.cu.cc` | `//jaxlib/cuda:_prng` | `_prng.so` | jax-cuda-plugin |
| 4 | `threefry_uniform__gpu__f64` | threefry_uniform | gpu | f64 | `jaxlib/gpu/prng_kernels.cu.cc` | `//jaxlib/cuda:_prng` | `_prng.so` | jax-cuda-plugin |
| 5 | `cholesky__gpu__c128` | cholesky | gpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 6 | `cholesky__gpu__c64` | cholesky | gpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 7 | `cholesky__gpu__f32` | cholesky | gpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 8 | `cholesky__gpu__f64` | cholesky | gpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 9 | `det__gpu__c128` | det | gpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 10 | `det__gpu__c64` | det | gpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 11 | `det__gpu__f32` | det | gpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 12 | `det__gpu__f64` | det | gpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 13 | `eigh__gpu__c128` | eigh | gpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 14 | `eigh__gpu__c64` | eigh | gpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 15 | `eigh__gpu__f32` | eigh | gpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 16 | `eigh__gpu__f64` | eigh | gpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 17 | `expm__gpu__f32` | expm | gpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 18 | `expm__gpu__f64` | expm | gpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 19 | `householder_product__gpu__c128` | householder_product | gpu | c128 | `jaxlib/gpu/householder_kernels.cu.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 20 | `householder_product__gpu__c64` | householder_product | gpu | c64 | `jaxlib/gpu/householder_kernels.cu.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 21 | `householder_product__gpu__f32` | householder_product | gpu | f32 | `jaxlib/gpu/householder_kernels.cu.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 22 | `householder_product__gpu__f64` | householder_product | gpu | f64 | `jaxlib/gpu/householder_kernels.cu.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 23 | `inv__gpu__c128` | inv | gpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 24 | `inv__gpu__c64` | inv | gpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 25 | `inv__gpu__f32` | inv | gpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 26 | `inv__gpu__f64` | inv | gpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 27 | `lstsq__gpu__f32` | lstsq | gpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 28 | `lstsq__gpu__f64` | lstsq | gpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 29 | `lu__gpu__c128` | lu | gpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 30 | `lu__gpu__c64` | lu | gpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 31 | `lu__gpu__f32` | lu | gpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 32 | `lu__gpu__f64` | lu | gpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 33 | `lu_solve__gpu__c128` | lu_solve | gpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 34 | `lu_solve__gpu__c64` | lu_solve | gpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 35 | `lu_solve__gpu__f32` | lu_solve | gpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 36 | `lu_solve__gpu__f64` | lu_solve | gpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 37 | `qr__gpu__c128` | qr | gpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 38 | `qr__gpu__c64` | qr | gpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 39 | `qr__gpu__f32` | qr | gpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 40 | `qr__gpu__f64` | qr | gpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 41 | `slogdet__gpu__f32` | slogdet | gpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 42 | `slogdet__gpu__f64` | slogdet | gpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 43 | `solve__gpu__c128` | solve | gpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 44 | `solve__gpu__c64` | solve | gpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 45 | `solve__gpu__f32` | solve | gpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 46 | `solve__gpu__f64` | solve | gpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 47 | `svd__gpu__c128` | svd | gpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 48 | `svd__gpu__c64` | svd | gpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 49 | `svd__gpu__f32` | svd | gpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 50 | `svd__gpu__f64` | svd | gpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cuda:_solver` | `_solver.so` | jax-cuda-plugin |
| 51 | `qr_pivoting__gpu__f32` | qr_pivoting | gpu | f32 | `jaxlib/gpu/hybrid_kernels.cc` | `//jaxlib/cuda:_hybrid` | `_hybrid.so` | jax-cuda-plugin |
| 52 | `qr_pivoting__gpu__f64` | qr_pivoting | gpu | f64 | `jaxlib/gpu/hybrid_kernels.cc` | `//jaxlib/cuda:_hybrid` | `_hybrid.so` | jax-cuda-plugin |
| 53 | `csr_matmul__cpu__f32` | csr_matmul | cpu | f32 | `jaxlib/gpu/sparse_kernels.cc` | `//jaxlib/cpu:_sparse` | `_sparse.so` | jaxlib |
| 54 | `csr_matmul__cpu__f64` | csr_matmul | cpu | f64 | `jaxlib/gpu/sparse_kernels.cc` | `//jaxlib/cpu:_sparse` | `_sparse.so` | jaxlib |
| 55 | `csr_matmul__gpu__f32` | csr_matmul | gpu | f32 | `jaxlib/gpu/sparse_kernels.cc` | `//jaxlib/cuda:_sparse` | `_sparse.so` | jax-cuda-plugin |
| 56 | `csr_matmul__gpu__f64` | csr_matmul | gpu | f64 | `jaxlib/gpu/sparse_kernels.cc` | `//jaxlib/cuda:_sparse` | `_sparse.so` | jax-cuda-plugin |
| 57 | `cholesky_update__gpu__f32` | cholesky_update | gpu | f32 | `jaxlib/gpu/linalg_kernels.cu.cc` | `//jaxlib/cuda:_linalg` | `_linalg.so` | jax-cuda-plugin |
| 58 | `cholesky_update__gpu__f64` | cholesky_update | gpu | f64 | `jaxlib/gpu/linalg_kernels.cu.cc` | `//jaxlib/cuda:_linalg` | `_linalg.so` | jax-cuda-plugin |
| 59 | `lu_pivots_to_permutation__gpu__i32` | lu_pivots_to_permutation | gpu | i32 | `jaxlib/gpu/linalg_kernels.cu.cc` | `//jaxlib/cuda:_linalg` | `_linalg.so` | jax-cuda-plugin |
| 60 | `tridiagonal_solve__gpu__f32` | tridiagonal_solve | gpu | f32 | `jaxlib/gpu/linalg_kernels.cu.cc` | `//jaxlib/cuda:_linalg` | `_linalg.so` | jax-cuda-plugin |
| 61 | `tridiagonal_solve__gpu__f64` | tridiagonal_solve | gpu | f64 | `jaxlib/gpu/linalg_kernels.cu.cc` | `//jaxlib/cuda:_linalg` | `_linalg.so` | jax-cuda-plugin |
| 62 | `tridiagonal_solve_perturbed__gpu__f32` | tridiagonal_solve_perturbed | gpu | f32 | `jaxlib/tridiagonal_solve_perturbed.h` | `//jaxlib/cuda:_linalg` | `_linalg.so` | jax-cuda-plugin |
| 63 | `tridiagonal_solve_perturbed__gpu__f64` | tridiagonal_solve_perturbed | gpu | f64 | `jaxlib/tridiagonal_solve_perturbed.h` | `//jaxlib/cuda:_linalg` | `_linalg.so` | jax-cuda-plugin |
| 64 | `cholesky__cpu__c128` | cholesky | cpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 65 | `cholesky__cpu__c64` | cholesky | cpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 66 | `cholesky__cpu__f32` | cholesky | cpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 67 | `cholesky__cpu__f64` | cholesky | cpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 68 | `det__cpu__c128` | det | cpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 69 | `det__cpu__c64` | det | cpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 70 | `det__cpu__f32` | det | cpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 71 | `det__cpu__f64` | det | cpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 72 | `eigh__cpu__c128` | eigh | cpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 73 | `eigh__cpu__c64` | eigh | cpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 74 | `eigh__cpu__f32` | eigh | cpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 75 | `eigh__cpu__f64` | eigh | cpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 76 | `expm__cpu__f32` | expm | cpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 77 | `expm__cpu__f64` | expm | cpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 78 | `inv__cpu__c128` | inv | cpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 79 | `inv__cpu__c64` | inv | cpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 80 | `inv__cpu__f32` | inv | cpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 81 | `inv__cpu__f64` | inv | cpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 82 | `lstsq__cpu__f32` | lstsq | cpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 83 | `lstsq__cpu__f64` | lstsq | cpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 84 | `lu__cpu__c128` | lu | cpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 85 | `lu__cpu__c64` | lu | cpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 86 | `lu__cpu__f32` | lu | cpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 87 | `lu__cpu__f64` | lu | cpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 88 | `lu_solve__cpu__c128` | lu_solve | cpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 89 | `lu_solve__cpu__c64` | lu_solve | cpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 90 | `lu_solve__cpu__f32` | lu_solve | cpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 91 | `lu_solve__cpu__f64` | lu_solve | cpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 92 | `qr__cpu__c128` | qr | cpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 93 | `qr__cpu__c64` | qr | cpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 94 | `qr__cpu__f32` | qr | cpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 95 | `qr__cpu__f64` | qr | cpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 96 | `slogdet__cpu__f32` | slogdet | cpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 97 | `slogdet__cpu__f64` | slogdet | cpu | f64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 98 | `solve__cpu__c128` | solve | cpu | c128 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 99 | `solve__cpu__c64` | solve | cpu | c64 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |
| 100 | `solve__cpu__f32` | solve | cpu | f32 | `jaxlib/gpu/solver_kernels_ffi.cc` | `//jaxlib/cpu:_lapack` | `_lapack.so` | jaxlib |

## Metrics recorded per task

- **correctness_pass** — max residual over the (N x dtype) sweep is below tolerance (gate).
- **latency_ms** — median wall time per call, post-warmup, device-synchronised.
- **throughput_gflops** — op-specific FLOPs / latency; machine-comparable.
- **speedup_vs_baseline** — baseline_latency / candidate_latency (the optimisation target).
