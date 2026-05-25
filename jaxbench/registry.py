"""JaxBench task registry.

A *task* is a single (operation x dtype x device) benchmark unit, in the spirit of
KernelBench. Each task names:

  * the JAX-level operation under test (what the user actually calls),
  * the jaxlib C++/CUDA source file that implements it (the ShinkaEvolve mutation
    surface),
  * the Bazel build target + the resulting extension `.so` (for targeted rebuild +
    hot-swap),
  * the device, dtype and problem sizes to sweep,
  * the correctness reference and tolerance,
  * the evaluation metrics and whether multi-GPU sharding applies.

The registry is the single source of truth: the correctness suite, the perf
harness, the runner and the ShinkaEvolve adapter are all parameterised over it.
`tasks()` returns the fully-expanded list (100 tasks); `dump_yaml()` serialises it.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Sequence


# --- backend kernel files & build targets (verified against the fork) ----------
# (build_target, so_path_in_plugin, wheel)
BACKENDS = {
    "gpu_solver":   ("//jaxlib/cuda:_solver", "jax_cuda12_plugin/_solver.so", "jax-cuda-plugin"),
    "gpu_linalg":   ("//jaxlib/cuda:_linalg", "jax_cuda12_plugin/_linalg.so", "jax-cuda-plugin"),
    "gpu_hybrid":   ("//jaxlib/cuda:_hybrid", "jax_cuda12_plugin/_hybrid.so", "jax-cuda-plugin"),
    "gpu_prng":     ("//jaxlib/cuda:_prng",   "jax_cuda12_plugin/_prng.so",   "jax-cuda-plugin"),
    "gpu_sparse":   ("//jaxlib/cuda:_sparse", "jax_cuda12_plugin/_sparse.so", "jax-cuda-plugin"),
    "cpu_lapack":   ("//jaxlib/cpu:_lapack",  "jaxlib/cpu/_lapack.so",        "jaxlib"),
    "cpu_sparse":   ("//jaxlib/cpu:_sparse",  "jaxlib/cpu/_sparse.so",        "jaxlib"),
}


@dataclass(frozen=True)
class Op:
    """An operation under test and where it is implemented."""
    name: str                 # op key, resolved in jaxbench.ops
    family: str               # linalg | prng | sparse | tridiagonal | rnn
    file: str                 # jaxlib source file = mutation surface
    gpu_backend: str | None   # BACKENDS key for the GPU build, or None
    cpu_backend: str | None   # BACKENDS key for the CPU build, or None
    dtypes: tuple[str, ...]   # dtypes this op supports
    ref: str                  # reference implementation key (jaxbench.reference)
    shardable: bool = False   # has a meaningful multi-GPU (sharded-batch) variant
    note: str = ""


# Problem-size sweeps per family (square matrix dimension N, or vector length).
SIZES = {
    "linalg":      (64, 128, 256, 512, 1024),
    "tridiagonal": (256, 1024, 4096, 16384),
    "prng":        (1 << 16, 1 << 20, 1 << 24),
    "sparse":      (1024, 4096, 16384),
    "rnn":         (128, 256, 512),
}

# Per-dtype relative tolerances for correctness. f32/c64 scale with problem size
# (residual ~ sqrt(N)*eps); 2e-3 cleanly separates a correct kernel (~3e-4 at N=1024)
# from a broken one (O(1)). PRNG is a *statistical* check, not a numeric residual, so
# it gets a loose family override below.
TOL = {"f32": 2e-3, "f64": 1e-10, "c64": 2e-3, "c128": 1e-10, "i32": 0.0}
FAMILY_TOL = {"prng": 0.05}  # finite-sample mean/var deviation


# --- operation catalogue (maps directly onto the fork's EVOLVE-BLOCK files) -----
OPS: tuple[Op, ...] = (
    # dense LAPACK/cuSOLVER decompositions: GPU -> _solver, CPU -> _lapack
    Op("lu",       "linalg", "jaxlib/gpu/solver_kernels_ffi.cc", "gpu_solver", "cpu_lapack",
       ("f32", "f64", "c64", "c128"), "lu", shardable=True,
       note="LU (getrf). jax.scipy.linalg.lu; backend cuSOLVER / LAPACK."),
    Op("qr",       "linalg", "jaxlib/gpu/solver_kernels_ffi.cc", "gpu_solver", "cpu_lapack",
       ("f32", "f64", "c64", "c128"), "qr", shardable=True,
       note="QR (geqrf+orgqr). jnp.linalg.qr."),
    Op("svd",      "linalg", "jaxlib/gpu/solver_kernels_ffi.cc", "gpu_solver", "cpu_lapack",
       ("f32", "f64", "c64", "c128"), "svd", shardable=True,
       note="SVD (gesvd/gesvdj). jnp.linalg.svd."),
    Op("eigh",     "linalg", "jaxlib/gpu/solver_kernels_ffi.cc", "gpu_solver", "cpu_lapack",
       ("f32", "f64", "c64", "c128"), "eigh", shardable=True,
       note="Symmetric/Hermitian eig (syevd/heevd). jnp.linalg.eigh."),
    Op("solve",    "linalg", "jaxlib/gpu/solver_kernels_ffi.cc", "gpu_solver", "cpu_lapack",
       ("f32", "f64", "c64", "c128"), "solve", shardable=True,
       note="Linear solve (getrf+getrs). jnp.linalg.solve."),
    Op("inv",      "linalg", "jaxlib/gpu/solver_kernels_ffi.cc", "gpu_solver", "cpu_lapack",
       ("f32", "f64", "c64", "c128"), "inv", shardable=True,
       note="Matrix inverse. jnp.linalg.inv."),
    Op("cholesky", "linalg", "jaxlib/gpu/solver_kernels_ffi.cc", "gpu_solver", "cpu_lapack",
       ("f32", "f64", "c64", "c128"), "cholesky", shardable=True,
       note="Cholesky (potrf). jnp.linalg.cholesky."),
    Op("slogdet",  "linalg", "jaxlib/gpu/solver_kernels_ffi.cc", "gpu_solver", "cpu_lapack",
       ("f32", "f64"), "slogdet", note="Sign+log|det| via LU. jnp.linalg.slogdet."),
    Op("det",      "linalg", "jaxlib/gpu/solver_kernels_ffi.cc", "gpu_solver", "cpu_lapack",
       ("f32", "f64", "c64", "c128"), "det", note="Determinant via LU. jnp.linalg.det."),
    Op("lu_solve", "linalg", "jaxlib/gpu/solver_kernels_ffi.cc", "gpu_solver", "cpu_lapack",
       ("f32", "f64", "c64", "c128"), "lu_solve",
       note="Solve with precomputed LU (getrs). jax.scipy.linalg.lu_solve."),
    Op("lstsq",    "linalg", "jaxlib/gpu/solver_kernels_ffi.cc", "gpu_solver", "cpu_lapack",
       ("f32", "f64"), "lstsq", note="Least squares (gelsd/QR). jnp.linalg.lstsq."),
    Op("expm",     "linalg", "jaxlib/gpu/solver_kernels_ffi.cc", "gpu_solver", "cpu_lapack",
       ("f32", "f64"), "expm", note="Matrix exponential (Pade/eig). jax.scipy.linalg.expm."),

    # GPU custom linalg kernels -> _linalg (the hot, evolvable .cu.cc / .cc)
    Op("cholesky_update", "linalg", "jaxlib/gpu/linalg_kernels.cu.cc", "gpu_linalg", None,
       ("f32", "f64"), "cholesky_update",
       note="Rank-1 Cholesky update (drotg). jax.lax.linalg.cholesky_update."),
    Op("lu_pivots_to_permutation", "linalg", "jaxlib/gpu/linalg_kernels.cu.cc", "gpu_linalg", None,
       ("i32",), "lu_pivots_to_permutation",
       note="Pivots->permutation. jax.lax.linalg.lu_pivots_to_permutation."),

    # tridiagonal solvers (GPU _linalg device + CPU _lapack); perturbed = proven win
    Op("tridiagonal_solve", "tridiagonal", "jaxlib/gpu/linalg_kernels.cu.cc", "gpu_linalg", "cpu_lapack",
       ("f32", "f64"), "tridiagonal_solve", note="jax.lax.linalg.tridiagonal_solve."),
    Op("tridiagonal_solve_perturbed", "tridiagonal", "jaxlib/tridiagonal_solve_perturbed.h",
       "gpu_linalg", "cpu_lapack", ("f32", "f64"), "tridiagonal_solve",
       note="Perturbed-pivot tridiagonal solve (header; fans out, gen-68 +41.73%)."),

    # Householder product (QR building block) -> _solver
    Op("householder_product", "linalg", "jaxlib/gpu/householder_kernels.cu.cc", "gpu_solver", None,
       ("f32", "f64", "c64", "c128"), "householder_product",
       note="Apply elementary Householder reflectors. jax.lax.linalg.householder_product."),

    # pivoted QR -> _hybrid (CPU/GPU hybrid)
    Op("qr_pivoting", "linalg", "jaxlib/gpu/hybrid_kernels.cc", "gpu_hybrid", None,
       ("f32", "f64"), "qr_pivoting",
       note="Column-pivoted QR. jax.scipy.linalg.qr(pivoting=True)."),

    # PRNG (Threefry) -> _prng (GPU) / lapack-side (CPU handled by XLA, kept GPU here)
    Op("threefry_uniform", "prng", "jaxlib/gpu/prng_kernels.cu.cc", "gpu_prng", None,
       ("f32", "f64"), "threefry_uniform", shardable=True,
       note="Uniform via Threefry2x32. jax.random.uniform."),
    Op("threefry_normal", "prng", "jaxlib/gpu/prng_kernels.cu.cc", "gpu_prng", None,
       ("f32", "f64"), "threefry_normal", shardable=True,
       note="Normal via Threefry2x32. jax.random.normal."),

    # sparse CSR x dense -> _sparse
    Op("csr_matmul", "sparse", "jaxlib/gpu/sparse_kernels.cc", "gpu_sparse", "cpu_sparse",
       ("f32", "f64"), "csr_matmul", note="CSR @ dense. jax.experimental.sparse."),
)


@dataclass(frozen=True)
class Task:
    id: str
    op: str
    family: str
    file: str
    device: str            # "gpu" | "cpu"
    dtype: str
    backend: str           # BACKENDS key
    build_target: str
    so_path: str
    wheel: str
    sizes: tuple[int, ...]
    tol: float
    ref: str
    metrics: tuple[str, ...]
    shardable: bool
    note: str

    @property
    def cuda_compiler(self) -> str:
        # device kernels (.cu.cc) compile ~25-36% faster with clang; host .cc unaffected
        return "clang" if self.device == "gpu" and self.file.endswith(".cu.cc") else "default"


METRICS = ("latency_ms", "throughput_gflops", "speedup_vs_baseline", "correctness_pass")


def _expand() -> list[Task]:
    out: list[Task] = []
    for op in OPS:
        for device in ("gpu", "cpu"):
            backend_key = op.gpu_backend if device == "gpu" else op.cpu_backend
            if backend_key is None:
                continue
            target, so_path, wheel = BACKENDS[backend_key]
            for dtype in op.dtypes:
                tol = FAMILY_TOL.get(op.family, TOL.get(dtype, 1e-4))
                out.append(Task(
                    id=f"{op.name}__{device}__{dtype}",
                    op=op.name, family=op.family, file=op.file, device=device,
                    dtype=dtype, backend=backend_key, build_target=target,
                    so_path=so_path, wheel=wheel,
                    sizes=SIZES[op.family], tol=tol, ref=op.ref,
                    metrics=METRICS, shardable=op.shardable, note=op.note,
                ))
    return out


# Build, rank by "iterate-fast value" (cheap rebuild first), then cap at exactly 100.
_REBUILD_RANK = {  # measured per-edit cost (s); lower = better loop target
    "//jaxlib/cuda:_prng": 4.8, "//jaxlib/cuda:_linalg": 12.9,
    "//jaxlib/cuda:_solver": 7.0, "//jaxlib/cuda:_hybrid": 7.5,
    "//jaxlib/cuda:_sparse": 8.4, "//jaxlib/cpu:_lapack": 22.7,
    "//jaxlib/cpu:_sparse": 8.4,
}


def tasks(limit: int = 100) -> list[Task]:
    allt = _expand()
    allt.sort(key=lambda t: (_REBUILD_RANK.get(t.build_target, 99), t.op, t.device, t.dtype))
    return allt[:limit]


def task_by_id(tid: str) -> Task:
    for t in tasks(10**9):
        if t.id == tid:
            return t
    raise KeyError(tid)


def dump_yaml(path: str, limit: int = 100) -> int:
    import yaml  # optional dep; only needed to (re)generate tasks.yaml
    rows = [asdict(t) for t in tasks(limit)]
    with open(path, "w") as f:
        yaml.safe_dump({"version": 1, "n_tasks": len(rows), "tasks": rows},
                       f, sort_keys=False)
    return len(rows)


if __name__ == "__main__":
    ts = tasks()
    print(f"{len(ts)} tasks")
    from collections import Counter
    print("by family:", dict(Counter(t.family for t in ts)))
    print("by device:", dict(Counter(t.device for t in ts)))
    print("by build target:", dict(Counter(t.build_target for t in ts)))
    for t in ts[:8]:
        print(f"  {t.id:38} {t.file:40} {t.build_target}")
