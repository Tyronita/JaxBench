"""JaxBench challenge dataset.

JaxBench is **not** a benchmark of how fast JAX is. It is a **dataset of optimization
challenges on JAX's own hot paths**: each challenge hands you the *current* C++/CUDA
source as the reference, you submit a *modified* implementation, and we **rebuild JAX**
and score the **speedup of the real library** — gated on correctness against stock JAX.

A challenge spans two tiers of hot path:

  * `jaxlib_kernel` — leaf C++/CUDA kernels (linalg solvers, PRNG, sparse). Cheap to
    rebuild (~4–23 s) via a targeted extension-`.so` build + hot-swap.
  * `xla_core` — XLA compiler-graph logic (scan/while lowering, algebraic simplifier,
    fusion, HLO IR). This is where the *computational* wins are — making `jax.lax.scan`
    and expensive graph ops genuinely faster. Costlier to rebuild (~27–134 s, via
    `--override_repository`), which is exactly why the loop runs serverless + parallel.

Each challenge records: the editable file + region, *why* optimizing it makes JAX more
effective, the quick build instructions + config, the speed-up test (a JAX workload
that exercises the hot path), how correctness is checked, and the devices it runs on.
This module is the single source of truth for the dataset (`challenges()`), the
serverless runner, and the docs.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict


# ---- build recipes (the "quick build instructions + config") ------------------
COMMON_FLAGS = ["--repo_env=HERMETIC_PYTHON_VERSION=3.12", "--disk_cache=/data/bazel-disk",
                "--features=-layering_check"]
GPU_FLAGS = ["--config=cuda_libraries_from_stubs"]
CLANG_DEVICE = ["--config=build_cuda_with_clang"]   # 25–36% faster .cu.cc rebuilds


@dataclass(frozen=True)
class Build:
    target: str                 # bazel target rebuilt
    flags: tuple[str, ...]      # the config that makes the rebuild fast/correct
    rebuild_s: float            # measured per-edit rebuild cost on the A100 box
    so_path: str | None         # extension .so to hot-swap (jaxlib tier), else None
    xla_override: bool = False  # XLA tier: build with --override_repository=xla=<copy>

    def command(self) -> str:
        flags = list(COMMON_FLAGS) + list(self.flags)
        if self.xla_override:
            flags.append("--override_repository=xla=/data/xla-local")
        return "bazel build " + " ".join(flags) + " " + self.target


@dataclass(frozen=True)
class SpeedTest:
    workload: str               # key in jaxbench.workloads — the program exercising the hot path
    sizes: tuple                # problem sizes to sweep
    dtypes: tuple[str, ...]
    devices: tuple[str, ...]    # gpu | cpu | tpu
    correctness: str            # how the rebuilt result is checked (reference key)


@dataclass(frozen=True)
class Challenge:
    id: str
    tier: str                   # jaxlib_kernel | xla_core
    title: str
    file: str                   # editable hot path (reference impl lives here)
    region: str                 # what to edit (EVOLVE-BLOCK / function / pass)
    why: str                    # why optimizing this makes JAX more effective
    build: Build
    test: SpeedTest

    @property
    def pytests(self) -> tuple[str, ...]:
        """The pytest node-ids that gate this challenge's correctness."""
        base = (f"tests/test_challenges.py::test_challenge_correct[{self.id}]",)
        if self.tier == "jaxlib_kernel":
            return base + (
                f"tests/test_correctness.py::test_task_correct[{self.test.workload}*]",
                "tests/test_integration.py",
            )
        return base + ("tests/test_xla_core.py",)

    @property
    def vals(self) -> tuple[str, ...]:
        """The values we measure for this challenge (the eval metrics)."""
        if self.test.workload.startswith("threefry"):
            return ("determinism", "uniform_mean_var", "latency_ms", "speedup_vs_stock")
        if self.tier == "xla_core":
            return ("correctness_residual", "compile_time_ms", "exec_latency_ms",
                    "speedup_vs_stock")
        flop_ops = {"lu", "qr", "svd", "eigh", "cholesky", "solve", "inv",
                    "cholesky_update", "householder_product", "tridiagonal_solve"}
        if self.test.workload in flop_ops:
            return ("correctness_residual", "latency_ms", "throughput_gflops",
                    "speedup_vs_stock")
        return ("correctness_residual", "latency_ms", "speedup_vs_stock")

    @property
    def validation(self) -> tuple:
        """(pytests, vals) — the correctness gate and the measured values."""
        return (self.pytests, self.vals)


# Hot-swap .so paths for the jaxlib extensions
_SO = {"_solver": "jax_cuda12_plugin/_solver.so", "_linalg": "jax_cuda12_plugin/_linalg.so",
       "_prng": "jax_cuda12_plugin/_prng.so", "_hybrid": "jax_cuda12_plugin/_hybrid.so",
       "_sparse": "jax_cuda12_plugin/_sparse.so", "_lapack": "jaxlib/cpu/_lapack.so"}


def _gpu_kernel(target, so, rebuild, cu=False):
    flags = tuple(GPU_FLAGS + (CLANG_DEVICE if cu else []))
    return Build(target=target, flags=flags, rebuild_s=rebuild, so_path=_SO[so])


CHALLENGES: tuple[Challenge, ...] = (
    # ===== Tier 1: jaxlib leaf kernels (cheap, targeted .so + hot-swap) ==========
    Challenge(
        "linalg_lu_getrf", "jaxlib_kernel", "LU decomposition (getrf)",
        "jaxlib/gpu/solver_kernels_ffi.cc", "GetrfImpl / EVOLVE-BLOCK",
        "LU backs jnp.linalg.solve/inv/det and every linear-solve-heavy program.",
        _gpu_kernel("//jaxlib/cuda:_solver.so", "_solver", 7.0),
        SpeedTest("lu", (256, 512, 1024, 2048), ("f32", "f64", "c64", "c128"),
                  ("gpu", "cpu", "tpu"), "lu_reconstruct")),
    Challenge(
        "linalg_qr_geqrf", "jaxlib_kernel", "QR decomposition (geqrf/orgqr)",
        "jaxlib/gpu/solver_kernels_ffi.cc", "GeqrfImpl / OrgqrImpl",
        "QR backs least-squares, orthogonalization, many iterative solvers.",
        _gpu_kernel("//jaxlib/cuda:_solver.so", "_solver", 7.0),
        SpeedTest("qr", (256, 512, 1024, 2048), ("f32", "f64", "c64", "c128"),
                  ("gpu", "cpu", "tpu"), "qr_reconstruct")),
    Challenge(
        "linalg_svd_gesvd", "jaxlib_kernel", "SVD (gesvd/gesvdj)",
        "jaxlib/gpu/solver_kernels_ffi.cc", "GesvdImpl",
        "SVD backs PCA, low-rank, pinv, spectral methods.",
        _gpu_kernel("//jaxlib/cuda:_solver.so", "_solver", 7.0),
        SpeedTest("svd", (256, 512, 1024), ("f32", "f64"),
                  ("gpu", "cpu", "tpu"), "svd_reconstruct")),
    Challenge(
        "linalg_eigh_syevd", "jaxlib_kernel", "Symmetric/Hermitian eig (syevd/heevd)",
        "jaxlib/gpu/solver_kernels_ffi.cc", "SyevdImpl",
        "eigh backs spectral methods, PCA, physics; a common bottleneck.",
        _gpu_kernel("//jaxlib/cuda:_solver.so", "_solver", 7.0),
        SpeedTest("eigh", (256, 512, 1024), ("f32", "f64", "c64", "c128"),
                  ("gpu", "cpu", "tpu"), "eigh_residual")),
    Challenge(
        "linalg_cholesky_update", "jaxlib_kernel", "Rank-1 Cholesky update (drotg)",
        "jaxlib/gpu/linalg_kernels.cu.cc", "drotg / CholeskyUpdateKernel (EVOLVE-BLOCK)",
        "Online/streaming covariance, Kalman filters, GP updates.",
        _gpu_kernel("//jaxlib/cuda:_linalg.so", "_linalg", 12.9, cu=True),
        SpeedTest("cholesky_update", (256, 512, 1024), ("f32", "f64"),
                  ("gpu",), "cholesky_update_residual")),
    Challenge(
        "linalg_tridiagonal_solve", "jaxlib_kernel", "Tridiagonal solve (perturbed pivots)",
        "jaxlib/tridiagonal_solve_perturbed.h", "MaybePerturbPivot (EVOLVE-BLOCK)",
        "PDE/ODE solvers, time series, banded systems; proven +41.7% in earlier work.",
        _gpu_kernel("//jaxlib/cuda:_linalg.so", "_linalg", 17.0, cu=True),
        SpeedTest("tridiagonal_solve", (1024, 4096, 16384), ("f32", "f64"),
                  ("gpu", "cpu"), "tridiagonal_residual")),
    Challenge(
        "linalg_householder", "jaxlib_kernel", "Householder reflector product",
        "jaxlib/gpu/householder_kernels.cu.cc", "ProductOf...Reflectors...Kernel",
        "QR application step; orthogonal transforms.",
        _gpu_kernel("//jaxlib/cuda:_solver.so", "_solver", 5.5, cu=True),
        SpeedTest("householder_product", (256, 512, 1024), ("f32", "f64", "c64", "c128"),
                  ("gpu",), "householder_orthonormal")),
    Challenge(
        "prng_threefry", "jaxlib_kernel", "Threefry2x32 PRNG",
        "jaxlib/gpu/prng_kernels.cu.cc", "ThreeFry2x32Kernel (EVOLVE-BLOCK)",
        "JAX's RNG — threaded through every program (init, dropout, sampling). Cheapest "
        "kernel to rebuild (~4.8s) => ideal first challenge.",
        _gpu_kernel("//jaxlib/cuda:_prng.so", "_prng", 4.8, cu=True),
        SpeedTest("threefry_uniform", (1 << 18, 1 << 22, 1 << 24), ("f32",),
                  ("gpu",), "prng_statistical")),
    Challenge(
        "sparse_csr_matmul", "jaxlib_kernel", "CSR sparse × dense matmul",
        "jaxlib/gpu/sparse_kernels.cc", "CsrMatmul (EVOLVE-BLOCK)",
        "GNNs, scientific computing, sparse linear algebra.",
        _gpu_kernel("//jaxlib/cuda:_sparse.so", "_sparse", 8.4),
        SpeedTest("csr_matmul", (1024, 4096, 16384), ("f32", "f64"),
                  ("gpu", "cpu"), "csr_vs_dense")),
    Challenge(
        "lapack_cpu_backend", "jaxlib_kernel", "CPU LAPACK backend (lu/qr/svd/eigh)",
        "jaxlib/cpu/lapack_kernels.cc", "TriMatrixEquationSolver / EVOLVE-BLOCK",
        "The CPU backend for all of jnp.linalg; most-maintained kernel upstream. NOTE: "
        "22.7s monolith — split the hot routine into its own .cc before iterating.",
        Build("//jaxlib/cpu:_lapack.so", tuple(), 22.7, _SO["_lapack"]),
        SpeedTest("lu", (256, 512, 1024), ("f32", "f64"), ("cpu",), "lu_reconstruct")),

    # ===== Tier 2: XLA core graph (the computational wins; serverless rebuild) ===
    Challenge(
        "xla_scan_expander", "xla_core", "Scan lowering (jax.lax.scan)",
        "xla/service/scan_expander.cc", "ExpandInstruction / scan->while lowering",
        "THE 'scan expensive operations' target. jax.lax.scan lowers through this; "
        "RNNs, ODE integrators, cumulative ops, associative scans. Optimizing the "
        "lowering makes a huge class of programs faster.",
        Build("//jaxlib/tools:jax_cuda12_plugin_wheel", tuple(GPU_FLAGS), 60.0,
              None, xla_override=True),
        SpeedTest("scan_cumsum", (1024, 8192, 65536), ("f32", "f64"),
                  ("gpu", "cpu", "tpu"), "scan_vs_reference")),
    Challenge(
        "xla_while_loop_simplifier", "xla_core", "While-loop simplification",
        "xla/service/while_loop_simplifier.cc", "while-loop simplification pass",
        "jax.lax.while_loop / fori_loop / scan all lower to While; trip-count and "
        "invariant-hoisting optimizations speed up every loop-heavy program.",
        Build("//jaxlib/tools:jax_cuda12_plugin_wheel", tuple(GPU_FLAGS), 60.0,
              None, xla_override=True),
        SpeedTest("while_loop_iter", (1000, 10000, 100000), ("f32",),
                  ("gpu", "cpu", "tpu"), "while_vs_reference")),
    Challenge(
        "xla_algebraic_simplifier", "xla_core", "Algebraic simplification pass",
        "xla/hlo/transforms/simplifiers/algebraic_simplifier.cc", "the simplifier rules",
        "The biggest single optimization pass (428KB). Better/cheaper algebraic "
        "rewrites speed up compilation AND the compiled graph for nearly all programs.",
        Build("//jaxlib/tools:jax_cuda12_plugin_wheel", tuple(GPU_FLAGS), 43.0,
              None, xla_override=True),
        SpeedTest("algebra_graph", (512, 1024, 2048), ("f32",),
                  ("gpu", "cpu", "tpu"), "graph_vs_reference")),
    Challenge(
        "xla_instruction_fusion", "xla_core", "Instruction fusion",
        "xla/service/instruction_fusion.cc", "fusion decisions",
        "Fusion eliminates memory traffic for elementwise/reduction chains — the core "
        "lever for memory-bound DL graphs (activations, normalizations, softmax).",
        Build("//jaxlib/tools:jax_cuda12_plugin_wheel", tuple(GPU_FLAGS), 60.0,
              None, xla_override=True),
        SpeedTest("elementwise_chain", (1 << 20, 1 << 22, 1 << 24), ("f32",),
                  ("gpu", "tpu"), "graph_vs_reference")),
    Challenge(
        "xla_hlo_instruction", "xla_core", "Core HLO IR (hlo_instruction)",
        "xla/hlo/ir/hlo_instruction.cc", "core IR routines",
        "The IR every pass touches; micro-optimizations here speed up *compilation* of "
        "all programs. Highest fan-out (the .h is 134s/98 libs) — edit the .cc only.",
        Build("//jaxlib/tools:jax_cuda12_plugin_wheel", tuple(GPU_FLAGS), 27.0,
              None, xla_override=True),
        SpeedTest("compile_time_graph", (256, 512), ("f32",),
                  ("gpu", "cpu"), "graph_vs_reference")),
)


def challenges(tier: str | None = None) -> list[Challenge]:
    return [c for c in CHALLENGES if tier is None or c.tier == tier]


def by_id(cid: str) -> Challenge:
    for c in CHALLENGES:
        if c.id == cid:
            return c
    raise KeyError(cid)


def dump_yaml(path: str) -> int:
    import yaml
    rows = []
    for c in CHALLENGES:
        d = asdict(c)
        d["build"]["command"] = c.build.command()
        d["pytests"] = list(c.pytests)
        d["vals"] = list(c.vals)
        rows.append(d)
    with open(path, "w") as f:
        yaml.safe_dump({"version": 2, "n_challenges": len(rows),
                        "tiers": {"jaxlib_kernel": len(challenges("jaxlib_kernel")),
                                  "xla_core": len(challenges("xla_core"))},
                        "challenges": rows}, f, sort_keys=False)
    return len(rows)


if __name__ == "__main__":
    cs = challenges()
    print(f"{len(cs)} challenges  "
          f"(jaxlib_kernel={len(challenges('jaxlib_kernel'))}, xla_core={len(challenges('xla_core'))})")
    for c in cs:
        print(f"\n[{c.tier:13}] {c.id}")
        print(f"  file:  {c.file}")
        print(f"  build: {c.build.command()[:96]}...")
        print(f"  test:  {c.test.workload}  devices={c.test.devices}  rebuild~{c.build.rebuild_s}s")
