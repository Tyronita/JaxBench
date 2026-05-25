"""Evaluation metrics for JaxBench.

JaxBench scores a mutated kernel like KernelBench: a candidate counts only if it is
**correct**, and is then ranked by **speedup** over the unmutated baseline. We record
absolute latency and an op-specific GFLOP/s so results are comparable across machines.
"""
from __future__ import annotations
import math

# Approximate floating-point op counts per call, as a function of matrix dim n.
# (Leading-order; complex dtypes carry a ~4x flop multiplier applied in flops().)
_FLOPS = {
    "lu":        lambda n: 2/3 * n**3,
    "qr":        lambda n: 4/3 * n**3,
    "svd":       lambda n: 22 * n**3,
    "eigh":      lambda n: 9 * n**3,
    "solve":     lambda n: 2/3 * n**3 + 2 * n**2,
    "lu_solve":  lambda n: 2 * n**2,
    "inv":       lambda n: 2 * n**3,
    "cholesky":  lambda n: 1/3 * n**3,
    "det":       lambda n: 2/3 * n**3,
    "slogdet":   lambda n: 2/3 * n**3,
    "expm":      lambda n: 20 * n**3,
    "lstsq":     lambda n: 2 * n**3,
    "cholesky_update": lambda n: 6 * n**2,
    "lu_pivots_to_permutation": lambda n: n,
    "householder_product": lambda n: 2 * n**3,
    "qr_pivoting": lambda n: 4/3 * n**3,
    "tridiagonal_solve": lambda n: 8 * n,
    "tridiagonal_solve_perturbed": lambda n: 9 * n,
    "csr_matmul": lambda n: 2 * n * n * 0.02 * 16,
}


def flops(op: str, n: int, dtype: str) -> float:
    base = _FLOPS.get(op, lambda n: float(n))(n)
    return base * (4.0 if dtype.startswith("c") else 1.0)


def gflops(op: str, n: int, dtype: str, latency_s: float) -> float:
    if latency_s <= 0:
        return float("nan")
    return flops(op, n, dtype) / latency_s / 1e9


def speedup(baseline_s: float, candidate_s: float) -> float:
    if candidate_s <= 0:
        return float("nan")
    return baseline_s / candidate_s


# Metric metadata surfaced in the registry / docs.
DEFINITIONS = {
    "correctness_pass": "max residual over the (N x dtype) sweep is below tolerance (gate).",
    "latency_ms":       "median wall time per call, post-warmup, device-synchronised.",
    "throughput_gflops":"op-specific FLOPs / latency; machine-comparable.",
    "speedup_vs_baseline":"baseline_latency / candidate_latency (the optimisation target).",
}
