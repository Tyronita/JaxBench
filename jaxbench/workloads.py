"""Speed-up test workloads — the JAX programs that exercise each challenge's hot path.

Tier-1 (jaxlib kernel) workloads reuse `ops.py` (the op dispatches straight to the
kernel under test). Tier-2 (xla_core) workloads are whole-program JAX computations that
route through the targeted compiler pass:

  scan_cumsum       -> jax.lax.scan           (xla_scan_expander)
  while_loop_iter   -> jax.lax.while_loop     (xla_while_loop_simplifier)
  algebra_graph     -> a simplifiable graph   (xla_algebraic_simplifier)
  elementwise_chain -> fusible elementwise     (xla_instruction_fusion)
  compile_time_graph-> compilation stress      (xla_hlo_instruction)

Each workload exposes `inputs(size, dtype, seed)`, `fn()` (jitted), and a host
`reference(inputs)` so correctness is checked against a trusted result — for xla_core
that proves a compiler change didn't alter program semantics.
"""
from __future__ import annotations
import numpy as np

from . import ops as _ops

_NP = {"f32": np.float32, "f64": np.float64, "c64": np.complex64, "c128": np.complex128}

# tier-1 workload keys are op names handled by ops.py
_KERNEL_OPS = {"lu", "qr", "svd", "eigh", "cholesky", "solve", "inv", "det",
               "cholesky_update", "tridiagonal_solve", "householder_product",
               "threefry_uniform", "csr_matmul"}


def is_kernel(workload: str) -> bool:
    return workload in _KERNEL_OPS


# ---- tier-2 XLA workloads -----------------------------------------------------
def _rng(seed): return np.random.default_rng(seed)


def inputs(workload: str, size: int, dtype: str, seed: int = 0) -> dict:
    if is_kernel(workload):
        return _ops.make_inputs(workload, size, dtype, seed)
    dt = _NP[dtype]; r = _rng(seed)
    if workload == "scan_cumsum":
        return {"xs": r.standard_normal(size).astype(dt)}
    if workload == "while_loop_iter":
        return {"x0": np.asarray(1.0, dtype=dt), "n": int(size)}
    if workload in ("algebra_graph", "compile_time_graph"):
        return {"a": r.standard_normal((size,)).astype(dt)}
    if workload == "elementwise_chain":
        return {"a": r.standard_normal((size,)).astype(dt)}
    raise KeyError(workload)


def fn(workload: str):
    import jax, jax.numpy as jnp
    if is_kernel(workload):
        return _ops.jax_callable(workload)
    if workload == "scan_cumsum":
        def f(xs):
            def step(carry, x):
                carry = carry + x
                return carry, carry
            _, ys = jax.lax.scan(step, jnp.asarray(0.0, xs.dtype), xs)
            return ys
        return jax.jit(f)
    if workload == "while_loop_iter":
        def f(x0, n):
            def cond(state): i, x = state; return i < n
            def body(state): i, x = state; return (i + 1, x + jnp.cos(x))
            _, x = jax.lax.while_loop(cond, body, (0, x0))
            return x
        return jax.jit(f, static_argnums=1)
    if workload in ("algebra_graph", "compile_time_graph"):
        def f(a):  # algebra a + 0, *1, x-x etc. that the simplifier should fold
            b = a + 0.0
            b = b * 1.0
            b = b - (a - a)
            for _ in range(8):
                b = (b + a) - a
            return b + jnp.sum(a) * 0.0
        return jax.jit(f)
    if workload == "elementwise_chain":
        def f(a):  # long fusible elementwise chain (memory-bound without fusion)
            b = a
            for _ in range(16):
                b = jnp.tanh(b * 1.001 + 0.001)
            return b
        return jax.jit(f)
    raise KeyError(workload)


def reference(workload: str, inp: dict, out) -> float:
    """Relative residual of the rebuilt result vs a trusted reference (host)."""
    from . import reference as _ref
    if is_kernel(workload):
        return _ref.residual(workload, inp, out)
    out = np.asarray(out)
    if workload == "scan_cumsum":
        return _rel(out, np.cumsum(inp["xs"]))
    if workload == "while_loop_iter":
        x = float(inp["x0"])
        for _ in range(inp["n"]):
            x = x + np.cos(x)
        return _rel(out, x)
    if workload in ("algebra_graph", "compile_time_graph"):
        return _rel(out, inp["a"])             # all the algebra is identity on a
    if workload == "elementwise_chain":
        b = inp["a"].astype(np.float64)
        for _ in range(16):
            b = np.tanh(b * 1.001 + 0.001)
        return _rel(out, b)
    raise KeyError(workload)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1.0))
