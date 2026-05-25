"""Correctness gate: run each task's op across its (N) sweep and check residuals.

Returns the worst-case residual and a pass/fail against the dtype tolerance. This is
the gate KernelBench-style scoring requires: a fast-but-wrong kernel scores zero.
"""
from __future__ import annotations
from dataclasses import dataclass

from . import ops as _ops
from . import reference as _ref


@dataclass
class Correctness:
    task_id: str
    passed: bool
    worst_residual: float
    tol: float
    per_size: dict


def _to_host(out):
    import numpy as np, jax
    out = jax.block_until_ready(out)
    # JAX returns NamedTuple results (QRResult, SVDResult, ...); flatten to a plain
    # tuple of host arrays so references can unpack positionally.
    if isinstance(out, tuple):
        return tuple(_to_host(o) for o in out)
    if isinstance(out, list):
        return [_to_host(o) for o in out]
    return np.asarray(out)


def check_task(task, seed: int = 0) -> Correctness:
    fn = _ops.jax_callable(task.op)
    worst, per = 0.0, {}
    for n in task.sizes:
        host = _ops.make_inputs(task.op, n, task.dtype, seed)
        out = _to_host(fn(**{k: (v) for k, v in host.items()}))
        r = _ref.residual(task.op, host, out)
        per[n] = r
        worst = max(worst, r)
    return Correctness(task.id, worst <= task.tol, worst, task.tol, per)
