"""Latency/throughput measurement with correct JAX timing hygiene.

Pitfalls handled: async dispatch (we `block_until_ready`), JIT warmup (excluded),
input transfer (done once, outside the loop), and run-to-run noise (median of trials).
"""
from __future__ import annotations
import statistics, time
from dataclasses import dataclass, asdict

from . import ops as _ops
from . import metrics as _metrics


@dataclass
class Measurement:
    op: str
    device: str
    dtype: str
    n: int
    latency_ms: float
    gflops: float
    trials: int


def _to_device(inp: dict, device: str):
    import jax, jax.numpy as jnp
    dev = jax.devices(device)[0]
    out = {}
    for k, v in inp.items():
        if hasattr(v, "shape") or isinstance(v, (list, tuple)) and k not in ("shape",):
            try:
                out[k] = jax.device_put(jnp.asarray(v), dev)
                continue
            except Exception:
                pass
        out[k] = v
    return out


def measure(op: str, n: int, dtype: str, device: str = "gpu",
            seed: int = 0, warmup: int = 3, trials: int = 20) -> Measurement:
    """Time one op at one size; returns median latency + GFLOP/s."""
    import jax
    fn = _ops.jax_callable(op)
    host = _ops.make_inputs(op, n, dtype, seed)
    dev_inp = _to_device(host, device)

    def call():
        out = fn(**dev_inp)
        return jax.block_until_ready(out)

    for _ in range(warmup):
        call()
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        call()
        samples.append(time.perf_counter() - t0)
    lat = statistics.median(samples)
    return Measurement(op, device, dtype, n, lat * 1e3,
                       _metrics.gflops(op, n, dtype, lat), trials)


def sweep(task, seed: int = 0, **kw) -> list[Measurement]:
    return [measure(task.op, n, task.dtype, task.device, seed, **kw) for n in task.sizes]
