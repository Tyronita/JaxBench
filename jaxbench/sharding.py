"""Multi-GPU sharding evaluation for batched linear algebra.

Many linalg ops are applied to *batches* of matrices; JAX shards the batch across
devices and runs the per-matrix kernel locally on each. JaxBench measures the sharded
throughput so a kernel optimisation is validated to hold under data-parallel sharding
(and to surface any host-sync / collective regressions).

Only tasks with `shardable=True` in the registry get a sharded variant.
"""
from __future__ import annotations
import statistics, time
from dataclasses import dataclass

from . import ops as _ops


@dataclass
class ShardedMeasurement:
    op: str
    dtype: str
    n: int
    batch: int
    n_devices: int
    latency_ms: float
    per_device_throughput: float


def measure_sharded(op: str, n: int, dtype: str, batch: int | None = None,
                    seed: int = 0, warmup: int = 3, trials: int = 10) -> ShardedMeasurement:
    import jax, jax.numpy as jnp, numpy as np
    from jax.sharding import PositionalSharding
    devs = jax.devices("gpu")
    nd = len(devs)
    batch = batch or (nd * 8)
    sharding = PositionalSharding(devs).reshape(nd, 1, 1)

    # build a batched input (batch, n, n) and shard along the batch axis
    single = _ops.make_inputs(op, n, dtype, seed)
    a = np.stack([single["a"]] * batch) if "a" in single else None
    if a is None:
        raise ValueError(f"op {op!r} has no batched matrix input for sharding")
    a = jax.device_put(jnp.asarray(a), sharding)

    base = _ops.jax_callable(op)
    fn = jax.jit(jax.vmap(lambda m: base(a=m)))

    def call(): return jax.block_until_ready(fn(a))
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter(); call(); samples.append(time.perf_counter() - t0)
    lat = statistics.median(samples)
    return ShardedMeasurement(op, dtype, n, batch, nd, lat * 1e3,
                              per_device_throughput=batch / nd / lat)
