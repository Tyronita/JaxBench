"""Correctness of the xla_core speed-up-test workloads (scan / while / algebra / fusion).

These verify a candidate XLA graph-pass change preserves program semantics: the JAX
workload that routes through the pass must still match its reference. Run on stock JAX
here (no rebuild); the rebuilt-JAX check happens in the challenge runner / serverless.
"""
import os
import numpy as np
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")
pytest.importorskip("jax")

from jaxbench import challenges, workloads


XLA = challenges.challenges("xla_core")


@pytest.mark.parametrize("c", XLA, ids=[c.id for c in XLA])
def test_xla_workload_semantics(c):
    wl = c.test.workload
    dtype = c.test.dtypes[0]
    worst = 0.0
    for n in c.test.sizes:
        inp = workloads.inputs(wl, n, dtype, seed=1)
        out = workloads.fn(wl)(**inp)
        import jax
        out = jax.block_until_ready(out)
        worst = max(worst, workloads.reference(wl, inp, np.asarray(out)))
    assert worst < 1e-3, f"{c.id} workload {wl}: residual {worst:.3e}"
