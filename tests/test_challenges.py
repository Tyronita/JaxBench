"""Correctness gate for every challenge in the dataset (parameterised by id).

Runs each challenge's speed-up-test workload on stock JAX (no rebuild) and checks its
residual. This is the `pytests` node referenced by each challenge's validation tuple.
CPU runs everywhere; GPU-only challenges are skipped without a GPU.
"""
import pytest

from jaxbench import challenges, challenge_runner

ALL = challenges.challenges()


def _device_for(c):
    try:
        import jax
        have = {d.platform for d in jax.devices()}
    except Exception:
        have = {"cpu"}
    for d in c.test.devices:
        if d in have:
            return d
    return None


@pytest.mark.parametrize("c", ALL, ids=[c.id for c in ALL])
def test_challenge_correct(c):
    dev = _device_for(c)
    if dev is None:
        pytest.skip(f"no available device in {c.test.devices}")
    rec = challenge_runner.evaluate(c, device=dev, do_build=False)
    cor = rec.get("correctness")
    assert cor and cor["passed"], f"{c.id} on {dev}: {rec}"


@pytest.mark.parametrize("c", ALL, ids=[c.id for c in ALL])
def test_challenge_validation_tuple(c):
    """Each challenge advertises (pytests, vals)."""
    pytests, vals = c.validation
    assert pytests and vals
    assert "speedup_vs_stock" in vals
