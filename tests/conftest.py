"""Shared pytest fixtures. Enables float64 and skips GPU tasks when no GPU is present."""
import os
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")


def _has_gpu() -> bool:
    try:
        import jax
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


HAS_GPU = _has_gpu()


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="no GPU device available")
    for item in items:
        if "gpu" in item.nodeid and not HAS_GPU:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def tasks():
    from jaxbench import registry
    return registry.tasks()
