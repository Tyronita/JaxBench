"""Build / hot-swap smoke tests.

Skipped unless a jax checkout + bazel are present (set JAXBENCH_JAX_REPO). These verify
the targeted-rebuild recipe and that the registry's build targets are well-formed.
"""
import os
import shutil
import pytest

from jaxbench import registry, build


def _can_build() -> bool:
    return shutil.which(build.BAZEL.split("/")[-1]) is not None or os.path.exists(build.BAZEL)


def test_registry_targets_wellformed():
    for t in registry.tasks():
        assert t.build_target.startswith("//jaxlib/"), t.id
        assert t.so_path.endswith(".so"), t.id
        assert t.wheel in ("jaxlib", "jax-cuda-plugin"), t.id


def test_cuda_compiler_choice():
    # device .cu.cc kernels should use clang; host code uses default
    cu = registry.task_by_id("cholesky_update__gpu__f32")
    assert cu.cuda_compiler == "clang"
    host = registry.task_by_id("qr_pivoting__gpu__f32")
    assert host.cuda_compiler == "default"


@pytest.mark.skipif(os.environ.get("JAXBENCH_RUN_BUILD") != "1"
                    or not (_can_build() and os.path.isdir(build.JAX_REPO)),
                    reason="set JAXBENCH_RUN_BUILD=1 (needs bazel + a jax checkout)")
def test_targeted_build_smoke():
    t = registry.task_by_id("cholesky_update__gpu__f32")
    res = build.build_target(t.build_target)
    assert res.rc == 0, res.log
    assert os.path.exists(res.so_src)
