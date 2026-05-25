"""Targeted rebuild + .so hot-swap — the JaxBench fast inner loop.

Validated recipe (A100, hermetic toolchain): rebuild only the one affected extension
`.so` with clang as the CUDA device compiler, then copy it over the installed plugin
extension instead of repackaging a wheel. ~13s for a device kernel vs ~21s for a full
nvcc wheel rebuild. See docs/METHODOLOGY.md.

Side-effect discipline: the build is hermetic and content-addressed (disk_cache);
the only mutation to the environment is the atomic replacement of a single `.so`,
which `restore()` reverts. Nothing else in the venv is touched.
"""
from __future__ import annotations
import os, shutil, subprocess, sys, time
from dataclasses import dataclass

JAX_REPO = os.environ.get("JAXBENCH_JAX_REPO", "/home/oleary/jax")
BAZEL = os.environ.get("JAXBENCH_BAZEL", "/usr/local/bin/bazel")
DISK_CACHE = os.environ.get("JAXBENCH_DISK_CACHE", "/data/bazel-disk")
PY_VER = os.environ.get("HERMETIC_PYTHON_VERSION", "3.12")
# Local, editable copy of the XLA source for the xla_core tier (in-place edits to the
# http_archive cache are NOT detected by bazel; --override_repository is required).
XLA_LOCAL = os.environ.get("JAXBENCH_XLA_LOCAL", "/data/xla-local")


@dataclass
class BuildResult:
    target: str
    rc: int
    wall_s: float
    elapsed_s: float | None
    so_src: str
    log: str


def _bazel_elapsed(text: str):
    import re
    m = re.search(r"Elapsed time: ([\d.]+)s", text)
    return float(m.group(1)) if m else None


def ensure_xla_local() -> str:
    """Copy the bazel-fetched XLA source to an editable location (once)."""
    import glob
    if os.path.isdir(XLA_LOCAL):
        return XLA_LOCAL
    cand = glob.glob(os.path.expanduser("~/.cache/bazel/**/external/xla", recursive=True)) \
        or glob.glob("/mnt/bazel/**/external/xla", recursive=True)
    if not cand:
        raise FileNotFoundError("fetched external/xla not found; run a jax build first")
    shutil.copytree(cand[0], XLA_LOCAL, symlinks=True)
    return XLA_LOCAL


def stage_xla_candidate(xla_relpath: str, source: str) -> str:
    """Write a candidate XLA source into the editable copy; return backup path."""
    ensure_xla_local()
    dest = os.path.join(XLA_LOCAL, xla_relpath)
    backup = dest + ".jaxbench.bak"
    if not os.path.exists(backup):
        shutil.copy2(dest, backup)
    with open(dest, "w") as f:
        f.write(source)
    return dest


def restore_xla(xla_relpath: str) -> None:
    dest = os.path.join(XLA_LOCAL, xla_relpath)
    backup = dest + ".jaxbench.bak"
    if os.path.exists(backup):
        shutil.copy2(backup, dest)


def build_target(target: str, *, clang_device: bool = True,
                 libs_from_stubs: bool = True, xla_override: bool = False,
                 log_path: str | None = None) -> BuildResult:
    """Rebuild a target. jaxlib tier -> extension .so. xla_core tier (xla_override)
    -> the plugin wheel built against the editable XLA copy (reinstall to apply)."""
    cmd = [BAZEL, "build", f"--repo_env=HERMETIC_PYTHON_VERSION={PY_VER}",
           f"--disk_cache={DISK_CACHE}", "--features=-layering_check"]
    if libs_from_stubs and ("//jaxlib/cuda" in target or "plugin" in target):
        cmd.append("--config=cuda_libraries_from_stubs")
    if clang_device and "//jaxlib/cuda" in target:
        cmd.append("--config=build_cuda_with_clang")  # ~25-36% faster device compiles
    if xla_override:
        ensure_xla_local()
        cmd.append(f"--override_repository=xla={XLA_LOCAL}")
    cmd.append(target)

    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=JAX_REPO, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True,
                       env={**os.environ, "HERMETIC_PYTHON_VERSION": PY_VER})
    wall = time.monotonic() - t0
    if log_path:
        with open(log_path, "w") as f:
            f.write(" ".join(cmd) + "\n\n" + p.stdout)
    # the extension .so lands at bazel-bin/<pkg>/<name>.so
    pkg = target.split("//", 1)[1].split(":")[0]
    name = target.split(":")[1]
    so_src = os.path.join(JAX_REPO, "bazel-bin", pkg, f"{name}.so")
    return BuildResult(target, p.returncode, wall, _bazel_elapsed(p.stdout), so_src, p.stdout[-4000:])


def plugin_dir(import_name: str) -> str:
    """Locate the installed package dir (e.g. jax_cuda12_plugin) in the active venv."""
    import importlib, os
    mod = importlib.import_module(import_name.split("/")[0])
    return os.path.dirname(mod.__file__)


def hot_swap(so_src: str, so_rel: str) -> str:
    """Copy a freshly built .so over the installed extension. Returns backup path."""
    base = so_rel.split("/")[0]
    dest_dir = plugin_dir(base)
    dest = os.path.join(dest_dir, os.path.basename(so_rel))
    backup = dest + ".jaxbench.bak"
    if not os.path.exists(backup) and os.path.exists(dest):
        shutil.copy2(dest, backup)
    shutil.copy2(so_src, dest)  # atomic-enough single-file replace
    return dest


def restore(so_rel: str) -> None:
    base = so_rel.split("/")[0]
    dest = os.path.join(plugin_dir(base), os.path.basename(so_rel))
    backup = dest + ".jaxbench.bak"
    if os.path.exists(backup):
        shutil.copy2(backup, dest)
