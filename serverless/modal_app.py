"""Modal deployment of JaxBench (serverless GPU + CPU fan-out).

Modal gives ephemeral A100s and CPUs with a mounted Volume for the warm bazel cache —
a clean fit for evolutionary fan-out (one candidate per function call). TPUs are not on
Modal; use GCP for the TPU device class (see serverless/README.md).

    pip install modal && modal deploy serverless/modal_app.py
    modal run serverless/modal_app.py::evaluate_gpu --task-id cholesky_update__gpu__f32
"""
from __future__ import annotations
import modal

# warm bazel disk_cache + a jax checkout persist across invocations via a Volume
cache = modal.Volume.from_name("jaxbench-bazel-cache", create_if_missing=True)
jaxsrc = modal.Volume.from_name("jaxbench-jax-src", create_if_missing=True)

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "clang", "lld", "curl")
    .run_commands("curl -fsSL -o /usr/local/bin/bazel "
                  "https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 "
                  "&& chmod +x /usr/local/bin/bazel")
    .pip_install("jax[cuda12]", "numpy", "scipy", "pyyaml")
    .add_local_dir(".", "/opt/jaxbench")
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("jax[cpu]", "numpy", "scipy", "pyyaml")
    .add_local_dir(".", "/opt/jaxbench")
)

app = modal.App("jaxbench")
ENV = {"JAXBENCH_DISK_CACHE": "/cache/bazel-disk", "JAXBENCH_JAX_REPO": "/jaxsrc/jax",
       "JAX_ENABLE_X64": "1", "HERMETIC_PYTHON_VERSION": "3.12", "PYTHONPATH": "/opt/jaxbench"}


@app.function(image=gpu_image, gpu="A100-80GB", timeout=1800,
              volumes={"/cache": cache, "/jaxsrc": jaxsrc}, env=ENV)
def evaluate_gpu(event: dict) -> dict:
    from serverless.handler import handle
    return handle(event)


@app.function(image=cpu_image, cpu=16.0, timeout=1800,
              volumes={"/cache": cache, "/jaxsrc": jaxsrc}, env=ENV)
def evaluate_cpu(event: dict) -> dict:
    from serverless.handler import handle
    return handle(event)


@app.local_entrypoint()
def main(task_id: str = "lu__cpu__f64", device: str = "cpu", build: bool = False):
    fn = evaluate_gpu if device == "gpu" else evaluate_cpu
    print(fn.remote({"task_id": task_id, "build": build}))
