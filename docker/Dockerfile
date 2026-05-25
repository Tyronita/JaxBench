# JaxBench build+eval image, parameterised by device class.
#   DEVICE=cpu   -> jax[cpu]      (any x86 box / serverless CPU)
#   DEVICE=cuda  -> jax[cuda12]   (A100/H100 serverless GPU; needs --gpus all)
#   DEVICE=tpu   -> jax[tpu]      (GCP TPU VM / Cloud Run TPU)
#
# Two roles:
#   * EVAL image (default): runs correctness + perf for a device class. Light.
#   * BUILD image (--build-arg WITH_BUILD=1): also installs bazel + clang so it can
#     recompile a mutated kernel .so. Heavier; used by the build/hot-swap loop.
#
# Side-effect discipline: the only state a container mutates is one extension .so in
# its own filesystem; mount a warm bazel disk_cache READ-ONLY so the cold build is
# paid once globally and each invocation only does the seconds-long targeted rebuild.

ARG DEVICE=cpu
ARG BASE_CPU=python:3.12-slim
ARG BASE_CUDA=nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

# pick base by device (BuildKit selects the matching stage)
FROM ${BASE_CUDA} AS base-cuda
FROM ${BASE_CPU} AS base-cpu
FROM ${BASE_CPU} AS base-tpu

FROM base-${DEVICE} AS final
ARG DEVICE
ARG WITH_BUILD=0

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 \
    JAX_ENABLE_X64=1 HERMETIC_PYTHON_VERSION=3.12 \
    JAXBENCH_DISK_CACHE=/cache/bazel-disk JAXBENCH_JAX_REPO=/opt/jax

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates python3 python3-pip build-essential \
    && rm -rf /var/lib/apt/lists/*

# Optional build toolchain (bazelisk + clang) for the targeted-rebuild loop.
RUN if [ "$WITH_BUILD" = "1" ]; then \
      curl -fsSL -o /usr/local/bin/bazel \
        https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
      && chmod +x /usr/local/bin/bazel \
      && apt-get update && apt-get install -y --no-install-recommends clang lld \
      && rm -rf /var/lib/apt/lists/* ; \
    fi

WORKDIR /opt/jaxbench
COPY . /opt/jaxbench

# JaxBench + jax for the chosen device. (CUDA wheels self-installed; TPU via libtpu.)
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -e . \
    && case "$DEVICE" in \
         cpu)  python3 -m pip install "jax[cpu]" ;; \
         cuda) python3 -m pip install "jax[cuda12]" ;; \
         tpu)  python3 -m pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html ;; \
       esac

# warm cache + a jax checkout are mounted at runtime (read-only cache, scratch repo):
#   docker run --rm [--gpus all] \
#     -v $PWD/bazel-disk:/cache/bazel-disk:ro -v $PWD/jax:/opt/jax \
#     jaxbench:cuda python -m jaxbench.runner cholesky_update__gpu__f32 --build
ENTRYPOINT ["python3", "-m", "jaxbench.runner"]
CMD ["--help"]
