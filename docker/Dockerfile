# JaxBench cross-device image, parameterised by device class.
#   DEVICE=cpu   -> jax[cpu]      (any x86 box / serverless CPU)
#   DEVICE=cuda  -> jax[cuda12]   (A100/H100; runs on slim base via pip CUDA wheels + --gpus all)
#   DEVICE=tpu   -> jax[tpu]      (GCP TPU VM / Cloud Run TPU)
#
# Roles:
#   * EVAL  (default):            run correctness + perf for a device class. Light.
#   * BUILD (--build-arg WITH_BUILD=1): + bazelisk + the jax fork checkout, so it can
#     rebuild a mutated hot path. The JAX build is HERMETIC (bazel fetches CUDA 12.9 +
#     llvm18 itself), so no nvidia/cuda base is needed — slim + jax[cuda12] runs on GPU.
#
# Side-effect discipline: a container only ever mutates one extension .so / a wheel in
# its own fs; mount a warm bazel disk_cache READ-ONLY so the cold build is paid once
# globally and each candidate does only the seconds-long targeted rebuild.

FROM python:3.12-slim AS final
ARG DEVICE=cpu
ARG WITH_BUILD=0
ARG JAX_FORK=https://github.com/Tyronita/jax.git
ARG JAX_BRANCH=shinka/targets-v1

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 \
    JAX_ENABLE_X64=1 HERMETIC_PYTHON_VERSION=3.12 \
    JAXBENCH_DISK_CACHE=/cache/bazel-disk JAXBENCH_JAX_REPO=/opt/jax \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/jaxbench
COPY . /opt/jaxbench

# JaxBench + jax for the chosen device.
RUN python3 -m pip install --upgrade pip && python3 -m pip install -e . \
    && case "$DEVICE" in \
         cpu)  python3 -m pip install "jax[cpu]" ;; \
         cuda) python3 -m pip install "jax[cuda12]" ;; \
         tpu)  python3 -m pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html ;; \
       esac

# Build variant: bazelisk + the jax fork (hermetic toolchain is fetched at build time).
RUN if [ "$WITH_BUILD" = "1" ]; then \
      curl -fsSL -o /usr/local/bin/bazel \
        https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
      && chmod +x /usr/local/bin/bazel \
      && git clone --depth 1 --branch "$JAX_BRANCH" "$JAX_FORK" /opt/jax ; \
    fi

# EVAL:  docker run --rm [--gpus all] <img> <challenge_id> --device {cpu|cuda|tpu}
# BUILD: docker run --rm --gpus all -v $PWD/bazel-disk:/cache/bazel-disk:ro \
#          <build-img> <challenge_id> --device cuda --build
ENTRYPOINT ["python3", "-m", "jaxbench.challenge_runner"]
CMD ["--help"]
